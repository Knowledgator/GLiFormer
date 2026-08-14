#!/usr/bin/env python3
"""Evaluate a GLiNExT structuring checkpoint on the text2json task.

The evaluator uses GLiNExT's native ``model.structure`` API.  Multi-level
checkpoints receive the complete recursive JSON template.  Flat checkpoints
receive dot-qualified leaf paths and their output is re-nested, which mirrors
the flat-schema GLiNER2 baseline used for this dataset.

The Hugging Face token is read from ``HF_TOKEN`` or
``HUGGING_FACE_HUB_TOKEN``; credentials are never embedded in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from huggingface_hub import hf_hub_download

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

EVAL_DATASET_REPO = "knowledgator/text2json-training-data"
EVAL_DATASET_FILENAME = "top_1000_en_annotated_changed_eval.jsonl"
DEFAULT_OUTPUT_DIR = Path("eval_results_glinext_text2json")
STRUCTURE_NAME = "record"

SIMILARITY_THRESHOLD = 0.5
DEPTH_DECAY = 2.0
LEAF_WEIGHT = 3.0
WL_ITERATIONS = 3
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOGGER = logging.getLogger("eval_glinext_text2json")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_type_string(value: Any) -> str:
    if value is None:
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "string"


def convert_to_json_template(data: Any) -> Any:
    """Build a recursive typed template from one target JSON value."""

    if isinstance(data, dict):
        return {str(key): convert_to_json_template(value) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_to_json_template(data[0])] if data else ["string"]
    return get_type_string(data)


_TYPE_ALIASES = {
    "any": "string",
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "date": "date",
    "datetime": "datetime",
}


def normalize_template(template: Any) -> Any:
    """Normalize common text2json type spellings for GLiNExT."""

    if isinstance(template, dict):
        if _is_field_descriptor(template):
            normalized = dict(template)
            raw_type = normalized.get("$type")
            if isinstance(raw_type, str):
                normalized["$type"] = _TYPE_ALIASES.get(
                    raw_type.strip().casefold().strip("<>"), "string"
                )
            if "$items" in normalized:
                normalized["$items"] = normalize_template(normalized["$items"])
            return normalized
        normalized = {}
        for key, value in template.items():
            # ``$required`` contains literal field names, not type specs.
            normalized[str(key)] = value if key == "$required" else normalize_template(value)
        return normalized
    if isinstance(template, list):
        return [normalize_template(template[0])] if template else ["string"]
    if isinstance(template, str):
        candidate = template.strip().casefold().strip("<>")
        return _TYPE_ALIASES.get(candidate, "string")
    return get_type_string(template)


def parse_json_value(value: Any) -> Any | None:
    """Parse JSON strings defensively while accepting native containers."""

    parsed = value
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            return None
    return parsed


def prepare_eval_records(
    rows: list[dict[str, Any]],
    *,
    num_samples: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Validate source rows and create typed per-example schemas."""

    rows = list(rows)
    if shuffle:
        random.Random(seed).shuffle(rows)
    if num_samples is not None:
        rows = rows[:num_samples]

    prepared: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        text = item.get("text")
        raw_output = item.get("output")
        if raw_output is None:
            raw_output = item.get("extracted")
        solution = parse_json_value(raw_output)
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(solution, (dict, list)):
            continue

        raw_template = item.get("template")
        template = parse_json_value(raw_template) if isinstance(raw_template, str) else raw_template
        if template is None:
            template = convert_to_json_template(solution)
        if not isinstance(template, (dict, list)):
            LOGGER.warning("Skipping row %d with a non-container template", index)
            continue
        prepared.append(
            {
                "text": text,
                "template": normalize_template(template),
                "solution": solution,
            }
        )
    return prepared


def load_eval_data(
    *,
    repo: str,
    filename: str,
    data_path: Path | None,
    token: str | None,
    num_samples: int | None,
    seed: int,
    shuffle: bool,
) -> list[dict[str, Any]]:
    if data_path is None:
        LOGGER.info("Downloading evaluation data: %s/%s", repo, filename)
        data_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                token=token or None,
            )
        )
    rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at line %d: %s", line_number, exc)
                continue
            if isinstance(row, dict):
                rows.append(row)
    prepared = prepare_eval_records(
        rows,
        num_samples=num_samples,
        seed=seed,
        shuffle=shuffle,
    )
    LOGGER.info("Prepared %d/%d evaluation examples", len(prepared), len(rows))
    return prepared


def is_oom_error(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _is_field_descriptor(value: Any) -> bool:
    return isinstance(value, dict) and bool(set(value) & {"$type", "$items", "$enum"})


def build_flat_fields(template: Any) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []

    def walk(value: Any, prefix: str, under_list: bool) -> None:
        if not isinstance(value, dict) or _is_field_descriptor(value):
            return
        for raw_key, child in value.items():
            if raw_key == "$required":
                continue
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict) and not _is_field_descriptor(child):
                walk(child, path, under_list)
            elif isinstance(child, list):
                if child and isinstance(child[0], dict) and not _is_field_descriptor(child[0]):
                    walk(child[0], path, True)
                else:
                    item_type = child[0] if child else "string"
                    fields.append(
                        {"path": path, "name": path, "is_list": True, "type": item_type}
                    )
            else:
                fields.append(
                    {
                        "path": path,
                        "name": path,
                        "is_list": under_list,
                        "type": child,
                    }
                )

    if isinstance(template, dict):
        walk(template, "", False)
    elif isinstance(template, list) and template and isinstance(template[0], dict):
        walk(template[0], "", True)
    return fields


def build_flat_schema(fields: list[dict[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field in fields:
        field_type = normalize_template(field["type"])
        record[field["name"]] = [field_type] if field["is_list"] else field_type
    return {STRUCTURE_NAME: record}


def build_native_schema(template: Any) -> Any:
    if isinstance(template, list):
        return template
    if isinstance(template, dict):
        return {"$root": template}
    raise TypeError("A text2json template must be an object or list")


def flatten_json_by_path(obj: Any) -> dict[str, list[Any]]:
    output: dict[str, list[Any]] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, (dict, list)):
                    walk(child, path)
                else:
                    output.setdefault(path, []).append(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child, prefix)
                else:
                    output.setdefault(prefix, []).append(child)
        elif prefix:
            output.setdefault(prefix, []).append(value)

    walk(obj, "")
    return output


def unflatten_by_path(flat: dict[str, list[Any]], fields: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        parts = field["path"].split(".")
        values = flat.get(field["path"], [])
        cursor = output
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = values if field["is_list"] else (values[0] if values else None)
    return output


def extract_flat(
    model: Any,
    text: str,
    fields: list[dict[str, Any]],
    **inference_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    raw = model.structure(text, build_flat_schema(fields), **inference_kwargs)
    instances = raw.get(STRUCTURE_NAME, []) if isinstance(raw, dict) else []
    record = instances[0] if instances and isinstance(instances[0], dict) else {}
    flat: dict[str, list[Any]] = {}
    for field in fields:
        value = record.get(field["name"])
        if field["is_list"]:
            flat[field["path"]] = value if isinstance(value, list) else ([] if value is None else [value])
        else:
            flat[field["path"]] = [] if value is None else [value]
    return unflatten_by_path(flat, fields), flat


def extract_native(
    model: Any,
    text: str,
    template: Any,
    **inference_kwargs: Any,
) -> tuple[Any, dict[str, list[Any]]]:
    prediction = model.structure(text, build_native_schema(template), **inference_kwargs)
    return prediction, flatten_json_by_path(prediction)


def model_supports_multi_level(model: Any) -> bool:
    config = getattr(model, "config", None)
    for name in ("structuring_config", "set_structuring_config"):
        task_config = getattr(config, name, None)
        if task_config is None:
            continue
        if bool(getattr(task_config, "multi_level", False)):
            return True
        mode = getattr(task_config, "structure_mode", None)
        if bool(getattr(mode, "is_multi_level", False)):
            return True
        if isinstance(mode, str) and mode == "multi_level":
            return True
    return False


_EMBEDDING_MODEL: dict[str, Any] = {"model": None, "name": None, "device": None}
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}
_EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL
_EMBEDDING_DEVICE = "cpu"


def configure_embedding_model(name: str, device: str) -> None:
    global _EMBEDDING_MODEL_NAME, _EMBEDDING_DEVICE
    _EMBEDDING_MODEL_NAME = name
    _EMBEDDING_DEVICE = device


def get_embedding_model() -> Any:
    if (
        _EMBEDDING_MODEL["model"] is None
        or _EMBEDDING_MODEL["name"] != _EMBEDDING_MODEL_NAME
        or _EMBEDDING_MODEL["device"] != _EMBEDDING_DEVICE
    ):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(_EMBEDDING_MODEL_NAME, device=_EMBEDDING_DEVICE)
        model.eval()
        _EMBEDDING_MODEL.update(
            {"model": model, "name": _EMBEDDING_MODEL_NAME, "device": _EMBEDDING_DEVICE}
        )
        _EMBEDDING_CACHE.clear()
    return _EMBEDDING_MODEL["model"]


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    missing = list(dict.fromkeys(text for text in texts if text not in _EMBEDDING_CACHE))
    if missing:
        with torch.inference_mode():
            embeddings = get_embedding_model().encode(
                missing,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=False,
            )
        for text, embedding in zip(missing, embeddings, strict=True):
            _EMBEDDING_CACHE[text] = np.asarray(embedding, dtype=np.float32)
    return np.stack([_EMBEDDING_CACHE[text] for text in texts])


def build_tree(obj: Any) -> tuple[dict[int, dict[str, Any]], list[tuple[int, int]]]:
    nodes: dict[int, dict[str, Any]] = {}
    edges: list[tuple[int, int]] = []

    def add(label: Any, value: str | None = None, leaf: bool = False, kind: str = "key") -> int:
        node_id = len(nodes)
        nodes[node_id] = {
            "label": str(label),
            "value_str": value,
            "is_leaf": leaf,
            "node_type": kind,
        }
        return node_id

    def scalar(value: Any) -> str:
        if value is None:
            return "__null__"
        if isinstance(value, bool):
            return "__true__" if value else "__false__"
        return str(value)

    def walk(value: Any, parent: int | None = None) -> None:
        if isinstance(value, dict):
            if parent is None:
                parent = add("__root__", kind="root")
            for key, child in value.items():
                key_id = add(key)
                edges.append((parent, key_id))
                if isinstance(child, (dict, list)):
                    walk(child, key_id)
                else:
                    text = scalar(child)
                    value_id = add(text, value=text, leaf=True, kind="value")
                    edges.append((key_id, value_id))
        elif isinstance(value, list):
            if parent is None:
                parent = add("__root__", kind="root")
            for index, child in enumerate(value):
                item_id = add(f"[{index}]", kind="list_idx")
                edges.append((parent, item_id))
                if isinstance(child, (dict, list)):
                    walk(child, item_id)
                else:
                    text = scalar(child)
                    value_id = add(text, value=text, leaf=True, kind="value")
                    edges.append((item_id, value_id))
        else:
            text = scalar(value)
            value_id = add(text, value=text, leaf=True, kind="value")
            if parent is not None:
                edges.append((parent, value_id))

    walk(obj)
    return nodes, edges


def compute_depths(nodes: dict[int, Any], edges: list[tuple[int, int]]) -> dict[int, int]:
    adjacency = {node_id: [] for node_id in nodes}
    for parent, child in edges:
        adjacency[parent].append(child)
    children = {child for _, child in edges}
    roots = [node_id for node_id in nodes if node_id not in children]
    if not roots:
        return {node_id: 0 for node_id in nodes}
    depths = {node_id: 0 for node_id in nodes}
    queue = deque([roots[0]])
    visited = {roots[0]}
    while queue:
        node_id = queue.popleft()
        for child in adjacency[node_id]:
            if child not in visited:
                visited.add(child)
                depths[child] = depths[node_id] + 1
                queue.append(child)
    return depths


def build_adjacency(nodes: dict[int, Any], edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    adjacency = {node_id: [] for node_id in nodes}
    for parent, child in edges:
        adjacency[parent].append(child)
        adjacency[child].append(parent)
    return adjacency


def initial_embeddings(nodes: dict[int, dict[str, Any]]) -> dict[int, np.ndarray]:
    texts = [
        node["value_str"] if node["is_leaf"] and node["value_str"] else node["label"]
        for node in nodes.values()
    ]
    unique = list(dict.fromkeys(texts))
    matrix = embed_texts(unique)
    lookup = {text: matrix[index] for index, text in enumerate(unique)}
    return {node_id: lookup[texts[index]].copy() for index, node_id in enumerate(nodes)}


def wl_aggregate(
    embeddings: dict[int, np.ndarray],
    adjacency: dict[int, list[int]],
    iterations: int = WL_ITERATIONS,
) -> dict[int, np.ndarray]:
    current = {key: value.copy() for key, value in embeddings.items()}
    for _ in range(iterations):
        updated: dict[int, np.ndarray] = {}
        for node_id, embedding in current.items():
            neighbors = adjacency.get(node_id, [])
            combined = (
                embedding + np.mean([current[neighbor] for neighbor in neighbors], axis=0)
                if neighbors
                else embedding.copy()
            )
            norm = np.linalg.norm(combined)
            updated[node_id] = combined / norm if norm > 1e-9 else combined
        current = updated
    return current


def bipartite_match(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = similarity.shape
    matched_rows = np.zeros(rows, dtype=bool)
    matched_columns = np.zeros(columns, dtype=bool)
    forward = np.zeros(rows)
    backward = np.zeros(columns)
    for flat_index in np.argsort(similarity.ravel(), kind="stable")[::-1]:
        row, column = divmod(int(flat_index), columns)
        if not matched_rows[row] and not matched_columns[column]:
            score = similarity[row, column]
            if score < SIMILARITY_THRESHOLD:
                break
            forward[row] = score
            backward[column] = score
            matched_rows[row] = True
            matched_columns[column] = True
    return forward, backward


def _subscore(matrix: np.ndarray, pred_ids: list[int], target_ids: list[int]) -> float:
    if not pred_ids and not target_ids:
        return 1.0
    if not pred_ids or not target_ids:
        return 0.0
    forward, backward = bipartite_match(matrix[np.ix_(pred_ids, target_ids)])
    precision = forward.sum() / len(pred_ids)
    recall = backward.sum() / len(target_ids)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def compute_wl_graph_score(prediction: Any, target: Any) -> dict[str, float]:
    pred_nodes, pred_edges = build_tree(prediction)
    target_nodes, target_edges = build_tree(target)
    if not pred_nodes and not target_nodes:
        return dict(precision=1.0, recall=1.0, f1=1.0, structural_score=1.0, semantic_score=1.0)
    if not pred_nodes or not target_nodes:
        return dict(precision=0.0, recall=0.0, f1=0.0, structural_score=0.0, semantic_score=0.0)

    pred_embeddings = wl_aggregate(initial_embeddings(pred_nodes), build_adjacency(pred_nodes, pred_edges))
    target_embeddings = wl_aggregate(
        initial_embeddings(target_nodes), build_adjacency(target_nodes, target_edges)
    )
    pred_depths = compute_depths(pred_nodes, pred_edges)
    target_depths = compute_depths(target_nodes, target_edges)
    pred_ids = sorted(pred_nodes)
    target_ids = sorted(target_nodes)
    pred_matrix = np.asarray([pred_embeddings[node_id] for node_id in pred_ids])
    target_matrix = np.asarray([target_embeddings[node_id] for node_id in target_ids])
    pred_depth = np.asarray([pred_depths[node_id] for node_id in pred_ids], dtype=float)
    target_depth = np.asarray([target_depths[node_id] for node_id in target_ids], dtype=float)
    pred_weights = np.asarray([LEAF_WEIGHT if pred_nodes[node_id]["is_leaf"] else 1.0 for node_id in pred_ids])
    target_weights = np.asarray([LEAF_WEIGHT if target_nodes[node_id]["is_leaf"] else 1.0 for node_id in target_ids])
    similarities = pred_matrix @ target_matrix.T
    max_depth = max(float(pred_depth.max()), float(target_depth.max()), 1.0)
    penalized = similarities * np.exp(
        -DEPTH_DECAY * np.abs(pred_depth[:, None] - target_depth[None, :]) / max_depth
    )
    forward, backward = bipartite_match(penalized)
    precision = float(np.sum(forward * pred_weights) / np.sum(pred_weights))
    recall = float(np.sum(backward * target_weights) / np.sum(target_weights))
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))
    pred_leaf = [index for index, node_id in enumerate(pred_ids) if pred_nodes[node_id]["is_leaf"]]
    target_leaf = [index for index, node_id in enumerate(target_ids) if target_nodes[node_id]["is_leaf"]]
    pred_key = [index for index in range(len(pred_ids)) if index not in pred_leaf]
    target_key = [index for index in range(len(target_ids)) if index not in target_leaf]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "structural_score": _subscore(penalized, pred_key, target_key),
        "semantic_score": _subscore(penalized, pred_leaf, target_leaf),
    }


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            current[index] = (
                previous[index - 1] + 1
                if left_token == right_token
                else max(current[index - 1], previous[index])
            )
        previous = current
    return previous[-1]


def compute_rouge_l(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    length = lcs_length(pred_tokens, ref_tokens)
    precision = length / len(pred_tokens)
    recall = length / len(ref_tokens)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def extract_leaf_values(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [leaf for value in obj.values() for leaf in extract_leaf_values(value)]
    if isinstance(obj, list):
        return [leaf for value in obj for leaf in extract_leaf_values(value)]
    return [obj]


def compute_attribution(prediction: Any, source_text: str) -> dict[str, Any]:
    leaves = [value for value in extract_leaf_values(prediction) if value is not None]
    if not leaves:
        return {"attribution_score": 1.0, "grounded": 0, "total": 0, "ungrounded": 0}
    folded_text = source_text.casefold()
    grounded = 0
    for value in leaves:
        value_text = str(value).casefold().strip()
        if not value_text or isinstance(value, bool) or value_text in folded_text:
            grounded += 1
        elif isinstance(value, (int, float)):
            grounded += int(str(value) in source_text)
        else:
            tokens = value_text.split()
            grounded += int(bool(tokens) and sum(token in folded_text for token in tokens) / len(tokens) >= 0.75)
    return {
        "attribution_score": grounded / len(leaves),
        "grounded": grounded,
        "total": len(leaves),
        "ungrounded": len(leaves) - grounded,
    }


def semantic_value_score(pred_values: list[Any], target_values: list[Any]) -> dict[str, float]:
    pred_strings = [str(value) for value in pred_values if value is not None]
    target_strings = [str(value) for value in target_values if value is not None]
    if not pred_strings and not target_strings:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_strings or not target_strings:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    unique = list(dict.fromkeys(pred_strings + target_strings))
    matrix = embed_texts(unique)
    lookup = {text: matrix[index] for index, text in enumerate(unique)}
    pred_matrix = np.asarray([lookup[text] for text in pred_strings])
    target_matrix = np.asarray([lookup[text] for text in target_strings])
    forward, backward = bipartite_match(pred_matrix @ target_matrix.T)
    precision = float(forward.sum() / len(pred_strings))
    recall = float(backward.sum() / len(target_strings))
    return {
        "precision": precision,
        "recall": recall,
        "f1": float(2 * precision * recall / (precision + recall + 1e-9)),
    }


def compute_flat_value_score(prediction: Any, target: Any) -> dict[str, float]:
    return semantic_value_score(extract_leaf_values(prediction), extract_leaf_values(target))


def compute_flat_field_score(
    prediction: dict[str, list[Any]], target: dict[str, list[Any]]
) -> dict[str, float]:
    pred_count = target_count = 0
    forward_total = backward_total = 0.0
    for path in set(prediction) | set(target):
        pred_values = [value for value in prediction.get(path, []) if value is not None]
        target_values = [value for value in target.get(path, []) if value is not None]
        pred_count += len(pred_values)
        target_count += len(target_values)
        if not pred_values or not target_values:
            continue
        score = semantic_value_score(pred_values, target_values)
        forward_total += score["precision"] * len(pred_values)
        backward_total += score["recall"] * len(target_values)
    precision = forward_total / pred_count if pred_count else float(target_count == 0)
    recall = backward_total / target_count if target_count else float(pred_count == 0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-9),
    }


METRIC_KEYS = [
    "graph_precision",
    "graph_recall",
    "graph_f1",
    "structural_score",
    "semantic_score",
    "rouge_l_f1",
    "attribution_score",
    "flat_value_precision",
    "flat_value_recall",
    "flat_value_f1",
    "flat_field_precision",
    "flat_field_recall",
    "flat_field_f1",
]


def evaluate_single(
    prediction: Any,
    prediction_by_path: dict[str, list[Any]],
    solution: Any,
    source_text: str,
) -> dict[str, float]:
    metrics = {key: 0.0 for key in METRIC_KEYS}
    prediction_text = json.dumps(prediction, ensure_ascii=False) if prediction is not None else ""
    solution_text = json.dumps(solution, ensure_ascii=False)
    metrics["rouge_l_f1"] = compute_rouge_l(prediction_text, solution_text)["f1"]
    if prediction is None:
        return metrics
    graph = compute_wl_graph_score(prediction, solution)
    metrics.update(
        {
            "graph_precision": graph["precision"],
            "graph_recall": graph["recall"],
            "graph_f1": graph["f1"],
            "structural_score": graph["structural_score"],
            "semantic_score": graph["semantic_score"],
        }
    )
    value_score = compute_flat_value_score(prediction, solution)
    field_score = compute_flat_field_score(prediction_by_path, flatten_json_by_path(solution))
    metrics.update(
        {
            "flat_value_precision": value_score["precision"],
            "flat_value_recall": value_score["recall"],
            "flat_value_f1": value_score["f1"],
            "flat_field_precision": field_score["precision"],
            "flat_field_recall": field_score["recall"],
            "flat_field_f1": field_score["f1"],
            "attribution_score": compute_attribution(prediction, source_text)["attribution_score"],
        }
    )
    return metrics


def walk_json(
    obj: Any,
    keys: list[tuple[str, int]] | None = None,
    leaves: list[Any] | None = None,
    depth: int = 0,
    max_depth: list[int] | None = None,
) -> tuple[list[tuple[str, int]], list[Any], int]:
    keys = [] if keys is None else keys
    leaves = [] if leaves is None else leaves
    max_depth = [0] if max_depth is None else max_depth
    max_depth[0] = max(max_depth[0], depth)
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.append((str(key), depth))
            if isinstance(value, (dict, list)):
                walk_json(value, keys, leaves, depth + 1, max_depth)
            else:
                leaves.append(value)
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                walk_json(value, keys, leaves, depth + 1, max_depth)
            else:
                leaves.append(value)
    else:
        leaves.append(obj)
    return keys, leaves, max_depth[0]


def compute_diagnostics(item: dict[str, Any], prediction: Any) -> dict[str, Any]:
    template_keys, template_leaves, template_depth = walk_json(item["template"])
    target_keys, target_leaves, target_depth = walk_json(item["solution"])
    diagnostics = {
        "template_depth": template_depth,
        "pred_depth": 0,
        "gt_depth": target_depth,
        "depth_mismatch": template_depth,
        "n_template_keys": len(template_keys),
        "n_pred_keys": 0,
        "n_gt_keys": len(target_keys),
        "n_missing_keys": len({key for key, _ in template_keys}),
        "n_extra_keys": 0,
        "key_exact_match_rate": 0.0,
        "n_template_leaves": len(template_leaves),
        "n_pred_leaves": 0,
        "n_gt_leaves": len(target_leaves),
        "n_null_leaves_pred": 0,
        "null_rate": 0.0,
    }
    if prediction is None:
        return diagnostics
    pred_keys, pred_leaves, pred_depth = walk_json(prediction)
    template_key_set = {key for key, _ in template_keys}
    pred_key_set = {key for key, _ in pred_keys}
    null_count = sum(value is None for value in pred_leaves)
    diagnostics.update(
        {
            "pred_depth": pred_depth,
            "depth_mismatch": abs(pred_depth - template_depth),
            "n_pred_keys": len(pred_keys),
            "n_missing_keys": len(template_key_set - pred_key_set),
            "n_extra_keys": len(pred_key_set - template_key_set),
            "key_exact_match_rate": (
                len(template_key_set & pred_key_set) / len(template_key_set)
                if template_key_set
                else 1.0
            ),
            "n_pred_leaves": len(pred_leaves),
            "n_null_leaves_pred": null_count,
            "null_rate": null_count / len(pred_leaves) if pred_leaves else 0.0,
        }
    )
    return diagnostics


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def load_model(
    model_path: str,
    *,
    device: str,
    dtype: str | None,
    local_files_only: bool,
) -> Any:
    from glinext import GLiNExT

    resolved_dtype = getattr(torch, dtype) if dtype else None
    model = GLiNExT.from_pretrained(
        model_path,
        map_location="cpu",
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        load_tokenizer=True,
    )
    model = model.to(device)
    model.eval()
    return model


def _model_key(model_path: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_path).strip("._") or "model"
    digest = hashlib.sha256(model_path.encode()).hexdigest()[:8]
    return f"{readable[-100:]}__{digest}"


def _bucket(value: float, edges: list[float], labels: list[str]) -> str:
    for index, edge in enumerate(edges):
        if value < edge:
            return labels[index]
    return labels[-1]


def build_segments(results: list[dict[str, Any]]) -> dict[str, Any]:
    def segment(edges: list[float], labels: list[str], field: str) -> dict[str, Any]:
        buckets = {label: [] for label in labels}
        for result in results:
            label = _bucket(result["diagnostics"].get(field, 0), edges, labels)
            buckets[label].append(result["metrics"]["flat_field_f1"])
        return {
            label: {"n": len(values), "flat_field_f1_mean": safe_mean(values)}
            for label, values in buckets.items()
        }

    return {
        "by_template_depth": segment([2, 3, 4], ["d<2", "d=2", "d=3", "d>=4"], "template_depth"),
        "by_n_template_leaves": segment([5, 15, 40], ["<5", "5-14", "15-39", ">=40"], "n_template_leaves"),
        "by_null_rate": segment([0.25, 0.5, 0.75], ["<0.25", "0.25-0.49", "0.5-0.74", ">=0.75"], "null_rate"),
    }


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["aggregate_metrics"]
    width = 59
    print("\n" + "=" * width)
    print("GLiNExT TEXT2JSON EVALUATION")
    print("=" * width)
    print(f"  Model:              {report['model']}")
    print(f"  Device:             {report['device']}")
    print(f"  Schema mode:        {report['schema_mode']}")
    print(f"  Samples:            {report['num_samples']}")
    print(f"  Avg fields/schema:  {report['avg_schema_fields']:.1f}")
    print(f"  Output hash:        {report['output_hash']}")
    print(f"  Total time:         {report['total_time_s']:.1f}s ({report['samples_per_second']:.2f} samples/s)")
    print(f"  Extract errors:     {report['num_extract_errors']}")
    print(f"  Of which OOM:       {report['num_oom_errors']}")
    print("-" * width)
    print("WL GRAPH METRIC")
    print(f"  Precision:          {metrics['graph_precision']:.4f}")
    print(f"  Recall:             {metrics['graph_recall']:.4f}")
    print(f"  F1:                 {metrics['graph_f1']:.4f}")
    print(f"  Structural Score:   {metrics['structural_score']:.4f}")
    print(f"  Semantic Score:     {metrics['semantic_score']:.4f}")
    print("-" * width)
    print("FLAT FIELD METRIC (matched by schema path)")
    print(f"  Precision:          {metrics['flat_field_precision']:.4f}")
    print(f"  Recall:             {metrics['flat_field_recall']:.4f}")
    print(f"  F1:                 {metrics['flat_field_f1']:.4f}")
    print("-" * width)
    print("FLAT VALUE METRIC (global bag)")
    print(f"  Precision:          {metrics['flat_value_precision']:.4f}")
    print(f"  Recall:             {metrics['flat_value_recall']:.4f}")
    print(f"  F1:                 {metrics['flat_value_f1']:.4f}")
    print("-" * width)
    print(f"ROUGE-L F1:           {metrics['rouge_l_f1']:.4f}")
    print(f"ATTRIBUTION:          {metrics['attribution_score']:.4f}")
    print("=" * width)


def evaluate_model(
    model_path: str,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seeds(args.seed)
    LOGGER.info("Loading GLiNExT model: %s -> %s", model_path, args.device)
    model = load_model(
        model_path,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    schema_mode = args.schema_mode
    if schema_mode == "auto":
        schema_mode = "native" if model_supports_multi_level(model) else "flat"
    if schema_mode == "native" and not model_supports_multi_level(model):
        LOGGER.warning(
            "Native mode was requested, but the checkpoint does not advertise multi-level "
            "structuring; nested examples may fail"
        )
    LOGGER.info("Using %s schema mode", schema_mode)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_key = _model_key(model_path)
    results_path = output_dir / f"{model_key}.results.jsonl"
    summary_path = output_dir / f"{model_key}.summary.json"
    totals = {key: 0.0 for key in METRIC_KEYS}
    results: list[dict[str, Any]] = []
    extract_errors = oom_errors = field_count = 0
    output_hasher = hashlib.sha256()
    inference_kwargs = {
        "threshold": args.threshold,
        "objectness_threshold": args.objectness_threshold,
        "batch_size": args.batch_size,
        "structuring_dedup": not args.no_structuring_dedup,
    }

    try:
        from tqdm import tqdm

        progress = tqdm(items, desc=f"GLiNExT {schema_mode}")
    except ImportError:
        progress = items

    started = time.time()
    with results_path.open("w", encoding="utf-8") as output_file:
        for item in progress:
            fields = build_flat_fields(item["template"])
            field_count += len(fields)
            prediction = None
            prediction_by_path: dict[str, list[Any]] = {}
            error: str | None = None
            was_oom = False
            inference_started = time.time()
            for attempt in range(2):
                try:
                    with torch.inference_mode():
                        if schema_mode == "native":
                            prediction, prediction_by_path = extract_native(
                                model, item["text"], item["template"], **inference_kwargs
                            )
                        else:
                            prediction, prediction_by_path = extract_flat(
                                model, item["text"], fields, **inference_kwargs
                            )
                    error = None
                    break
                except Exception as exc:
                    if is_oom_error(exc):
                        was_oom = True
                        error = f"{type(exc).__name__}: {exc}"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if attempt == 0:
                            LOGGER.warning("CUDA OOM; clearing cache and retrying once")
                            continue
                    else:
                        error = f"{type(exc).__name__}: {exc}"
                    if args.fail_fast:
                        raise
                    LOGGER.warning("Extraction failed: %s", error)
                    break
            inference_time = time.time() - inference_started

            metrics = evaluate_single(
                prediction, prediction_by_path, item["solution"], item["text"]
            )
            diagnostics = compute_diagnostics(item, prediction)
            diagnostics.update(
                {
                    "extract_error": error,
                    "was_oom": was_oom,
                    "gen_time_s": round(inference_time, 4),
                    "n_schema_fields": len(fields),
                }
            )
            for key in METRIC_KEYS:
                totals[key] += metrics[key]
            extract_errors += int(error is not None)
            oom_errors += int(was_oom)
            row = {
                "text": item["text"],
                "template": item["template"],
                "solution": item["solution"],
                "pred_json": prediction,
                "metrics": metrics,
                "diagnostics": diagnostics,
            }
            results.append(row)
            serialized = json.dumps(row, ensure_ascii=False)
            output_file.write(serialized + "\n")
            output_file.flush()
            output_hasher.update(json.dumps(prediction, ensure_ascii=False).encode("utf-8"))
            output_hasher.update(b"\n---\n")

    elapsed = time.time() - started
    sample_count = len(results)
    aggregate = {
        key: totals[key] / sample_count if sample_count else 0.0 for key in METRIC_KEYS
    }
    worst = sorted(results, key=lambda row: row["metrics"]["flat_field_f1"])[:10]
    worst_examples = [
        {
            "idx": results.index(row),
            "flat_field_f1": row["metrics"]["flat_field_f1"],
            "graph_f1": row["metrics"]["graph_f1"],
            "extract_error": row["diagnostics"]["extract_error"],
            "was_oom": row["diagnostics"]["was_oom"],
            "template_depth": row["diagnostics"]["template_depth"],
            "n_template_leaves": row["diagnostics"]["n_template_leaves"],
            "n_schema_fields": row["diagnostics"]["n_schema_fields"],
            "null_rate": round(row["diagnostics"]["null_rate"], 3),
            "text_head": row["text"][:200],
            "pred_head": json.dumps(row["pred_json"], ensure_ascii=False)[:300]
            if row["pred_json"] is not None
            else None,
        }
        for row in worst
    ]
    report = {
        "model": model_path,
        "device": args.device,
        "schema_mode": schema_mode,
        "num_samples": sample_count,
        "output_hash": output_hasher.hexdigest()[:16],
        "total_time_s": round(elapsed, 2),
        "samples_per_second": sample_count / max(elapsed, 0.1),
        "avg_schema_fields": field_count / max(sample_count, 1),
        "aggregate_metrics": {key: round(value, 4) for key, value in aggregate.items()},
        "num_extract_errors": extract_errors,
        "num_oom_errors": oom_errors,
        "segments": build_segments(results),
        "worst_10_examples": worst_examples,
        "results_jsonl_path": str(results_path),
        "config": {
            "dataset_repo": args.dataset_repo,
            "dataset_filename": args.dataset_filename,
            "threshold": args.threshold,
            "objectness_threshold": args.objectness_threshold,
            "batch_size": args.batch_size,
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "seed": args.seed,
        },
    }
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    LOGGER.info("Per-sample results: %s", results_path)
    LOGGER.info("Summary report: %s", summary_path)
    print_summary(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path",
        "--model_path",
        action="append",
        required=True,
        help="Local checkpoint or Hub model ID; repeat to evaluate multiple models.",
    )
    parser.add_argument("--dataset-repo", default=EVAL_DATASET_REPO)
    parser.add_argument("--dataset-filename", default=EVAL_DATASET_FILENAME)
    parser.add_argument("--data-path", type=Path, help="Use a local JSONL instead of downloading.")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates all examples.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--objectness-threshold", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--schema-mode", choices=("auto", "native", "flat"), default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Metric embedding device; CPU avoids competing with GLiNExT for VRAM.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-structuring-dedup", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_samples < 0:
        raise SystemExit("--num-samples must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.objectness_threshold is not None and not 0 <= args.objectness_threshold <= 1:
        raise SystemExit("--objectness-threshold must be between 0 and 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_embedding_model(args.embedding_model, args.embedding_device)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    items = load_eval_data(
        repo=args.dataset_repo,
        filename=args.dataset_filename,
        data_path=args.data_path,
        token=token,
        num_samples=args.num_samples or None,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    if not items:
        raise SystemExit("No valid evaluation examples were found")
    for model_path in args.model_path:
        evaluate_model(model_path, items, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
