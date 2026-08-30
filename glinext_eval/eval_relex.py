#!/usr/bin/env python3
"""Evaluate a GLiNExT joint-relation checkpoint on DocRED, CrossRE, and FewRel.

The dataset conversion and strict relation matching follow the reference
``gliner_eval`` benchmark. A prediction is correct when its normalized head
text, tail text, and relation label match a gold triple. DocRED aliases are
accepted so any annotated coreference mention can match the gold entity.

Example:
    python glinext_eval/eval_relex.py \
        --model logs/multitask_text/checkpoint-90000 \
        --max-samples 100
"""

# Console output is the primary progress report produced by this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DOCRED_REPO_ID = "thunlp/docred"
CROSSRE_BASE_URL = "https://raw.githubusercontent.com/mainlp/CrossRE/main/crossre_data"
FEWREL_BASE_URL = "https://raw.githubusercontent.com/thunlp/FewRel/master/data"
DEFAULT_CROSSRE_DOMAINS = ("ai", "news", "science")
ALL_CROSSRE_DOMAINS = ("ai", "literature", "music", "news", "politics", "science")
DEFAULT_OUTPUT = Path("eval_results/glinext_relex.json")

_NO_SPACE_BEFORE = {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "'",
    ")",
    "]",
    "}",
    "'s",
    "n't",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "%",
    "''",
}
_NO_SPACE_AFTER = {"(", "[", "{", "``"}
_CROSSRE_ENTITY_TYPE_MAP = {
    "academicjournal": "academic journal",
    "astronomicalobject": "astronomical object",
    "chemicalcompound": "chemical compound",
    "chemicalelement": "chemical element",
    "programlang": "programming language",
}


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldRelation:
    head: Entity
    tail: Entity
    relation: str


@dataclass(frozen=True)
class RelationSample:
    id: str
    text: str
    relations: tuple[GoldRelation, ...]


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    samples: tuple[RelationSample, ...]
    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]


@dataclass(frozen=True)
class PredictedRelation:
    head_text: str
    head_type: str
    tail_text: str
    tail_type: str
    relation: str
    score: float = 1.0


@dataclass(frozen=True)
class F1Score:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass(frozen=True)
class MetricsResult:
    micro: F1Score = field(default_factory=F1Score)
    macro: F1Score = field(default_factory=F1Score)
    per_label: dict[str, F1Score] = field(default_factory=dict)
    cardinality: dict[str, F1Score] = field(default_factory=dict)
    total_gold: int = 0
    total_predicted: int = 0
    total_true_positives: int = 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one GLiNExT joint-relation model on DocRED, CrossRE, and FewRel."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local GLiNExT checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("docred", "crossre", "fewrel"),
        default=["docred", "crossre", "fewrel"],
        help="Datasets to evaluate (default: all three).",
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        dest="max_samples",
        type=positive_int,
        help="Evaluate at most this many randomly selected samples from each dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic per-dataset sampling (default: 42).",
    )
    parser.add_argument(
        "--threshold",
        type=unit_float,
        default=0.5,
        help="Entity and relation confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=positive_int,
        default=1,
        help="GLiNExT inference batch size (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as 'cuda:0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--flat-ner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enforce non-overlapping entity spans (default: false).",
    )
    parser.add_argument(
        "--max-text-chars",
        type=non_negative_int,
        default=3000,
        help="Truncate input texts to this many characters; 0 disables it (default: 3000).",
    )
    parser.add_argument(
        "--max-relations-per-pair",
        type=positive_int,
        default=3,
        help="Keep at most this many highest-scoring labels per entity pair (default: 3).",
    )
    parser.add_argument(
        "--schema-name",
        default="relations",
        help="Name of the joint-relation schema group (default: relations).",
    )
    parser.add_argument(
        "--crossre-domains",
        nargs="+",
        choices=ALL_CROSSRE_DOMAINS,
        default=list(DEFAULT_CROSSRE_DOMAINS),
        help="CrossRE domains (default: ai news science, matching gliner_eval).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional cache root for benchmark data and Hugging Face downloads.",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face access token for the model and DocRED.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download the model or datasets; require them in their caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def reconstruct_text(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens naturally and return each token's character offsets."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    previous: str | None = None
    for token in tokens:
        if (
            previous is not None
            and token not in _NO_SPACE_BEFORE
            and previous not in _NO_SPACE_AFTER
        ):
            parts.append(" ")
            position += 1
        start = position
        position += len(token)
        offsets.append((start, position))
        parts.append(token)
        previous = token
    return "".join(parts), offsets


def _sample(
    items: list[RelationSample], max_samples: int | None, seed: int
) -> tuple[RelationSample, ...]:
    if max_samples is not None and max_samples < len(items):
        items = random.Random(seed).sample(items, max_samples)
    return tuple(items)


def _data_cache_root(cache_dir: Path | None) -> Path:
    return cache_dir if cache_dir is not None else Path.home() / ".cache" / "gliner_eval"


def _download_url(url: str, destination: Path, *, local_files_only: bool) -> Path:
    if destination.is_file():
        return destination
    if local_files_only:
        raise FileNotFoundError(f"Required cached dataset file does not exist: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "GLiNExT relation evaluator"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        return json.load(input_file)


def _docred_text(sents: Sequence[Sequence[str]]) -> tuple[str, list[list[tuple[int, int]]]]:
    flat_tokens: list[str] = []
    boundaries: list[tuple[int, int]] = []
    for sentence in sents:
        boundaries.append((len(flat_tokens), len(flat_tokens) + len(sentence)))
        flat_tokens.extend(sentence)
    text, flat_offsets = reconstruct_text(flat_tokens)
    return text, [flat_offsets[start:end] for start, end in boundaries]


def load_docred(
    *,
    max_samples: int | None,
    seed: int,
    cache_dir: Path | None,
    token: str | None,
    local_files_only: bool,
) -> DatasetInfo:
    """Load the labeled DocRED validation split from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    download_kwargs: dict[str, Any] = {
        "repo_id": DOCRED_REPO_ID,
        "repo_type": "dataset",
        "token": token,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)

    dev_path = Path(hf_hub_download(filename="data/dev.json.gz", **download_kwargs))
    info_path = Path(hf_hub_download(filename="data/rel_info.json.gz", **download_kwargs))
    documents = _load_gzip_json(dev_path)
    relation_names = _load_gzip_json(info_path)

    samples: list[RelationSample] = []
    entity_types: set[str] = set()
    relation_types: set[str] = set()
    for document_index, document in enumerate(documents):
        text, offsets = _docred_text(document["sents"])
        entities: list[Entity] = []
        for mentions in document["vertexSet"]:
            entity_type = str(mentions[0]["type"]).lower()
            entity_types.add(entity_type)
            mention_texts: list[str] = []
            for mention in mentions:
                sentence_id = int(mention["sent_id"])
                token_start, token_end = mention["pos"]
                if sentence_id < len(offsets) and token_start < len(offsets[sentence_id]):
                    char_start = offsets[sentence_id][token_start][0]
                    end_index = min(token_end - 1, len(offsets[sentence_id]) - 1)
                    char_end = offsets[sentence_id][end_index][1]
                    mention_texts.append(text[char_start:char_end])
                else:
                    mention_texts.append(str(mention["name"]))
            unique_mentions = list(dict.fromkeys(mention_texts))
            primary = max(unique_mentions, key=len)
            aliases = tuple(value for value in unique_mentions if value != primary)
            entities.append(Entity(primary, entity_type, aliases))

        relations: list[GoldRelation] = []
        for label in document.get("labels", []):
            relation_id = label["r"]
            relation = str(relation_names.get(relation_id, relation_id)).lower()
            relation_types.add(relation)
            relations.append(
                GoldRelation(
                    head=entities[int(label["h"])],
                    tail=entities[int(label["t"])],
                    relation=relation,
                )
            )
        if relations:
            samples.append(RelationSample(f"docred_{document_index}", text, tuple(relations)))

    return DatasetInfo(
        name="docred",
        samples=_sample(samples, max_samples, seed),
        entity_types=tuple(sorted(entity_types)),
        relation_types=tuple(sorted(relation_types)),
    )


def _parse_crossre(path: Path, domain: str) -> tuple[list[RelationSample], set[str], set[str]]:
    samples: list[RelationSample] = []
    entity_types: set[str] = set()
    relation_types: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file):
            if not line.strip():
                continue
            record = json.loads(line)
            text, token_offsets = reconstruct_text(record["sentence"])
            entities: dict[tuple[int, int], Entity] = {}
            for start, end, raw_type, *_ in record.get("ner", []):
                char_start = token_offsets[start][0]
                char_end = token_offsets[end][1]
                entity_type = _CROSSRE_ENTITY_TYPE_MAP.get(
                    str(raw_type).lower(), str(raw_type).lower()
                )
                entity_types.add(entity_type)
                entities[(start, end)] = Entity(text[char_start:char_end], entity_type)

            relations: list[GoldRelation] = []
            for start1, end1, start2, end2, raw_relation, *_ in record.get("relations", []):
                head = entities.get((start1, end1))
                if head is None:
                    head = Entity(
                        text[token_offsets[start1][0] : token_offsets[end1][1]], "unknown"
                    )
                tail = entities.get((start2, end2))
                if tail is None:
                    tail = Entity(
                        text[token_offsets[start2][0] : token_offsets[end2][1]], "unknown"
                    )
                relation = str(raw_relation).lower()
                relation_types.add(relation)
                relations.append(GoldRelation(head, tail, relation))
            if relations:
                samples.append(
                    RelationSample(f"crossre_{domain}_{line_number}", text, tuple(relations))
                )
    return samples, entity_types, relation_types


def load_crossre(
    *,
    domains: Sequence[str],
    max_samples: int | None,
    seed: int,
    cache_dir: Path | None,
    local_files_only: bool,
) -> DatasetInfo:
    """Load and combine the selected CrossRE test domains."""
    cache_root = _data_cache_root(cache_dir) / "crossre"
    samples: list[RelationSample] = []
    entity_types: set[str] = set()
    relation_types: set[str] = set()
    for domain in domains:
        filename = f"{domain}-test.json"
        path = _download_url(
            f"{CROSSRE_BASE_URL}/{filename}",
            cache_root / filename,
            local_files_only=local_files_only,
        )
        domain_samples, domain_entity_types, domain_relation_types = _parse_crossre(path, domain)
        samples.extend(domain_samples)
        entity_types.update(domain_entity_types)
        relation_types.update(domain_relation_types)

    return DatasetInfo(
        name="crossre",
        samples=_sample(samples, max_samples, seed),
        entity_types=tuple(sorted(entity_types)),
        relation_types=tuple(sorted(relation_types)),
    )


def load_fewrel(
    *,
    max_samples: int | None,
    seed: int,
    cache_dir: Path | None,
    local_files_only: bool,
) -> DatasetInfo:
    """Load the FewRel ``val_wiki`` split."""
    cache_root = _data_cache_root(cache_dir) / "fewrel"
    data_path = _download_url(
        f"{FEWREL_BASE_URL}/val_wiki.json",
        cache_root / "val_wiki.json",
        local_files_only=local_files_only,
    )
    names_path = _download_url(
        f"{FEWREL_BASE_URL}/pid2name.json",
        cache_root / "pid2name.json",
        local_files_only=local_files_only,
    )
    with data_path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    with names_path.open(encoding="utf-8") as input_file:
        relation_names = json.load(input_file)

    samples: list[RelationSample] = []
    relation_types: set[str] = set()
    sample_index = 0
    for relation_id, instances in data.items():
        raw_name = relation_names.get(relation_id, [relation_id])
        relation = str(raw_name[0] if isinstance(raw_name, list) else raw_name).lower()
        relation_types.add(relation)
        for instance in instances:
            text, token_offsets = reconstruct_text(instance["tokens"])
            head_indices = instance["h"][2][0] if instance["h"][2] else []
            tail_indices = instance["t"][2][0] if instance["t"][2] else []
            if head_indices:
                head_text = text[
                    token_offsets[head_indices[0]][0] : token_offsets[head_indices[-1]][1]
                ]
            else:
                head_text = str(instance["h"][0])
            if tail_indices:
                tail_text = text[
                    token_offsets[tail_indices[0]][0] : token_offsets[tail_indices[-1]][1]
                ]
            else:
                tail_text = str(instance["t"][0])
            gold = GoldRelation(Entity(head_text, "entity"), Entity(tail_text, "entity"), relation)
            samples.append(RelationSample(f"fewrel_{sample_index}", text, (gold,)))
            sample_index += 1

    return DatasetInfo(
        name="fewrel",
        samples=_sample(samples, max_samples, seed),
        entity_types=("entity",),
        relation_types=tuple(sorted(relation_types)),
    )


def load_datasets(args: argparse.Namespace) -> list[DatasetInfo]:
    loaders = {
        "docred": lambda: load_docred(
            max_samples=args.max_samples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            token=args.token,
            local_files_only=args.local_files_only,
        ),
        "crossre": lambda: load_crossre(
            domains=args.crossre_domains,
            max_samples=args.max_samples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ),
        "fewrel": lambda: load_fewrel(
            max_samples=args.max_samples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ),
    }
    datasets: list[DatasetInfo] = []
    for name in dict.fromkeys(args.datasets):
        print(f"Loading {name} ...")
        dataset = loaders[name]()
        if not dataset.samples:
            raise ValueError(f"{name} contains no relation-bearing samples")
        datasets.append(dataset)
        print(
            f"  {len(dataset.samples):,} samples, {len(dataset.entity_types):,} entity types, "
            f"{len(dataset.relation_types):,} relation types"
        )
    return datasets


def normalize(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+([,.:;!?)])", r"\1", normalized)
    normalized = re.sub(r"([\[(])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return re.sub(r"^(the|a|an)\s+", "", normalized)


def _entity_matches(gold: Entity, predicted_text: str) -> bool:
    predicted = normalize(predicted_text)
    return any(normalize(candidate) == predicted for candidate in (gold.text, *gold.aliases))


def _strict_match(gold: GoldRelation, predicted: PredictedRelation) -> bool:
    return (
        _entity_matches(gold.head, predicted.head_text)
        and _entity_matches(gold.tail, predicted.tail_text)
        and normalize(gold.relation) == normalize(predicted.relation)
    )


def _match_relations(
    gold_relations: Sequence[GoldRelation],
    predicted_relations: Sequence[PredictedRelation],
) -> tuple[
    list[tuple[GoldRelation, PredictedRelation]], list[PredictedRelation], list[GoldRelation]
]:
    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()
    true_positives: list[tuple[GoldRelation, PredictedRelation]] = []
    for prediction_index, prediction in enumerate(predicted_relations):
        for gold_index, gold in enumerate(gold_relations):
            if gold_index not in matched_gold and _strict_match(gold, prediction):
                matched_gold.add(gold_index)
                matched_predictions.add(prediction_index)
                true_positives.append((gold, prediction))
                break
    false_positives = [
        prediction
        for index, prediction in enumerate(predicted_relations)
        if index not in matched_predictions
    ]
    false_negatives = [
        gold for index, gold in enumerate(gold_relations) if index not in matched_gold
    ]
    return true_positives, false_positives, false_negatives


def _f1(true_positives: int, false_positives: int, false_negatives: int) -> F1Score:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return F1Score(precision, recall, score)


def _cardinality(relations: Sequence[GoldRelation]) -> str:
    if len(relations) <= 1:
        return "one-to-one"
    heads: dict[str, int] = defaultdict(int)
    tails: dict[str, int] = defaultdict(int)
    for relation in relations:
        heads[normalize(relation.head.text)] += 1
        tails[normalize(relation.tail.text)] += 1
    repeated_head = any(count > 1 for count in heads.values())
    repeated_tail = any(count > 1 for count in tails.values())
    if repeated_head and repeated_tail:
        return "many-to-many"
    if repeated_head or repeated_tail:
        return "one-to-many"
    return "one-to-one"


def compute_metrics(
    samples: Sequence[RelationSample],
    predictions: Sequence[Sequence[PredictedRelation]],
) -> MetricsResult:
    """Compute strict micro, macro/per-label, and cardinality F1."""
    if len(samples) != len(predictions):
        raise ValueError(
            f"Gold/prediction sample count mismatch: {len(samples)} != {len(predictions)}"
        )

    total_tp = total_fp = total_fn = 0
    label_tp: dict[str, int] = defaultdict(int)
    label_fp: dict[str, int] = defaultdict(int)
    label_fn: dict[str, int] = defaultdict(int)
    card_tp: dict[str, int] = defaultdict(int)
    card_fp: dict[str, int] = defaultdict(int)
    card_fn: dict[str, int] = defaultdict(int)

    for sample, sample_predictions in zip(samples, predictions, strict=True):
        matched, false_positives, false_negatives = _match_relations(
            sample.relations, sample_predictions
        )
        total_tp += len(matched)
        total_fp += len(false_positives)
        total_fn += len(false_negatives)
        for gold, _ in matched:
            label_tp[normalize(gold.relation)] += 1
        for prediction in false_positives:
            label_fp[normalize(prediction.relation)] += 1
        for gold in false_negatives:
            label_fn[normalize(gold.relation)] += 1

        cardinality = _cardinality(sample.relations)
        card_tp[cardinality] += len(matched)
        card_fp[cardinality] += len(false_positives)
        card_fn[cardinality] += len(false_negatives)

    labels = sorted(set(label_tp) | set(label_fp) | set(label_fn))
    per_label = {label: _f1(label_tp[label], label_fp[label], label_fn[label]) for label in labels}
    macro = F1Score(
        precision=sum(score.precision for score in per_label.values()) / len(per_label)
        if per_label
        else 0.0,
        recall=sum(score.recall for score in per_label.values()) / len(per_label)
        if per_label
        else 0.0,
        f1=sum(score.f1 for score in per_label.values()) / len(per_label) if per_label else 0.0,
    )
    cardinality = {
        name: _f1(card_tp[name], card_fp[name], card_fn[name])
        for name in ("one-to-one", "one-to-many", "many-to-many")
        if card_tp[name] + card_fp[name] + card_fn[name]
    }
    return MetricsResult(
        micro=_f1(total_tp, total_fp, total_fn),
        macro=macro,
        per_label=per_label,
        cardinality=cardinality,
        total_gold=total_tp + total_fn,
        total_predicted=total_tp + total_fp,
        total_true_positives=total_tp,
    )


def _iter_triples(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if all(key in value for key in ("head", "tail", "relation")):
            yield value
        return
    if isinstance(value, list):
        for child in value:
            yield from _iter_triples(child)


def _endpoint(triple: dict[str, Any], role: str) -> tuple[str, str] | None:
    endpoint = triple.get(role)
    if not isinstance(endpoint, dict):
        return None
    text = endpoint.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    entity_type = endpoint.get("type", endpoint.get("label", ""))
    return text, str(entity_type) if entity_type is not None else ""


def convert_predictions(
    raw_predictions: Any,
    relation_types: Sequence[str],
    *,
    max_relations_per_pair: int,
) -> list[PredictedRelation]:
    """Normalize native GLiNExT triples and prune labels per entity pair."""
    allowed_relations = {label.casefold(): label for label in relation_types}
    unique: dict[tuple[str, str, str], PredictedRelation] = {}
    for triple in _iter_triples(raw_predictions):
        head = _endpoint(triple, "head")
        tail = _endpoint(triple, "tail")
        raw_relation = triple.get("relation")
        if head is None or tail is None or not isinstance(raw_relation, str):
            continue
        relation = allowed_relations.get(raw_relation.casefold())
        if relation is None:
            continue
        try:
            score = float(triple.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        prediction = PredictedRelation(head[0], head[1], tail[0], tail[1], relation, score)
        key = (prediction.head_text, prediction.tail_text, prediction.relation)
        previous = unique.get(key)
        if previous is None or prediction.score > previous.score:
            unique[key] = prediction

    by_pair: dict[tuple[str, str], list[PredictedRelation]] = defaultdict(list)
    for prediction in unique.values():
        by_pair[(prediction.head_text, prediction.tail_text)].append(prediction)
    output: list[PredictedRelation] = []
    for pair_predictions in by_pair.values():
        pair_predictions.sort(key=lambda prediction: prediction.score, reverse=True)
        output.extend(pair_predictions[:max_relations_per_pair])
    return output


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    import torch  # noqa: PLC0415 - keep --help and metric imports lightweight

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(
    model_id: str,
    *,
    device: str,
    cache_dir: Path | None,
    token: str | None,
    local_files_only: bool,
) -> Any:
    from glinext import GLiNExT  # noqa: PLC0415 - defer heavyweight model imports

    model = GLiNExT.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        load_tokenizer=True,
        map_location=device,
    )
    if getattr(model.config, "joint_relex_config", None) is None:
        raise ValueError(f"GLiNExT checkpoint {model_id!r} does not enable a joint_relex head")
    model.to(device)
    model.eval()
    return model


def _batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_dataset(
    model: Any,
    dataset: DatasetInfo,
    *,
    threshold: float,
    batch_size: int,
    flat_ner: bool,
    max_text_chars: int,
    max_relations_per_pair: int,
    schema_name: str,
) -> tuple[list[list[PredictedRelation]], float]:
    """Run batched joint-relation inference for one prepared dataset."""
    schema = {
        schema_name: {
            "entities": list(dataset.entity_types),
            "relations": list(dataset.relation_types),
        }
    }
    all_predictions: list[list[PredictedRelation]] = []
    batches = (len(dataset.samples) + batch_size - 1) // batch_size
    progress_interval = max(1, batches // 100)
    started = time.perf_counter()
    for batch_index, batch in enumerate(_batched(dataset.samples, batch_size), start=1):
        texts = [
            sample.text[:max_text_chars] if max_text_chars else sample.text for sample in batch
        ]
        results = model.inference(
            texts,
            joint_relations=schema,
            threshold=threshold,
            flat_ner=flat_ner,
            batch_size=batch_size,
        )
        raw_batch = results.get("joint_relex")
        if not isinstance(raw_batch, list) or len(raw_batch) != len(batch):
            actual = len(raw_batch) if isinstance(raw_batch, list) else type(raw_batch).__name__
            raise RuntimeError(
                f"GLiNExT returned invalid joint_relex output for {dataset.name}: "
                f"expected {len(batch)} samples, got {actual}"
            )
        all_predictions.extend(
            convert_predictions(
                raw_sample,
                dataset.relation_types,
                max_relations_per_pair=max_relations_per_pair,
            )
            for raw_sample in raw_batch
        )
        if batch_index == 1 or batch_index == batches or batch_index % progress_interval == 0:
            print(f"\r  batch {batch_index:,}/{batches:,}", end="", flush=True)
    print()
    return all_predictions, time.perf_counter() - started


def print_report(results: dict[str, tuple[MetricsResult, float]]) -> None:
    name_width = max(10, *(len(name) for name in results))
    print()
    print(
        f"{'Dataset':<{name_width}}  {'Gold':>8}  {'Micro-F1':>10}  "
        f"{'Macro-F1':>10}  {'Precision':>10}  {'Recall':>10}  {'Seconds':>10}"
    )
    print("-" * (name_width + 76))
    for name, (metrics, elapsed) in results.items():
        print(
            f"{name:<{name_width}}  {metrics.total_gold:>8,d}  {metrics.micro.f1:>9.2%}  "
            f"{metrics.macro.f1:>9.2%}  {metrics.micro.precision:>9.2%}  "
            f"{metrics.micro.recall:>9.2%}  {elapsed:>10.1f}"
        )


def save_report(
    output_path: Path,
    *,
    args: argparse.Namespace,
    datasets: Sequence[DatasetInfo],
    predictions: dict[str, list[list[PredictedRelation]]],
    results: dict[str, tuple[MetricsResult, float]],
    device: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": args.model,
        "device": device,
        "settings": {
            "datasets": list(dict.fromkeys(args.datasets)),
            "max_samples_per_dataset": args.max_samples,
            "seed": args.seed,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "flat_ner": args.flat_ner,
            "max_text_chars": args.max_text_chars,
            "max_relations_per_pair": args.max_relations_per_pair,
            "schema_name": args.schema_name,
            "crossre_domains": args.crossre_domains,
        },
        "datasets": {},
    }
    for dataset in datasets:
        metrics, elapsed = results[dataset.name]
        payload["datasets"][dataset.name] = {
            "samples": len(dataset.samples),
            "entity_types": list(dataset.entity_types),
            "relation_types": list(dataset.relation_types),
            "elapsed_seconds": elapsed,
            "metrics": asdict(metrics),
            "predictions": [
                {
                    "sample_id": sample.id,
                    "relations": [asdict(prediction) for prediction in sample_predictions],
                }
                for sample, sample_predictions in zip(
                    dataset.samples,
                    predictions[dataset.name],
                    strict=True,
                )
            ],
        }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def main() -> None:
    args = create_parser().parse_args()
    if not args.schema_name.strip():
        raise ValueError("--schema-name cannot be empty")

    datasets = load_datasets(args)
    device = resolve_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    model = load_model(
        args.model,
        device=device,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )

    predictions: dict[str, list[list[PredictedRelation]]] = {}
    results: dict[str, tuple[MetricsResult, float]] = {}
    for index, dataset in enumerate(datasets, start=1):
        print(
            f"[{index}/{len(datasets)}] Evaluating {dataset.name} on {len(dataset.samples):,} samples"
        )
        dataset_predictions, elapsed = evaluate_dataset(
            model,
            dataset,
            threshold=args.threshold,
            batch_size=args.batch_size,
            flat_ner=args.flat_ner,
            max_text_chars=args.max_text_chars,
            max_relations_per_pair=args.max_relations_per_pair,
            schema_name=args.schema_name,
        )
        metrics = compute_metrics(dataset.samples, dataset_predictions)
        predictions[dataset.name] = dataset_predictions
        results[dataset.name] = (metrics, elapsed)
        print(
            f"  Micro-F1={metrics.micro.f1:.2%}  P={metrics.micro.precision:.2%}  "
            f"R={metrics.micro.recall:.2%}  time={elapsed:.1f}s"
        )

    print_report(results)
    save_report(
        args.output,
        args=args,
        datasets=datasets,
        predictions=predictions,
        results=results,
        device=device,
    )
    print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()
