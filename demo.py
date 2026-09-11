"""GLiFormer Gradio Demo — Multi-task Information Extraction

Run it with `python demo.py`; point it at a checkpoint with `GLIFORMER_MODEL_ID`.

Everything the demo needs lives in this one file, in three sections: the
**Metrics** that score a prediction against a gold annotation, the
**Examples** — ten annotated ones per tab — and the **Demo app** itself.
Loading an example fills in the text and the tab's inputs (label groups, or
the nested JSON template on the multi-level structuring tab); running it
scores the prediction against that example's gold annotation.
"""

import functools
import json
import os
import re
import traceback
from dataclasses import dataclass
from html import escape
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import gradio as gr
import torch

from gliformer import GLiFormer, FieldType, GLiFormerPDFProcessor, StructuringOutputFormatter

# ── Model loading ────────────────────────────────────────────────────────────

# Override the checkpoint without editing this file:
#   GLIFORMER_MODEL_ID=logs/multitask_deberta2d/checkpoint-10000/ python3 demo.py
DEFAULT_MODEL_ID = "./logs/multitask_deberta2d/post1"


def _resolve_model_id(model_id: str) -> str:
    """Fall back to the newest local checkpoint when a pinned step is gone.

    Training rotates `checkpoint-*` directories away, so a pinned step stops
    existing after a few saves and the demo would fail to start. Hub ids and
    paths without sibling checkpoints are returned untouched.
    """
    if os.path.exists(model_id):
        return model_id
    root = os.path.dirname(os.path.normpath(model_id))
    if not os.path.isdir(root):
        return model_id
    checkpoints = []
    for name in os.listdir(root):
        prefix, _, step = name.partition("-")
        path = os.path.join(root, name)
        if prefix == "checkpoint" and step.isdigit() and os.path.isdir(path):
            checkpoints.append((int(step), path))
    return max(checkpoints)[1] if checkpoints else model_id


MODEL_ID = _resolve_model_id(os.environ.get("GLIFORMER_MODEL_ID", DEFAULT_MODEL_ID))
model: Optional[GLiFormer] = None


def get_model():
    global model
    if model is None:
        model = GLiFormer.from_pretrained(MODEL_ID, load_tokenizer=True)
        model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Metrics — task-specific scoring of a prediction against a gold annotation
# ═════════════════════════════════════════════════════════════════════════════
#
# Predictions are matched against the hand-written gold annotations of the
# Examples section below by *surface form* — the normalised mention/value text
# — rather than by character offsets. That keeps the examples readable and
# editable by hand at the cost of conflating repeated mentions of the same
# string, which is an acceptable trade-off for a demo.
#
# Every extraction task reports strict micro precision / recall / F1 over a set
# of task-specific keys:
#
# | Task            | Match key                                  |
# |-----------------|--------------------------------------------|
# | NER             | (mention, label)                           |
# | Classification  | class name (plus exact-set accuracy)       |
# | Relations       | (head, relation, tail)                     |
# | Structuring     | (field path, value) after instance align   |
# | Embedding       | ranking: P@1 / MRR / MAP                   |

_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\"'`(\[{.,;:!?-]+|[\s\"'`)\]}.,;:!?-]+$")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize(value: Any) -> str:
    """Normalise a mention or field value for surface-form comparison."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return " | ".join(sorted(normalize(item) for item in value))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).casefold().replace("’", "'")
    text = _WHITESPACE.sub(" ", text).strip()
    text = _EDGE_PUNCT.sub("", text)
    return _THOUSANDS.sub("", text).strip()


# ── Metrics: precision / recall / F1 ─────────────────────────────────────────

@dataclass
class PRF:
    """Micro counts for one task; instances sum with ``+``."""

    tp: int = 0
    predicted: int = 0
    gold: int = 0

    def __add__(self, other: "PRF") -> "PRF":
        return PRF(self.tp + other.tp, self.predicted + other.predicted, self.gold + other.gold)

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.tp,
            "predicted": self.predicted,
            "gold": self.gold,
        }


def _prf(gold_keys: Iterable[Tuple], predicted_keys: Iterable[Tuple]) -> PRF:
    gold_set, pred_set = set(gold_keys), set(predicted_keys)
    return PRF(len(gold_set & pred_set), len(pred_set), len(gold_set))


def prf_from_sets(gold_keys: Iterable[Tuple], predicted_keys: Iterable[Tuple]) -> PRF:
    """Public wrapper around set-based PRF counting."""
    return _prf(gold_keys, predicted_keys)


def iter_dicts(obj: Any) -> Iterator[dict]:
    """Yield every dict inside an arbitrarily nested list structure.

    Decoders return either a flat list per text or a list of per-group lists
    depending on the task, so metrics flatten defensively.
    """
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from iter_dicts(item)


# ── Metrics: NER ─────────────────────────────────────────────────────────────

def _entity_key(entity: dict) -> Tuple[str, str] | None:
    text = entity.get("text")
    label = entity.get("label", entity.get("type"))
    if text is None or label is None:
        return None
    return normalize(text), normalize(label)


def ner_key_sets(gold: Sequence[dict], predicted: Any) -> Tuple[set, set]:
    """Gold and predicted (mention, label) key sets for a single text."""
    gold_keys = {key for key in (_entity_key(e) for e in gold) if key}
    pred_keys = {key for key in (_entity_key(e) for e in iter_dicts(predicted)) if key}
    return gold_keys, pred_keys


def ner_prf(gold: Sequence[dict], predicted: Any) -> PRF:
    """Strict (mention, label) micro PRF for a single text."""
    return _prf(*ner_key_sets(gold, predicted))


# ── Metrics: Classification ──────────────────────────────────────────────────

def classification_key_sets(gold: Sequence[str], predicted: Any) -> Tuple[set, set]:
    """Gold and predicted class-name key sets for a single text."""
    gold_keys = {(normalize(label),) for label in gold}
    pred_keys = {
        (normalize(item["class_name"]),)
        for item in iter_dicts(predicted)
        if "class_name" in item
    }
    return gold_keys, pred_keys


def classification_prf(gold: Sequence[str], predicted: Any) -> Tuple[PRF, bool]:
    """Label-set micro PRF plus exact-set match for a single text."""
    gold_keys, pred_keys = classification_key_sets(gold, predicted)
    return _prf(gold_keys, pred_keys), gold_keys == pred_keys


# ── Metrics: Relations (open + joint) ────────────────────────────────────────

def _endpoint_text(endpoint: Any) -> str:
    if isinstance(endpoint, dict):
        return normalize(endpoint.get("text", ""))
    return normalize(endpoint)


def _triple_key(triple: dict) -> Tuple[str, str, str] | None:
    if "relation" not in triple or "head" not in triple or "tail" not in triple:
        return None
    return (
        _endpoint_text(triple["head"]),
        normalize(triple["relation"]),
        _endpoint_text(triple["tail"]),
    )


def relation_key_sets(gold: Sequence[dict], predicted: Any) -> Tuple[set, set]:
    """Gold and predicted (head, relation, tail) key sets for a single text."""
    gold_keys = {key for key in (_triple_key(t) for t in gold) if key}
    pred_keys = {key for key in (_triple_key(t) for t in iter_dicts(predicted)) if key}
    return gold_keys, pred_keys


def relation_prf(gold: Sequence[dict], predicted: Any) -> PRF:
    """Strict (head, relation, tail) micro PRF for a single text."""
    return _prf(*relation_key_sets(gold, predicted))


# ── Metrics: Structuring ─────────────────────────────────────────────────────

def _child_instances(instance: dict) -> Dict[str, List[dict]]:
    """Nested object children of one instance, keyed by the field holding them.

    A multi-level schema nests either a single object (``"seller": {...}``) or
    a repeated child (``"items": [{...}, ...]``); both are scored as an
    instance list under that field.
    """
    children: Dict[str, List[dict]] = {}
    for field, value in instance.items():
        if isinstance(value, dict):
            children[field] = [value]
        elif isinstance(value, (list, tuple)) and any(isinstance(i, dict) for i in value):
            children[field] = [i for i in value if isinstance(i, dict)]
    return children


def _scalar_pairs(instance: dict) -> set:
    """(field, value) pairs for the scalar fields of one instance level."""
    pairs = set()
    for field, value in instance.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, (list, tuple)) and any(isinstance(i, dict) for i in value):
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                pairs.add((normalize(field), normalize(item)))
        else:
            pairs.add((normalize(field), normalize(value)))
    return pairs


def _instance_pairs(instance: dict, prefix: str = "") -> set:
    """Flatten one instance into (field path, value) pairs, nesting included."""
    pairs = {(prefix + field, value) for field, value in _scalar_pairs(instance)}
    for field, children in _child_instances(instance).items():
        for child in children:
            pairs |= _instance_pairs(child, f"{prefix}{normalize(field)}.")
    return pairs


def _instance_size(instance: dict) -> int:
    """Number of scored field values in an instance and its whole subtree."""
    return len(_scalar_pairs(instance)) + sum(
        _instance_size(child)
        for children in _child_instances(instance).values()
        for child in children
    )


def _instance_prf(gold: dict, predicted: dict) -> PRF:
    """PRF of one aligned instance pair, counting its whole subtree.

    Children are aligned within the field that holds them, so a value only
    counts when it sits at the same place in the hierarchy on both sides.
    """
    gold_pairs, predicted_pairs = _scalar_pairs(gold), _scalar_pairs(predicted)
    total = PRF(len(gold_pairs & predicted_pairs), len(predicted_pairs), len(gold_pairs))
    gold_children, predicted_children = _child_instances(gold), _child_instances(predicted)
    for field in set(gold_children) | set(predicted_children):
        total += _align_instances(
            gold_children.get(field, []), predicted_children.get(field, [])
        )
    return total


def _align_instances(gold_instances: Sequence[Any], predicted_instances: Sequence[Any]) -> PRF:
    """Greedily align predicted instances to gold ones by field-value overlap."""
    gold_list = [i for i in gold_instances if isinstance(i, dict)]
    predicted_list = [i for i in predicted_instances if isinstance(i, dict)]

    scored = sorted(
        (
            (_instance_prf(gold, predicted), gi, pi)
            for gi, gold in enumerate(gold_list)
            for pi, predicted in enumerate(predicted_list)
        ),
        key=lambda item: -item[0].tp,
    )
    used_gold, used_predicted, total = set(), set(), PRF()
    for prf, gi, pi in scored:
        if prf.tp == 0:
            break
        if gi in used_gold or pi in used_predicted:
            continue
        used_gold.add(gi)
        used_predicted.add(pi)
        total += prf

    total += PRF(gold=sum(
        _instance_size(instance)
        for gi, instance in enumerate(gold_list) if gi not in used_gold
    ))
    total += PRF(predicted=sum(
        _instance_size(instance)
        for pi, instance in enumerate(predicted_list) if pi not in used_predicted
    ))
    return total


def _as_instance_list(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def structuring_key_sets(gold: Dict[str, Any], predicted: Any) -> Tuple[set, set]:
    """Schema-qualified (schema, field path, value) key sets, for display only.

    Scoring uses `structuring_prf`, which aligns instances before counting;
    this flattened view ignores instance boundaries and only answers "which
    field values were missed or invented". Nested values keep their dotted
    path (``depots.routes.destination``) so a multi-level diff stays readable.
    """
    predicted_map = predicted if isinstance(predicted, dict) else {}

    def flatten(mapping: Dict[str, Any]) -> set:
        keys = set()
        for schema, instances in mapping.items():
            for instance in _as_instance_list(instances):
                if isinstance(instance, dict):
                    keys.update((schema, field, value) for field, value in _instance_pairs(instance))
        return keys

    return flatten(gold), flatten(predicted_map)


def structuring_prf(gold: Dict[str, Any], predicted: Any) -> PRF:
    """Field-value micro PRF over all schemas, after greedy instance alignment."""
    predicted_map = predicted if isinstance(predicted, dict) else {}
    total = PRF()
    for schema, gold_instances in gold.items():
        total += _align_instances(
            _as_instance_list(gold_instances),
            _as_instance_list(predicted_map.get(schema)),
        )
    for schema, predicted_instances in predicted_map.items():
        if schema in gold:
            continue
        total += _align_instances([], _as_instance_list(predicted_instances))
    return total


# ── Metrics: Embedding / retrieval ───────────────────────────────────────────

def ranking_metrics(ranked_texts: Sequence[str], relevant: Iterable[str]) -> Dict[str, float]:
    """Ranking quality of a similarity-sorted candidate list."""
    relevant_set = {normalize(text) for text in relevant if normalize(text)}
    hits = [normalize(text) in relevant_set for text in ranked_texts]

    precision_at_1 = float(hits[0]) if hits else 0.0
    mrr = next((1.0 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0)

    found, average_precision = 0, 0.0
    for rank, hit in enumerate(hits):
        if hit:
            found += 1
            average_precision += found / (rank + 1)
    average_precision = average_precision / len(relevant_set) if relevant_set else 0.0

    k = max(len(relevant_set), 1)
    recall_at_k = sum(hits[:k]) / len(relevant_set) if relevant_set else 0.0

    return {
        "p@1": precision_at_1,
        "mrr": mrr,
        "map": average_precision,
        "recall@k": recall_at_k,
    }


# ── Metrics: Reporting ───────────────────────────────────────────────────────

def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a markdown table."""
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def format_prf_report(
    title: str,
    per_example: Sequence[Tuple[str, Dict[str, PRF]]],
    extra_columns: Dict[str, Sequence[Any]] | None = None,
) -> str:
    """Render per-example and aggregate PRF for one or more metric families.

    Args:
        title: Heading for the report.
        per_example: (example name, {metric family: PRF}) pairs.
        extra_columns: Optional extra columns keyed by header, one value per example.
    """
    if not per_example:
        return "No examples to score."

    families = list(per_example[0][1].keys())
    extra_columns = extra_columns or {}

    headers = ["#", "Example"]
    for family in families:
        prefix = f"{family} " if len(families) > 1 else ""
        headers += [f"{prefix}P", f"{prefix}R", f"{prefix}F1"]
    headers += list(extra_columns)

    rows, totals = [], {family: PRF() for family in families}
    macro = {family: [] for family in families}
    for index, (name, metrics) in enumerate(per_example, start=1):
        row = [index, name]
        for family in families:
            prf = metrics[family]
            totals[family] += prf
            macro[family].append(prf.f1)
            row += [_pct(prf.precision), _pct(prf.recall), _pct(prf.f1)]
        for header, values in extra_columns.items():
            row.append(values[index - 1] if index - 1 < len(values) else "")
        rows.append(row)

    micro_row = ["", "**Micro average**"]
    macro_row = ["", "**Macro average**"]
    for family in families:
        prf = totals[family]
        micro_row += [f"**{_pct(prf.precision)}**", f"**{_pct(prf.recall)}**", f"**{_pct(prf.f1)}**"]
        scores = macro[family]
        mean_f1 = sum(scores) / len(scores) if scores else 0.0
        macro_row += ["", "", f"**{_pct(mean_f1)}**"]
    for header, values in extra_columns.items():
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        micro_row.append(f"**{sum(numeric) / len(numeric):.3f}**" if numeric else "")
        macro_row.append("")
    rows.extend([micro_row, macro_row])

    counts = ", ".join(
        f"{family}: {totals[family].tp}/{totals[family].predicted} predicted, "
        f"{totals[family].tp}/{totals[family].gold} gold matched"
        for family in families
    )
    return f"### {title}\n\n{render_table(headers, rows)}\n\n_{counts}_"


def format_ranking_report(title: str, per_example: Sequence[Tuple[str, Dict[str, float]]]) -> str:
    """Render per-example and mean ranking metrics."""
    if not per_example:
        return "No examples to score."

    keys = list(per_example[0][1].keys())
    headers = ["#", "Example"] + [key.upper() for key in keys]
    rows = [
        [index, name] + [f"{metrics[key]:.3f}" for key in keys]
        for index, (name, metrics) in enumerate(per_example, start=1)
    ]
    means = ["", "**Mean**"] + [
        f"**{sum(metrics[key] for _, metrics in per_example) / len(per_example):.3f}**"
        for key in keys
    ]
    rows.append(means)
    return f"### {title}\n\n{render_table(headers, rows)}"


def _format_key(key: Tuple) -> str:
    if len(key) == 1:
        return f"`{key[0]}`"
    if len(key) == 2:
        return f"`{key[0]}` [{key[1]}]"
    return f"`{key[0]}` —{key[1]}→ `{key[2]}`"


def format_key_diff(gold_keys: set, predicted_keys: set, limit: int = 8) -> str:
    """Render the missed and spurious items of one prediction."""
    lines = []
    for title, keys in (("Missed", gold_keys - predicted_keys), ("Spurious", predicted_keys - gold_keys)):
        if not keys:
            continue
        shown = sorted(keys)[:limit]
        suffix = f" … (+{len(keys) - limit} more)" if len(keys) > limit else ""
        lines.append(f"- **{title} ({len(keys)}):** " + ", ".join(_format_key(k) for k in shown) + suffix)
    return "\n".join(lines) if lines else "- **Exact match** — no missed or spurious items."


def format_single_report(metrics: Dict[str, PRF], notes: Sequence[str] = ()) -> str:
    """Render one example's metrics as a compact markdown table plus notes."""
    rows = [
        [
            family,
            _pct(prf.precision),
            _pct(prf.recall),
            _pct(prf.f1),
            f"{prf.tp}/{prf.predicted}",
            f"{prf.tp}/{prf.gold}",
        ]
        for family, prf in metrics.items()
    ]
    table = render_table(["Metric", "P", "R", "F1", "TP/pred", "TP/gold"], rows)
    body = "\n".join(notes)
    return f"{table}\n\n{body}" if body else table


# ═════════════════════════════════════════════════════════════════════════════
# Examples — curated inputs with gold annotations, ten per demo tab
# ═════════════════════════════════════════════════════════════════════════════
#
# Each example carries the inputs a tab needs plus a `gold` block the metrics
# above score the model's predictions against. Gold mentions are written as
# surface strings that occur verbatim in `text`, so they match by normalised
# text rather than by character offsets.
#
# Shapes
# ------
# NER / open relex / classification: ``groups`` is the label-group list the
# demo serialises into its group manager
# (``{"parent": ..., "labels": [...]}``).
# Joint relex groups additionally carry ``entities`` and ``relations``.
# Structuring labels use the ``field:type`` syntax the demo already
# understands; multi-level structuring instead carries a nested ``schema``
# template and a matching nested ``gold`` tree.
# Layout examples replace the plain paragraph with ``words``: one
# ``(word, x0, y0, x1, y1)`` row per word, in 0-1000 page coordinates.

# ── Examples: NER ────────────────────────────────────────────────────────────

NER_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Tech acquisition",
        "text": (
            "Satya Nadella, chief executive of Microsoft, said in Redmond on March 3, 2024 "
            "that the company would acquire Nuance Communications for $19.7 billion."
        ),
        "groups": [{"parent": "news", "labels": ["person", "organization", "location", "date", "money"]}],
        "gold": {
            "ner": [
                {"text": "Satya Nadella", "label": "person"},
                {"text": "Microsoft", "label": "organization"},
                {"text": "Redmond", "label": "location"},
                {"text": "March 3, 2024", "label": "date"},
                {"text": "Nuance Communications", "label": "organization"},
                {"text": "$19.7 billion", "label": "money"},
            ]
        },
    },
    {
        "name": "Clinical trial report",
        "text": (
            "Patients treated with metformin showed a marked reduction in HbA1c levels, "
            "while those receiving insulin glargine reported more frequent hypoglycemia."
        ),
        "groups": [{"parent": "biomedical", "labels": ["drug", "biomarker", "adverse event"]}],
        "gold": {
            "ner": [
                {"text": "metformin", "label": "drug"},
                {"text": "HbA1c", "label": "biomarker"},
                {"text": "insulin glargine", "label": "drug"},
                {"text": "hypoglycemia", "label": "adverse event"},
            ]
        },
    },
    {
        "name": "Court ruling",
        "text": (
            "The Supreme Court of California ruled on June 12, 1998 in Smith v. Nakamura "
            "that the arbitration clause in the employment contract was unenforceable."
        ),
        "groups": [{"parent": "legal", "labels": ["court", "date", "case", "legal provision"]}],
        "gold": {
            "ner": [
                {"text": "Supreme Court of California", "label": "court"},
                {"text": "June 12, 1998", "label": "date"},
                {"text": "Smith v. Nakamura", "label": "case"},
                {"text": "arbitration clause", "label": "legal provision"},
            ]
        },
    },
    {
        "name": "Market move",
        "text": (
            "Shares of Banco Santander fell 4.2% in Madrid trading after the ECB raised "
            "its benchmark rate to 4.5% on September 14, 2023."
        ),
        "groups": [{"parent": "finance", "labels": ["company", "institution", "city", "percentage", "date"]}],
        "gold": {
            "ner": [
                {"text": "Banco Santander", "label": "company"},
                {"text": "4.2%", "label": "percentage"},
                {"text": "Madrid", "label": "city"},
                {"text": "ECB", "label": "institution"},
                {"text": "4.5%", "label": "percentage"},
                {"text": "September 14, 2023", "label": "date"},
            ]
        },
    },
    {
        "name": "Match report",
        "text": (
            "Lionel Messi scored twice for Inter Miami against Orlando City at Chase Stadium "
            "on Saturday, sealing a 3-1 win."
        ),
        "groups": [{"parent": "sports", "labels": ["player", "team", "venue", "day", "score"]}],
        "gold": {
            "ner": [
                {"text": "Lionel Messi", "label": "player"},
                {"text": "Inter Miami", "label": "team"},
                {"text": "Orlando City", "label": "team"},
                {"text": "Chase Stadium", "label": "venue"},
                {"text": "Saturday", "label": "day"},
                {"text": "3-1", "label": "score"},
            ]
        },
    },
    {
        "name": "Recipe steps",
        "text": (
            "Preheat the oven to 180°C, whisk two eggs with 200 g of caster sugar, "
            "then fold in the sifted flour and bake for 25 minutes."
        ),
        "groups": [{"parent": "cooking", "labels": ["ingredient", "quantity", "temperature", "duration", "equipment"]}],
        "gold": {
            "ner": [
                {"text": "oven", "label": "equipment"},
                {"text": "180°C", "label": "temperature"},
                {"text": "two", "label": "quantity"},
                {"text": "eggs", "label": "ingredient"},
                {"text": "200 g", "label": "quantity"},
                {"text": "caster sugar", "label": "ingredient"},
                {"text": "flour", "label": "ingredient"},
                {"text": "25 minutes", "label": "duration"},
            ]
        },
    },
    {
        "name": "Travel itinerary",
        "text": (
            "The overnight train from Zurich to Milan departs at 22:40 from platform 7 "
            "and arrives at Milano Centrale early the next morning."
        ),
        "groups": [{"parent": "travel", "labels": ["transport", "city", "time", "station", "platform"]}],
        "gold": {
            "ner": [
                {"text": "train", "label": "transport"},
                {"text": "Zurich", "label": "city"},
                {"text": "Milan", "label": "city"},
                {"text": "22:40", "label": "time"},
                {"text": "platform 7", "label": "platform"},
                {"text": "Milano Centrale", "label": "station"},
            ]
        },
    },
    {
        "name": "Resume snippet",
        "text": (
            "Priya Raman worked as a senior data engineer at Infosys in Bengaluru from 2016 "
            "to 2021 and holds an M.Sc. in Computer Science from IIT Madras."
        ),
        "groups": [{"parent": "resume", "labels": ["person", "job title", "company", "city", "year", "degree", "university"]}],
        "gold": {
            "ner": [
                {"text": "Priya Raman", "label": "person"},
                {"text": "senior data engineer", "label": "job title"},
                {"text": "Infosys", "label": "company"},
                {"text": "Bengaluru", "label": "city"},
                {"text": "2016", "label": "year"},
                {"text": "2021", "label": "year"},
                {"text": "M.Sc. in Computer Science", "label": "degree"},
                {"text": "IIT Madras", "label": "university"},
            ]
        },
    },
    {
        "name": "Product listing",
        "text": (
            "The Anker 737 Power Bank with 24,000 mAh capacity is listed at $109.99 on Amazon "
            "and ships free to Germany."
        ),
        "groups": [{"parent": "ecommerce", "labels": ["product", "specification", "price", "retailer", "country"]}],
        "gold": {
            "ner": [
                {"text": "Anker 737 Power Bank", "label": "product"},
                {"text": "24,000 mAh", "label": "specification"},
                {"text": "$109.99", "label": "price"},
                {"text": "Amazon", "label": "retailer"},
                {"text": "Germany", "label": "country"},
            ]
        },
    },
    {
        "name": "ML paper abstract",
        "text": (
            "We fine-tuned a RoBERTa-large encoder on the CoNLL-2003 corpus using 8 NVIDIA A100 "
            "GPUs, reaching an F1 score of 93.4 after three epochs."
        ),
        "groups": [{"parent": "scientific", "labels": ["model", "dataset", "hardware", "metric", "value"]}],
        "gold": {
            "ner": [
                {"text": "RoBERTa-large", "label": "model"},
                {"text": "CoNLL-2003", "label": "dataset"},
                {"text": "NVIDIA A100", "label": "hardware"},
                {"text": "F1 score", "label": "metric"},
                {"text": "93.4", "label": "value"},
            ]
        },
    },
]


# ── Examples: Classification ─────────────────────────────────────────────────

CLASSIFICATION_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Product review sentiment",
        "text": "The battery lasts barely three hours and the screen flickers constantly — I regret this purchase.",
        "groups": [{"parent": "sentiment", "labels": ["positive", "negative", "neutral"]}],
        "multi_label": False,
        "gold": {"classification": ["negative"]},
    },
    {
        "name": "Financial news topic",
        "text": "The central bank raised interest rates by 25 basis points in an effort to curb persistent inflation.",
        "groups": [{"parent": "topic", "labels": ["economics", "sports", "health", "entertainment"]}],
        "multi_label": False,
        "gold": {"classification": ["economics"]},
    },
    {
        "name": "Support intent (multi-label)",
        "text": "Can you cancel my subscription and refund the payment you took last week?",
        "groups": [{"parent": "intent", "labels": ["cancel subscription", "request refund", "technical support", "greeting"]}],
        "multi_label": True,
        "gold": {"classification": ["cancel subscription", "request refund"]},
    },
    {
        "name": "Spam detection",
        "text": "CONGRATULATIONS! You have been selected to receive a free iPhone. Click here now to claim your prize!!!",
        "groups": [{"parent": "spam", "labels": ["spam", "legitimate"]}],
        "multi_label": False,
        "gold": {"classification": ["spam"]},
    },
    {
        "name": "Register / tone",
        "text": "Per our previous correspondence, kindly find attached the revised statement of work for your review.",
        "groups": [{"parent": "register", "labels": ["formal", "informal"]}],
        "multi_label": False,
        "gold": {"classification": ["formal"]},
    },
    {
        "name": "Emotion",
        "text": "I can't believe they cancelled the show right before the finale. I'm absolutely furious about it.",
        "groups": [{"parent": "emotion", "labels": ["anger", "joy", "sadness", "fear"]}],
        "multi_label": False,
        "gold": {"classification": ["anger"]},
    },
    {
        "name": "News section",
        "text": "Real Madrid secured a late victory over Atletico in a tense Madrid derby at the Bernabeu.",
        "groups": [{"parent": "section", "labels": ["sports", "politics", "technology", "business"]}],
        "multi_label": False,
        "gold": {"classification": ["sports"]},
    },
    {
        "name": "Toxicity",
        "text": "Thanks for the detailed review — I have addressed all of your comments in the latest commit.",
        "groups": [{"parent": "moderation", "labels": ["toxic", "non-toxic"]}],
        "multi_label": False,
        "gold": {"classification": ["non-toxic"]},
    },
    {
        "name": "Medical triage",
        "text": "Patient reports crushing chest pain radiating to the left arm together with shortness of breath.",
        "groups": [{"parent": "triage", "labels": ["emergency", "routine", "follow-up"]}],
        "multi_label": False,
        "gold": {"classification": ["emergency"]},
    },
    {
        "name": "Policy brief (multi-label)",
        "text": (
            "The new EU regulation on artificial intelligence will require companies to audit their "
            "models, raising compliance costs across the technology sector."
        ),
        "groups": [{"parent": "tags", "labels": ["regulation", "technology", "sports", "agriculture"]}],
        "multi_label": True,
        "gold": {"classification": ["regulation", "technology"]},
    },
]


# ── Examples: Open relation extraction ───────────────────────────────────────

OPEN_RELEX_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Biography",
        "text": "Marie Curie was born in Warsaw and later worked at the University of Paris.",
        "groups": [{"parent": "bio", "labels": ["born in", "worked at"]}],
        "gold": {
            "relations": [
                {"head": "Marie Curie", "relation": "born in", "tail": "Warsaw"},
                {"head": "Marie Curie", "relation": "worked at", "tail": "University of Paris"},
            ]
        },
    },
    {
        "name": "Acquisition",
        "text": "Google acquired DeepMind in 2014 for a reported 500 million dollars.",
        "groups": [{"parent": "business", "labels": ["acquired", "founded by"]}],
        "gold": {"relations": [{"head": "Google", "relation": "acquired", "tail": "DeepMind"}]},
    },
    {
        "name": "Collaboration",
        "text": "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.",
        "groups": [{"parent": "history", "labels": ["collaborated with", "worked on"]}],
        "gold": {
            "relations": [
                {"head": "Ada Lovelace", "relation": "collaborated with", "tail": "Charles Babbage"},
                {"head": "Ada Lovelace", "relation": "worked on", "tail": "Analytical Engine"},
            ]
        },
    },
    {
        "name": "Geography",
        "text": "The Amazon River flows through Brazil before emptying into the Atlantic Ocean.",
        "groups": [{"parent": "geo", "labels": ["flows through", "empties into"]}],
        "gold": {
            "relations": [
                {"head": "Amazon River", "relation": "flows through", "tail": "Brazil"},
                {"head": "Amazon River", "relation": "empties into", "tail": "Atlantic Ocean"},
            ]
        },
    },
    {
        "name": "Drug label",
        "text": "Aspirin is used to treat mild pain and is contraindicated in patients with peptic ulcers.",
        "groups": [{"parent": "pharma", "labels": ["treats", "contraindicated in"]}],
        "gold": {
            "relations": [
                {"head": "Aspirin", "relation": "treats", "tail": "mild pain"},
                {"head": "Aspirin", "relation": "contraindicated in", "tail": "peptic ulcers"},
            ]
        },
    },
    {
        "name": "Executive succession",
        "text": "Satya Nadella succeeded Steve Ballmer as chief executive of Microsoft in 2014.",
        "groups": [{"parent": "corporate", "labels": ["succeeded", "CEO of"]}],
        "gold": {
            "relations": [
                {"head": "Satya Nadella", "relation": "succeeded", "tail": "Steve Ballmer"},
                {"head": "Satya Nadella", "relation": "CEO of", "tail": "Microsoft"},
            ]
        },
    },
    {
        "name": "Manufacturing",
        "text": "Toyota manufactures the Corolla in Derbyshire.",
        "groups": [{"parent": "industry", "labels": ["manufactures", "manufactured in"]}],
        "gold": {
            "relations": [
                {"head": "Toyota", "relation": "manufactures", "tail": "Corolla"},
                {"head": "Corolla", "relation": "manufactured in", "tail": "Derbyshire"},
            ]
        },
    },
    {
        "name": "Open source",
        "text": "Python was created by Guido van Rossum and is maintained by the Python Software Foundation.",
        "groups": [{"parent": "software", "labels": ["created by", "maintained by"]}],
        "gold": {
            "relations": [
                {"head": "Python", "relation": "created by", "tail": "Guido van Rossum"},
                {"head": "Python", "relation": "maintained by", "tail": "Python Software Foundation"},
            ]
        },
    },
    {
        "name": "Landmark",
        "text": "The Eiffel Tower is located in Paris and was designed by Gustave Eiffel.",
        "groups": [{"parent": "culture", "labels": ["located in", "designed by"]}],
        "gold": {
            "relations": [
                {"head": "Eiffel Tower", "relation": "located in", "tail": "Paris"},
                {"head": "Eiffel Tower", "relation": "designed by", "tail": "Gustave Eiffel"},
            ]
        },
    },
    {
        "name": "Streaming",
        "text": "Netflix produced Stranger Things, which stars Millie Bobby Brown.",
        "groups": [{"parent": "media", "labels": ["produced", "stars"]}],
        "gold": {
            "relations": [
                {"head": "Netflix", "relation": "produced", "tail": "Stranger Things"},
                {"head": "Stranger Things", "relation": "stars", "tail": "Millie Bobby Brown"},
            ]
        },
    },
]


# ── Examples: Joint NER + relations ──────────────────────────────────────────

JOINT_RELEX_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Apple leadership",
        "text": "Tim Cook has led Apple since 2011, when he took over from Steve Jobs in Cupertino.",
        "groups": [{
            "parent": "corporate",
            "entities": ["person", "organization", "location"],
            "relations": ["CEO of", "succeeded"],
        }],
        "gold": {
            "ner": [
                {"text": "Tim Cook", "label": "person"},
                {"text": "Apple", "label": "organization"},
                {"text": "Steve Jobs", "label": "person"},
                {"text": "Cupertino", "label": "location"},
            ],
            "relations": [
                {"head": "Tim Cook", "relation": "CEO of", "tail": "Apple"},
                {"head": "Tim Cook", "relation": "succeeded", "tail": "Steve Jobs"},
            ],
        },
    },
    {
        "name": "Head of state",
        "text": "Angela Merkel served as Chancellor of Germany and studied physics in Leipzig.",
        "groups": [{
            "parent": "politics",
            "entities": ["person", "country", "city"],
            "relations": ["leader of", "studied in"],
        }],
        "gold": {
            "ner": [
                {"text": "Angela Merkel", "label": "person"},
                {"text": "Germany", "label": "country"},
                {"text": "Leipzig", "label": "city"},
            ],
            "relations": [
                {"head": "Angela Merkel", "relation": "leader of", "tail": "Germany"},
                {"head": "Angela Merkel", "relation": "studied in", "tail": "Leipzig"},
            ],
        },
    },
    {
        "name": "Vaccine partnership",
        "text": "Pfizer developed Comirnaty in partnership with BioNTech, which is based in Mainz.",
        "groups": [{
            "parent": "pharma",
            "entities": ["company", "product", "city"],
            "relations": ["developed", "partnered with", "based in"],
        }],
        "gold": {
            "ner": [
                {"text": "Pfizer", "label": "company"},
                {"text": "Comirnaty", "label": "product"},
                {"text": "BioNTech", "label": "company"},
                {"text": "Mainz", "label": "city"},
            ],
            "relations": [
                {"head": "Pfizer", "relation": "developed", "tail": "Comirnaty"},
                {"head": "Pfizer", "relation": "partnered with", "tail": "BioNTech"},
                {"head": "BioNTech", "relation": "based in", "tail": "Mainz"},
            ],
        },
    },
    {
        "name": "Tennis final",
        "text": "Serena Williams won Wimbledon in 2016, defeating Angelique Kerber in the final.",
        "groups": [{
            "parent": "sports",
            "entities": ["player", "tournament", "year"],
            "relations": ["won", "defeated"],
        }],
        "gold": {
            "ner": [
                {"text": "Serena Williams", "label": "player"},
                {"text": "Wimbledon", "label": "tournament"},
                {"text": "2016", "label": "year"},
                {"text": "Angelique Kerber", "label": "player"},
            ],
            "relations": [
                {"head": "Serena Williams", "relation": "won", "tail": "Wimbledon"},
                {"head": "Serena Williams", "relation": "defeated", "tail": "Angelique Kerber"},
            ],
        },
    },
    {
        "name": "Cloud infrastructure",
        "text": "Amazon Web Services runs a data centre in Dublin that is managed by Sarah O'Brien.",
        "groups": [{
            "parent": "infrastructure",
            "entities": ["company", "facility", "city", "person"],
            "relations": ["operates", "located in", "managed by"],
        }],
        "gold": {
            "ner": [
                {"text": "Amazon Web Services", "label": "company"},
                {"text": "data centre", "label": "facility"},
                {"text": "Dublin", "label": "city"},
                {"text": "Sarah O'Brien", "label": "person"},
            ],
            "relations": [
                {"head": "Amazon Web Services", "relation": "operates", "tail": "data centre"},
                {"head": "data centre", "relation": "located in", "tail": "Dublin"},
                {"head": "data centre", "relation": "managed by", "tail": "Sarah O'Brien"},
            ],
        },
    },
    {
        "name": "Book metadata",
        "text": "The novel Beloved was written by Toni Morrison and published by Alfred A. Knopf in 1987.",
        "groups": [{
            "parent": "publishing",
            "entities": ["book", "person", "publisher", "year"],
            "relations": ["written by", "published by"],
        }],
        "gold": {
            "ner": [
                {"text": "Beloved", "label": "book"},
                {"text": "Toni Morrison", "label": "person"},
                {"text": "Alfred A. Knopf", "label": "publisher"},
                {"text": "1987", "label": "year"},
            ],
            "relations": [
                {"head": "Beloved", "relation": "written by", "tail": "Toni Morrison"},
                {"head": "Beloved", "relation": "published by", "tail": "Alfred A. Knopf"},
            ],
        },
    },
    {
        "name": "Drug safety",
        "text": "Ibuprofen relieves inflammation but may cause gastric bleeding in elderly patients.",
        "groups": [{
            "parent": "pharmacovigilance",
            "entities": ["drug", "symptom", "adverse event", "population"],
            "relations": ["treats", "causes"],
        }],
        "gold": {
            "ner": [
                {"text": "Ibuprofen", "label": "drug"},
                {"text": "inflammation", "label": "symptom"},
                {"text": "gastric bleeding", "label": "adverse event"},
                {"text": "elderly patients", "label": "population"},
            ],
            "relations": [
                {"head": "Ibuprofen", "relation": "treats", "tail": "inflammation"},
                {"head": "Ibuprofen", "relation": "causes", "tail": "gastric bleeding"},
            ],
        },
    },
    {
        "name": "Automotive alliance",
        "text": "Renault formed an alliance with Nissan and moved its headquarters to Amsterdam.",
        "groups": [{
            "parent": "corporate",
            "entities": ["company", "city"],
            "relations": ["allied with", "headquartered in"],
        }],
        "gold": {
            "ner": [
                {"text": "Renault", "label": "company"},
                {"text": "Nissan", "label": "company"},
                {"text": "Amsterdam", "label": "city"},
            ],
            "relations": [
                {"head": "Renault", "relation": "allied with", "tail": "Nissan"},
                {"head": "Renault", "relation": "headquartered in", "tail": "Amsterdam"},
            ],
        },
    },
    {
        "name": "Trade route",
        "text": "Kenya exports tea to the United Kingdom through the port of Mombasa.",
        "groups": [{
            "parent": "trade",
            "entities": ["country", "product", "city"],
            "relations": ["exports", "exports to", "ships through"],
        }],
        "gold": {
            "ner": [
                {"text": "Kenya", "label": "country"},
                {"text": "tea", "label": "product"},
                {"text": "United Kingdom", "label": "country"},
                {"text": "Mombasa", "label": "city"},
            ],
            "relations": [
                {"head": "Kenya", "relation": "exports", "tail": "tea"},
                {"head": "Kenya", "relation": "exports to", "tail": "United Kingdom"},
                {"head": "Kenya", "relation": "ships through", "tail": "Mombasa"},
            ],
        },
    },
    {
        "name": "Academic profile",
        "text": "Professor Hiroshi Tanaka teaches robotics at the University of Tokyo and advises Sony.",
        "groups": [{
            "parent": "academia",
            "entities": ["person", "field", "university", "company"],
            "relations": ["teaches", "works at", "advises"],
        }],
        "gold": {
            "ner": [
                {"text": "Hiroshi Tanaka", "label": "person"},
                {"text": "robotics", "label": "field"},
                {"text": "University of Tokyo", "label": "university"},
                {"text": "Sony", "label": "company"},
            ],
            "relations": [
                {"head": "Hiroshi Tanaka", "relation": "teaches", "tail": "robotics"},
                {"head": "Hiroshi Tanaka", "relation": "works at", "tail": "University of Tokyo"},
                {"head": "Hiroshi Tanaka", "relation": "advises", "tail": "Sony"},
            ],
        },
    },
]


# ── Examples: Structuring ────────────────────────────────────────────────────

STRUCTURING_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Flight schedule (3 records)",
        "text": (
            "Flight LH441 departs Frankfurt at 10:15 and lands in Houston at 14:50. "
            "Flight LH442 leaves Houston at 17:20 and reaches Frankfurt at 09:35. "
            "Flight LH610 departs Munich at 07:05 and arrives in Cairo at 11:40."
        ),
        "groups": [{
            "parent": "flight",
            "labels": ["flight_number:str", "origin:str", "departure_time:str", "destination:str", "arrival_time:str"],
        }],
        "gold": {
            "structuring": {
                "flight": [
                    {
                        "flight_number": "LH441",
                        "origin": "Frankfurt",
                        "departure_time": "10:15",
                        "destination": "Houston",
                        "arrival_time": "14:50",
                    },
                    {
                        "flight_number": "LH442",
                        "origin": "Houston",
                        "departure_time": "17:20",
                        "destination": "Frankfurt",
                        "arrival_time": "09:35",
                    },
                    {
                        "flight_number": "LH610",
                        "origin": "Munich",
                        "departure_time": "07:05",
                        "destination": "Cairo",
                        "arrival_time": "11:40",
                    },
                ]
            }
        },
    },
    {
        "name": "Staff directory (4 records)",
        "text": (
            "Maria Gonzalez, 34, is a cardiologist in Madrid. Her colleague Ahmed Farouk, 41, "
            "works as a radiologist in Cairo. Jonas Weber, 29, is a paediatrician in Bern, and "
            "Sofia Rossi, 52, practises as a neurologist in Bologna."
        ),
        "groups": [{
            "parent": "person",
            "labels": ["name:str", "age:int", "occupation:str", "city:str"],
        }],
        "gold": {
            "structuring": {
                "person": [
                    {"name": "Maria Gonzalez", "age": "34", "occupation": "cardiologist", "city": "Madrid"},
                    {"name": "Ahmed Farouk", "age": "41", "occupation": "radiologist", "city": "Cairo"},
                    {"name": "Jonas Weber", "age": "29", "occupation": "paediatrician", "city": "Bern"},
                    {"name": "Sofia Rossi", "age": "52", "occupation": "neurologist", "city": "Bologna"},
                ]
            }
        },
    },
    {
        "name": "Purchase order lines (5 records)",
        "text": (
            "Order 44-B contains 3 keyboards at 45 euros each, 2 monitors at 210 euros each, "
            "5 USB hubs at 19 euros each, 1 docking station at 175 euros, and 4 webcams at "
            "62 euros each."
        ),
        "groups": [{
            "parent": "line_item",
            "labels": ["item:str", "quantity:int", "unit_price:str"],
        }],
        "gold": {
            "structuring": {
                "line_item": [
                    {"item": "keyboards", "quantity": "3", "unit_price": "45 euros"},
                    {"item": "monitors", "quantity": "2", "unit_price": "210 euros"},
                    {"item": "USB hubs", "quantity": "5", "unit_price": "19 euros"},
                    {"item": "docking station", "quantity": "1", "unit_price": "175 euros"},
                    {"item": "webcams", "quantity": "4", "unit_price": "62 euros"},
                ]
            }
        },
    },
    {
        "name": "Conference programme (3 records)",
        "text": (
            "The morning programme opens with Scaling Retrieval by Ana Petrova at 09:30 in Hall A, "
            "continues with Robust Evaluation by Kenji Sato at 11:00 in Hall B, and closes with "
            "Open Weights by Leila Haddad at 14:15 in Hall C."
        ),
        "groups": [{
            "parent": "talk",
            "labels": ["title:str", "speaker:str", "time:str", "room:str"],
        }],
        "gold": {
            "structuring": {
                "talk": [
                    {"title": "Scaling Retrieval", "speaker": "Ana Petrova", "time": "09:30", "room": "Hall A"},
                    {"title": "Robust Evaluation", "speaker": "Kenji Sato", "time": "11:00", "room": "Hall B"},
                    {"title": "Open Weights", "speaker": "Leila Haddad", "time": "14:15", "room": "Hall C"},
                ]
            }
        },
    },
    {
        "name": "Match results (4 records)",
        "text": (
            "Saturday results: Arsenal 2-1 Chelsea, Liverpool 0-0 Everton, "
            "Manchester City 3-2 Tottenham, Newcastle 1-4 Brighton."
        ),
        "groups": [{
            "parent": "match",
            "labels": ["home_team:str", "away_team:str", "score:str"],
        }],
        "gold": {
            "structuring": {
                "match": [
                    {"home_team": "Arsenal", "away_team": "Chelsea", "score": "2-1"},
                    {"home_team": "Liverpool", "away_team": "Everton", "score": "0-0"},
                    {"home_team": "Manchester City", "away_team": "Tottenham", "score": "3-2"},
                    {"home_team": "Newcastle", "away_team": "Brighton", "score": "1-4"},
                ]
            }
        },
    },
    {
        "name": "Discharge medication (2 records)",
        "text": (
            "On discharge the patient was prescribed metformin 500 mg twice daily and "
            "ramipril 5 mg once daily."
        ),
        "groups": [{
            "parent": "prescription",
            "labels": ["medication:str", "dose:str", "frequency:str"],
        }],
        "gold": {
            "structuring": {
                "prescription": [
                    {"medication": "metformin", "dose": "500 mg", "frequency": "twice daily"},
                    {"medication": "ramipril", "dose": "5 mg", "frequency": "once daily"},
                ]
            }
        },
    },
    {
        "name": "Hotel availability (4 records)",
        "text": (
            "Available rooms: the Grand Hotel offers a double at 180 euros, Pension Alpina a "
            "single at 95 euros, Hotel Marina a suite at 340 euros, and Casa Verde a twin at "
            "120 euros."
        ),
        "groups": [{
            "parent": "room",
            "labels": ["hotel:str", "room_type:str", "price:str"],
        }],
        "gold": {
            "structuring": {
                "room": [
                    {"hotel": "Grand Hotel", "room_type": "double", "price": "180 euros"},
                    {"hotel": "Pension Alpina", "room_type": "single", "price": "95 euros"},
                    {"hotel": "Hotel Marina", "room_type": "suite", "price": "340 euros"},
                    {"hotel": "Casa Verde", "room_type": "twin", "price": "120 euros"},
                ]
            }
        },
    },
    {
        "name": "Weekly forecast (5 records)",
        "text": (
            "Forecast: Monday sunny with a high of 24°C, Tuesday cloudy at 19°C, Wednesday rainy "
            "at 16°C, Thursday windy at 18°C, Friday sunny again at 26°C."
        ),
        "groups": [{
            "parent": "forecast",
            "labels": ["day:str", "conditions:str", "temperature:str"],
        }],
        "gold": {
            "structuring": {
                "forecast": [
                    {"day": "Monday", "conditions": "sunny", "temperature": "24°C"},
                    {"day": "Tuesday", "conditions": "cloudy", "temperature": "19°C"},
                    {"day": "Wednesday", "conditions": "rainy", "temperature": "16°C"},
                    {"day": "Thursday", "conditions": "windy", "temperature": "18°C"},
                    {"day": "Friday", "conditions": "sunny", "temperature": "26°C"},
                ]
            }
        },
    },
    {
        "name": "Course catalogue (3 records)",
        "text": (
            "Autumn courses: CS101 Introduction to Programming taught by Alan Reyes on Mondays, "
            "CS204 Data Structures taught by Mei Lin on Wednesdays, and CS310 Machine Learning "
            "taught by Omar Aziz on Fridays."
        ),
        "groups": [{
            "parent": "course",
            "labels": ["code:str", "title:str", "instructor:str", "day:str"],
        }],
        "gold": {
            "structuring": {
                "course": [
                    {"code": "CS101", "title": "Introduction to Programming", "instructor": "Alan Reyes", "day": "Mondays"},
                    {"code": "CS204", "title": "Data Structures", "instructor": "Mei Lin", "day": "Wednesdays"},
                    {"code": "CS310", "title": "Machine Learning", "instructor": "Omar Aziz", "day": "Fridays"},
                ]
            }
        },
    },
    {
        "name": "Parcel tracking (2 records)",
        "text": (
            "Parcel DE-8891 left Rotterdam on 2 February 2025 bound for Lyon; parcel DE-8892 "
            "left Hamburg on 3 February 2025 bound for Prague."
        ),
        "groups": [{
            "parent": "parcel",
            "labels": ["parcel_id:str", "origin:str", "destination:str", "ship_date:str"],
        }],
        "gold": {
            "structuring": {
                "parcel": [
                    {
                        "parcel_id": "DE-8891",
                        "origin": "Rotterdam",
                        "destination": "Lyon",
                        "ship_date": "2 February 2025",
                    },
                    {
                        "parcel_id": "DE-8892",
                        "origin": "Hamburg",
                        "destination": "Prague",
                        "ship_date": "3 February 2025",
                    },
                ]
            }
        },
    },
]


# ── Examples: Multi-level structuring ────────────────────────────────────────

# Nested schemas, written as the JSON template `GLiFormerSchema.add_structure`
# compiles: a scalar field maps to its type name, an object to a template, and
# a repeated child to a one-element list holding the child's template. Gold
# instances mirror that shape, so a child value only counts when it sits under
# the same parent record on both sides.

MULTI_LEVEL_STRUCTURING_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Logistics depots and routes (3 levels)",
        "text": (
            "Northwind Logistics operates two depots. Depot Rotterdam is managed by Iris Bakker "
            "and handles route R-12 to Lyon and route R-19 to Milan. Depot Hamburg is managed by "
            "Lars Petersen and handles route R-04 to Prague."
        ),
        "schema": {
            "company": {
                "name": "str",
                "depots": [{
                    "depot": "str",
                    "manager": "str",
                    "routes": [{"route_id": "str", "destination": "str"}],
                }],
            }
        },
        "gold": {
            "structuring": {
                "company": [{
                    "name": "Northwind Logistics",
                    "depots": [
                        {
                            "depot": "Rotterdam",
                            "manager": "Iris Bakker",
                            "routes": [
                                {"route_id": "R-12", "destination": "Lyon"},
                                {"route_id": "R-19", "destination": "Milan"},
                            ],
                        },
                        {
                            "depot": "Hamburg",
                            "manager": "Lars Petersen",
                            "routes": [{"route_id": "R-04", "destination": "Prague"}],
                        },
                    ],
                }]
            }
        },
    },
    {
        "name": "University outline (4 levels)",
        "text": (
            "University Ostmark University\n"
            "  Faculty Natural Sciences, dean Anika Vogel\n"
            "    Department Physics, head Marek Novak\n"
            "      Module PHY-201 Thermodynamics, credits 6\n"
            "      Module PHY-305 Quantum Optics, credits 8\n"
            "    Department Chemistry, head Elif Kaur\n"
            "      Module CHE-110 Organic Synthesis, credits 5"
        ),
        "schema": {
            "university": {
                "university": "str",
                "faculties": [{
                    "faculty": "str",
                    "dean": "str",
                    "departments": [{
                        "department": "str",
                        "head": "str",
                        "modules": [{"code": "str", "title": "str", "credits": "int"}],
                    }],
                }],
            }
        },
        "gold": {
            "structuring": {
                "university": [{
                    "university": "Ostmark University",
                    "faculties": [{
                        "faculty": "Natural Sciences",
                        "dean": "Anika Vogel",
                        "departments": [
                            {
                                "department": "Physics",
                                "head": "Marek Novak",
                                "modules": [
                                    {"code": "PHY-201", "title": "Thermodynamics", "credits": "6"},
                                    {"code": "PHY-305", "title": "Quantum Optics", "credits": "8"},
                                ],
                            },
                            {
                                "department": "Chemistry",
                                "head": "Elif Kaur",
                                "modules": [
                                    {"code": "CHE-110", "title": "Organic Synthesis", "credits": "5"},
                                ],
                            },
                        ],
                    }],
                }]
            }
        },
    },
    {
        "name": "Invoice with seller object and lines",
        "text": (
            "Invoice INV-2041 was issued by Baltic Supplies, VAT number LT44920, to Meridian Labs. "
            "The order lists 12 pipettes at 14 euros, 4 centrifuge rotors at 320 euros and "
            "30 sample vials at 3 euros."
        ),
        "schema": {
            "invoice": {
                "invoice_id": "str",
                "seller": {"name": "str", "vat_number": "str"},
                "buyer": "str",
                "lines": [{"item": "str", "quantity": "int", "unit_price": "str"}],
            }
        },
        "gold": {
            "structuring": {
                "invoice": [{
                    "invoice_id": "INV-2041",
                    "seller": {"name": "Baltic Supplies", "vat_number": "LT44920"},
                    "buyer": "Meridian Labs",
                    "lines": [
                        {"item": "pipettes", "quantity": "12", "unit_price": "14 euros"},
                        {"item": "centrifuge rotors", "quantity": "4", "unit_price": "320 euros"},
                        {"item": "sample vials", "quantity": "30", "unit_price": "3 euros"},
                    ],
                }]
            }
        },
    },
    {
        "name": "Ward round (3 patients with medication)",
        "text": (
            "Ward 3B round. Helena Marsh, 68, takes metformin 500 mg twice daily and ramipril "
            "5 mg once daily. Otto Brandt, 74, takes furosemide 40 mg once daily. Priya Raman, "
            "55, takes levothyroxine 75 mcg every morning."
        ),
        "schema": {
            "patient": {
                "name": "str",
                "age": "int",
                "medications": [{"drug": "str", "dose": "str", "frequency": "str"}],
            }
        },
        "gold": {
            "structuring": {
                "patient": [
                    {
                        "name": "Helena Marsh",
                        "age": "68",
                        "medications": [
                            {"drug": "metformin", "dose": "500 mg", "frequency": "twice daily"},
                            {"drug": "ramipril", "dose": "5 mg", "frequency": "once daily"},
                        ],
                    },
                    {
                        "name": "Otto Brandt",
                        "age": "74",
                        "medications": [
                            {"drug": "furosemide", "dose": "40 mg", "frequency": "once daily"},
                        ],
                    },
                    {
                        "name": "Priya Raman",
                        "age": "55",
                        "medications": [
                            {"drug": "levothyroxine", "dose": "75 mcg", "frequency": "every morning"},
                        ],
                    },
                ]
            }
        },
    },
    {
        "name": "Conference tracks and sessions",
        "text": (
            "Track Retrieval opens with Scaling Dense Indexes by Ana Petrova at 09:30 and "
            "continues with Hybrid Rerankers by Kenji Sato at 11:00. Track Safety runs "
            "Red Teaming at Scale by Leila Haddad at 13:15 and Auditing Agents by Tomas Ruiz "
            "at 15:00."
        ),
        "schema": {
            "track": {
                "track": "str",
                "sessions": [{"title": "str", "speaker": "str", "time": "str"}],
            }
        },
        "gold": {
            "structuring": {
                "track": [
                    {
                        "track": "Retrieval",
                        "sessions": [
                            {"title": "Scaling Dense Indexes", "speaker": "Ana Petrova", "time": "09:30"},
                            {"title": "Hybrid Rerankers", "speaker": "Kenji Sato", "time": "11:00"},
                        ],
                    },
                    {
                        "track": "Safety",
                        "sessions": [
                            {"title": "Red Teaming at Scale", "speaker": "Leila Haddad", "time": "13:15"},
                            {"title": "Auditing Agents", "speaker": "Tomas Ruiz", "time": "15:00"},
                        ],
                    },
                ]
            }
        },
    },
    {
        "name": "League teams and squads",
        "text": (
            "Northern Division. Aurora FC, coached by Greta Lindholm, fields Dario Almeida as "
            "striker and Yara Toledano as goalkeeper. Harbour United, coached by Boris Radu, "
            "fields Noor Kaur as midfielder and Hugo Fontaine as defender."
        ),
        "schema": {
            "team": {
                "team": "str",
                "coach": "str",
                "players": [{"player": "str", "position": "str"}],
            }
        },
        "gold": {
            "structuring": {
                "team": [
                    {
                        "team": "Aurora FC",
                        "coach": "Greta Lindholm",
                        "players": [
                            {"player": "Dario Almeida", "position": "striker"},
                            {"player": "Yara Toledano", "position": "goalkeeper"},
                        ],
                    },
                    {
                        "team": "Harbour United",
                        "coach": "Boris Radu",
                        "players": [
                            {"player": "Noor Kaur", "position": "midfielder"},
                            {"player": "Hugo Fontaine", "position": "defender"},
                        ],
                    },
                ]
            }
        },
    },
    {
        "name": "Release notes (typed date, nested issues)",
        "text": (
            "Release 4.2.0 shipped on 2026-03-14. Component parser fixes issue BUG-118 with "
            "severity high and issue BUG-140 with severity low. Component scheduler fixes issue "
            "BUG-207 with severity critical."
        ),
        "schema": {
            "release": {
                "version": "str",
                "release_date": "date",
                "components": [{
                    "component": "str",
                    "issues": [{"issue_id": "str", "severity": "str"}],
                }],
            }
        },
        "gold": {
            "structuring": {
                "release": [{
                    "version": "4.2.0",
                    "release_date": "2026-03-14",
                    "components": [
                        {
                            "component": "parser",
                            "issues": [
                                {"issue_id": "BUG-118", "severity": "high"},
                                {"issue_id": "BUG-140", "severity": "low"},
                            ],
                        },
                        {
                            "component": "scheduler",
                            "issues": [{"issue_id": "BUG-207", "severity": "critical"}],
                        },
                    ],
                }]
            }
        },
    },
    {
        "name": "Menu sections and dishes",
        "text": (
            "Casa Verde menu. Starters: gazpacho at 6 euros and croquettes at 7 euros. "
            "Mains: sea bass at 19 euros and lamb shank at 22 euros. Desserts: flan at 5 euros."
        ),
        "schema": {
            "menu": {
                "restaurant": "str",
                "sections": [{
                    "section": "str",
                    "dishes": [{"dish": "str", "price": "str"}],
                }],
            }
        },
        "gold": {
            "structuring": {
                "menu": [{
                    "restaurant": "Casa Verde",
                    "sections": [
                        {
                            "section": "Starters",
                            "dishes": [
                                {"dish": "gazpacho", "price": "6 euros"},
                                {"dish": "croquettes", "price": "7 euros"},
                            ],
                        },
                        {
                            "section": "Mains",
                            "dishes": [
                                {"dish": "sea bass", "price": "19 euros"},
                                {"dish": "lamb shank", "price": "22 euros"},
                            ],
                        },
                        {
                            "section": "Desserts",
                            "dishes": [{"dish": "flan", "price": "5 euros"}],
                        },
                    ],
                }]
            }
        },
    },
    {
        "name": "Itinerary days and activities",
        "text": (
            "Itinerary for Kyoto. Day 1: temple walk at 09:00, tea ceremony at 14:00. "
            "Day 2: bamboo grove at 08:30, river cruise at 16:45. Day 3: pottery workshop at 10:15."
        ),
        "schema": {
            "itinerary": {
                "destination": "str",
                "days": [{
                    "day": "str",
                    "activities": [{"activity": "str", "time": "str"}],
                }],
            }
        },
        "gold": {
            "structuring": {
                "itinerary": [{
                    "destination": "Kyoto",
                    "days": [
                        {
                            "day": "Day 1",
                            "activities": [
                                {"activity": "temple walk", "time": "09:00"},
                                {"activity": "tea ceremony", "time": "14:00"},
                            ],
                        },
                        {
                            "day": "Day 2",
                            "activities": [
                                {"activity": "bamboo grove", "time": "08:30"},
                                {"activity": "river cruise", "time": "16:45"},
                            ],
                        },
                        {
                            "day": "Day 3",
                            "activities": [{"activity": "pottery workshop", "time": "10:15"}],
                        },
                    ],
                }]
            }
        },
    },
    {
        "name": "Org chart outline (4 levels)",
        "text": (
            "Company Helios Robotics\n"
            "  Division Hardware, director Camille Chevalier\n"
            "    Team Actuators, lead Rasmus Haugen, headcount 9\n"
            "    Team Sensors, lead Frida Zeller, headcount 6\n"
            "  Division Software, director Dmitri Novak\n"
            "    Team Perception, lead Saoirse Walsh, headcount 12"
        ),
        "schema": {
            "company": {
                "company": "str",
                "divisions": [{
                    "division": "str",
                    "director": "str",
                    "teams": [{"team": "str", "lead": "str", "headcount": "int"}],
                }],
            }
        },
        "gold": {
            "structuring": {
                "company": [{
                    "company": "Helios Robotics",
                    "divisions": [
                        {
                            "division": "Hardware",
                            "director": "Camille Chevalier",
                            "teams": [
                                {"team": "Actuators", "lead": "Rasmus Haugen", "headcount": "9"},
                                {"team": "Sensors", "lead": "Frida Zeller", "headcount": "6"},
                            ],
                        },
                        {
                            "division": "Software",
                            "director": "Dmitri Novak",
                            "teams": [
                                {"team": "Perception", "lead": "Saoirse Walsh", "headcount": "12"},
                            ],
                        },
                    ],
                }]
            }
        },
    },
]


# ── Examples: Embedding / retrieval ──────────────────────────────────────────

EMBEDDING_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "EV factory news",
        "query": "Tesla is investing in a new factory in Texas.",
        "candidates": [
            "A new automotive plant is being built in Austin.",
            "SpaceX launched a rocket to the International Space Station.",
            "The stock market saw significant gains today.",
            "Electric vehicle production is expanding in the US.",
            "The weather in New York is sunny and warm.",
        ],
        "gold": {
            "relevant": [
                "A new automotive plant is being built in Austin.",
                "Electric vehicle production is expanding in the US.",
            ]
        },
    },
    {
        "name": "Password reset",
        "query": "How do I reset my password?",
        "candidates": [
            "Click 'Forgot password' to receive a reset link by email.",
            "Our office hours are 9am to 5pm on weekdays.",
            "Steps to recover access to your account credentials.",
            "The new pricing plan includes additional storage.",
            "We accept Visa and Mastercard for all subscriptions.",
        ],
        "gold": {
            "relevant": [
                "Click 'Forgot password' to receive a reset link by email.",
                "Steps to recover access to your account credentials.",
            ]
        },
    },
    {
        "name": "Vitamin D symptoms",
        "query": "symptoms of vitamin D deficiency",
        "candidates": [
            "Low vitamin D can cause bone pain and muscle weakness.",
            "Fatigue and frequent illness may signal insufficient vitamin D.",
            "Vitamin C is abundant in citrus fruit.",
            "Regular strength training builds lean muscle mass.",
            "The recommended daily intake of sodium is under 2 grams.",
        ],
        "gold": {
            "relevant": [
                "Low vitamin D can cause bone pain and muscle weakness.",
                "Fatigue and frequent illness may signal insufficient vitamin D.",
            ]
        },
    },
    {
        "name": "Python testing",
        "query": "best practices for unit testing in Python",
        "candidates": [
            "Use pytest fixtures to isolate test dependencies.",
            "Keep each test focused on a single behaviour.",
            "Django templates render HTML on the server.",
            "NumPy arrays support vectorised arithmetic.",
            "Docker images should be kept small for faster pulls.",
        ],
        "gold": {
            "relevant": [
                "Use pytest fixtures to isolate test dependencies.",
                "Keep each test focused on a single behaviour.",
            ]
        },
    },
    {
        "name": "Flight search",
        "query": "cheap flights from Berlin to Rome",
        "candidates": [
            "Budget airlines offer low-cost routes between BER and FCO.",
            "Compare ticket prices for Italian destinations departing Germany.",
            "The Colosseum opens to visitors at 8:30 every morning.",
            "Rail passes cover unlimited travel across Switzerland.",
            "Hotel occupancy in Barcelona rose sharply last summer.",
        ],
        "gold": {
            "relevant": [
                "Budget airlines offer low-cost routes between BER and FCO.",
                "Compare ticket prices for Italian destinations departing Germany.",
            ]
        },
    },
    {
        "name": "Factual lookup",
        "query": "What is the capital of Australia?",
        "candidates": [
            "Canberra has been the seat of the Australian government since 1927.",
            "Sydney is the largest city in New South Wales.",
            "Australia is home to unique marsupials such as the quokka.",
            "The capital of New Zealand is Wellington.",
            "Melbourne hosts the Australian Open every January.",
        ],
        "gold": {"relevant": ["Canberra has been the seat of the Australian government since 1927."]},
    },
    {
        "name": "Gluten-free baking",
        "query": "recipe for gluten-free banana bread",
        "candidates": [
            "Bake a banana loaf using almond flour instead of wheat.",
            "A wheat-free quick bread made with very ripe bananas.",
            "Sourdough starter needs daily feeding with rye flour.",
            "Bananas are a good source of potassium.",
            "Roast the vegetables at 200°C for forty minutes.",
        ],
        "gold": {
            "relevant": [
                "Bake a banana loaf using almond flour instead of wheat.",
                "A wheat-free quick bread made with very ripe bananas.",
            ]
        },
    },
    {
        "name": "Coral reefs",
        "query": "climate change impact on coral reefs",
        "candidates": [
            "Rising ocean temperatures cause widespread coral bleaching.",
            "Warmer seas threaten reef ecosystems worldwide.",
            "Deforestation in the Amazon slowed slightly last year.",
            "Solar panel efficiency has improved steadily since 2010.",
            "Scuba diving certification takes about four days.",
        ],
        "gold": {
            "relevant": [
                "Rising ocean temperatures cause widespread coral bleaching.",
                "Warmer seas threaten reef ecosystems worldwide.",
            ]
        },
    },
    {
        "name": "NER training",
        "query": "how to train a named entity recognition model",
        "candidates": [
            "Fine-tune a transformer encoder on labelled entity spans.",
            "Annotate a corpus with entity types before training.",
            "Convolutional networks dominate image classification benchmarks.",
            "Gradient boosting works well on tabular data.",
            "Set the learning rate schedule with a linear warmup.",
        ],
        "gold": {
            "relevant": [
                "Fine-tune a transformer encoder on labelled entity spans.",
                "Annotate a corpus with entity types before training.",
            ]
        },
    },
    {
        "name": "Refund policy",
        "query": "refund policy for damaged goods",
        "candidates": [
            "Items that arrive broken can be returned within 30 days for a full refund.",
            "Damaged shipments qualify for replacement or reimbursement.",
            "Orders above 50 euros ship free within the EU.",
            "Our warehouse is closed on public holidays.",
            "Gift cards are valid for twenty-four months after purchase.",
        ],
        "gold": {
            "relevant": [
                "Items that arrive broken can be returned within 30 days for a full refund.",
                "Damaged shipments qualify for replacement or reimbursement.",
            ]
        },
    },
]


# ── Examples: Multi-task ─────────────────────────────────────────────────────

MULTITASK_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Earnings call",
        "text": (
            "Nvidia reported revenue of $26.0 billion for the quarter ending July 28, 2024, "
            "and chief executive Jensen Huang said demand from data centre customers remained strong."
        ),
        "ner": [{"parent": "news", "labels": ["company", "person", "money", "date"]}],
        "classification": [{"parent": "topic", "labels": ["finance", "sports", "health"]}],
        "structuring": [{"parent": "earnings", "labels": ["company:str", "revenue:str", "period:str", "CEO:str"]}],
        "gold": {
            "ner": [
                {"text": "Nvidia", "label": "company"},
                {"text": "$26.0 billion", "label": "money"},
                {"text": "July 28, 2024", "label": "date"},
                {"text": "Jensen Huang", "label": "person"},
            ],
            "classification": ["finance"],
            "structuring": {
                "earnings": [{
                    "company": "Nvidia",
                    "revenue": "$26.0 billion",
                    "period": "July 28, 2024",
                    "CEO": "Jensen Huang",
                }]
            },
        },
    },
    {
        "name": "Hospital discharge",
        "text": (
            "Mrs. Helena Novak, 62, was discharged from Charles University Hospital on 4 April 2023 "
            "after treatment for pneumonia with intravenous amoxicillin."
        ),
        "ner": [{"parent": "clinical", "labels": ["person", "hospital", "date", "condition", "drug"]}],
        "classification": [{"parent": "document type", "labels": ["discharge summary", "invoice", "news article"]}],
        "structuring": [{"parent": "discharge", "labels": ["patient:str", "age:int", "hospital:str", "diagnosis:str", "medication:str"]}],
        "gold": {
            "ner": [
                {"text": "Helena Novak", "label": "person"},
                {"text": "Charles University Hospital", "label": "hospital"},
                {"text": "4 April 2023", "label": "date"},
                {"text": "pneumonia", "label": "condition"},
                {"text": "amoxicillin", "label": "drug"},
            ],
            "classification": ["discharge summary"],
            "structuring": {
                "discharge": [{
                    "patient": "Helena Novak",
                    "age": "62",
                    "hospital": "Charles University Hospital",
                    "diagnosis": "pneumonia",
                    "medication": "amoxicillin",
                }]
            },
        },
    },
    {
        "name": "Startup funding",
        "text": (
            "Berlin-based Helsing raised 209 million euros in a Series B round led by General Catalyst, "
            "bringing its valuation to 1.7 billion euros."
        ),
        "ner": [{"parent": "vc", "labels": ["company", "city", "money", "investor"]}],
        "classification": [{"parent": "topic", "labels": ["venture capital", "sports", "agriculture"]}],
        "structuring": [{"parent": "funding", "labels": ["company:str", "amount:str", "round:str", "lead_investor:str", "valuation:str"]}],
        "gold": {
            "ner": [
                {"text": "Helsing", "label": "company"},
                {"text": "Berlin", "label": "city"},
                {"text": "209 million euros", "label": "money"},
                {"text": "General Catalyst", "label": "investor"},
                {"text": "1.7 billion euros", "label": "money"},
            ],
            "classification": ["venture capital"],
            "structuring": {
                "funding": [{
                    "company": "Helsing",
                    "amount": "209 million euros",
                    "round": "Series B",
                    "lead_investor": "General Catalyst",
                    "valuation": "1.7 billion euros",
                }]
            },
        },
    },
    {
        "name": "Support ticket",
        "text": (
            "Ticket #4821: customer Laura Beck cannot log into the mobile app since the 3.2.1 update "
            "and requests a callback on +49 30 555 0198."
        ),
        "ner": [{"parent": "support", "labels": ["person", "product version", "phone number", "ticket id"]}],
        "classification": [{"parent": "intent", "labels": ["technical support", "request refund", "sales enquiry"]}],
        "structuring": [{"parent": "ticket", "labels": ["ticket_id:str", "customer:str", "issue:str", "contact:str"]}],
        "gold": {
            "ner": [
                {"text": "Laura Beck", "label": "person"},
                {"text": "3.2.1", "label": "product version"},
                {"text": "+49 30 555 0198", "label": "phone number"},
                {"text": "#4821", "label": "ticket id"},
            ],
            "classification": ["technical support"],
            "structuring": {
                "ticket": [{
                    "ticket_id": "#4821",
                    "customer": "Laura Beck",
                    "issue": "cannot log into the mobile app",
                    "contact": "+49 30 555 0198",
                }]
            },
        },
    },
    {
        "name": "Restaurant review",
        "text": (
            "We ate at Osteria Francescana in Modena last Friday; the tasting menu cost 320 euros "
            "per person and every course was outstanding."
        ),
        "ner": [{"parent": "review", "labels": ["restaurant", "city", "day", "price"]}],
        "classification": [{"parent": "sentiment", "labels": ["positive", "negative", "neutral"]}],
        "structuring": [{"parent": "visit", "labels": ["restaurant:str", "city:str", "price:str"]}],
        "gold": {
            "ner": [
                {"text": "Osteria Francescana", "label": "restaurant"},
                {"text": "Modena", "label": "city"},
                {"text": "Friday", "label": "day"},
                {"text": "320 euros", "label": "price"},
            ],
            "classification": ["positive"],
            "structuring": {
                "visit": [{
                    "restaurant": "Osteria Francescana",
                    "city": "Modena",
                    "price": "320 euros",
                }]
            },
        },
    },
    {
        "name": "Job advert",
        "text": (
            "Spotify is hiring a Machine Learning Engineer in Stockholm, offering a hybrid contract "
            "and a salary band of 65,000 to 85,000 euros."
        ),
        "ner": [{"parent": "hiring", "labels": ["company", "job title", "city", "salary"]}],
        "classification": [{"parent": "document type", "labels": ["job advert", "press release", "research paper"]}],
        "structuring": [{"parent": "vacancy", "labels": ["company:str", "title:str", "location:str", "salary:str", "contract:str"]}],
        "gold": {
            "ner": [
                {"text": "Spotify", "label": "company"},
                {"text": "Machine Learning Engineer", "label": "job title"},
                {"text": "Stockholm", "label": "city"},
                {"text": "65,000 to 85,000 euros", "label": "salary"},
            ],
            "classification": ["job advert"],
            "structuring": {
                "vacancy": [{
                    "company": "Spotify",
                    "title": "Machine Learning Engineer",
                    "location": "Stockholm",
                    "salary": "65,000 to 85,000 euros",
                    "contract": "hybrid",
                }]
            },
        },
    },
    {
        "name": "Weather bulletin",
        "text": (
            "The Met Office issued an amber warning for Cornwall on 18 December 2023, forecasting "
            "gusts of 80 mph and heavy rainfall overnight."
        ),
        "ner": [{"parent": "weather", "labels": ["agency", "region", "date", "measurement"]}],
        "classification": [{"parent": "severity", "labels": ["severe", "moderate", "mild"]}],
        "structuring": [{"parent": "alert", "labels": ["agency:str", "level:str", "region:str", "date:str", "wind_speed:str"]}],
        "gold": {
            "ner": [
                {"text": "Met Office", "label": "agency"},
                {"text": "Cornwall", "label": "region"},
                {"text": "18 December 2023", "label": "date"},
                {"text": "80 mph", "label": "measurement"},
            ],
            "classification": ["severe"],
            "structuring": {
                "alert": [{
                    "agency": "Met Office",
                    "level": "amber",
                    "region": "Cornwall",
                    "date": "18 December 2023",
                    "wind_speed": "80 mph",
                }]
            },
        },
    },
    {
        "name": "Research abstract",
        "text": (
            "Our team at ETH Zurich trained a graph neural network on the QM9 dataset and cut the "
            "mean absolute error to 0.012 eV, outperforming the previous state of the art."
        ),
        "ner": [{"parent": "science", "labels": ["institution", "model", "dataset", "metric", "value"]}],
        "classification": [{"parent": "document type", "labels": ["research paper", "job advert", "invoice"]}],
        "structuring": [{"parent": "experiment", "labels": ["institution:str", "model:str", "dataset:str", "metric:str", "result:str"]}],
        "gold": {
            "ner": [
                {"text": "ETH Zurich", "label": "institution"},
                {"text": "graph neural network", "label": "model"},
                {"text": "QM9", "label": "dataset"},
                {"text": "mean absolute error", "label": "metric"},
                {"text": "0.012 eV", "label": "value"},
            ],
            "classification": ["research paper"],
            "structuring": {
                "experiment": [{
                    "institution": "ETH Zurich",
                    "model": "graph neural network",
                    "dataset": "QM9",
                    "metric": "mean absolute error",
                    "result": "0.012 eV",
                }]
            },
        },
    },
    {
        "name": "Shipping notice",
        "text": (
            "Order 55-A9 shipped from our Rotterdam warehouse on 2 February 2025 via DHL and should "
            "reach Lyon within three working days."
        ),
        "ner": [{"parent": "logistics", "labels": ["order id", "city", "date", "carrier", "duration"]}],
        "classification": [{"parent": "document type", "labels": ["shipping notice", "discharge summary", "job advert"]}],
        "structuring": [{"parent": "shipment", "labels": ["order_id:str", "origin:str", "destination:str", "carrier:str", "ship_date:str"]}],
        "gold": {
            "ner": [
                {"text": "55-A9", "label": "order id"},
                {"text": "Rotterdam", "label": "city"},
                {"text": "2 February 2025", "label": "date"},
                {"text": "DHL", "label": "carrier"},
                {"text": "Lyon", "label": "city"},
                {"text": "three working days", "label": "duration"},
            ],
            "classification": ["shipping notice"],
            "structuring": {
                "shipment": [{
                    "order_id": "55-A9",
                    "origin": "Rotterdam",
                    "destination": "Lyon",
                    "carrier": "DHL",
                    "ship_date": "2 February 2025",
                }]
            },
        },
    },
    {
        "name": "Policy announcement",
        "text": (
            "The European Commission announced in Brussels on 10 January 2026 a 4 billion euro fund "
            "for semiconductor research, welcomed by commissioner Thierry Breton."
        ),
        "ner": [{"parent": "policy", "labels": ["institution", "city", "date", "money", "person"]}],
        "classification": [{"parent": "topic", "labels": ["public policy", "sports", "entertainment"]}],
        "structuring": [{"parent": "programme", "labels": ["institution:str", "budget:str", "field:str", "announced_on:str"]}],
        "gold": {
            "ner": [
                {"text": "European Commission", "label": "institution"},
                {"text": "Brussels", "label": "city"},
                {"text": "10 January 2026", "label": "date"},
                {"text": "4 billion euro", "label": "money"},
                {"text": "Thierry Breton", "label": "person"},
            ],
            "classification": ["public policy"],
            "structuring": {
                "programme": [{
                    "institution": "European Commission",
                    "budget": "4 billion euro",
                    "field": "semiconductor research",
                    "announced_on": "10 January 2026",
                }]
            },
        },
    },
]


# ── Examples: Document layout (PDF) NER ──────────────────────────────────────

# Each example is a page rather than a paragraph: `words` carries every word
# with its `[x0, y0, x1, y1]` box in LayoutLM-style 0-1000 page coordinates,
# which is what the layout head reads in place of reading order. Boxes come
# from the same corpora the layout model trains on -- real scans (FUNSD) and
# synthetic pages whose values are deliberately emitted after all their labels.
LAYOUT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Bank statement",
        # Synthetic page. Every block label comes first in the token stream and
        # every value after it, so only the boxes tie a value to its block.
        "words": [
            ("Balance", 44, 349, 106, 358),
            ("Reference", 355, 982, 401, 988),
            ("DF1BF2B8", 405, 982, 452, 988),
            ("•", 456, 982, 461, 986),
            ("Page", 466, 982, 488, 989),
            ("1", 492, 982, 498, 988),
            ("of", 502, 982, 512, 988),
            ("1", 516, 982, 522, 988),
            ("•", 526, 982, 531, 986),
            ("Retain", 536, 982, 565, 988),
            ("for", 569, 982, 582, 988),
            ("your", 586, 982, 607, 988),
            ("records", 611, 982, 645, 988),
            ("Statement", 505, 103, 577, 110),
            ("Period", 580, 103, 624, 111),
            ("Account", 501, 196, 556, 203),
            ("Document", 48, 44, 129, 52),
            ("Header", 132, 44, 189, 53),
            ("Description", 370, 455, 461, 466),
            ("Opening", 44, 336, 110, 347),
            ("Closing", 114, 336, 172, 347),
            ("Date", 122, 455, 159, 463),
            ("Bank", 22, 3, 45, 10),
            ("statement", 49, 3, 96, 8),
            ("•", 100, 3, 105, 8),
            ("Finance", 110, 3, 145, 9),
            ("&", 149, 3, 156, 9),
            ("Accounting", 161, 3, 211, 10),
            ("Account", 39, 202, 103, 210),
            ("Holder", 106, 202, 159, 211),
            ("Transactions", 53, 411, 146, 419),
            ("Table", 149, 411, 188, 419),
            ("Balance", 836, 455, 898, 464),
            ("Bank", 40, 90, 80, 99),
            ("Info", 83, 90, 114, 99),
            ("Summary", 501, 207, 565, 217),
            ("Amount", 660, 455, 722, 463),
            ("CHF", 628, 231, 654, 239),
            ("8002", 39, 285, 82, 294),
            ("Zurich", 87, 285, 141, 295),
            ("Checking", 628, 214, 688, 225),
            ("14,250.00", 243, 336, 319, 346),
            ("25/10/2004", 57, 508, 138, 517),
            ("18,130.50", 783, 508, 854, 517),
            ("Wire", 233, 473, 264, 481),
            ("Inward", 271, 473, 318, 481),
            ("POS", 233, 490, 261, 498),
            ("Debit", 268, 490, 306, 499),
            ("-319.50", 607, 490, 661, 498),
            ("31/10/2004", 877, 124, 952, 133),
            ("Kenjiro", 39, 234, 97, 247),
            ("Mori", 101, 234, 138, 244),
            ("Zurich", 40, 132, 91, 142),
            ("-800.00", 607, 508, 661, 516),
            ("12/10/2004", 57, 490, 138, 500),
            ("Seestrasse", 39, 259, 132, 269),
            ("14", 136, 259, 158, 269),
            ("+5,000.00", 607, 473, 681, 481),
            ("01/10/2004", 877, 103, 952, 112),
            ("04/10/2004", 57, 473, 138, 482),
            ("Vanguard", 40, 112, 116, 124),
            ("Trust", 123, 112, 163, 121),
            ("8820-4491-02", 628, 196, 720, 203),
            ("STATEMENT", 247, 44, 386, 56),
            ("18,930.50", 783, 490, 854, 499),
            ("18,130.50", 243, 354, 319, 364),
            ("Utility", 233, 508, 274, 519),
            ("Pmt", 281, 508, 309, 516),
            ("19,250.00", 783, 473, 854, 481),
        ],
        "groups": [
            {
                "parent": "layout_blocks",
                "labels": [
                    "document_header",
                    "bank_information",
                    "account_holder",
                    "account_summary",
                    "statement_period",
                    "transactions_table",
                    "opening_closing_balance",
                    "fees_interest_summary",
                    "footer_legal",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "CHF", "label": "account_summary"},
                {"text": "8002 Zurich", "label": "account_holder"},
                {"text": "Checking", "label": "account_summary"},
                {"text": "14,250.00", "label": "opening_closing_balance"},
                {"text": "25/10/2004", "label": "transactions_table"},
                {"text": "18,130.50", "label": "transactions_table"},
                {"text": "Wire Inward", "label": "transactions_table"},
                {"text": "POS Debit", "label": "transactions_table"},
                {"text": "-319.50", "label": "transactions_table"},
                {"text": "31/10/2004", "label": "statement_period"},
                {"text": "Kenjiro Mori", "label": "account_holder"},
                {"text": "Zurich", "label": "bank_information"},
                {"text": "-800.00", "label": "transactions_table"},
                {"text": "12/10/2004", "label": "transactions_table"},
                {"text": "Seestrasse 14", "label": "account_holder"},
                {"text": "+5,000.00", "label": "transactions_table"},
                {"text": "01/10/2004", "label": "statement_period"},
                {"text": "04/10/2004", "label": "transactions_table"},
                {"text": "Vanguard Trust", "label": "bank_information"},
                {"text": "8820-4491-02", "label": "account_summary"},
                {"text": "STATEMENT", "label": "document_header"},
                {"text": "18,930.50", "label": "transactions_table"},
                {"text": "18,130.50", "label": "opening_closing_balance"},
                {"text": "Utility Pmt", "label": "transactions_table"},
                {"text": "19,250.00", "label": "transactions_table"},
            ],
        },
    },
    {
        "name": "Insurance declarations page",
        # Two-column declarations page: coverage types on the left, limits and
        # deductibles in columns to their right.
        "words": [
            ("Insurance", 22, 3, 67, 9),
            ("policy", 71, 3, 98, 11),
            ("declarations", 102, 3, 158, 10),
            ("page", 162, 3, 185, 9),
            ("•", 189, 3, 194, 8),
            ("Insurance", 199, 3, 244, 9),
            ("Policy", 47, 47, 94, 58),
            ("Header", 97, 47, 154, 56),
            ("POL-7739201-A", 520, 47, 669, 56),
            ("Insured", 527, 127, 580, 135),
            ("Info", 583, 127, 610, 135),
            ("Insurer", 60, 144, 117, 152),
            ("Info", 120, 144, 151, 153),
            ("Apex", 283, 144, 329, 157),
            ("Mutual", 334, 144, 395, 155),
            ("Elena", 527, 156, 565, 164),
            ("Vance", 569, 156, 610, 163),
            ("Columbus,", 275, 167, 370, 179),
            ("OH", 376, 167, 404, 177),
            ("404", 527, 176, 552, 183),
            ("Elm", 556, 176, 582, 184),
            ("St,", 586, 176, 604, 184),
            ("Dayton,", 609, 176, 660, 185),
            ("OH", 665, 176, 687, 183),
            ("Policy", 43, 258, 83, 268),
            ("Metadata", 87, 258, 151, 266),
            ("2025-04-01", 240, 258, 316, 265),
            ("2026-04-01", 240, 277, 316, 285),
            ("Annual", 240, 297, 285, 306),
            ("Coverage", 51, 343, 125, 353),
            ("Table", 129, 343, 171, 352),
            ("Type", 231, 388, 288, 404),
            ("Limit", 686, 388, 749, 401),
            ("Dwelling", 55, 413, 150, 429),
            ("$300,000", 474, 413, 579, 428),
            ("Liability", 55, 437, 141, 454),
            ("$100,000", 474, 437, 579, 452),
            ("Limits", 54, 593, 95, 601),
            ("Deductibles", 99, 593, 179, 601),
            ("$1,000", 914, 630, 959, 638),
            ("$500", 926, 652, 959, 660),
            ("Premium", 617, 746, 672, 754),
            ("Summary", 675, 746, 730, 755),
            ("Endorsements", 43, 748, 148, 756),
            ("Reference", 151, 748, 225, 756),
            ("$1,240.00", 617, 786, 693, 796),
            ("END-01", 43, 788, 96, 796),
            ("END-04", 43, 808, 96, 816),
            ("$1,240.00", 617, 809, 693, 819),
            ("Reference", 355, 982, 401, 988),
            ("4D8FF94C", 405, 982, 451, 988),
            ("•", 456, 982, 461, 986),
            ("Page", 466, 982, 488, 989),
            ("1", 492, 982, 498, 988),
            ("of", 502, 982, 512, 988),
            ("1", 516, 982, 522, 988),
            ("•", 526, 982, 531, 986),
            ("Retain", 536, 982, 565, 988),
            ("for", 569, 982, 582, 988),
            ("your", 586, 982, 607, 988),
            ("records", 611, 982, 645, 988),
        ],
        "groups": [
            {
                "parent": "layout_blocks",
                "labels": [
                    "policy_header",
                    "insured_information",
                    "insurer_information",
                    "policy_metadata",
                    "coverage_table",
                    "limits_deductibles",
                    "insured_property_or_risk",
                    "endorsements_reference",
                    "premium_summary",
                    "agent_information",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "POL-7739201-A", "label": "policy_header"},
                {"text": "Apex Mutual", "label": "insurer_information"},
                {"text": "Elena Vance", "label": "insured_information"},
                {"text": "Columbus, OH", "label": "insurer_information"},
                {"text": "404 Elm St, Dayton, OH", "label": "insured_information"},
                {"text": "2025-04-01", "label": "policy_metadata"},
                {"text": "2026-04-01", "label": "policy_metadata"},
                {"text": "Annual", "label": "policy_metadata"},
                {"text": "Dwelling", "label": "coverage_table"},
                {"text": "$300,000", "label": "coverage_table"},
                {"text": "Liability", "label": "coverage_table"},
                {"text": "$100,000", "label": "coverage_table"},
                {"text": "$1,000", "label": "limits_deductibles"},
                {"text": "$500", "label": "limits_deductibles"},
                {"text": "$1,240.00", "label": "premium_summary"},
                {"text": "END-01", "label": "endorsements_reference"},
                {"text": "END-04", "label": "endorsements_reference"},
            ],
        },
    },
    {
        "name": "Payment receipt",
        # Line items, a totals stack and payment metadata, all of which read as
        # bare numbers until the geometry says which column they sit in.
        "words": [
            ("12500.00", 800, 350, 848, 356),
            ("TXN-98421", 118, 233, 197, 241),
            ("18/04/2007", 203, 233, 284, 242),
            ("…", 290, 233, 304, 235),
            ("Apex", 353, 44, 430, 63),
            ("Ledger", 436, 44, 546, 64),
            ("Associates", 552, 44, 717, 60),
            ("2", 544, 380, 550, 386),
            ("Statutory", 98, 380, 145, 387),
            ("Verification", 150, 380, 206, 387),
            ("1", 544, 365, 550, 371),
            ("Filing", 98, 365, 124, 373),
            ("Charges", 130, 365, 171, 373),
            ("Q4", 177, 365, 191, 372),
            ("22460.00", 474, 904, 531, 911),
            ("Cheque", 455, 870, 501, 880),
            ("440912", 506, 870, 552, 877),
            ("1", 544, 350, 550, 356),
            ("20000.00", 819, 704, 908, 713),
            ("Tax", 98, 350, 116, 356),
            ("Audit", 121, 350, 147, 357),
            ("Retainer", 153, 350, 196, 357),
            ("12500.00", 672, 350, 720, 356),
            ("Cleared", 480, 921, 526, 929),
            ("2000.00", 672, 380, 713, 386),
            ("18/04/2007", 468, 887, 537, 895),
            ("Fort,", 395, 129, 439, 141),
            ("Mumbai", 446, 129, 521, 140),
            ("400001", 529, 129, 598, 139),
            ("2460.00", 831, 725, 909, 735),
            ("4000.00", 800, 380, 841, 386),
            ("3500.00", 800, 365, 841, 371),
            ("3500.00", 672, 365, 713, 371),
            ("22460.00", 819, 746, 908, 756),
            ("HDFC", 468, 853, 502, 860),
            ("Bank", 507, 853, 537, 861),
            ("Payment", 111, 820, 180, 830),
            ("Info", 183, 820, 214, 829),
            ("Transaction", 118, 208, 210, 217),
            ("Metadata", 213, 208, 288, 217),
            ("Service", 296, 335, 338, 342),
            ("Rate", 718, 335, 744, 341),
            ("Merchant", 196, 129, 273, 138),
            ("Address", 276, 129, 340, 138),
            ("Subtotal", 437, 669, 504, 677),
            ("Tax", 508, 669, 534, 677),
            ("Total", 538, 669, 576, 677),
            ("Amount", 839, 335, 883, 341),
            ("Line", 94, 293, 115, 300),
            ("Items", 119, 293, 147, 299),
            ("Qty", 594, 335, 614, 343),
            ("Receipt", 22, 3, 56, 10),
            ("•", 60, 3, 65, 8),
            ("Finance", 70, 3, 105, 9),
            ("&", 109, 3, 116, 9),
            ("Accounting", 121, 3, 171, 10),
            ("Reference", 355, 982, 401, 988),
            ("4BA41F91", 405, 982, 451, 988),
            ("•", 456, 982, 461, 986),
            ("Page", 466, 982, 488, 989),
            ("1", 492, 982, 498, 988),
            ("of", 502, 982, 512, 988),
            ("1", 516, 982, 522, 988),
            ("•", 526, 982, 531, 986),
            ("Retain", 536, 982, 565, 988),
            ("for", 569, 982, 582, 988),
            ("your", 586, 982, 607, 988),
            ("records", 611, 982, 645, 988),
            ("Merchant", 154, 44, 231, 53),
            ("Header", 234, 44, 292, 53),
        ],
        "groups": [
            {
                "parent": "layout_blocks",
                "labels": [
                    "merchant_header",
                    "transaction_metadata",
                    "line_items",
                    "subtotal_tax_total",
                    "payment_information",
                    "merchant_address",
                    "discounts",
                    "loyalty_information",
                    "footer",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "12500.00", "label": "line_items"},
                {"text": "TXN-98421 18/04/2007 …", "label": "transaction_metadata"},
                {"text": "Apex Ledger Associates", "label": "merchant_header"},
                {"text": "2", "label": "line_items"},
                {"text": "Statutory Verification", "label": "line_items"},
                {"text": "1", "label": "line_items"},
                {"text": "Filing Charges Q4", "label": "line_items"},
                {"text": "22460.00", "label": "payment_information"},
                {"text": "Cheque 440912", "label": "payment_information"},
                {"text": "20000.00", "label": "subtotal_tax_total"},
                {"text": "Tax Audit Retainer", "label": "line_items"},
                {"text": "Cleared", "label": "payment_information"},
                {"text": "2000.00", "label": "line_items"},
                {"text": "18/04/2007", "label": "payment_information"},
                {"text": "Fort, Mumbai 400001", "label": "merchant_address"},
                {"text": "2460.00", "label": "subtotal_tax_total"},
                {"text": "4000.00", "label": "line_items"},
                {"text": "3500.00", "label": "line_items"},
                {"text": "22460.00", "label": "subtotal_tax_total"},
                {"text": "HDFC Bank", "label": "payment_information"},
            ],
        },
    },
    {
        "name": "Income statement",
        # Six currency figures whose only distinguishing feature is the row they
        # belong to.
        "words": [
            ("Tax", 58, 727, 77, 732),
            ("Section", 80, 727, 122, 733),
            ("Reporting", 67, 122, 145, 133),
            ("Period", 149, 122, 200, 131),
            ("Revenue", 59, 171, 127, 179),
            ("Section", 131, 171, 190, 179),
            ("Cost", 58, 330, 93, 338),
            ("Of", 96, 330, 115, 339),
            ("Sales", 118, 330, 160, 339),
            ("Document", 60, 54, 141, 62),
            ("Header", 144, 54, 201, 63),
            ("Income", 22, 3, 56, 9),
            ("statement", 60, 3, 107, 8),
            ("/", 111, 3, 115, 9),
            ("P&L", 119, 3, 136, 9),
            ("report", 141, 3, 169, 10),
            ("•", 173, 3, 178, 8),
            ("Finance", 183, 3, 218, 9),
            ("&", 222, 3, 229, 9),
            ("Accounting", 234, 3, 284, 10),
            ("Operating", 54, 642, 133, 653),
            ("Income", 137, 642, 194, 650),
            ("Reference", 354, 982, 400, 988),
            ("1B95EECB", 404, 982, 451, 988),
            ("•", 456, 982, 461, 986),
            ("Page", 466, 982, 488, 989),
            ("1", 492, 982, 498, 988),
            ("of", 502, 982, 512, 988),
            ("1", 516, 982, 522, 988),
            ("•", 526, 982, 531, 986),
            ("Retain", 536, 982, 565, 988),
            ("for", 569, 982, 582, 988),
            ("your", 586, 982, 607, 988),
            ("records", 611, 982, 645, 988),
            ("Operating", 57, 456, 127, 467),
            ("Expenses", 131, 456, 194, 465),
            ("Net", 58, 844, 83, 852),
            ("Income", 87, 844, 138, 852),
            ("01/04/2003", 266, 122, 352, 131),
            ("-", 358, 122, 363, 126),
            ("31/03/2004", 369, 122, 455, 131),
            ("£420,000", 256, 844, 343, 856),
            ("£180,000", 58, 754, 115, 762),
            ("£218,400", 58, 383, 130, 393),
            ("£630,700", 58, 404, 130, 414),
            ("£412,300", 58, 362, 130, 372),
            ("£395,000", 57, 556, 124, 566),
            ("Kestrel", 259, 54, 372, 70),
            ("Precision", 378, 54, 523, 70),
            ("Ltd", 529, 54, 580, 70),
            ("£1,240,500", 59, 206, 156, 217),
            ("£96,500", 57, 518, 116, 527),
            ("£114,300", 57, 537, 124, 546),
            ("£385,200", 59, 227, 140, 237),
            ("£184,200", 57, 499, 124, 508),
            ("£1,625,700", 59, 247, 156, 258),
            ("£600,000", 865, 642, 937, 652),
        ],
        "groups": [
            {
                "parent": "layout_blocks",
                "labels": [
                    "document_header",
                    "reporting_period",
                    "revenue_section",
                    "cost_of_sales",
                    "operating_expenses",
                    "operating_income",
                    "other_income_expenses",
                    "tax_section",
                    "net_income",
                    "comparative_period",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "01/04/2003 - 31/03/2004", "label": "reporting_period"},
                {"text": "£420,000", "label": "net_income"},
                {"text": "£180,000", "label": "tax_section"},
                {"text": "£218,400", "label": "cost_of_sales"},
                {"text": "£630,700", "label": "cost_of_sales"},
                {"text": "£412,300", "label": "cost_of_sales"},
                {"text": "£395,000", "label": "operating_expenses"},
                {"text": "Kestrel Precision Ltd", "label": "document_header"},
                {"text": "£1,240,500", "label": "revenue_section"},
                {"text": "£96,500", "label": "operating_expenses"},
                {"text": "£114,300", "label": "operating_expenses"},
                {"text": "£385,200", "label": "revenue_section"},
                {"text": "£184,200", "label": "operating_expenses"},
                {"text": "£1,625,700", "label": "revenue_section"},
                {"text": "£600,000", "label": "operating_income"},
            ],
        },
    },
    {
        "name": "Driver's licence",
        # Identity document with a portrait block, personal data and a barcode
        # line placed around the card rather than in reading order.
        "words": [
            ("PENTLAND", 347, 407, 417, 414),
            ("01", 667, 672, 690, 682),
            ("115", 697, 672, 731, 682),
            ("License", 351, 641, 410, 650),
            ("Classes", 414, 641, 473, 650),
            ("Reference", 354, 982, 400, 988),
            ("0A66AD2B", 404, 982, 452, 988),
            ("•", 457, 982, 462, 986),
            ("Page", 467, 982, 489, 989),
            ("1", 493, 982, 499, 988),
            ("of", 503, 982, 513, 988),
            ("1", 517, 982, 523, 988),
            ("•", 527, 982, 532, 986),
            ("Retain", 537, 982, 566, 988),
            ("for", 570, 982, 583, 988),
            ("your", 587, 982, 608, 988),
            ("records", 612, 982, 646, 988),
            ("GREAT", 236, 39, 313, 51),
            ("BRITAIN", 320, 39, 414, 51),
            ("Header", 39, 52, 96, 61),
            ("DRIVER", 236, 64, 324, 75),
            ("LICENCE", 332, 64, 430, 75),
            ("Document", 39, 39, 120, 47),
            ("Identity", 123, 39, 185, 50),
            ("PENTL705128AG9DE", 521, 160, 688, 170),
            ("34", 696, 160, 716, 170),
            ("12/05/1998", 702, 258, 783, 267),
            ("Restrictions", 667, 642, 762, 650),
            ("Endorsements", 765, 642, 878, 650),
            ("License", 344, 160, 403, 169),
            ("Number", 407, 160, 470, 169),
            ("KENDAL", 347, 492, 400, 499),
            ("[PHOTO-ID", 53, 374, 146, 385),
            ("AP-28491]", 152, 374, 237, 385),
            ("12/05/1970", 347, 449, 422, 458),
            ("A", 351, 672, 362, 681),
            ("B", 365, 672, 375, 681),
            ("B1", 379, 672, 399, 681),
            ("BE", 402, 672, 422, 681),
            ("Portrait", 53, 165, 115, 174),
            ("14", 347, 471, 364, 478),
            ("CROWN", 367, 471, 418, 478),
            ("YARD", 421, 471, 457, 478),
            ("Personal", 347, 364, 406, 372),
            ("Data", 409, 364, 441, 371),
            ("*PENTL705128AG9DE*", 56, 671, 216, 679),
            ("11/05/2008", 702, 280, 783, 289),
            ("ALISTAIR", 347, 428, 404, 435),
            ("GRAHAM", 407, 428, 464, 435),
            ("Issue", 349, 258, 388, 265),
            ("Expiry", 391, 258, 437, 268),
            ("Dates", 440, 258, 483, 265),
            ("LA9", 347, 513, 372, 520),
            ("4DX", 375, 513, 402, 520),
            ("Barcode", 56, 531, 116, 539),
            ("Driver's", 22, 3, 57, 9),
            ("license", 61, 3, 93, 10),
            ("•", 97, 3, 102, 8),
            ("Government", 107, 3, 164, 9),
            ("&", 168, 3, 175, 9),
            ("Identity", 180, 3, 215, 11),
        ],
        "groups": [
            {
                "parent": "layout_blocks",
                "labels": [
                    "document_identity_header",
                    "portrait",
                    "personal_data",
                    "license_number",
                    "address",
                    "issue_expiry_dates",
                    "license_classes",
                    "restrictions_endorsements",
                    "barcode",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "PENTLAND", "label": "personal_data"},
                {"text": "01 115", "label": "restrictions_endorsements"},
                {"text": "GREAT BRITAIN", "label": "document_identity_header"},
                {"text": "DRIVER LICENCE", "label": "document_identity_header"},
                {"text": "PENTL705128AG9DE 34", "label": "license_number"},
                {"text": "12/05/1998", "label": "issue_expiry_dates"},
                {"text": "KENDAL", "label": "personal_data"},
                {"text": "[PHOTO-ID AP-28491]", "label": "portrait"},
                {"text": "12/05/1970", "label": "personal_data"},
                {"text": "A B B1 BE", "label": "license_classes"},
                {"text": "14 CROWN YARD", "label": "personal_data"},
                {"text": "*PENTL705128AG9DE*", "label": "barcode"},
                {"text": "11/05/2008", "label": "issue_expiry_dates"},
                {"text": "ALISTAIR GRAHAM", "label": "personal_data"},
                {"text": "LA9 4DX", "label": "personal_data"},
            ],
        },
    },
    {
        "name": "Restaurant menu",
        # Dish names in one column, prices in another, plus footnotes set apart
        # at the foot of the page.
        "words": [
            ("Prices", 702, 182, 743, 190),
            ("Restaurant", 22, 3, 72, 9),
            ("menu", 76, 3, 102, 8),
            ("•", 106, 3, 111, 8),
            ("Highly", 116, 3, 145, 11),
            ("Layout-Dependent", 149, 3, 233, 11),
            ("Operational", 237, 3, 290, 11),
            ("Documents", 294, 3, 346, 9),
            ("Menu", 74, 56, 119, 64),
            ("Sections", 122, 56, 189, 65),
            ("Footnotes", 72, 898, 151, 906),
            ("Disclaimers", 155, 898, 246, 906),
            ("Reference", 355, 982, 401, 988),
            ("2614BF08", 405, 982, 451, 988),
            ("•", 455, 982, 460, 986),
            ("Page", 465, 982, 487, 989),
            ("1", 491, 982, 497, 988),
            ("of", 501, 982, 511, 988),
            ("1", 515, 982, 521, 988),
            ("•", 525, 982, 530, 986),
            ("Retain", 535, 982, 564, 988),
            ("for", 568, 982, 581, 988),
            ("your", 585, 982, 606, 988),
            ("records", 610, 982, 644, 988),
            ("Item", 67, 188, 103, 196),
            ("Names", 106, 188, 160, 196),
            ("10%", 271, 898, 307, 907),
            ("service", 313, 898, 370, 907),
            ("charge", 377, 898, 432, 909),
            ("extra.", 439, 898, 486, 906),
            ("VAT", 492, 898, 522, 907),
            ("applicable.", 528, 898, 615, 909),
            ("Est.", 622, 898, 652, 907),
            ("14/08/2004.", 658, 898, 755, 908),
            ("Rs", 702, 289, 718, 297),
            ("225.00", 721, 289, 766, 297),
            ("Rs", 702, 267, 718, 274),
            ("110.00", 721, 267, 766, 274),
            ("Jeera", 67, 324, 114, 337),
            ("Rice", 121, 324, 159, 335),
            ("Rs", 702, 222, 718, 229),
            ("140.00", 721, 222, 766, 229),
            ("Rs", 702, 244, 718, 252),
            ("195.00", 721, 244, 766, 252),
            ("Butter", 67, 301, 124, 311),
            ("Chicken", 131, 301, 202, 312),
            ("Rs", 702, 335, 718, 342),
            ("40.00", 721, 335, 758, 342),
            ("Murgh", 67, 254, 124, 268),
            ("Malai", 131, 254, 179, 265),
            ("Kebab", 186, 254, 242, 265),
            ("Rs", 702, 357, 718, 365),
            ("65.00", 721, 357, 758, 365),
            ("Garlic", 67, 347, 119, 358),
            ("Naan", 127, 347, 174, 358),
            ("Rs", 702, 312, 718, 319),
            ("85.00", 721, 312, 758, 319),
            ("Gulab", 67, 371, 120, 382),
            ("Jamun", 127, 371, 185, 384),
            ("Paneer", 67, 231, 130, 241),
            ("Tikka", 137, 231, 185, 242),
            ("Dal", 67, 277, 97, 288),
            ("Makhani", 104, 277, 180, 288),
            ("Tandoor", 273, 56, 328, 64),
            ("Curries", 334, 56, 383, 63),
            ("Rice", 390, 56, 419, 63),
            ("&", 425, 56, 437, 63),
            ("Breads", 443, 56, 489, 64),
            ("…", 496, 56, 509, 58),
        ],
        "groups": [
            {
                "parent": "layout_blocks",
                "labels": [
                    "restaurant_header",
                    "menu_sections",
                    "item_names",
                    "item_descriptions",
                    "prices",
                    "dietary_markers",
                    "add_ons_options",
                    "footnotes_disclaimers",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "10% service charge extra. VAT applicable. Est. 14/08/2004.", "label": "footnotes_disclaimers"},
                {"text": "Rs 225.00", "label": "prices"},
                {"text": "Rs 110.00", "label": "prices"},
                {"text": "Jeera Rice", "label": "item_names"},
                {"text": "Rs 140.00", "label": "prices"},
                {"text": "Rs 195.00", "label": "prices"},
                {"text": "Butter Chicken", "label": "item_names"},
                {"text": "Rs 40.00", "label": "prices"},
                {"text": "Murgh Malai Kebab", "label": "item_names"},
                {"text": "Rs 65.00", "label": "prices"},
                {"text": "Garlic Naan", "label": "item_names"},
                {"text": "Rs 85.00", "label": "prices"},
                {"text": "Gulab Jamun", "label": "item_names"},
                {"text": "Paneer Tikka", "label": "item_names"},
                {"text": "Dal Makhani", "label": "item_names"},
                {"text": "Tandoor Curries Rice & Breads …", "label": "menu_sections"},
            ],
        },
    },
    {
        "name": "Approval routing sheet",
        # Real scan (FUNSD). Question labels and the answers written next to them
        # are far apart in OCR order.
        "words": [
            ("DATE:", 119, 158, 175, 179),
            ("SUBJECT:", 116, 187, 209, 207),
            ("WRITER:", 116, 240, 194, 260),
            ("APPROVALS:", 114, 286, 233, 306),
            ("Other:", 114, 508, 187, 528),
            ("YES", 576, 426, 616, 447),
            ("NO", 748, 433, 774, 450),
            ("DATE", 790, 323, 845, 341),
            ("PUBLIC", 286, 91, 355, 109),
            ("COMMUNICATIONS", 364, 91, 522, 111),
            ("APPROVAL", 530, 94, 619, 111),
            ("SHEET", 629, 96, 688, 113),
            ("The", 364, 120, 400, 135),
            ("Tobacco", 405, 122, 488, 137),
            ("Institute", 493, 123, 599, 140),
            ("Attachment", 707, 53, 818, 75),
            ("A", 824, 53, 841, 75),
            ("NAME", 544, 314, 598, 338),
            ("OR", 602, 318, 629, 335),
            ("INITIALS", 637, 318, 729, 336),
            ("Division", 115, 342, 204, 359),
            ("Head", 213, 344, 259, 361),
            ("Bill", 114, 367, 160, 384),
            ("Kloepfer", 170, 370, 262, 388),
            ("Sam", 115, 399, 152, 414),
            ("Chilcote", 161, 397, 250, 415),
            ("Legal", 115, 425, 174, 442),
            ("Approval", 181, 425, 269, 443),
            ("Recommended/", 280, 426, 418, 446),
            ("Required:", 419, 429, 517, 447),
            ("SH", 115, 448, 137, 466),
            ("&B", 137, 453, 161, 468),
            ("C&", 114, 479, 136, 496),
            ("B", 135, 482, 154, 495),
        ],
        "groups": [
            {
                "parent": "funsd",
                "labels": [
                    "ANSWER",
                    "HEADER",
                    "QUESTION",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "DATE: SUBJECT: WRITER: APPROVALS: Other: YES NO DATE", "label": "QUESTION"},
                {"text": "PUBLIC COMMUNICATIONS APPROVAL SHEET The Tobacco Institute Attachment A", "label": "HEADER"},
                {"text": "NAME OR INITIALS", "label": "QUESTION"},
                {"text": "Division Head Bill Kloepfer Sam Chilcote", "label": "ANSWER"},
                {"text": "Legal Approval Recommended/ Required:", "label": "HEADER"},
            ],
        },
    },
    {
        "name": "Tar & nicotine change form",
        # Real scan with a four-column measurement table under a shared header.
        "words": [
            ("Date", 526, 165, 580, 189),
            ("Signature", 451, 433, 559, 457),
            ("From", 276, 295, 335, 312),
            ("To", 645, 293, 682, 310),
            ("NOTE:", 165, 547, 229, 567),
            ("3", 202, 385, 219, 402),
            ("2", 562, 390, 582, 403),
            ("A.", 300, 94, 324, 114),
            ("T.", 324, 96, 347, 113),
            ("Co.", 346, 94, 383, 111),
            ("Tar", 392, 95, 429, 112),
            ("&", 438, 95, 456, 112),
            ("Nicotine", 461, 96, 555, 113),
            ("Change", 560, 95, 629, 112),
            ("Form", 643, 98, 690, 113),
            ("7/", 639, 161, 653, 178),
            ("24/", 654, 159, 682, 180),
            ("90", 683, 159, 712, 180),
            ("CARLTON", 374, 229, 444, 250),
            ("100's", 448, 228, 500, 248),
            ("FMSP", 503, 230, 548, 248),
            ("Brand", 162, 233, 224, 258),
            ("&", 232, 232, 254, 257),
            ("Style", 258, 230, 317, 257),
            ("Tar", 199, 323, 240, 343),
            ("(Mg", 166, 348, 199, 370),
            ("/Cigt)", 203, 351, 272, 371),
            ("Nicotine", 351, 323, 443, 343),
            ("(Mg", 347, 348, 385, 369),
            ("/Cigt)", 383, 352, 456, 367),
            ("Tar", 564, 323, 606, 340),
            ("(Mg", 536, 348, 569, 370),
            ("/Cigt)", 566, 351, 640, 371),
            ("Nicotine", 712, 324, 809, 342),
            ("(Mg/", 716, 352, 751, 370),
            ("/Cigt)", 751, 352, 820, 370),
            ("0", 742, 387, 757, 401),
            (".2", 757, 387, 783, 404),
            ("0", 374, 387, 388, 404),
            (".3", 387, 388, 409, 403),
            ("Use", 245, 547, 287, 565),
            ("Separate", 295, 546, 387, 564),
            ("Form", 397, 547, 447, 565),
            ("For", 456, 549, 492, 564),
            ("Each", 500, 546, 547, 563),
            ("Change", 559, 547, 627, 565),
        ],
        "groups": [
            {
                "parent": "funsd",
                "labels": [
                    "ANSWER",
                    "HEADER",
                    "QUESTION",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "Date Signature", "label": "QUESTION"},
                {"text": "From To", "label": "HEADER"},
                {"text": "NOTE:", "label": "QUESTION"},
                {"text": "3 2", "label": "ANSWER"},
                {"text": "A. T. Co. Tar & Nicotine Change Form", "label": "HEADER"},
                {"text": "7/ 24/ 90 CARLTON 100's FMSP", "label": "ANSWER"},
                {"text": "Brand & Style Tar (Mg /Cigt) Nicotine (Mg /Cigt) Tar (Mg /Cigt) Nicotine (Mg/ /Cigt)", "label": "QUESTION"},
                {"text": "0 .2 0 .3 Use Separate Form For Each Change", "label": "ANSWER"},
            ],
        },
    },
    {
        "name": "Project initiation form",
        # Real scan: a form whose answers sit to the right of, or below, their
        # printed question.
        "words": [
            ("Date", 657, 141, 702, 156),
            ("Marketing", 258, 295, 350, 312),
            ("Mann", 220, 270, 266, 283),
            ("Project", 392, 82, 463, 97),
            ("Initiation", 472, 81, 569, 98),
            ("Form", 576, 81, 620, 96),
            ("November", 717, 142, 795, 157),
            ("15,", 800, 141, 833, 159),
            ("1968", 842, 142, 884, 157),
            ("CST-", 220, 142, 261, 153),
            ("N-", 262, 142, 279, 156),
            ("68", 280, 141, 300, 154),
            ("Project", 70, 141, 140, 159),
            ("Code", 148, 142, 188, 157),
            ("Project", 69, 209, 139, 226),
            ("Name", 146, 211, 188, 225),
            ("Charcoal", 223, 208, 300, 223),
            ("Smoking", 309, 209, 377, 226),
            ("Tobacco", 383, 207, 458, 224),
            ("Project", 70, 270, 144, 284),
            ("Leader", 146, 267, 208, 282),
            ("Work", 69, 299, 114, 314),
            ("Requested", 119, 299, 208, 314),
            ("by", 212, 299, 236, 316),
            ("Project", 70, 356, 141, 373),
            ("Objective", 150, 355, 237, 373),
            ("A", 262, 358, 276, 372),
            ("pipe", 282, 358, 324, 372),
            ("tobacco", 328, 356, 401, 367),
            ("containing", 406, 356, 502, 373),
            ("activated", 511, 353, 599, 370),
            ("charcoal", 604, 353, 685, 370),
            ("-", 691, 356, 706, 369),
            ("must", 711, 353, 749, 368),
            ("show", 758, 353, 800, 367),
            ("vapor", 804, 355, 859, 369),
            ("phase", 263, 385, 311, 402),
            ("reduction.", 317, 385, 416, 399),
            ("Smoking", 248, 432, 398, 482),
            ("Deval", 555, 432, 665, 460),
            ("Tobacco", 241, 465, 379, 494),
            ("Other", 74, 747, 129, 760),
            ("personnel", 135, 744, 223, 762),
            ("assigned", 232, 744, 311, 762),
            ("Analytical", 333, 742, 434, 757),
            ("Section", 442, 742, 507, 757),
            ("is", 514, 743, 534, 756),
            ("doing", 544, 742, 595, 759),
            ("most", 599, 742, 641, 757),
            ("of", 648, 743, 670, 757),
            ("the", 675, 742, 707, 756),
            ("work", 716, 742, 754, 755),
            ("on", 762, 744, 786, 757),
            ("vapor", 790, 742, 841, 760),
            ("phase.", 849, 742, 905, 759),
            ("PDL", 97, 772, 128, 789),
            ("and", 133, 772, 165, 789),
            ("John", 171, 774, 213, 788),
            ("Brooks", 220, 774, 280, 788),
            ("preparing", 287, 775, 375, 790),
            ("charocal", 383, 771, 458, 785),
            ("impregnated", 469, 772, 577, 787),
            ("RC.", 582, 772, 614, 783),
            ("Estimated", 80, 849, 166, 866),
            ("Man", 174, 848, 209, 863),
            ("Hours", 212, 848, 263, 863),
            ("for", 269, 849, 301, 864),
            ("Completion", 309, 848, 406, 865),
        ],
        "groups": [
            {
                "parent": "funsd",
                "labels": [
                    "ANSWER",
                    "HEADER",
                    "QUESTION",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "Date", "label": "QUESTION"},
                {"text": "Marketing Mann", "label": "ANSWER"},
                {"text": "Project Initiation Form", "label": "HEADER"},
                {"text": "November 15, 1968 CST- N- 68", "label": "ANSWER"},
                {"text": "Project Code Project Name", "label": "QUESTION"},
                {"text": "Charcoal Smoking Tobacco", "label": "ANSWER"},
                {"text": "Project Leader Work Requested by Project Objective", "label": "QUESTION"},
                {"text": "A pipe tobacco containing activated charcoal - must show vapor phase reduction.", "label": "ANSWER"},
                {"text": "Other personnel assigned", "label": "QUESTION"},
                {"text": "Analytical Section is doing most of the work on vapor phase. PDL and John Brooks preparing charocal impregnated RC.", "label": "ANSWER"},
                {"text": "Estimated Man Hours for Completion", "label": "QUESTION"},
            ],
        },
    },
    {
        "name": "Fax transmission cover page",
        # Real scan mixing a machine-printed status report with the addressee
        # block of a fax cover page.
        "words": [
            ("OPTION", 319, 128, 381, 140),
            ("PAGE", 726, 126, 769, 138),
            ("SHB", 519, 146, 550, 159),
            ("03", 725, 148, 749, 160),
            ("(AUTO)", 790, 110, 843, 122),
            ("207422272", 879, 821, 903, 914),
            ("Date:", 282, 822, 339, 840),
            ("Attention:", 233, 849, 340, 866),
            ("Company:", 236, 875, 342, 890),
            ("Subject:", 249, 927, 339, 942),
            ("11/12/96", 387, 822, 473, 839),
            ("MEMORY", 380, 73, 440, 83),
            ("STORAGE", 446, 71, 510, 84),
            ("REPORT", 519, 68, 572, 85),
            ("(NOV", 687, 71, 729, 83),
            ("12", 729, 75, 751, 83),
            ("96", 757, 73, 786, 85),
            ("05:49PM", 797, 71, 871, 84),
            ("FILE", 90, 130, 136, 140),
            ("FILE", 145, 126, 189, 139),
            ("TYPE", 193, 128, 230, 143),
            ("083", 92, 145, 129, 158),
            ("MEMORY", 145, 148, 212, 160),
            ("TX", 209, 148, 240, 163),
            ("TEL", 515, 130, 546, 140),
            ("NO.", 550, 130, 576, 140),
            ("REMAINING", 599, 505, 680, 518),
            ("CALL", 689, 505, 725, 518),
            ("CAPACITY", 736, 505, 806, 518),
            ("299", 812, 502, 843, 519),
            ("Facsimile", 322, 718, 454, 743),
            ("Transmission", 456, 719, 629, 739),
            ("Legal", 379, 757, 442, 772),
            ("Department", 446, 757, 572, 774),
            ("120", 383, 776, 423, 788),
            ("Park", 427, 776, 484, 788),
            ("Avenue", 480, 774, 561, 787),
            ("New", 326, 792, 375, 807),
            ("York,", 387, 791, 440, 808),
            ("NY", 452, 792, 485, 805),
            ("10017-", 502, 791, 572, 806),
            ("5592", 575, 792, 622, 805),
            ("PM", 458, 613, 509, 645),
            ("PHILIP MORRIS", 416, 658, 533, 671),
            ("John J.", 387, 847, 450, 864),
            ("Mulderig", 460, 850, 541, 865),
            ("C/O", 543, 850, 567, 863),
            ("Mike", 572, 850, 623, 863),
            ("Baker", 629, 849, 683, 864),
            ("Philip", 383, 874, 446, 889),
            ("Morris", 446, 872, 510, 894),
            ("Management", 510, 874, 627, 891),
            ("Corp.", 633, 875, 687, 890),
            ("816/545-7473", 389, 900, 517, 918),
            ("Fax", 387, 927, 423, 944),
            ("Received", 425, 925, 511, 940),
            ("Fax", 269, 900, 311, 917),
            ("#:", 312, 900, 339, 917),
        ],
        "groups": [
            {
                "parent": "funsd",
                "labels": [
                    "ANSWER",
                    "HEADER",
                    "QUESTION",
                ],
            },
        ],
        "gold": {
            "ner": [
                {"text": "OPTION PAGE", "label": "QUESTION"},
                {"text": "SHB 03", "label": "ANSWER"},
                {"text": "Date: Attention: Company: Subject:", "label": "QUESTION"},
                {"text": "11/12/96", "label": "ANSWER"},
                {"text": "FILE FILE TYPE", "label": "QUESTION"},
                {"text": "083 MEMORY TX", "label": "ANSWER"},
                {"text": "TEL NO.", "label": "QUESTION"},
                {"text": "Facsimile Transmission Legal Department 120 Park Avenue New York, NY 10017- 5592 PM PHILIP MORRIS", "label": "HEADER"},
                {"text": "John J. Mulderig C/O Mike Baker Philip Morris Management Corp. 816/545-7473 Fax Received", "label": "ANSWER"},
                {"text": "Fax #:", "label": "QUESTION"},
            ],
        },
    },
]


for _layout_example in LAYOUT_EXAMPLES:
    # `parse_pdf` joins the words with single spaces before decoding, so the
    # flat text every other tab carries is derived rather than stored twice.
    _layout_example["text"] = " ".join(word for word, *_ in _layout_example["words"])


# ── Examples: helpers ────────────────────────────────────────────────────────

def _preview(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def dataset_samples(examples: List[Dict[str, Any]]) -> List[List[str]]:
    """Rows for the `gr.Dataset` example picker: name + text preview."""
    return [
        [example["name"], _preview(example.get("text") or example.get("query", ""))]
        for example in examples
    ]


def groups_json(example: Dict[str, Any], key: str = "groups") -> str:
    """Serialise an example's label groups the way the demo's group manager stores them."""
    return json.dumps(example.get(key) or [])


# ═════════════════════════════════════════════════════════════════════════════
# Demo app — parsing, inference handlers and the Gradio UI
# ═════════════════════════════════════════════════════════════════════════════


# ── Shared helpers ───────────────────────────────────────────────────────────

def parse_label_groups(groups_json: str) -> Optional[Union[List[str], Dict[str, List[str]]]]:
    """Parse label groups JSON into the format expected by GLiFormer.

    Accepts:
      [{"parent": "name", "labels": ["a", "b"]}, ...]
    Returns:
      - None if empty
      - List[str] if single unnamed group
      - Dict[str, List[str]] if named groups
    """
    if not groups_json or groups_json.strip() in ("", "[]"):
        return None
    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return None
    if not groups:
        return None
    # Filter out groups with no labels
    groups = [g for g in groups if g.get("labels")]
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]["labels"]
    return {g["parent"]: g["labels"] for g in groups}


def _parse_field_type(type_spec: str) -> Union[str, FieldType]:
    """Parse a field type like `str` or `list[int]` into a formatter spec."""
    normalized = type_spec.strip().replace(" ", "")
    if not normalized:
        return "str"
    if normalized == "string":
        return "str"

    if normalized.startswith("list[") and normalized.endswith("]"):
        item_type = normalized[5:-1].strip() or "str"
        if item_type == "string":
            item_type = "str"
        return FieldType("list", list_item_type=item_type)

    return normalized


def _parse_structure_field(label: str) -> tuple[str, Optional[Union[str, FieldType]]]:
    """Parse a structuring field label, optionally with `field:type` syntax."""
    field_name, sep, type_spec = label.rpartition(":")
    if not sep:
        return label.strip(), None
    return field_name.strip(), _parse_field_type(type_spec)


def parse_structure_groups(groups_json: str) -> Optional[Dict[str, Union[List[str], dict]]]:
    """Parse structuring groups, preserving optional per-field type hints."""
    if not groups_json or groups_json.strip() in ("", "[]"):
        return None
    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return None
    if not groups:
        return None
    groups = [g for g in groups if g.get("labels")]
    if not groups:
        return None

    parsed_groups = {}
    for group in groups:
        field_names = []
        field_types = {}
        for raw_label in group["labels"]:
            field_name, field_type = _parse_structure_field(raw_label)
            if not field_name:
                continue
            field_names.append(field_name)
            if field_type is not None:
                field_types[field_name] = field_type

        if not field_names:
            continue

        parsed_groups[group["parent"]] = {
            "fields": field_names,
            "field_types": field_types or None,
        }

    return parsed_groups or None


def build_structuring_formatter(
    structures: Optional[Dict[str, Union[List[str], dict]]]
) -> Optional[StructuringOutputFormatter]:
    """Build a formatter when any structuring field carries type metadata."""
    if not structures:
        return None

    schema_types = {}
    for schema_name, schema_spec in structures.items():
        if not isinstance(schema_spec, dict):
            continue
        field_types = schema_spec.get("field_types")
        if field_types:
            schema_types[schema_name] = field_types

    if not schema_types:
        return None

    return StructuringOutputFormatter(schema_types)


def parse_joint_relex_groups(groups_json: str) -> Optional[Dict[str, dict]]:
    """Parse joint relex groups.

    Accepts:
      [{"parent": "name", "entities": ["a"], "relations": ["r"]}, ...]
    """
    if not groups_json or groups_json.strip() in ("", "[]"):
        return None
    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return None
    if not groups:
        return None
    groups = [g for g in groups if g.get("entities") and g.get("relations")]
    if not groups:
        return None
    return {
        g["parent"]: {"entities": g["entities"], "relations": g["relations"]}
        for g in groups
    }


def create_label_group_manager(
    task_name: str,
    default_parent: str = "default",
    default_labels: str = "",
    extra_fields: bool = False,
    initial_groups: Optional[list] = None,
):
    """Create a dynamic label group manager with add/remove buttons.

    Returns (state, table) — the gr.State holding the groups JSON and the
    Dataframe mirroring it, so example loaders can update both.
    """
    initial_groups = initial_groups or []
    groups_state = gr.State(value=json.dumps(initial_groups))

    with gr.Column():
        gr.Markdown(f"### Label Groups")

        # Container for rendered groups
        groups_display = gr.JSON(
            label="Current groups",
            visible=False,
        )

        if not extra_fields:
            with gr.Row():
                parent_input = gr.Textbox(
                    label="Group name",
                    placeholder=default_parent,
                    scale=1,
                )
                labels_input = gr.Textbox(
                    label="Labels (comma-separated)",
                    placeholder=default_labels,
                    scale=3,
                )
                add_btn = gr.Button("+", scale=0, min_width=50, variant="primary")
        else:
            # Joint relex: entities + relations
            with gr.Row():
                parent_input = gr.Textbox(
                    label="Group name",
                    placeholder=default_parent,
                    scale=1,
                )
                labels_input = gr.Textbox(
                    label="Entity types (comma-separated)",
                    placeholder="person, organization, location",
                    scale=2,
                )
                relations_input = gr.Textbox(
                    label="Relation types (comma-separated)",
                    placeholder="works_at, born_in, located_in",
                    scale=2,
                )
                add_btn = gr.Button("+", scale=0, min_width=50, variant="primary")

        # Display current groups as a dataframe
        groups_table = gr.Dataframe(
            value=_groups_to_table(initial_groups, extra_fields),
            headers=["#", "Group", "Labels"] if not extra_fields else ["#", "Group", "Entities", "Relations"],
            datatype=["number", "str", "str"] if not extra_fields else ["number", "str", "str", "str"],
            label="Groups",
            interactive=False,
            col_count=(3 if not extra_fields else 4, "fixed"),
        )

        with gr.Row():
            remove_idx = gr.Number(label="Remove group #", precision=0, minimum=1, scale=1)
            remove_btn = gr.Button("Remove", scale=0, min_width=80, variant="stop")
            clear_btn = gr.Button("Clear all", scale=0, min_width=80)

    def add_group(current_json, parent, labels, *args):
        try:
            groups = json.loads(current_json) if current_json else []
        except json.JSONDecodeError:
            groups = []

        parent = parent.strip() if parent.strip() else f"group_{len(groups) + 1}"
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        if not label_list:
            return current_json, _groups_to_table(groups, extra_fields)

        if extra_fields:
            rel_list = [r.strip() for r in args[0].split(",") if r.strip()] if args else []
            groups.append({"parent": parent, "entities": label_list, "relations": rel_list})
        else:
            groups.append({"parent": parent, "labels": label_list})

        new_json = json.dumps(groups)
        return new_json, _groups_to_table(groups, extra_fields)

    def remove_group(current_json, idx):
        try:
            groups = json.loads(current_json) if current_json else []
        except json.JSONDecodeError:
            groups = []
        idx = int(idx) - 1
        if 0 <= idx < len(groups):
            groups.pop(idx)
        new_json = json.dumps(groups)
        return new_json, _groups_to_table(groups, extra_fields)

    def clear_groups():
        return "[]", _groups_to_table([], extra_fields)

    if not extra_fields:
        add_btn.click(
            fn=add_group,
            inputs=[groups_state, parent_input, labels_input],
            outputs=[groups_state, groups_table],
        )
    else:
        add_btn.click(
            fn=add_group,
            inputs=[groups_state, parent_input, labels_input, relations_input],
            outputs=[groups_state, groups_table],
        )

    remove_btn.click(
        fn=remove_group,
        inputs=[groups_state, remove_idx],
        outputs=[groups_state, groups_table],
    )

    clear_btn.click(
        fn=clear_groups,
        inputs=[],
        outputs=[groups_state, groups_table],
    )

    return groups_state, groups_table


def _groups_to_table(groups: list, extra_fields: bool = False) -> list:
    """Convert groups list to a table for display."""
    if not groups:
        return []
    rows = []
    for i, g in enumerate(groups):
        if extra_fields:
            rows.append([
                i + 1,
                g.get("parent", ""),
                ", ".join(g.get("entities", [])),
                ", ".join(g.get("relations", [])),
            ])
        else:
            rows.append([
                i + 1,
                g.get("parent", ""),
                ", ".join(g.get("labels", [])),
            ])
    return rows


# ── Example loading & metric plumbing ────────────────────────────────────────

NO_GOLD_NOTE = (
    "_Load one of the examples below to score the prediction against a gold annotation._"
)


def guard(num_outputs: int = 2):
    """Surface handler errors in the UI instead of raising inside Gradio.

    A checkpoint may not carry every task head, so a tab can legitimately fail
    while the rest of the demo keeps working.
    """

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — demo surface
                traceback.print_exc()
                message = f"**Error:** {type(exc).__name__}: {exc}"
                if num_outputs == 1:
                    return message
                return (message, *[""] * (num_outputs - 1))

        return wrapper

    return decorate


def gold_payload(gold: dict, *inputs: Any) -> str:
    """Pack an example's gold block together with the inputs it belongs to."""
    return json.dumps({"inputs": [str(value) for value in inputs], "gold": gold}, default=str)


def gold_for(payload: str, *current_inputs: Any) -> Optional[dict]:
    """Return the gold block, or None if the tab inputs no longer match it."""
    if not payload:
        return None
    try:
        stored = json.loads(payload)
    except json.JSONDecodeError:
        return None
    stored_inputs = stored.get("inputs", [])
    if len(stored_inputs) != len(current_inputs):
        return None
    if any(a.strip() != str(b).strip() for a, b in zip(stored_inputs, current_inputs)):
        return None
    return stored.get("gold")


def metrics_block(families: Dict[str, PRF], notes: Sequence[str] = ()) -> str:
    return "#### Metrics vs. gold\n\n" + format_single_report(families, notes)


def load_example(
    examples: List[dict],
    index: int,
    extra_fields: bool = False,
) -> Tuple[str, str, list, str, str]:
    """Load a single-task example into (text, groups JSON, table, gold state, gold view)."""
    example = examples[int(index)]
    groups = example.get("groups") or []
    serialized = json.dumps(groups)
    return (
        example["text"],
        serialized,
        _groups_to_table(groups, extra_fields),
        gold_payload(example["gold"], example["text"], serialized),
        json.dumps(example["gold"], indent=2, default=str),
    )


def build_structuring_schema(model, structures: Dict[str, Union[List[str], dict]]):
    """Turn parsed structuring groups into a GLiFormerSchema."""
    schema = model.create_schema()
    for schema_name, schema_spec in structures.items():
        if isinstance(schema_spec, dict) and schema_spec.get("field_types"):
            schema.add_structure(schema_name, schema_spec["field_types"])
        elif isinstance(schema_spec, dict):
            schema.add_structure(schema_name, schema_spec.get("fields", []))
        else:
            schema.add_structure(schema_name, schema_spec)
    return schema


def parse_multi_level_schema(schema_json: str) -> Optional[Dict[str, Any]]:
    """Parse the nested schema template the multi-level tab edits.

    The template is the JSON shape `GLiFormerSchema.add_structure` compiles: a
    scalar field maps to its type name, an object to a nested template, and a
    repeated child to a one-element list holding the child's template.
    """
    if not schema_json or not schema_json.strip():
        return None
    try:
        schemas = json.loads(schema_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(schemas, dict) or not schemas:
        return None
    return schemas


def build_multi_level_schema(model, templates: Dict[str, Any]):
    """Turn nested schema templates into a GLiFormerSchema."""
    schema = model.create_schema()
    for schema_name, template in templates.items():
        schema.add_structure(schema_name, template)
    return schema


def structures_for_inference(
    structures: Optional[Dict[str, Union[List[str], dict]]]
) -> Optional[Dict[str, List[str]]]:
    """Reduce parsed structuring groups to the mapping `inference()` expects.

    `parse_structure_groups` produces the `{"fields", "field_types"}` envelope
    the schema builder takes. `inference(structures=...)` instead compiles the
    mapping as a template, so that envelope would be read as two literal fields
    named "fields" and "field_types". Pass plain field lists here; the type
    hints are reapplied afterwards by `build_structuring_formatter`.
    """
    if not structures:
        return None
    return {
        name: (spec.get("fields", []) if isinstance(spec, dict) else spec)
        for name, spec in structures.items()
    }


def rank_candidates(model, query: str, candidates: List[str]) -> List[dict]:
    """Cosine-similarity ranking of candidates against a query."""
    embeddings = model.embed_text([query] + candidates)
    query_emb, cand_embs = embeddings[0:1], embeddings[1:]
    query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cand_norm = cand_embs / cand_embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    similarities = (query_norm @ cand_norm.T).squeeze(0).tolist()
    return [
        {"text": text, "similarity": round(similarity, 4)}
        for text, similarity in sorted(zip(candidates, similarities), key=lambda pair: -pair[1])
    ]


# ── Task inference functions ─────────────────────────────────────────────────

@guard()
def run_ner(text: str, groups_value: str, threshold: float, flat_ner: bool, multi_label: bool, gold_state: str):
    if not text.strip():
        return "Please enter some text.", ""
    entities = parse_label_groups(groups_value)
    if entities is None:
        return "Please add at least one label group.", ""

    results = get_model().inference(
        text, entities=entities, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label
    )
    predictions = results.get("ner", [[]])[0]

    gold = gold_for(gold_state, text, groups_value)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        gold_keys, pred_keys = ner_key_sets(gold["ner"], predictions)
        report = metrics_block(
            {"NER": prf_from_sets(gold_keys, pred_keys)},
            [format_key_diff(gold_keys, pred_keys)],
        )
    return json.dumps(predictions, indent=2, default=str), report


@guard()
def run_classification(text: str, groups_value: str, threshold: float, multi_label: bool, gold_state: str):
    if not text.strip():
        return "Please enter some text.", ""
    classes = parse_label_groups(groups_value)
    if classes is None:
        return "Please add at least one label group.", ""

    results = get_model().inference(text, classes=classes, threshold=threshold, multi_label=multi_label)
    predictions = results.get("classification", [[]])[0]

    gold = gold_for(gold_state, text, groups_value)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        gold_keys, pred_keys = classification_key_sets(gold["classification"], predictions)
        exact = "yes" if gold_keys == pred_keys else "no"
        report = metrics_block(
            {"Classification": prf_from_sets(gold_keys, pred_keys)},
            [format_key_diff(gold_keys, pred_keys), f"- **Exact label set:** {exact}"],
        )
    return json.dumps(predictions, indent=2, default=str), report


@guard()
def run_joint_relex(text: str, groups_value: str, threshold: float, flat_ner: bool, gold_state: str):
    if not text.strip():
        return "Please enter some text.", ""
    joint_relations = parse_joint_relex_groups(groups_value)
    if joint_relations is None:
        return "Please add at least one group with both entities and relations.", ""

    results = get_model().inference(
        text, joint_relations=joint_relations, threshold=threshold, flat_ner=flat_ner
    )
    entities = results.get("ner", [[]])[0]
    triples = results.get("joint_relex", [[]])[0]
    output = {"ner": entities, "joint_relex": triples}

    gold = gold_for(gold_state, text, groups_value)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        gold_entities, pred_entities = ner_key_sets(gold["ner"], entities)
        gold_triples, pred_triples = relation_key_sets(gold["relations"], triples)
        report = metrics_block(
            {
                "NER": prf_from_sets(gold_entities, pred_entities),
                "Relations": prf_from_sets(gold_triples, pred_triples),
            },
            [
                "**Entities**",
                format_key_diff(gold_entities, pred_entities),
                "**Relations**",
                format_key_diff(gold_triples, pred_triples),
            ],
        )
    return json.dumps(output, indent=2, default=str), report


@guard()
def run_structuring(text: str, groups_value: str, threshold: float, flat_ner: bool, gold_state: str):
    if not text.strip():
        return "Please enter some text.", ""
    structures = parse_structure_groups(groups_value)
    if structures is None:
        return "Please add at least one schema group.", ""

    model = get_model()
    schema = build_structuring_schema(model, structures)
    results = model.inference_from_schema(
        schema=schema, texts=text, threshold=threshold, flat_ner=flat_ner
    )
    predictions = results.get("structuring", [{}])[0]

    gold = gold_for(gold_state, text, groups_value)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        gold_keys, pred_keys = structuring_key_sets(gold["structuring"], predictions)
        report = metrics_block(
            {"Structuring": structuring_prf(gold["structuring"], predictions)},
            [format_key_diff(gold_keys, pred_keys)],
        )
    return json.dumps(predictions, indent=2, default=str), report


@guard()
def run_multi_level_structuring(
    text: str, schema_value: str, threshold: float, flat_ner: bool, gold_state: str
):
    if not text.strip():
        return "Please enter some text.", ""
    templates = parse_multi_level_schema(schema_value)
    if templates is None:
        return "Please provide a JSON object mapping schema names to templates.", ""

    model = get_model()
    schema = build_multi_level_schema(model, templates)
    results = model.inference_from_schema(
        schema=schema, texts=text, threshold=threshold, flat_ner=flat_ner
    )
    predictions = results.get("structuring", [{}])[0]

    gold = gold_for(gold_state, text, schema_value)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        gold_keys, pred_keys = structuring_key_sets(gold["structuring"], predictions)
        report = metrics_block(
            {"Structuring": structuring_prf(gold["structuring"], predictions)},
            [format_key_diff(gold_keys, pred_keys)],
        )
    return json.dumps(predictions, indent=2, default=str), report


@guard()
def run_embedding(query: str, candidates: str, gold_state: str):
    if not query.strip():
        return "Please enter a query text.", ""
    candidate_list = [line.strip() for line in candidates.split("\n") if line.strip()]
    if not candidate_list:
        return "Please enter at least one candidate text.", ""

    ranking = rank_candidates(get_model(), query, candidate_list)

    gold = gold_for(gold_state, query, candidates)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        scores = ranking_metrics([item["text"] for item in ranking], gold["relevant"])
        rows = [[key.upper(), f"{value:.3f}"] for key, value in scores.items()]
        relevant = ", ".join(f"`{text}`" for text in gold["relevant"])
        report = (
            "#### Metrics vs. gold\n\n"
            + render_table(["Metric", "Value"], rows)
            + f"\n\n- **Relevant candidates:** {relevant}"
        )
    return json.dumps(ranking, indent=2, default=str), report


@guard()
def run_multitask(
    text: str,
    ner_value: str,
    cls_value: str,
    joint_value: str,
    struct_value: str,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
    gold_state: str,
):
    if not text.strip():
        return "Please enter some text.", ""
    entities = parse_label_groups(ner_value)
    classes = parse_label_groups(cls_value)
    joint_relations = parse_joint_relex_groups(joint_value)
    structures = parse_structure_groups(struct_value)

    if all(value is None for value in (entities, classes, joint_relations, structures)):
        return "Please configure at least one task.", ""

    results = get_model().inference(
        text,
        entities=entities,
        classes=classes,
        joint_relations=joint_relations,
        structures=structures_for_inference(structures),
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
    )
    formatter = build_structuring_formatter(structures)
    if formatter is not None and "structuring" in results:
        results["structuring"] = formatter.format_batch(results["structuring"])

    output = {key: value[0] for key, value in results.items() if value}

    gold = gold_for(gold_state, text, ner_value, cls_value, struct_value)
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        families, notes = {}, []
        if "ner" in gold:
            gold_keys, pred_keys = ner_key_sets(gold["ner"], output.get("ner", []))
            families["NER"] = prf_from_sets(gold_keys, pred_keys)
            notes += ["**Entities**", format_key_diff(gold_keys, pred_keys)]
        if "classification" in gold:
            gold_keys, pred_keys = classification_key_sets(
                gold["classification"], output.get("classification", [])
            )
            families["Classification"] = prf_from_sets(gold_keys, pred_keys)
            notes += ["**Classes**", format_key_diff(gold_keys, pred_keys)]
        if "structuring" in gold:
            predictions = output.get("structuring", {})
            gold_keys, pred_keys = structuring_key_sets(gold["structuring"], predictions)
            families["Structuring"] = structuring_prf(gold["structuring"], predictions)
            notes += ["**Fields**", format_key_diff(gold_keys, pred_keys)]
        report = metrics_block(families, notes)
    return json.dumps(output, indent=2, default=str), report


# ── Batch scoring over all examples of a tab ─────────────────────────────────

@guard(1)
def score_all_ner(threshold: float, flat_ner: bool, multi_label: bool):
    model = get_model()
    per_example = []
    for example in NER_EXAMPLES:
        results = model.inference(
            example["text"],
            entities=parse_label_groups(groups_json(example)),
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        prf = ner_prf(example["gold"]["ner"], results.get("ner", [[]])[0])
        per_example.append((example["name"], {"NER": prf}))
    return format_prf_report(
        f"NER — {len(NER_EXAMPLES)} examples @ threshold {threshold:g}", per_example
    )


@guard(1)
def score_all_classification(threshold: float):
    model = get_model()
    per_example, exact_flags = [], []
    for example in CLASSIFICATION_EXAMPLES:
        results = model.inference(
            example["text"],
            classes=parse_label_groups(groups_json(example)),
            threshold=threshold,
            multi_label=example["multi_label"],
        )
        prf, exact = classification_prf(
            example["gold"]["classification"], results.get("classification", [[]])[0]
        )
        per_example.append((example["name"], {"Classification": prf}))
        exact_flags.append(1 if exact else 0)
    return format_prf_report(
        f"Classification — {len(CLASSIFICATION_EXAMPLES)} examples @ threshold {threshold:g}",
        per_example,
        extra_columns={"Exact set": exact_flags},
    )


@guard(1)
def score_all_joint_relex(threshold: float, flat_ner: bool):
    model = get_model()
    per_example = []
    for example in JOINT_RELEX_EXAMPLES:
        results = model.inference(
            example["text"],
            joint_relations=parse_joint_relex_groups(groups_json(example)),
            threshold=threshold,
            flat_ner=flat_ner,
        )
        per_example.append((
            example["name"],
            {
                "NER": ner_prf(example["gold"]["ner"], results.get("ner", [[]])[0]),
                "Rel": relation_prf(
                    example["gold"]["relations"], results.get("joint_relex", [[]])[0]
                ),
            },
        ))
    return format_prf_report(
        f"Joint NER + relations — {len(JOINT_RELEX_EXAMPLES)} examples @ threshold {threshold:g}",
        per_example,
    )


@guard(1)
def score_all_structuring(threshold: float, flat_ner: bool):
    model = get_model()
    per_example = []
    for example in STRUCTURING_EXAMPLES:
        structures = parse_structure_groups(groups_json(example))
        results = model.inference_from_schema(
            schema=build_structuring_schema(model, structures),
            texts=example["text"],
            threshold=threshold,
            flat_ner=flat_ner,
        )
        prf = structuring_prf(
            example["gold"]["structuring"], results.get("structuring", [{}])[0]
        )
        per_example.append((example["name"], {"Structuring": prf}))
    return format_prf_report(
        f"Structuring — {len(STRUCTURING_EXAMPLES)} examples @ threshold {threshold:g}", per_example
    )


@guard(1)
def score_all_multi_level_structuring(threshold: float, flat_ner: bool):
    model = get_model()
    per_example = []
    for example in MULTI_LEVEL_STRUCTURING_EXAMPLES:
        results = model.inference_from_schema(
            schema=build_multi_level_schema(model, example["schema"]),
            texts=example["text"],
            threshold=threshold,
            flat_ner=flat_ner,
        )
        prf = structuring_prf(
            example["gold"]["structuring"], results.get("structuring", [{}])[0]
        )
        per_example.append((example["name"], {"Structuring": prf}))
    return format_prf_report(
        f"Multi-level structuring — {len(MULTI_LEVEL_STRUCTURING_EXAMPLES)} examples "
        f"@ threshold {threshold:g}",
        per_example,
    )


@guard(1)
def score_all_embedding():
    model = get_model()
    per_example = []
    for example in EMBEDDING_EXAMPLES:
        ranking = rank_candidates(model, example["query"], example["candidates"])
        scores = ranking_metrics(
            [item["text"] for item in ranking], example["gold"]["relevant"]
        )
        per_example.append((example["name"], scores))
    return format_ranking_report(
        f"Embedding retrieval — {len(EMBEDDING_EXAMPLES)} queries", per_example
    )


@guard(1)
def score_all_multitask(threshold: float, flat_ner: bool, multi_label: bool):
    model = get_model()
    per_example = []
    for example in MULTITASK_EXAMPLES:
        structures = parse_structure_groups(groups_json(example, "structuring"))
        results = model.inference(
            example["text"],
            entities=parse_label_groups(groups_json(example, "ner")),
            classes=parse_label_groups(groups_json(example, "classification")),
            structures=structures_for_inference(structures),
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        formatter = build_structuring_formatter(structures)
        if formatter is not None and "structuring" in results:
            results["structuring"] = formatter.format_batch(results["structuring"])
        gold = example["gold"]
        per_example.append((
            example["name"],
            {
                "NER": ner_prf(gold["ner"], results.get("ner", [[]])[0]),
                "Cls": classification_prf(
                    gold["classification"], results.get("classification", [[]])[0]
                )[0],
                "Struct": structuring_prf(
                    gold["structuring"], results.get("structuring", [{}])[0]
                ),
            },
        ))
    return format_prf_report(
        f"Multi-task — {len(MULTITASK_EXAMPLES)} examples @ threshold {threshold:g}", per_example
    )


# ── Document layout (PDF) NER ────────────────────────────────────────────────

# Word boxes use LayoutLM-style page coordinates: both axes normalized to
# 0-1000, which is also the space the page preview draws in.
LAYOUT_COORD_SCALE = 1000

# ``parse_pdf`` only records the source path, so the inline path needs a label
# rather than a real file.
LAYOUT_SOURCE_LABEL = "<inline layout document>"

_LAYOUT_SPAN_COLORS = (
    "#2563eb", "#db2777", "#059669", "#d97706",
    "#7c3aed", "#0891b2", "#dc2626", "#4d7c0f",
)


def model_reads_layout(loaded_model) -> bool:
    """Whether the checkpoint consumes word boxes rather than ignoring them."""
    config = loaded_model.config
    return (
        getattr(config, "model_variant", "text") == "layout"
        or bool(getattr(config, "use_layout", False))
    )


def layout_document_json(words: Sequence[Sequence[Any]]) -> str:
    """Render ``(word, x0, y0, x1, y1)`` rows as the document box's JSON."""
    entries = []
    for row in words:
        entry: Dict[str, Any] = {"word": row[0], "bbox": [int(v) for v in row[1:5]]}
        if len(row) > 5 and row[5] is not None:
            entry["page"] = int(row[5])
        entries.append(json.dumps(entry, ensure_ascii=False))
    return '{\n  "layout": [\n    ' + ",\n    ".join(entries) + "\n  ]\n}"


def _coerce_bbox(bbox) -> List[int]:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            pass
    raise ValueError(f"Each bounding box needs four numbers [x0, y0, x1, y1], got {bbox!r}.")


def parse_layout_document(document_value: str) -> Tuple[List[str], List[List[int]], Optional[List[int]]]:
    """Parse a page into word, box and page-id lists.

    Accepts the layout corpus row shapes as well as the compact form the
    presets use::

        {"layout": [{"word": "Total", "bbox": [x0, y0, x1, y1], "page": 0}, ...]}
        {"tokenized_text": [...], "bboxes": [[x0, y0, x1, y1], ...]}
    """
    if not document_value or not document_value.strip():
        raise ValueError("Please provide a page as words with bounding boxes.")
    document = json.loads(document_value)

    entries: Any
    if isinstance(document, dict):
        entries = document.get("layout")
        if entries is None:
            words = document.get("tokenized_text", document.get("words"))
            boxes = document.get("bboxes", document.get("bbox", document.get("boxes")))
            if words is None or boxes is None:
                raise ValueError('The page needs a "layout" list, or "words" plus "bboxes".')
            if len(words) != len(boxes):
                raise ValueError(
                    f"Words and boxes must have the same length, got {len(words)} and {len(boxes)}."
                )
            raw_pages = document.get("page_ids", document.get("pages"))
            pages = [int(page) for page in raw_pages] if raw_pages is not None else None
            return [str(w) for w in words], [_coerce_bbox(b) for b in boxes], pages
    elif isinstance(document, list):
        entries = document
    else:
        raise ValueError("The page JSON must be an object or an array.")

    if not entries:
        raise ValueError("The page has no words.")

    words, boxes, pages = [], [], []
    for entry in entries:
        if isinstance(entry, dict):
            word = entry.get("word", entry.get("text", entry.get("token")))
            bbox = entry.get("bbox", entry.get("box"))
            page = entry.get("page", entry.get("page_id"))
            if word is None or bbox is None:
                raise ValueError(f'Layout entries need "word" and "bbox" keys, got {entry!r}.')
        elif isinstance(entry, (list, tuple)) and len(entry) == 5:
            word, bbox, page = entry[0], entry[1:5], None
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            word, bbox = entry[0], entry[1]
            page = entry[2] if len(entry) > 2 else None
        else:
            raise ValueError(f"Unsupported layout entry: {entry!r}")
        words.append(str(word))
        boxes.append(_coerce_bbox(bbox))
        pages.append(None if page is None else int(page))
    page_ids = [0 if p is None else p for p in pages] if any(p is not None for p in pages) else None
    return words, boxes, page_ids


def example_layout(example: dict) -> Tuple[List[str], List[List[int]], Optional[List[int]]]:
    """Split a preset's ``words`` rows into the lists inference needs."""
    words = [str(row[0]) for row in example["words"]]
    boxes = [[int(v) for v in row[1:5]] for row in example["words"]]
    pages = [int(row[5]) for row in example["words"] if len(row) > 5]
    return words, boxes, pages if len(pages) == len(words) else None


def _layout_pages(words, boxes, page_ids):
    """Group flat word/box lists into the per-page lists ``parse_pdf`` expects."""
    if not page_ids:
        return [list(words)], [list(boxes)], [0]
    page_words, page_boxes, pages = [], [], []
    for word, box, page in zip(words, boxes, page_ids):
        if not pages or page != pages[-1]:
            pages.append(page)
            page_words.append([])
            page_boxes.append([])
        page_words[-1].append(word)
        page_boxes[-1].append(box)
    return page_words, page_boxes, pages


def layout_predictions(
    model,
    words: List[str],
    boxes: List[List[int]],
    page_ids: Optional[List[int]],
    entities,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
    use_boxes: bool,
) -> List[dict]:
    """Run NER over a page, with or without feeding the word boxes."""
    page_words, page_boxes, pages = _layout_pages(words, boxes, page_ids)
    results = model.parse_pdf(
        LAYOUT_SOURCE_LABEL,
        words=page_words,
        bbox=page_boxes if use_boxes else None,
        pages=pages,
        entities=entities,
        add_image_token=False,
        return_word_bboxes=bool(use_boxes),
        return_page_ids=True,
        split_pages=False,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
    )
    return results.get("ner", [[]])[0]


# ── Page preview ─────────────────────────────────────────────────────────────

def _word_char_spans(words: Sequence[str]) -> Tuple[List[int], List[int]]:
    """Character offsets of each word in the space-joined page text.

    ``parse_pdf`` joins the words with single spaces before decoding, so spans
    come back as offsets into exactly that string.
    """
    starts, ends, cursor = [], [], 0
    for index, word in enumerate(words):
        if index:
            cursor += 1
        starts.append(cursor)
        cursor += len(word)
        ends.append(cursor)
    return starts, ends


def _span_word_range(starts, ends, span_start: int, span_end: int):
    """First and last word index covered by a character span."""
    first = last = None
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end <= span_start:
            continue
        if start >= span_end:
            break
        if first is None:
            first = index
        last = index
    return first, last


def _gold_word_ranges(words: Sequence[str], gold_entries: Sequence[dict]) -> List[Tuple[int, int]]:
    """Locate each gold mention in the word sequence.

    Gold is written as a surface string, so it is matched against the words it
    was built from rather than against character offsets.
    """
    lowered = [word.casefold() for word in words]
    ranges = []
    for entry in gold_entries or []:
        target = " ".join(str(entry.get("text", "")).split()).casefold()
        if not target:
            continue
        found = None
        for start in range(len(words)):
            joined = ""
            for end in range(start, len(words)):
                joined = lowered[end] if not joined else f"{joined} {lowered[end]}"
                if joined == target:
                    found = (start, end)
                    break
                if len(joined) >= len(target):
                    break
            if found is not None:
                break
        if found is not None:
            ranges.append(found)
    return ranges


def layout_preview_html(
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    page_ids: Optional[Sequence[int]],
    predictions: Sequence[dict],
    gold_entries: Sequence[dict] = (),
    *,
    boxes_used: bool = True,
) -> str:
    """Redraw the page and overlay predicted entities and gold spans."""
    if not words:
        return ""

    starts, ends = _word_char_spans(words)
    labels = sorted({str(p.get("label", "")) for p in predictions})
    color_of = {
        label: _LAYOUT_SPAN_COLORS[index % len(_LAYOUT_SPAN_COLORS)]
        for index, label in enumerate(labels)
    }

    # First prediction wins, so overlapping spans cannot repaint a word.
    predicted: Dict[int, Tuple[str, float]] = {}
    for entity in predictions:
        first, last = _span_word_range(
            starts, ends, int(entity.get("start", 0)), int(entity.get("end", 0))
        )
        if first is None:
            continue
        for index in range(first, last + 1):
            predicted.setdefault(
                index, (str(entity.get("label", "")), float(entity.get("score") or 0.0))
            )

    gold_words = set()
    for first, last in _gold_word_ranges(words, gold_entries):
        gold_words.update(range(first, last + 1))

    pages = list(page_ids) if page_ids else [0] * len(words)
    rendered = []
    for page in dict.fromkeys(pages):
        shapes = [
            f'<rect x="0" y="0" width="{LAYOUT_COORD_SCALE}" height="{LAYOUT_COORD_SCALE}" '
            'fill="#ffffff" stroke="#d1d5db" stroke-width="1.5"/>'
        ]
        for index, (word, box) in enumerate(zip(words, boxes)):
            if pages[index] != page:
                continue
            x0, y0, x1, y1 = box
            width, height = max(x1 - x0, 1), max(y1 - y0, 1)
            if index in gold_words:
                shapes.append(
                    f'<rect x="{x0 - 3}" y="{y0 - 3}" width="{width + 6}" height="{height + 6}" '
                    'rx="2" fill="none" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 3"/>'
                )
            hit = predicted.get(index)
            if hit is not None:
                label, score = hit
                color = color_of.get(label, _LAYOUT_SPAN_COLORS[0])
                shapes.append(
                    f'<rect x="{x0 - 1}" y="{y0 - 1}" width="{width + 2}" height="{height + 2}" '
                    f'rx="2" fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="1.4">'
                    f"<title>{escape(label)} · {score:.2f}</title></rect>"
                )
            else:
                color = "#111827"
                shapes.append(
                    f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" '
                    'stroke="#e5e7eb" stroke-width="0.7"/>'
                )
            font_size = min(max(height * 0.78, 5.0), 26.0)
            shapes.append(
                f'<text x="{x0}" y="{y1 - height * 0.16:.1f}" font-size="{font_size:.1f}" '
                f'textLength="{width}" lengthAdjust="spacingAndGlyphs" fill="{color}" '
                f'font-family="DejaVu Sans, Arial, sans-serif">{escape(word)}</text>'
            )
        label_line = f"Page {page}" if len(set(pages)) > 1 else "Page"
        rendered.append(
            f'<div style="margin-bottom:10px">'
            f'<div style="font-size:12px;color:#6b7280;margin-bottom:3px">{label_line}</div>'
            f'<svg viewBox="0 0 {LAYOUT_COORD_SCALE} {LAYOUT_COORD_SCALE}" '
            'style="width:100%;height:auto;background:#fff;border-radius:4px;'
            'box-shadow:0 1px 4px rgba(0,0,0,0.12)" xmlns="http://www.w3.org/2000/svg">'
            f'{"".join(shapes)}</svg></div>'
        )

    legend = "".join(
        f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:1px 6px;'
        f'border:1px solid {color};border-radius:4px;color:{color};font-size:12px">'
        f"{escape(label)}</span>"
        for label, color in color_of.items()
    )
    if gold_words:
        legend += (
            '<span style="display:inline-block;margin:2px 6px 2px 0;padding:1px 6px;'
            'border:1px dashed #9ca3af;border-radius:4px;color:#6b7280;font-size:12px">'
            "gold span</span>"
        )
    if predictions:
        caption = (
            "Filled boxes are predictions; hover one for its label and score."
            if boxes_used
            else "Text-only run — the page is drawn for reference, its boxes were not fed to the model."
        )
    else:
        caption = "No predictions yet — run the page to overlay them."
    return (
        '<div style="font-family:system-ui,sans-serif">'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:5px">{caption} '
        f"Coordinates are the normalized 0-{LAYOUT_COORD_SCALE} page space the model reads.</div>"
        f'{"".join(rendered)}<div style="margin-top:4px">{legend}</div></div>'
    )


# ── Layout handlers ──────────────────────────────────────────────────────────

@guard(3)
def run_layout_ner(
    document_value: str,
    groups_value: str,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
    use_boxes: bool,
    gold_state: str,
):
    words, boxes, page_ids = parse_layout_document(document_value)
    entities = parse_label_groups(groups_value)
    if entities is None:
        return "Please add at least one label group.", "", ""

    model = get_model()
    predictions = layout_predictions(
        model, words, boxes, page_ids, entities, threshold, flat_ner, multi_label, use_boxes
    )

    gold = gold_for(gold_state, document_value, groups_value)
    notes = []
    if use_boxes and not model_reads_layout(model):
        notes.append(
            "- **Note:** this checkpoint is not layout-aware (`model_variant` is not "
            "`layout`), so the boxes are ignored."
        )
    if gold is None:
        report = NO_GOLD_NOTE
    else:
        gold_keys, pred_keys = ner_key_sets(gold["ner"], predictions)
        report = metrics_block(
            {"Layout NER": prf_from_sets(gold_keys, pred_keys)},
            [format_key_diff(gold_keys, pred_keys), *notes],
        )
    preview = layout_preview_html(
        words, boxes, page_ids, predictions,
        (gold or {}).get("ner", ()), boxes_used=use_boxes,
    )
    return json.dumps(predictions, indent=2, default=str), report, preview


@guard(1)
def score_all_layout_ner(threshold: float, flat_ner: bool, multi_label: bool, use_boxes: bool):
    model = get_model()
    per_example, text_only_f1 = [], []
    for example in LAYOUT_EXAMPLES:
        words, boxes, page_ids = example_layout(example)
        entities = parse_label_groups(groups_json(example))
        predictions = layout_predictions(
            model, words, boxes, page_ids, entities, threshold, flat_ner, multi_label, use_boxes
        )
        per_example.append(
            (example["name"], {"Layout NER": ner_prf(example["gold"]["ner"], predictions)})
        )
        if use_boxes:
            # The same pages without their boxes, so the table shows what the
            # geometry is actually contributing.
            baseline = layout_predictions(
                model, words, boxes, page_ids, entities, threshold, flat_ner, multi_label, False
            )
            text_only_f1.append(round(ner_prf(example["gold"]["ner"], baseline).f1, 3))
    mode = "boxes on" if use_boxes else "text only"
    return format_prf_report(
        f"Layout NER — {len(LAYOUT_EXAMPLES)} pages @ threshold {threshold:g} ({mode})",
        per_example,
        extra_columns={"F1 without boxes": text_only_f1} if use_boxes else None,
    )


def load_layout_example(index: int):
    """Load a page preset into the document box, groups, gold and preview."""
    example = LAYOUT_EXAMPLES[int(index)]
    document = layout_document_json(example["words"])
    groups = example.get("groups") or []
    serialized = json.dumps(groups)
    words, boxes, page_ids = example_layout(example)
    return (
        document,
        serialized,
        _groups_to_table(groups),
        gold_payload(example["gold"], document, serialized),
        json.dumps(example["gold"], indent=2, default=str),
        layout_preview_html(words, boxes, page_ids, [], example["gold"]["ner"]),
        "",
        NO_GOLD_NOTE.replace("Load one of the examples below to", "Run the page to"),
    )


def _parse_page_selection(pages_spec: str) -> Optional[List[int]]:
    if not pages_spec or not pages_spec.strip():
        return None
    pages = []
    for chunk in pages_spec.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.lstrip("-").isdigit():
            raise ValueError(f"Page selection must be whole numbers, got {chunk!r}.")
        pages.append(int(chunk))
    return pages or None


def extract_layout_from_pdf(pdf_path: Optional[str], pages_spec: str, current_document: str):
    """Fill the document box with the words and boxes read from a PDF.

    Errors are returned as status text rather than raised, so a missing PDF
    backend or an image-only scan never clears the page already loaded.
    """
    if not pdf_path:
        return current_document, "", "Select a PDF file first."
    try:
        pages = _parse_page_selection(pages_spec)
        rows = GLiFormerPDFProcessor()(
            pdf_path,
            pages=pages,
            add_image_token=False,
            return_pixel_values=False,
            return_word_bboxes=True,
            return_page_ids=True,
            split_pages=False,
        )
    except ImportError as exc:
        return current_document, "", f"PDF text extraction needs an extra package: {exc}"
    except Exception as exc:  # noqa: BLE001 — depends on the uploaded file
        traceback.print_exc()
        return current_document, "", f"Could not read the PDF: {type(exc).__name__}: {exc}"

    row = rows[0] if rows else {}
    words = list(row.get("tokenized_text") or [])
    if not words:
        return current_document, "", "No text layer found — this PDF may need OCR first."
    boxes = list(row.get("bboxes") or [])
    page_ids = list(row.get("page_ids") or [])
    document = layout_document_json(
        [
            (word, *box, page_ids[index] if index < len(page_ids) else None)
            for index, (word, box) in enumerate(zip(words, boxes))
        ]
    )
    preview = layout_preview_html(words, boxes, page_ids or None, [])
    page_count = len(set(page_ids)) if page_ids else 1
    return document, preview, f"Extracted {len(words)} words from {page_count} page(s)."


# ── Example pickers ──────────────────────────────────────────────────────────

def create_example_picker(examples: List[dict], score_all_label: str) -> Tuple[gr.Dataset, gr.Button]:
    """Render the click-to-load example table and the batch-scoring button."""
    with gr.Accordion(f"Examples ({len(examples)}) — click a row to load it", open=False):
        picker = gr.Dataset(
            components=["textbox", "textbox"],
            headers=["Example", "Text"],
            samples=dataset_samples(examples),
            samples_per_page=len(examples),
            type="index",
            label=None,
        )
        score_all_btn = gr.Button(score_all_label, variant="secondary")
    return picker, score_all_btn


def create_gold_view(example: dict) -> gr.Code:
    with gr.Accordion("Gold annotation", open=False):
        return gr.Code(
            value=json.dumps(example["gold"], indent=2, default=str),
            label="Gold",
            language="json",
        )


# ── Build the UI ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="GLiFormer Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # GLiFormer — Multi-task Information Extraction
        Extract entities, classify text, find relations, and structure information into flat or
        nested JSON — all with a single model.

        Add label groups using the **+** button, then click **Run** to see results.
        Every tab carries **ten annotated examples**: click a row to load one, or score the whole
        set at once to get task-specific precision / recall / F1.
        """
    )

    # ── NER Tab ──────────────────────────────────────────────────────────
    with gr.Tab("NER"):
        first = NER_EXAMPLES[0]
        with gr.Row():
            with gr.Column(scale=1):
                ner_text = gr.Textbox(label="Text", lines=5, value=first["text"])
                ner_groups, ner_table = create_label_group_manager(
                    "ner",
                    default_parent="general",
                    default_labels="person, organization, location, date",
                    initial_groups=first["groups"],
                )
                with gr.Row():
                    ner_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    ner_flat = gr.Checkbox(value=True, label="Flat NER")
                    ner_multi = gr.Checkbox(value=False, label="Multi-label")
                ner_btn = gr.Button("Run NER", variant="primary")
                ner_picker, ner_score_all = create_example_picker(
                    NER_EXAMPLES, "Score all 10 NER examples"
                )
                ner_gold = create_gold_view(first)
            with gr.Column(scale=1):
                ner_output = gr.Code(label="Results", language="json")
                ner_metrics = gr.Markdown()

        ner_gold_state = gr.State(
            value=gold_payload(first["gold"], first["text"], json.dumps(first["groups"]))
        )

        ner_btn.click(
            fn=run_ner,
            inputs=[ner_text, ner_groups, ner_threshold, ner_flat, ner_multi, ner_gold_state],
            outputs=[ner_output, ner_metrics],
        )
        ner_picker.click(
            fn=lambda index: load_example(NER_EXAMPLES, index),
            inputs=[ner_picker],
            outputs=[ner_text, ner_groups, ner_table, ner_gold_state, ner_gold],
        )
        ner_score_all.click(
            fn=score_all_ner,
            inputs=[ner_threshold, ner_flat, ner_multi],
            outputs=ner_metrics,
        )

    # ── PDF / Layout NER Tab ─────────────────────────────────────────────
    with gr.Tab("PDF / Layout NER"):
        first = LAYOUT_EXAMPLES[0]
        first_words, first_boxes, first_pages = example_layout(first)
        first_document = layout_document_json(first["words"])
        gr.Markdown(
            "Entities over a **page** rather than a paragraph: each word carries its "
            "`[x0, y0, x1, y1]` box in 0-1000 page coordinates, which is what a "
            "`model_variant: layout` checkpoint reads alongside the text. The presets "
            "keep the word order their source produced — OCR order for the scans, all "
            "labels before all values for the synthetic pages — so **Use bounding boxes** "
            "toggles between the layout path and a text-only baseline on the same page."
        )
        with gr.Row():
            with gr.Column(scale=1):
                layout_document = gr.Textbox(
                    label="Page — words with boxes (JSON)",
                    lines=12,
                    value=first_document,
                )
                with gr.Accordion("Read words and boxes from a PDF", open=False):
                    layout_pdf = gr.File(label="PDF file", file_types=[".pdf"], type="filepath")
                    layout_pdf_pages = gr.Textbox(
                        label="Pages (comma-separated, blank = all)", value=""
                    )
                    layout_pdf_btn = gr.Button("Extract words + boxes")
                    layout_pdf_status = gr.Markdown()
                layout_groups, layout_table = create_label_group_manager(
                    "layout",
                    default_parent="layout_blocks",
                    default_labels="document_header, account_holder, transactions_table",
                    initial_groups=first["groups"],
                )
                with gr.Row():
                    layout_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    layout_flat = gr.Checkbox(value=True, label="Flat NER")
                    layout_multi = gr.Checkbox(value=False, label="Multi-label")
                    layout_boxes = gr.Checkbox(value=True, label="Use bounding boxes")
                layout_btn = gr.Button("Run Layout NER", variant="primary")
                layout_picker, layout_score_all = create_example_picker(
                    LAYOUT_EXAMPLES, "Score all 10 layout examples"
                )
                layout_gold = create_gold_view(first)
            with gr.Column(scale=1):
                layout_preview = gr.HTML(
                    value=layout_preview_html(
                        first_words, first_boxes, first_pages, [], first["gold"]["ner"]
                    ),
                    label="Page preview",
                )
                layout_output = gr.Code(label="Results", language="json")
                layout_metrics = gr.Markdown()

        layout_gold_state = gr.State(
            value=gold_payload(first["gold"], first_document, json.dumps(first["groups"]))
        )

        layout_btn.click(
            fn=run_layout_ner,
            inputs=[
                layout_document,
                layout_groups,
                layout_threshold,
                layout_flat,
                layout_multi,
                layout_boxes,
                layout_gold_state,
            ],
            outputs=[layout_output, layout_metrics, layout_preview],
        )
        layout_picker.click(
            fn=load_layout_example,
            inputs=[layout_picker],
            outputs=[
                layout_document,
                layout_groups,
                layout_table,
                layout_gold_state,
                layout_gold,
                layout_preview,
                layout_output,
                layout_metrics,
            ],
        )
        layout_score_all.click(
            fn=score_all_layout_ner,
            inputs=[layout_threshold, layout_flat, layout_multi, layout_boxes],
            outputs=layout_metrics,
        )
        layout_pdf_btn.click(
            fn=extract_layout_from_pdf,
            inputs=[layout_pdf, layout_pdf_pages, layout_document],
            outputs=[layout_document, layout_preview, layout_pdf_status],
        )

    # ── Classification Tab ───────────────────────────────────────────────
    with gr.Tab("Classification"):
        first = CLASSIFICATION_EXAMPLES[0]
        with gr.Row():
            with gr.Column(scale=1):
                cls_text = gr.Textbox(label="Text", lines=5, value=first["text"])
                cls_groups, cls_table = create_label_group_manager(
                    "classification",
                    default_parent="topic",
                    default_labels="technology, finance, politics, sports",
                    initial_groups=first["groups"],
                )
                with gr.Row():
                    cls_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    cls_multi = gr.Checkbox(value=first["multi_label"], label="Multi-label")
                cls_btn = gr.Button("Run Classification", variant="primary")
                cls_picker, cls_score_all = create_example_picker(
                    CLASSIFICATION_EXAMPLES, "Score all 10 classification examples"
                )
                cls_gold = create_gold_view(first)
            with gr.Column(scale=1):
                cls_output = gr.Code(label="Results", language="json")
                cls_metrics = gr.Markdown()

        cls_gold_state = gr.State(
            value=gold_payload(first["gold"], first["text"], json.dumps(first["groups"]))
        )

        def load_classification_example(index):
            example = CLASSIFICATION_EXAMPLES[int(index)]
            return (*load_example(CLASSIFICATION_EXAMPLES, index), example["multi_label"])

        cls_btn.click(
            fn=run_classification,
            inputs=[cls_text, cls_groups, cls_threshold, cls_multi, cls_gold_state],
            outputs=[cls_output, cls_metrics],
        )
        cls_picker.click(
            fn=load_classification_example,
            inputs=[cls_picker],
            outputs=[cls_text, cls_groups, cls_table, cls_gold_state, cls_gold, cls_multi],
        )
        cls_score_all.click(
            fn=score_all_classification,
            inputs=[cls_threshold],
            outputs=cls_metrics,
        )

    # ── Joint Relex Tab ──────────────────────────────────────────────────
    with gr.Tab("Joint NER + Relations"):
        first = JOINT_RELEX_EXAMPLES[0]
        with gr.Row():
            with gr.Column(scale=1):
                joint_text = gr.Textbox(label="Text", lines=5, value=first["text"])
                joint_groups, joint_table = create_label_group_manager(
                    "joint_relex",
                    default_parent="general",
                    extra_fields=True,
                    initial_groups=first["groups"],
                )
                with gr.Row():
                    joint_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    joint_flat = gr.Checkbox(value=True, label="Flat NER")
                joint_btn = gr.Button("Run Joint Extraction", variant="primary")
                joint_picker, joint_score_all = create_example_picker(
                    JOINT_RELEX_EXAMPLES, "Score all 10 joint examples"
                )
                joint_gold = create_gold_view(first)
            with gr.Column(scale=1):
                joint_output = gr.Code(label="Results", language="json")
                joint_metrics = gr.Markdown()

        joint_gold_state = gr.State(
            value=gold_payload(first["gold"], first["text"], json.dumps(first["groups"]))
        )

        joint_btn.click(
            fn=run_joint_relex,
            inputs=[joint_text, joint_groups, joint_threshold, joint_flat, joint_gold_state],
            outputs=[joint_output, joint_metrics],
        )
        joint_picker.click(
            fn=lambda index: load_example(JOINT_RELEX_EXAMPLES, index, extra_fields=True),
            inputs=[joint_picker],
            outputs=[joint_text, joint_groups, joint_table, joint_gold_state, joint_gold],
        )
        joint_score_all.click(
            fn=score_all_joint_relex,
            inputs=[joint_threshold, joint_flat],
            outputs=joint_metrics,
        )

    # ── Structuring Tab ──────────────────────────────────────────────────
    with gr.Tab("Structuring"):
        first = STRUCTURING_EXAMPLES[0]
        with gr.Row():
            with gr.Column(scale=1):
                struct_text = gr.Textbox(label="Text", lines=5, value=first["text"])
                gr.Markdown("Use `field:type` in labels for typed output, for example `name:str, founded_companies:list[str], employee_count:int`.")
                struct_groups, struct_table = create_label_group_manager(
                    "structuring",
                    default_parent="company",
                    default_labels="name:str, CEO:str, location:str, investment_amount:float",
                    initial_groups=first["groups"],
                )
                with gr.Row():
                    struct_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    struct_flat = gr.Checkbox(value=True, label="Flat NER")
                struct_btn = gr.Button("Run Structuring", variant="primary")
                struct_picker, struct_score_all = create_example_picker(
                    STRUCTURING_EXAMPLES, "Score all 10 structuring examples"
                )
                struct_gold = create_gold_view(first)
            with gr.Column(scale=1):
                struct_output = gr.Code(label="Results", language="json")
                struct_metrics = gr.Markdown()

        struct_gold_state = gr.State(
            value=gold_payload(first["gold"], first["text"], json.dumps(first["groups"]))
        )

        struct_btn.click(
            fn=run_structuring,
            inputs=[struct_text, struct_groups, struct_threshold, struct_flat, struct_gold_state],
            outputs=[struct_output, struct_metrics],
        )
        struct_picker.click(
            fn=lambda index: load_example(STRUCTURING_EXAMPLES, index),
            inputs=[struct_picker],
            outputs=[struct_text, struct_groups, struct_table, struct_gold_state, struct_gold],
        )
        struct_score_all.click(
            fn=score_all_structuring,
            inputs=[struct_threshold, struct_flat],
            outputs=struct_metrics,
        )

    # ── Multi-level Structuring Tab ──────────────────────────────────────
    with gr.Tab("Multi-level Structuring"):
        first = MULTI_LEVEL_STRUCTURING_EXAMPLES[0]
        with gr.Row():
            with gr.Column(scale=1):
                ml_text = gr.Textbox(label="Text", lines=8, value=first["text"])
                gr.Markdown(
                    "Nested schema, edited as JSON: a scalar field maps to its type "
                    "(`\"age\": \"int\"`), an object to a template (`\"seller\": {...}`), and a "
                    "repeated child to a one-element list holding the child template "
                    "(`\"items\": [{...}]`). Children are extracted under the parent record "
                    "they belong to."
                )
                ml_schema = gr.Code(
                    value=json.dumps(first["schema"], indent=2),
                    label="Schema (JSON)",
                    language="json",
                    lines=18,
                )
                with gr.Row():
                    ml_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
                    ml_flat = gr.Checkbox(value=True, label="Flat NER")
                ml_btn = gr.Button("Run Multi-level Structuring", variant="primary")
                ml_picker, ml_score_all = create_example_picker(
                    MULTI_LEVEL_STRUCTURING_EXAMPLES, "Score all 10 multi-level examples"
                )
                ml_gold = create_gold_view(first)
            with gr.Column(scale=1):
                ml_output = gr.Code(label="Results", language="json")
                ml_metrics = gr.Markdown()

        ml_gold_state = gr.State(
            value=gold_payload(
                first["gold"], first["text"], json.dumps(first["schema"], indent=2)
            )
        )

        def load_multi_level_example(index):
            """Load a multi-level example into (text, schema JSON, gold state, gold view)."""
            example = MULTI_LEVEL_STRUCTURING_EXAMPLES[int(index)]
            serialized = json.dumps(example["schema"], indent=2)
            return (
                example["text"],
                serialized,
                gold_payload(example["gold"], example["text"], serialized),
                json.dumps(example["gold"], indent=2, default=str),
            )

        ml_btn.click(
            fn=run_multi_level_structuring,
            inputs=[ml_text, ml_schema, ml_threshold, ml_flat, ml_gold_state],
            outputs=[ml_output, ml_metrics],
        )
        ml_picker.click(
            fn=load_multi_level_example,
            inputs=[ml_picker],
            outputs=[ml_text, ml_schema, ml_gold_state, ml_gold],
        )
        ml_score_all.click(
            fn=score_all_multi_level_structuring,
            inputs=[ml_threshold, ml_flat],
            outputs=ml_metrics,
        )

    # ── Embedding Tab ────────────────────────────────────────────────────
    with gr.Tab("Embedding"):
        first = EMBEDDING_EXAMPLES[0]
        first_candidates = "\n".join(first["candidates"])
        with gr.Row():
            with gr.Column(scale=1):
                emb_query = gr.Textbox(label="Query text", lines=3, value=first["query"])
                emb_candidates = gr.Textbox(
                    label="Candidate texts (one per line)",
                    lines=8,
                    value=first_candidates,
                )
                emb_btn = gr.Button("Compute Similarity", variant="primary")
                emb_picker, emb_score_all = create_example_picker(
                    EMBEDDING_EXAMPLES, "Score all 10 retrieval queries"
                )
                emb_gold = create_gold_view(first)
            with gr.Column(scale=1):
                emb_output = gr.Code(label="Results (sorted by similarity)", language="json")
                emb_metrics = gr.Markdown()

        emb_gold_state = gr.State(
            value=gold_payload(first["gold"], first["query"], first_candidates)
        )

        def load_embedding_example(index):
            example = EMBEDDING_EXAMPLES[int(index)]
            candidates = "\n".join(example["candidates"])
            return (
                example["query"],
                candidates,
                gold_payload(example["gold"], example["query"], candidates),
                json.dumps(example["gold"], indent=2, default=str),
            )

        emb_btn.click(
            fn=run_embedding,
            inputs=[emb_query, emb_candidates, emb_gold_state],
            outputs=[emb_output, emb_metrics],
        )
        emb_picker.click(
            fn=load_embedding_example,
            inputs=[emb_picker],
            outputs=[emb_query, emb_candidates, emb_gold_state, emb_gold],
        )
        emb_score_all.click(fn=score_all_embedding, inputs=[], outputs=emb_metrics)

    # ── Multi-task Tab ───────────────────────────────────────────────────
    with gr.Tab("Multi-task"):
        first = MULTITASK_EXAMPLES[0]
        mt_text = gr.Textbox(label="Text", lines=5, value=first["text"])
        with gr.Row():
            mt_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
            mt_flat = gr.Checkbox(value=True, label="Flat NER")
            mt_multi = gr.Checkbox(value=False, label="Multi-label")

        gr.Markdown("Configure any combination of tasks below:")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### NER")
                mt_ner, mt_ner_table = create_label_group_manager(
                    "mt_ner", default_parent="general",
                    default_labels="person, organization, location",
                    initial_groups=first["ner"],
                )
            with gr.Column():
                gr.Markdown("#### Classification")
                mt_cls, mt_cls_table = create_label_group_manager(
                    "mt_cls", default_parent="topic",
                    default_labels="technology, finance",
                    initial_groups=first["classification"],
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Joint NER + Relations")
                mt_joint, mt_joint_table = create_label_group_manager(
                    "mt_joint", default_parent="general",
                    extra_fields=True,
                )
            with gr.Column():
                gr.Markdown("#### Structuring")
                mt_struct, mt_struct_table = create_label_group_manager(
                    "mt_struct", default_parent="company",
                    default_labels="name:str, CEO:str, location:str",
                    initial_groups=first["structuring"],
                )

        with gr.Row():
            with gr.Column():
                mt_picker, mt_score_all = create_example_picker(
                    MULTITASK_EXAMPLES, "Score all 10 multi-task examples"
                )
                mt_gold = create_gold_view(first)

        mt_btn = gr.Button("Run Multi-task", variant="primary")
        mt_output = gr.Code(label="Results", language="json")
        mt_metrics = gr.Markdown()

        mt_gold_state = gr.State(
            value=gold_payload(
                first["gold"],
                first["text"],
                json.dumps(first["ner"]),
                json.dumps(first["classification"]),
                json.dumps(first["structuring"]),
            )
        )

        def load_multitask_example(index):
            """Load a multi-task example; the joint relation group is cleared."""
            example = MULTITASK_EXAMPLES[int(index)]
            ner_groups = example["ner"]
            cls_groups = example["classification"]
            struct_groups = example["structuring"]
            return (
                example["text"],
                json.dumps(ner_groups), _groups_to_table(ner_groups),
                json.dumps(cls_groups), _groups_to_table(cls_groups),
                "[]", _groups_to_table([], extra_fields=True),
                json.dumps(struct_groups), _groups_to_table(struct_groups),
                gold_payload(
                    example["gold"],
                    example["text"],
                    json.dumps(ner_groups),
                    json.dumps(cls_groups),
                    json.dumps(struct_groups),
                ),
                json.dumps(example["gold"], indent=2, default=str),
            )

        mt_btn.click(
            fn=run_multitask,
            inputs=[mt_text, mt_ner, mt_cls, mt_joint, mt_struct,
                    mt_threshold, mt_flat, mt_multi, mt_gold_state],
            outputs=[mt_output, mt_metrics],
        )
        mt_picker.click(
            fn=load_multitask_example,
            inputs=[mt_picker],
            outputs=[mt_text,
                     mt_ner, mt_ner_table,
                     mt_cls, mt_cls_table,
                     mt_joint, mt_joint_table,
                     mt_struct, mt_struct_table,
                     mt_gold_state, mt_gold],
        )
        mt_score_all.click(
            fn=score_all_multitask,
            inputs=[mt_threshold, mt_flat, mt_multi],
            outputs=mt_metrics,
        )

demo.queue()

if __name__ == "__main__":
    demo.launch(debug=True)
