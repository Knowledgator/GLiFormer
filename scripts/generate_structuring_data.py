"""Three-stage vLLM data generator for the GLiNExT structuring task.

Stage 1 — Text generation:
    Given (topic, subtopic, count_bucket, length_bucket), the model writes a
    natural-language passage containing the requested number of object
    instances. Stylistic variety is sampled (narrative, listing, log, report).

Stage 2 — Schema generation:
    Given the generated passage, the model produces a flat field schema in the
    structuring DSL used by GLiNExT:

        [schema_name]
        field_a::str
        field_b::int
        field_c::[option1|option2]::str

Stage 3 — Structured-extraction generation:
    Given the passage + the schema, the model emits a JSON array of objects,
    one per object instance found in the passage. Values are kept as strings
    (or string lists for `list`-typed fields), matching the shape used in
    `data/extraction_multi.json`.

The script samples uniformly across the cartesian product of
(topic × subtopic × length-bucket × count-bucket) so the final dataset is
diverse along each axis. Object counts vary from 2 to 30+.

Output:
    JSONL, one record per line, matching extraction_multi.json:
        {
            "tokenized_text": List[str],
            "text": str,
            "structuring": {
                "<schema_name>": [
                    {"<field>": "<value>" | [<value>, ...], ...},
                    ...
                ]
            },
            "schema_dsl": str,           # original DSL for reference
            "topic": str, "subtopic": str,
            "length_bucket": str, "count_bucket": str,
            "n_objects_target": int, "n_objects_extracted": int,
            "score": 0.4
        }

Example:
    python scripts/generate_structuring_data.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --num-samples 5000 \\
        --output data/structuring_synthetic.jsonl \\
        --tensor-parallel-size 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _log(msg: str) -> None:
    """Stderr log with timestamp + immediate flush so progress is visible."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------- #
# Topic ontology                                                        #
# --------------------------------------------------------------------- #

TOPICS: dict[str, list[str]] = {
    "email_correspondence": [
        "customer-support email threads",
        "internal team status emails",
        "sales outreach emails",
        "vendor invoice notification emails",
        "newsletter subscription confirmations",
        "phishing-report security emails",
        "recruiter outreach messages",
        "academic conference correspondence",
    ],
    "social_media": [
        "tech-influencer tweets",
        "linkedin products overview",
        "linkedin GitHub project reviews",
        "instagram product-launch posts",
        "linkedin job-announcement posts",
        "tiktok trend reposts with engagement metrics",
        "reddit thread top comments",
        "youtube video descriptions",
        "facebook event posts",
    ],
    "medical": [
        "lab-result reports",
        "prescription log entries",
        "patient encounter notes",
        "radiology report summaries",
        "vaccination records",
        "vital-signs monitoring logs",
        "discharge summaries",
        "clinical-trial enrollment lists",
        "drugs descriptions",
        "symptom checklists",
        "medical device inventory records",
        "gene regulator networks descriptions",
        "biological pathway descriptions",
        "mechanism of action descriptions",
        "disease symptom descriptions",
        "averse drug reaction reports",
        "patient medical history summaries",
        "chemical safety data sheets",
        "clinical guidelines summaries",
        "health insurance claim summaries",
        "medical research paper abstracts",
    ],
    "insurance": [
        "auto-insurance claim summaries",
        "home-insurance claim summaries",
        "health-insurance claim approvals",
        "life-insurance policy listings",
        "policy renewal notices",
        "underwriting risk reports",
    ],
    "ecommerce_products": [
        "online store best-sellers list",
        "product catalog page descriptions",
        "supply-chain SKU inventory snapshots",
        "marketplace returns log",
        "promotional campaign product lists",
        "products characteristic comparison",
        "products comparison tables with attributes and prices",
    ],
    "companies": [
        "startup investor briefings",
        "company employee directories",
        "competitor analysis briefs",
        "company funding-round announcements",
        "merger and acquisition press notes",
        "company office locations directory",
        "company products comparison",
        "company founders experiences",
    ],
    "reviews": [
        "restaurant reviews on a food blog",
        "movie reviews from a critic round-up",
        "ecommerce product reviews",
        "hotel reviews on a travel site",
        "app-store reviews summary",
        "book reviews in a literary magazine",
        "tech product reviews"
    ],
    "support_tickets": [
        "IT helpdesk ticket queue",
        "SaaS customer support backlog",
        "telco network outage tickets",
        "hardware RMA tickets",
        "dev-ops incident tickets",
    ],
    "legal_documents": [
        "court case docket summaries",
        "contract clause annotations",
        "patent filing abstracts",
        "trademark dispute notices",
        "compliance audit findings",
        "regulatory filings",
    ],
    "news": [
        "breaking-news brief roundups",
        "international affairs daily digests",
        "business news headlines",
        "science & technology news",
        "weather and natural-disaster reports",
        "sport news summaries",
    ],
    "politics": [
        "election results recap",
        "legislative votes summary",
        "political campaign finance disclosures",
        "government appointment announcements",
        "press-conference statement listings",
        "international treaty signings",
    ],
    "sports": [
        "soccer match-day results",
        "NBA box-score summaries",
        "tennis tournament round-up",
        "olympic medal events",
        "esports tournament brackets",
        "athlete transfer news",
    ],
    "finance": [
        "stock-market daily movers",
        "earnings-call result summaries",
        "crypto trading transactions log",
        "personal banking transaction list",
        "venture-capital round announcements",
        "credit-card statement line items",
        "currency-exchange rate snapshots",
        "companies financials comparison tables",
    ],
    "real_estate": [
        "property listings on a real-estate site",
        "rental availability board",
        "commercial leasing offers",
    ],
    "human_resources": [
        "new-hire onboarding rosters",
        "employee performance review summaries",
        "open job-requisition listings",
        "training program enrollments",
    ],
    "education": [
        "course catalog entries",
        "research-paper submission listings",
        "scholarship award announcements",
        "exam result publications",
    ],
    "events_logistics": [
        "tech conference schedules",
        "music festival lineups",
        "trade-show exhibitor lists",
        "shipping consignment manifests",
        "flight delay listings",
    ],
}


# --------------------------------------------------------------------- #
# Length and count buckets                                              #
# --------------------------------------------------------------------- #

LENGTH_BUCKETS: list[Tuple[str, str, int]] = [
    # (name, prose hint, max_new_tokens budget)
    ("short", "around 70-150 words, one tight paragraph", 360),
    ("medium", "around 200-400 words, 1-2 paragraphs", 700),
    ("long", "around 500-900 words with 2-4 paragraphs", 1400),
    ("very_long", "around 1000-1800 words, 4-7 paragraphs of mixed prose and listing", 2500),
]

COUNT_BUCKETS: list[Tuple[str, int, int]] = [
    # (name, min, max)
    ("small", 2, 5),
    ("medium", 6, 15),
    ("large", 16, 32),
]

STYLE_HINTS: list[str] = [
    "a flowing narrative paragraph",
    "a realistic dialogue between people",
    "a numbered list with mixed-length entries",
    "a bullet-point listing",
    "a semi-structured report with section headings",
    "a chat or email thread style",
    "a tabular dump rendered as plain text",
    "a news-brief round-up style",
    "a casual social-media stream",
    "a markdown table",
    "a json or log-like format with repeated keys",
    "a html-like format with angle-bracket tags",
    "a mix of prose and listing elements",
    "a structured data format with key-value pairs"
    "a YAML-like format with indentation and dashes",
    "a CSV-style format with rows of comma-separated values",
    "a report with labeled sections and subheadings"
]


# --------------------------------------------------------------------- #
# Prompt templates                                                      #
# --------------------------------------------------------------------- #

TEXT_SYSTEM_PROMPT = (
    "You generate realistic, diverse synthetic passages for an information-"
    "extraction dataset. Any private reasoning must come BEFORE a marker "
    "line; only the prose that follows the marker is kept."
)

TEXT_USER_TEMPLATE = """Domain: {topic}
Subtopic: {subtopic}
Object class to mention: {subtopic}
Number of distinct object instances to include: exactly {n}
Length: {length_desc}
Style: {style}

Write one realistic passage. Each of the {n} object instances must be \
clearly identifiable and carry at least 5 attributes (e.g. names, IDs, \
dates, amounts, statuses, descriptions). Make instance values realistic \
and varied (different people, dates, numbers, statuses) — do not repeat \
identical attribute values across instances. Use plausible names, \
addresses, codes. Plain English text only, no unicode artefacts.

Output format (strict):
1. Any planning, outline, or reasoning may appear FIRST.
2. Then a single line containing exactly: ===PASSAGE===
3. After that line, output ONLY the realistic passage prose. \
No headings, no bullet labels, no instance numbering, no commentary, \
no repetition of these instructions. Just natural narrative or \
structured-prose text that reads like a real document of the requested \
style. Whatever follows the marker is what gets saved verbatim."""


SCHEMA_SYSTEM_PROMPT = (
    "You design flat extraction schemas in a strict DSL. Any reasoning must "
    "come BEFORE a marker line; only what follows the marker is kept."
)

SCHEMA_USER_TEMPLATE = """Read the passage and produce a flat schema for the \
repeated object class it describes. Every line must end with a short \
description after a single `#` separator.

DSL rules:
- First line: [<schema_name>] # <one-sentence description of the object class>
- One line per field: <field_name>::<type> # <short description of the field>
- Allowed primitive types: str, int, float, bool, list, date, datetime
- For enum-like categorical fields: <field_name>::[option_a|option_b|option_c]::str # <description>
- Output 3 to 15 fields.
- Field names and the schema_name must be snake_case.
- Choose fields actually grounded in the passage's facts \
(do not invent fields with no values mentioned).
- Descriptions must be 3-15 words, plain ASCII, no `#` characters, no quotes, no newlines.
- Do NOT include the schema_name line as a field.

Example format:
[invoice] # Invoice records issued to clients with line-item totals
invoice_number::str # Unique invoice identifier as printed on the document
date::date # Date the invoice was issued
status::[paid|pending|overdue]::str # Current payment status
line_items::list # List of services or products billed
total::str # Final invoiced amount including tax

Passage:
\"\"\"
{text}
\"\"\"

Output format (strict):
1. Any planning may appear FIRST.
2. Then a single line containing exactly: ===SCHEMA===
3. After the marker, output ONLY the annotated DSL schema, starting with \
the [<schema_name>] line. Nothing else, no commentary, no fences."""


EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured records from a passage and emit a JSON array. "
    "Any reasoning must come BEFORE a marker line; only what follows the "
    "marker is parsed."
)

EXTRACTION_USER_TEMPLATE = """Extract structured records from the passage \
according to this schema.

Schema:
{schema_dsl}

Output rules:
- Output a JSON array (`[ ... ]`) containing one object per distinct instance \
mentioned in the passage.
- Each object's keys must be exactly the schema field names (snake_case).
- For `list` fields → output a JSON array of strings.
- For all other types (`str`, `int`, `float`, `bool`, `date`, `datetime`) → \
output a single string copying the value as it appears in the text.
- For enum-style fields shown as `field::[a|b|c]::str` → output exactly one \
of the listed options (lowercase, as written) as a string.
- If a field is not mentioned for a given instance, omit that key.
- Skip an instance entirely if fewer than two fields are grounded in the text.
- Keep the order of instances the same as their order in the passage.
- Output ONLY the JSON array.

Passage:
\"\"\"
{text}
\"\"\"

Output format (strict):
1. Any planning may appear FIRST.
2. Then a single line containing exactly: ===JSON===
3. After the marker, output ONLY the JSON array. No commentary, no \
markdown fences."""


# --------------------------------------------------------------------- #
# Schema parsing / validation                                           #
# --------------------------------------------------------------------- #

# Lenient regexes — we accept many model-side variations and post-normalize
# (lowercase names, alias types, etc.) rather than rejecting outright.
_SCHEMA_NAME_RE = re.compile(
    r"^\[\s*([A-Za-z][A-Za-z0-9_\- ]*)\s*\]"   # [Schema Name] — any case, hyphens, spaces
    r"(?:\s*[#:\-]+\s*(.+?))?"                 # description sep: #, :, -, --
    r"\s*[\.,;]?\s*$"
)
_FIELD_LINE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_\- ]*)\s*::?\s*"    # field_name with `::` or `:`
    r"(?:\[([^\]]+)\]\s*::?\s*)?"              # optional [a|b|c] enum
    r"([A-Za-z][A-Za-z0-9_]*)"                 # type word (alias-resolved later)
    r"(?:\s*[#:\-]+\s*(.+?))?"                 # optional description after #/:/-
    r"\s*[\.,;]?\s*$"
)

_TYPE_ALIASES: dict[str, str] = {
    "str": "str", "string": "str", "text": "str", "varchar": "str",
    "int": "int", "integer": "int", "long": "int",
    "float": "float", "double": "float", "decimal": "float", "number": "float",
    "bool": "bool", "boolean": "bool",
    "list": "list", "array": "list", "arr": "list",
    "date": "date",
    "datetime": "datetime", "timestamp": "datetime",
}

# Strip leading list/markdown markers so `- field::str` and `1. field::str` parse.
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-\*\+•·]+|\d+[\.\)])\s*")
# Drop **bold**, __bold__, `code`, _italic_ around field/schema names.
_MARKDOWN_INLINE_RE = re.compile(r"(\*\*|__|`|_)(.+?)\1")


_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", flags=re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_think_blocks(raw: str) -> str:
    """Drop ``<think>…</think>`` reasoning so the answer parsers see clean text.

    Handles three cases produced by reasoning-tuned chat models:
      - balanced ``<think>…</think>`` pairs (most common) — removed
      - orphan ``</think>`` (model emitted answer-only without an opener) —
        keep only the suffix after the last ``</think>``
      - orphan ``<think>`` (model truncated mid-reasoning) — keep only the
        prefix before the first ``<think>``; that's usually empty, which is
        still better than feeding raw reasoning to the next parser
    """
    if not raw:
        return ""
    text = _THINK_BLOCK_RE.sub("", raw)
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opens = list(_THINK_OPEN_RE.finditer(text))
    if opens:
        text = text[: opens[0].start()]
    return text.strip()


def _strip_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return cleaned


# Markers used in our prompts to separate (optional) reasoning from the
# final answer. Strip everything up to and including the LAST occurrence.
PASSAGE_MARKER = "===PASSAGE==="
SCHEMA_MARKER = "===SCHEMA==="
JSON_MARKER = "===JSON==="

# Loose fallback for when the model forgets the explicit marker but uses a
# natural-language hand-off line ("Final passage:", "Here's the passage:",
# "Let's write the passage:", "Passage:" on its own line).
_PASSAGE_HEURISTIC_RE = re.compile(
    r"\n\s*(?:"
    r"final\s+passage"
    r"|here(?:\s+is|\s*'s)?\s+(?:the\s+)?(?:final\s+)?passage"
    r"|let(?:'?s)?\s+write\s+(?:the\s+)?passage"
    r"|the\s+passage"
    r"|passage"
    r")\s*[:\-]?\s*\n",
    flags=re.IGNORECASE,
)


def _take_after_marker(text: str, marker: str) -> str:
    """Return content after the LAST occurrence of `marker` (rstripped)."""
    if not text or not marker:
        return text or ""
    idx = text.rfind(marker)
    if idx < 0:
        return text
    return text[idx + len(marker):].lstrip("\r\n").lstrip()


def extract_passage(raw: str) -> str:
    """Pull the actual passage out of a thinking-model output.

    Pipeline: strip ``<think>…</think>`` blocks → take content after the last
    ``===PASSAGE===`` marker → if no marker found, fall back to the last
    natural-language hand-off ("Final passage:", "Let's write the passage:").
    Returns the cleaned passage, or the raw text as a last resort.
    """
    text = _strip_think_blocks(raw)
    after_marker = _take_after_marker(text, PASSAGE_MARKER)
    if after_marker is not text:  # marker was present
        return after_marker.strip()
    matches = list(_PASSAGE_HEURISTIC_RE.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    return text.strip()


def _normalize_line(line: str) -> str:
    """Strip list markers and inline markdown so the regex sees a bare line."""
    s = line.strip()
    # Inline markdown first — otherwise leading `**` of `**name**` gets eaten
    # by the list-marker stripper.
    s = _MARKDOWN_INLINE_RE.sub(lambda m: m.group(2), s)
    s = _LINE_PREFIX_RE.sub("", s)
    return s.strip()


def _normalize_identifier(raw: str) -> str:
    """Lowercase, replace whitespace/hyphens with `_`, drop other non-ascii."""
    s = raw.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _clean_description(raw: Optional[str]) -> str:
    if not raw:
        return ""
    desc = raw.strip().rstrip(".").strip()
    desc = re.sub(r"\s+", " ", desc)
    return desc


# Field metadata: (type, enum_opts, description)
FieldSpec = Tuple[str, List[str], str]


def parse_schema_fields(
    dsl: str,
) -> Optional[Tuple[str, str, "dict[str, FieldSpec]"]]:
    """Parse the annotated DSL leniently.

    Returns (schema_name, schema_description, {field_name: (type, opts, desc)})
    or None if too few well-formed lines remain. Type aliases (e.g. ``string``
    → ``str``, ``boolean`` → ``bool``) and surface variations (markdown, list
    markers, single ``:`` separators, capitalised names) are normalised.
    """
    if not dsl:
        return None
    after_marker = _take_after_marker(_strip_think_blocks(dsl), SCHEMA_MARKER)
    cleaned = _strip_fences(after_marker)
    raw_lines = [_normalize_line(ln) for ln in cleaned.splitlines()]
    raw_lines = [ln for ln in raw_lines if ln]
    if not raw_lines:
        return None

    # Find the first line that looks like a schema header.
    head_idx = -1
    head_match = None
    for i, ln in enumerate(raw_lines):
        m = _SCHEMA_NAME_RE.match(ln)
        if m:
            head_idx = i
            head_match = m
            break
    if head_match is None:
        return None

    schema_name = _normalize_identifier(head_match.group(1))
    if not schema_name or not schema_name[0].isalpha():
        return None
    schema_description = _clean_description(head_match.group(2))

    fields: dict[str, FieldSpec] = {}
    for line in raw_lines[head_idx + 1:]:
        m = _FIELD_LINE_RE.match(line)
        if not m:
            # Skip the line — don't reject the whole schema.
            continue
        raw_name, options, raw_type, description = m.groups()
        name = _normalize_identifier(raw_name)
        if not name or not name[0].isalpha():
            continue
        type_ = _TYPE_ALIASES.get(raw_type.lower())
        if type_ is None:
            continue
        if name in fields:
            continue
        opts = (
            [o.strip() for o in options.split("|") if o.strip()]
            if options
            else []
        )
        fields[name] = (type_, opts, _clean_description(description))

    if not (3 <= len(fields) <= 20):
        return None
    return schema_name, schema_description, fields


def render_schema_dsl(
    schema_name: str,
    fields: "dict[str, FieldSpec]",
    schema_description: str = "",
    *,
    include_descriptions: bool = True,
) -> str:
    head = f"[{schema_name}]"
    if include_descriptions and schema_description:
        head = f"{head} # {schema_description}"
    lines = [head]
    for name, (type_, opts, desc) in fields.items():
        if opts:
            line = f"{name}::[{'|'.join(opts)}]::{type_}"
        else:
            line = f"{name}::{type_}"
        if include_descriptions and desc:
            line = f"{line} # {desc}"
        lines.append(line)
    return "\n".join(lines)


def parse_schema_dsl(raw: str) -> Optional[str]:
    """Backward-compatible: validate + canonicalize a DSL string."""
    parsed = parse_schema_fields(raw)
    if parsed is None:
        return None
    schema_name, schema_description, fields = parsed
    return render_schema_dsl(schema_name, fields, schema_description)


# --------------------------------------------------------------------- #
# Extraction parsing / validation                                       #
# --------------------------------------------------------------------- #

def _extract_json_array(raw: str) -> Optional[list]:
    """Locate the first `[...]` JSON array in the model output."""
    if not raw:
        return None
    after_marker = _take_after_marker(_strip_think_blocks(raw), JSON_MARKER)
    cleaned = _strip_fences(after_marker)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _coerce_value(val: Any, prefer_list: Optional[bool]) -> Optional[Any]:
    """Coerce one JSON value to a str (or List[str]).

    ``prefer_list=True`` forces list-of-strings shape; ``False`` forces
    scalar string; ``None`` infers from the value's runtime type.
    Returns ``None`` if the value cannot be safely coerced (e.g. nested dict
    in a scalar slot, empty after stripping).
    """
    if val is None:
        return None
    if isinstance(val, dict):
        return None
    if isinstance(val, bool):
        s = "true" if val else "false"
        return [s] if prefer_list else s
    if isinstance(val, list):
        flat: list[str] = []
        for x in val:
            if isinstance(x, (dict, list)) or x is None:
                continue
            if isinstance(x, bool):
                flat.append("true" if x else "false")
            else:
                s = str(x).strip()
                if s:
                    flat.append(s)
        if not flat:
            return None
        if prefer_list is False:
            return ", ".join(flat)
        return flat
    s = str(val).strip()
    if not s:
        return None
    return [s] if prefer_list else s


def validate_and_clean_extraction(
    items: list,
    fields: "dict[str, FieldSpec]",
    min_instances: int = 2,
    min_fields_per_item: int = 2,
) -> Optional[List[dict]]:
    """Coerce extraction output to ``{field: str | List[str]}`` format.

    The real correctness gate: stage 3 must return a list of dicts whose
    values are scalars or string lists. The schema is advisory — when
    ``fields`` is empty (parser fell back), per-key types are inferred from
    the runtime value type. Items keeping at least ``min_fields_per_item``
    fields survive; the sample is rejected when fewer than ``min_instances``
    items survive.
    """
    if not isinstance(items, list):
        return None

    cleaned_all: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cleaned: dict[str, Any] = {}
        for raw_key, raw_val in item.items():
            key = _normalize_identifier(str(raw_key)) if raw_key is not None else ""
            if not key:
                continue
            spec = fields.get(key) if fields else None
            prefer_list: Optional[bool]
            if spec is None:
                prefer_list = None  # infer
            else:
                prefer_list = (spec[0] == "list")
            coerced = _coerce_value(raw_val, prefer_list)
            if coerced is None:
                continue
            if key in cleaned:
                continue
            cleaned[key] = coerced
        if len(cleaned) >= min_fields_per_item:
            cleaned_all.append(cleaned)

    if len(cleaned_all) < min_instances:
        return None
    return cleaned_all


# --------------------------------------------------------------------- #
# Tokenization                                                          #
# --------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer matching extraction_multi.json."""
    return _TOKEN_RE.findall(text)


# --------------------------------------------------------------------- #
# Sample plan                                                           #
# --------------------------------------------------------------------- #

@dataclass
class SamplePlan:
    topic: str
    subtopic: str
    length_bucket: str
    length_desc: str
    max_text_tokens: int
    count_bucket: str
    n_objects: int
    style: str


def make_plan(num_samples: int, rng: random.Random) -> List[SamplePlan]:
    """Balanced sampling across topic × subtopic × length × count cells."""
    cells: list[tuple[str, str, tuple[str, str, int], tuple[str, int, int]]] = []
    for topic, subtopics in TOPICS.items():
        for subtopic in subtopics:
            for length in LENGTH_BUCKETS:
                for count in COUNT_BUCKETS:
                    cells.append((topic, subtopic, length, count))

    rng.shuffle(cells)
    plans: list[SamplePlan] = []
    while len(plans) < num_samples:
        for (topic, subtopic, length, count) in cells:
            if len(plans) >= num_samples:
                break
            n = rng.randint(count[1], count[2])
            plans.append(
                SamplePlan(
                    topic=topic,
                    subtopic=subtopic,
                    length_bucket=length[0],
                    length_desc=length[1],
                    max_text_tokens=length[2],
                    count_bucket=count[0],
                    n_objects=n,
                    style=rng.choice(STYLE_HINTS),
                )
            )
        rng.shuffle(cells)
    return plans


# --------------------------------------------------------------------- #
# vLLM driver                                                           #
# --------------------------------------------------------------------- #

def build_chat(tokenizer, system: str, user: str) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="HF model id or local path (instruct/chat tuned).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination JSONL file.")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of prompts per vLLM generate() call.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", type=float, default=0.4,
                        help="Sample weight written to each record.")
    parser.add_argument("--text-temperature", type=float, default=0.95)
    parser.add_argument("--schema-temperature", type=float, default=0.4)
    parser.add_argument("--extraction-temperature", type=float, default=0.2)
    parser.add_argument("--text-top-p", type=float, default=0.95)
    parser.add_argument("--schema-top-p", type=float, default=0.9)
    parser.add_argument("--extraction-top-p", type=float, default=0.9)
    # Defaults sized for thinking-tuned models (Qwen3 *-Thinking-2507 etc.) —
    # generous enough that the <think> block doesn't starve the final answer.
    # Tighten via env vars / CLI for non-thinking models if you want speed.
    parser.add_argument("--max-schema-tokens", type=int, default=4096)
    parser.add_argument("--max-extraction-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Retries per sample if any stage fails validation.")
    parser.add_argument("--min-instances", type=int, default=2,
                        help="Minimum extracted instances to keep a sample.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    plans = make_plan(args.num_samples, rng)
    _log(
        f"[plan] {len(plans)} samples across "
        f"{len(TOPICS)} topics × {len(LENGTH_BUCKETS)} lengths × "
        f"{len(COUNT_BUCKETS)} count buckets"
    )
    _log(f"[plan] batch_size={args.batch_size}, "
         f"first batch will hit stages 1→2→3 sequentially before any rows are written")

    # Lazy imports so --help works without vLLM installed.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    _log(f"[boot] loading tokenizer + vLLM engine for model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
    )

    written = 0
    pending: list[SamplePlan] = list(plans)
    retry_round = 0

    out_f = open(args.output, "a", encoding="utf-8")
    try:
        while pending and retry_round <= args.max_retries:
            failed: list[SamplePlan] = []

            for batch_start in range(0, len(pending), args.batch_size):
                batch = pending[batch_start:batch_start + args.batch_size]
                round_seed = args.seed + retry_round * 9973 + batch_start
                batch_idx = batch_start // args.batch_size + 1
                n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
                tag = f"round={retry_round} batch={batch_idx}/{n_batches} size={len(batch)}"

                # ---- Stage 1: text ----------------------------------- #
                _log(f"[stage 1/text] {tag} — generating passages")
                t0 = time.time()
                text_prompts = [
                    build_chat(
                        tokenizer,
                        TEXT_SYSTEM_PROMPT,
                        TEXT_USER_TEMPLATE.format(
                            topic=p.topic.replace("_", " "),
                            subtopic=p.subtopic,
                            n=p.n_objects,
                            length_desc=p.length_desc,
                            style=p.style,
                        ),
                    )
                    for p in batch
                ]
                text_params = [
                    SamplingParams(
                        temperature=args.text_temperature,
                        top_p=args.text_top_p,
                        max_tokens=p.max_text_tokens,
                        seed=round_seed + i,
                    )
                    for i, p in enumerate(batch)
                ]
                text_outputs = llm.generate(
                    text_prompts, text_params, use_tqdm=True
                )
                texts = [
                    extract_passage(o.outputs[0].text)
                    for o in text_outputs
                ]
                _log(f"[stage 1/text] done in {time.time() - t0:.1f}s")

                # ---- Stage 2: schema --------------------------------- #
                _log(f"[stage 2/schema] {tag} — generating DSL schemas")
                t0 = time.time()
                schema_prompts = [
                    build_chat(
                        tokenizer,
                        SCHEMA_SYSTEM_PROMPT,
                        SCHEMA_USER_TEMPLATE.format(text=t),
                    )
                    for t in texts
                ]
                schema_params = SamplingParams(
                    temperature=args.schema_temperature,
                    top_p=args.schema_top_p,
                    max_tokens=args.max_schema_tokens,
                    seed=round_seed + 31,
                )
                schema_outputs = llm.generate(
                    schema_prompts, schema_params, use_tqdm=True
                )
                raw_schemas = [o.outputs[0].text for o in schema_outputs]
                _log(f"[stage 2/schema] done in {time.time() - t0:.1f}s")

                # ---- Build stage-3 inputs ---------------------------- #
                # Schema parsing is advisory now: if it fails we fall back to
                # passing the (think-stripped) raw schema text to stage 3 and
                # let stage-3 JSON validation be the actual gate.
                stage3_inputs: list[tuple[
                    SamplePlan, str, str, str, dict, str
                ]] = []
                short_text_drops = 0
                schema_fallbacks = 0
                for plan, text, raw_schema in zip(batch, texts, raw_schemas):
                    if not text or len(text) < 60:
                        failed.append(plan)
                        short_text_drops += 1
                        continue
                    parsed = parse_schema_fields(raw_schema)
                    if parsed is not None:
                        schema_name, schema_description, fields = parsed
                        schema_dsl = render_schema_dsl(
                            schema_name, fields, schema_description
                        )
                    else:
                        # Fallback: derive a name from the subtopic, pass the
                        # think-stripped raw schema text into stage 3 verbatim.
                        schema_fallbacks += 1
                        schema_name = (
                            _normalize_identifier(plan.subtopic) or "object"
                        )
                        schema_description = ""
                        fields = {}
                        cleaned_raw = _strip_fences(
                            _take_after_marker(
                                _strip_think_blocks(raw_schema or ""),
                                SCHEMA_MARKER,
                            )
                        ).strip()
                        schema_dsl = (
                            cleaned_raw if cleaned_raw
                            else f"[{schema_name}]"
                        )
                    stage3_inputs.append(
                        (plan, text, schema_name, schema_description,
                         fields, schema_dsl)
                    )

                if short_text_drops:
                    _log(f"[stage 2/schema] {short_text_drops} samples had too-short texts")
                if schema_fallbacks:
                    _log(f"[stage 2/schema] {schema_fallbacks} schemas unparsed → "
                         f"using raw text + subtopic-derived name (will gate on stage 3)")
                if not stage3_inputs:
                    _log(f"[skip] {tag} — no usable inputs after stage 1, advancing")
                    continue

                # ---- Stage 3: structured extraction ------------------ #
                _log(f"[stage 3/extract] {tag} — extracting structured records "
                     f"({len(stage3_inputs)} valid schemas)")
                t0 = time.time()
                extraction_prompts = [
                    build_chat(
                        tokenizer,
                        EXTRACTION_SYSTEM_PROMPT,
                        EXTRACTION_USER_TEMPLATE.format(
                            schema_dsl=schema_dsl, text=text
                        ),
                    )
                    for (_p, text, _sn, _sd, _f, schema_dsl) in stage3_inputs
                ]
                extraction_params = SamplingParams(
                    temperature=args.extraction_temperature,
                    top_p=args.extraction_top_p,
                    max_tokens=args.max_extraction_tokens,
                    seed=round_seed + 4111,
                )
                extraction_outputs = llm.generate(
                    extraction_prompts, extraction_params, use_tqdm=True
                )
                raw_extractions = [
                    o.outputs[0].text for o in extraction_outputs
                ]
                _log(f"[stage 3/extract] done in {time.time() - t0:.1f}s")

                # ---- Validate stage 3 + write ------------------------ #
                batch_written = 0
                json_parse_drops = 0
                json_validate_drops = 0
                json_failure_samples: list[str] = []
                for (plan, text, schema_name, schema_description,
                     fields, schema_dsl), raw_ext in zip(
                    stage3_inputs, raw_extractions
                ):
                    items = _extract_json_array(raw_ext)
                    if items is None:
                        failed.append(plan)
                        json_parse_drops += 1
                        if len(json_failure_samples) < 2:
                            snip = (
                                _strip_fences(_strip_think_blocks(raw_ext or ""))
                                .strip()[:240]
                                .replace("\n", " ⏎ ")
                            )
                            json_failure_samples.append(f"unparseable: {snip}")
                        continue
                    cleaned = validate_and_clean_extraction(
                        items, fields, min_instances=args.min_instances
                    )
                    if cleaned is None:
                        failed.append(plan)
                        json_validate_drops += 1
                        if len(json_failure_samples) < 2:
                            json_failure_samples.append(
                                f"too few/short instances "
                                f"(parsed list of {len(items)})"
                            )
                        continue

                    field_descriptions = {
                        name: desc for name, (_t, _o, desc) in fields.items()
                        if desc
                    }
                    record = {
                        "tokenized_text": tokenize_text(text),
                        "text": text,
                        "structuring": {schema_name: cleaned},
                        "schema_dsl": schema_dsl,
                        "schema_description": schema_description,
                        "field_descriptions": field_descriptions,
                        "topic": plan.topic,
                        "subtopic": plan.subtopic,
                        "length_bucket": plan.length_bucket,
                        "count_bucket": plan.count_bucket,
                        "n_objects_target": plan.n_objects,
                        "n_objects_extracted": len(cleaned),
                        "score": args.score,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    batch_written += 1

                out_f.flush()
                if json_parse_drops or json_validate_drops:
                    _log(f"[stage 3/extract] {json_parse_drops} unparseable JSON, "
                         f"{json_validate_drops} too-few-instances after cleanup")
                    for i, snip in enumerate(json_failure_samples, 1):
                        _log(f"[stage 3/extract] failed sample {i}: {snip}")
                _log(
                    f"[batch] {tag} → +{batch_written} written, "
                    f"running total={written}, failures-this-round={len(failed)}"
                )

            _log(f"[round={retry_round}] complete — "
                 f"{len(failed)} samples queued for next retry round")
            pending = failed
            retry_round += 1

        skipped = len(pending)
    finally:
        out_f.close()

    _log(f"[done] wrote {written} samples to {args.output} "
         f"(skipped {skipped} after retries)")


if __name__ == "__main__":
    main()
