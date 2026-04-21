# GLiNExT — Multi-task Information Extraction Framework

Multi-task information extraction model built on top of GLiNER, supporting:
- Named-entity recognition (NER)
- Relation extraction (open & joint)
- Text classification
- Embedding (pairwise similarity)
- Grouping / JSON schema extraction (structuring)
- Instance counting

## Architecture Overview

### Three-component task design

Each task is a self-contained module with three components:
- **Processor** — data preparation, prompt contribution, label creation
- **Model (Head)** — task-specific neural layers, forward pass, loss computation
- **Decoder** — post-processing logits into structured predictions

### High-level orchestration

Four top-level classes coordinate the task modules:

| Class | File | Role |
|-------|------|------|
| `GLiNExT` | `glinext.py` | User-facing API: inference, training, model loading (extends `BaseEncoderGLiNER`) |
| `GLiNExTModel` | `model.py` | Shared encoder + task heads orchestrator (`nn.Module`) |
| `GLiNextProcessor` | `processor.py` | Delegates prompt/label work to per-task processors |
| `GLiNExTDecoder` | `decoder.py` | Factory assembling per-task decoders from config |
| `GLiNExTDataCollator` | `collator.py` | Bridges raw data → model-ready batches via processor |
| `GLiNExTSchema` | `schema.py` | Fluent builder for multi-task inference schemas |
| `GLiNExTTrainer` | `training.py` | Extends GLiNER Trainer for multi-task label handling |

### Model flow

1. **Encode text** — shared transformer backbone (`Encoder` or `BiEncoder`; task-specific labels batched into a single `encode_labels()` pass)
2. **Extract task-specific prompt embeddings** — [ENT], [CAT], [REL], [PARENT], [CHILD] tokens from encoder output
3. **Route to task-specific heads** — each head receives `SharedRepresentations` (word embeddings + prompt embeddings)
4. **Decode** — general method assembles outputs from all active heads into `GLiNExTOutput`

### Head execution order & dependencies

```
NER (no deps)
Joint Relex (no external deps — NER is built-in via inheritance from NERHead)
Open Relex, Classification, Count, Structuring, Embedding (no deps, run independently)
```

## Abstract Base Classes (`tasks/__init__.py`)

| ABC | Purpose | Key abstract method |
|-----|---------|-------------------|
| `TaskHead` | Neural module base | `forward(shared, dependency_outputs, **batch)` |
| `TaskProcessor` | Data processor base | `get_classes_mapping()`, `create_labels()` |
| `TaskDecoder` | Output decoder base | `decode(model_output, classes_mapping)` |

### Shared base classes

Extraction tasks share span resolution and decoding via intermediate bases:

```
TaskProcessor (ABC)                    TaskDecoder (ABC)
└── SpanProcessor                      └── SpanDecoder
    ├── NERProcessor                       ├── NERDecoder
    │   └── JointRelexProcessor            │   └── JointRelexDecoder
    ├── OpenRelexProcessor                 ├── OpenRelexDecoder
    └── StructuringProcessor               └── StructuringDecoder
```

**SpanProcessor** (`tasks/span_processor.py`):
- `_resolve_entity_spans()` — text mention → token index resolution
- `_resolve_text_span()` — single mention resolution
- `_tokenize_text()` — tokenize + cache
- `_generate_negative_spans()` — random non-overlapping negative sampling
- `_collect_span_candidates()` — positive + negative span collection

**SpanDecoder** (`tasks/span_decoder.py`):
- `decode_bio_spans()` — single-sample BIO (L, C, 3) → List[Span]
- `decode_bio_spans_single_class()` — single-class BIO (L, 3) → List[Span]
- `decode_bio_spans_batch()` — batched BIO decoding
- `decode_span_level()` — pre-computed span representations decoding
- `greedy_search()` — overlap removal (flat/nested modes)
- `resolve_span_text()` — token indices → text

## Task Modules

| Module | Path | Description |
|--------|------|-------------|
| **NER** | `tasks/ner/` | Named entity recognition via span scoring |
| **Joint Relex** | `tasks/joint_relex/` | Joint NER + relation extraction; inherits NERHead, adjacency-based entity pair scoring (GLiNER-relex style) |
| **Open Relex** | `tasks/open_relex/` | Anchor-based relation extraction; dual AnchoredSpanScorers, no NER dependency (GLiNER2-style) |
| **Classification** | `tasks/classification/` | Text classification via anchor paradigm + pooling (GLiClass-style) |
| **Embedding** | `tasks/embedding/` | Text pair similarity with configurable pooling and loss |
| **Structuring** | `tasks/structuring/` | Group entities into clusters for JSON schema extraction (GLiNER2-style) |
| **Count** | `tasks/count/` | Predict instance counts per parent group |

## File Structure

```
glinext/
├── __init__.py        — Public API re-exports
├── glinext.py         — GLiNExT user-facing class (inference, training, model I/O)
├── config.py          — GLiNextConfig with per-task sub-configs (NERHeadConfig, etc.)
├── model.py           — GLiNExTModel orchestrator, SharedRepresentations, GLiNExTOutput
├── training.py        — GLiNExTTrainer (multi-task label handling, OOM recovery)
├── utils.py           — Utilities
├── processing/        — Data preparation, collation, decoding, schema
│   ├── __init__.py
│   ├── processor.py       — GLiNextProcessor orchestrator, per-task processor delegation
│   ├── decoder.py         — GLiNExTDecoder factory (assembles per-task decoders)
│   ├── collator.py        — GLiNExTDataCollator (raw data → model-ready batches)
│   ├── schema.py          — GLiNExTSchema fluent builder for inference
│   └── mappings.py        — BatchClassesMapping, CatClassMapping, ExtractionClassMapping, etc.
├── layers/            — Shared neural primitives
│   ├── __init__.py        — Re-exports all layers
│   ├── mlp.py             — create_mlp, FeaturesProjector
│   ├── pair_rep.py        — PairRepLayer, PromptRelationExtractor
│   ├── anchored_scorer.py — AnchoredSpanScorer
│   ├── anchor_layer.py    — AnchorLayer factory (Parent, Fixed, Rotary, QueryLSTM, QueryTransformer)
│   ├── anchor_modeling.py — AnchorModeling factory (Linear, LSTM, MLP post-anchor fusion)
│   ├── groups.py          — RotaryGroupLSTM, QueryGroupLSTM, QueryGroupTransformer
│   ├── attention.py       — SelfAttentionBlock, CrossAttentionBlock, Fuser, LayerwiseAttention
│   ├── pooling.py         — Pooling registry (Mean, CLS, Max, Weighted)
│   ├── rotary.py          — RotaryEmbedding, rotate_half, apply_rotary_pos_emb
│   └── rnn.py             — LstmSeq2SeqEncoder
└── tasks/
    ├── __init__.py        — TaskHead, TaskProcessor, TaskDecoder ABCs; SharedRepresentations, TaskFlatInputs, TaskHeadOutput
    ├── span_processor.py  — SpanProcessor base (shared span resolution, negative sampling)
    ├── span_decoder.py    — SpanDecoder base (shared BIO decoding, greedy search, Span dataclass)
    ├── ner/
    │   ├── model.py       — NERHead
    │   ├── processor.py   — NERProcessor (inherits SpanProcessor)
    │   └── decoder.py     — NERDecoder (inherits SpanDecoder)
    ├── classification/
    │   ├── model.py       — ClassificationHead (anchor paradigm + Pooling)
    │   ├── processor.py   — ClassificationProcessor
    │   └── decoder.py     — ClassificationDecoder
    ├── joint_relex/
    │   ├── model.py       — JointRelexHead (inherits NERHead)
    │   ├── processor.py   — JointRelexProcessor (inherits NERProcessor)
    │   └── decoder.py     — JointRelexDecoder (inherits NERDecoder)
    ├── open_relex/
    │   ├── model.py       — OpenRelexHead (dual AnchoredSpanScorer + optional SpanRepLayer)
    │   ├── processor.py   — OpenRelexProcessor (inherits SpanProcessor)
    │   └── decoder.py     — OpenRelexDecoder (inherits SpanDecoder)
    ├── structuring/
    │   ├── model.py       — StructuringHead (AnchoredSpanScorer + optional SpanRepLayer)
    │   ├── processor.py   — StructuringProcessor (inherits SpanProcessor)
    │   └── decoder.py     — StructuringDecoder (inherits SpanDecoder)
    ├── embedding/
    │   ├── model.py       — EmbeddingHead, EmbeddingLoss registry
    │   ├── processor.py   — EmbeddingProcessor
    │   └── decoder.py     — EmbeddingDecoder
    └── count/
        ├── model.py       — CountHead, CountModule
        ├── processor.py   — CountProcessor
        └── decoder.py     — CountDecoder
```

## Unified Anchor Paradigm

All extraction tasks follow: **anchor + child → spans**:
- **Classification:** anchor = parent embedding, child = class embeddings
- **NER:** anchor = parent embedding, child = entity types ([ENT]) → spans in text
- **Relation extraction:** anchor = source entity span, child = relation types ([REL]) → target entity spans
- **Structuring:** anchor = instance rep, child = field types ([CHILD]) → value spans

### Anchor acquisition strategies (`layers/anchor_layer.py`)

Via `AnchorLayer.from_config(anchor_mode, hidden_size, **kwargs)`:

| Strategy | Class | Used by |
|----------|-------|---------|
| Parent embedding | `ParentAnchorLayer` | NER, Classification |
| Extracted entities (via NERHead inheritance) | — | Joint Relex |
| Fixed embeddings (learnable `nn.Embedding`) | `FixedAnchorLayer` | Structuring (fixed schema) |
| Rotary Groups (parent-conditioned) | `RotaryAnchorLayer` | Structuring (open-ended) |
| Query Groups (LSTM) | `QueryLSTMAnchorLayer` | Structuring |
| Query Groups (Transformer) | `QueryTransformerAnchorLayer` | Structuring |

### Anchor modeling layers (`layers/anchor_modeling.py`)

Via `AnchorModeling.from_config(modeling_type, hidden_size, **kwargs)`:

| Strategy | Class | Description |
|----------|-------|-------------|
| Linear | `LinearAnchorModeling` | Linear projection of concatenated anchor + child (default) |
| LSTM | `LSTMAnchorModeling` | GRU-based recurrent processing (GLiNER2-style) |
| MLP | `MLPAnchorModeling` | Multi-layer perceptron fusion |

### Dual scoring paths

Extraction heads support both token-level BIO and span-level scoring. When `represent_spans=True`, a `SpanRepLayer` extracts span representations, scores them against fused label embeddings, and adds a span-level loss. Decoders auto-select the best available path.

Configured per-task via: `represent_spans: bool`, `neg_spans_ratio: float`, `span_loss_coef: float`

| Task | Token-level shape | Span-level shape |
|------|-------------------|------------------|
| NER | `(B, L, C, 3)` | `(B, S, C)` |
| Structuring | `(B, X, L, C, 3)` | `(B, X, S, C)` |
| Open Relex | `(B, X, C, L, 2, 3)` | `(B, S, X, C, 2)` |

## Configuration

All anchor-based head configs inherit from `BaseHeadConfig`:
```python
@dataclass
class BaseHeadConfig:
    loss_coef: float = 1.0
    anchor_mode: str = "parent"
    anchor_modeling: str = "linear"
    anchor_refine_layers: int = 0
    anchor_refine_heads: int = 8
    represent_spans: bool = False
    neg_spans_ratio: float = 1.0
    span_loss_coef: float = 1.0
    parent_token_index: int = -1
    embed_parent_token: bool = True
```

Inheriting configs: `NERHeadConfig`, `ClassificationHeadConfig`, `JointRelexHeadConfig`, `OpenRelexHeadConfig`, `StructuringHeadConfig`.
Non-inheriting: `CountHeadConfig`, `EmbeddingHeadConfig` (no anchor pipeline).

Key per-task settings:
- `anchor_mode` — anchor acquisition strategy (default: "parent" for NER/Classification/JointRelex)
- `anchor_modeling` — post-anchor fusion strategy (default: "linear")
- `pooling_type` — text/label pooling (Classification, Embedding)
- `similarity_fn`, `loss_fn` — Embedding task-specific

Set a task sub-config to `None` to disable it entirely.

### Shared layers across tasks

`GLiNextConfig` supports sharing anchor modeling and cross-attention layers across all heads that use the anchor paradigm (NER, Classification, Open Relex, Structuring, Joint Relex):

- `shared_anchor_modeling: Optional[str]` — `"linear"`, `"lstm"`, or `"mlp"`. When set, a single `AnchorModeling` instance is created in `GLiNExTModel` and shared across all applicable task heads. Per-task `anchor_modeling` settings are ignored when this is set.
- `shared_anchor_refine_layers: int` — number of shared `AnchorCrossAttentionLayer` layers (0 = disabled). When > 0, a single cross-attention refinement layer is shared across all applicable heads. Per-task `anchor_refine_layers` settings are ignored.
- `shared_anchor_refine_heads: int` — number of attention heads for the shared cross-attention layer (default: 8).

When shared layers are not configured (`None` / `0`), each task head creates its own layers as before.

### Parent embedding modes

`GLiNextConfig.per_task_parents: Optional[bool]` controls how parent token embeddings are shared across tasks:

| Value | Behavior |
|-------|----------|
| `False` | All tasks share a single `[PARENT]` token. Parent offset within the shared tensor is computed via `_parent_offset_for_item()` using `_PROMPT_TASK_ORDER`. |
| `True` | Each task gets a distinct parent token (`[ENT_P]`, `[CAT_P]`, `[REL_P]`, `[STRUCT_P]` by default). Parent positions are task-local — no offset computation needed. |
| `None` (default) | Auto-detect from whether the resolved per-task parent tokens are distinct. Ensures backward compatibility with saved configs that predate this parameter. |

Per-task tokens can be overridden individually via `ner_parent_token`, `cat_parent_token`, `open_rel_parent_token`, `struct_parent_token`.

The resolved boolean is exposed as `config.uses_per_task_parents` (read-only property).

## Data Format

### Prompt structure

```
[TEXT]Prompt[SEP][P][CLS][CLS][CLS][SEP]<extraction groups>[SEP]Text...
```

Per-task prompt patterns:
- **NER:** `[P]name [ENT]type1 [ENT]type2`
- **Joint Relex:** `[P]name [ENT]type1 [ENT]type2 [REL]rel1 [REL]rel2`
- **Open Relex:** `[P]name [REL]rel1 [REL]rel2`
- **Structuring:** `[P]schema [CHILD]field1 [CHILD]field2`
- **Classification:** `[P]name [CAT]class1 [CAT]class2`

### Input data format

- **Text** — raw input text
- **Annotations:**
  - Classification: `[{name, all_labels, true_labels}]`
  - NER: `[{name, "ner": [text, (optional: start, end) label]}]`
  - Joint Relex: `[{name, ner, "relations": [(head_id, relation, tail_id)]}]` (in `extraction` key)
  - Open Relex: `[{name, "relations": [{"head": text, "relation": type, "tail": text}]}]` (in `open_relex` key)
  - Embedding: `[(text1, text2, score)]`
  - Structuring: `{schema_name: [{field: value_text, ...}, ...]}`

### Label tensor shapes

**Token-level BIO:**
- Classification: `(B*N, C)`
- NER: `(B*N, L, C+1, 3)`
- Joint Relex: `(B*N, E, E, C)`
- Open Relex: `(B*N, X, C, L, 2, 3)`
- Embedding: `(B, 1)`
- Structuring: `(B*N, X, L, C, 3)`

**Span-level (when `represent_spans=True`):**
- NER: `(B, S, C)`
- Structuring: `(B, X, S, C)`
- Open Relex: `(B, S, X, C, 2)`

### Inference API

#### Main entry point: `GLiNExT.inference()`

```python
def inference(
    texts: Union[str, List[str]],
    entities: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    classes: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    relations: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    joint_relations: Optional[Dict[str, dict]] = None,
    structures: Optional[Dict[str, Union[List[str], dict]]] = None,
    flat_ner: bool = True,
    threshold: float = 0.5,
    multi_label: bool = False,
    batch_size: int = 8,
    **kwargs,
) -> Dict[str, List]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `str` or `List[str]` | — | Input text(s) |
| `entities` | `List[str]` or `Dict[str, List[str]]` | `None` | NER entity types |
| `classes` | `List[str]` or `Dict[str, List[str]]` | `None` | Classification labels |
| `relations` | `List[str]` or `Dict[str, List[str]]` | `None` | Open relation types |
| `joint_relations` | `Dict[str, dict]` | `None` | Joint NER + relation extraction |
| `structures` | `Dict[str, Union[List[str], dict]]` | `None` | Structuring schemas |
| `flat_ner` | `bool` | `True` | Enforce non-overlapping spans |
| `threshold` | `float` | `0.5` | Detection confidence threshold |
| `multi_label` | `bool` | `False` | Allow overlapping spans with different labels |
| `batch_size` | `int` | `8` | Batch size for processing |

#### Input formats for label arguments

**Entities / Classes / Relations** — single unnamed group or multiple named groups:

```python
# Single group (auto-named)
entities = ["person", "org", "location"]

# Multiple named groups
entities = {
    "general": ["person", "org"],
    "medical": ["disease", "treatment"]
}
```

**Joint relation extraction** — groups with both entities and relations:

```python
joint_relations = {
    "general": {
        "entities": ["person", "org", "location"],
        "relations": ["works_at", "born_in", "located_in"]
    }
}
```

**Structuring** — schema name → field list, or schema name → dict with field types:

```python
# Simple field list
structures = {
    "person": ["name", "age", "occupation"]
}

# With field type specifications (used via schema builder)
structures = {
    "person": {
        "fields": ["name", "age", "salary"],
        "field_types": {
            "name": "str",
            "age": "int",
            "salary": "float"
        }
    }
}
```

#### Output format

```python
Dict[str, List]:
  "ner":            List[List[Dict]]  # [{start, end, text, label, score}, ...]
  "classification": List[List[Dict]]  # [{class_id, score}, ...]
  "open_relex":     List[List[Dict]]  # [{head: {start,end,text}, tail: {start,end,text}, relation, score}, ...]
  "structuring":    List[Dict[str, List[Dict]]]  # {schema_name: [{field: value}, ...]}
  "embedding":      List[torch.Tensor]  # embedding vectors
  "count":          List[int]           # predicted counts
```

Only keys for active tasks (non-None arguments) are present in the output.

#### Convenience methods

```python
# NER
entities = model.predict_entities(text, ["person", "org"])           # → List[Dict]
entities = model.batch_predict_entities(texts, ["person", "org"])    # → List[List[Dict]]

# Classification
labels = model.classify(text, ["positive", "negative"])              # → List[Dict]

# Open relation extraction
triples = model.predict_relations(text, ["works_at", "born_in"])     # → List[Dict]

# Structuring
structured = model.structure(text, {"person": ["name", "age"]})      # → Dict[str, List[Dict]]

# Embeddings
embeddings = model.embed_text(texts)                 # → Tensor (N, D) — mean-pooled word embeddings
label_embs = model.embed_labels(["person", "org"])   # → Tensor (N, D) — bi-encoder label embeddings
```

#### Schema-based inference

`GLiNExTSchema` is a fluent builder that composes multi-task queries and supports typed structuring output:

```python
schema = model.create_schema()

# Chain task definitions (all methods return self)
schema.add_entities(["person", "org"], parent="general", description="Extract entities")
schema.add_classes(["positive", "negative"], parent="sentiment")
schema.add_relations(["works_at", "born_in"], parent="relations")
schema.add_joint_entities_relations(
    entities=["person", "org"],
    relations=["works_at"],
    parent="joint_er"
)
schema.add_structure("person", {
    "name": "str",
    "age": "int",
    "email": "str"
})

results = model.inference_from_schema(texts, schema)
```

Typed structuring fields are automatically converted via `StructuringOutputFormatter`. Supported types: `"str"`, `"int"`, `"float"`, `"bool"`, `"list"`, `"date"`, `"datetime"`, or a custom `Callable`.

### Inference pipeline flow

1. **Input normalization** — single text → list; filter empty/whitespace texts
2. **Tokenization** — `prepare_inputs()` splits texts into word tokens with char-to-token index maps
3. **Input construction** — `_build_inference_input()` normalizes label groups (List → Dict), creates per-text dicts with task annotation stubs
4. **Batching** — `GLiNExTDataCollator` batches inputs, generates `BatchClassesMapping`, constructs prompts with task-specific special tokens, tokenizes via HF tokenizer, creates word masks; label creation skipped in inference mode
5. **Forward pass** — `GLiNExTModel.forward()` encodes text, extracts prompt embeddings, routes to active task heads → `GLiNExTOutput` with per-task logits
6. **Decoding** — `GLiNExTDecoder` delegates to per-task decoders (BIO span matching or span-level scoring), unflattens batch×groups → per-item results
7. **Post-processing** — span tasks: map token indices → char positions; structuring: organize into JSON per schema; optional typed formatting via `StructuringOutputFormatter`

## Architecture Components

- **Encoder:** input tokens `(B, L)` → features `(B, L, D)`
- **Labels Encoder:** input label tokens `(B*N, L)` → features `(B*N, L, D)` (bi-encoder mode)
- **SpanRepLayer:** token reps + span_idx → span features `(B, S, D)`
- **PairRepLayer:** entity pair representations (concat_proj/bilinear/additive/mlp)
- **AnchoredSpanScorer:** anchor + child fusion → span scoring
- **CountModule:** parent reps → count `(B, N, 1)`
- **Pooling:** registry-based (Mean, CLS, Max, Weighted)

## Key Design Patterns

1. **Modular heads** — each task is independent and optional; toggle via config (set to `None` to disable)
2. **Shared encoder** — single transformer backbone serving all tasks
3. **Unified anchor paradigm** — anchor + child → spans across all extraction tasks (NER, Classification, relations, structuring)
4. **Dependency ordering** — heads execute in declared order; Joint Relex has built-in NER via inheritance
5. **Task-specific label encoding** — optional BiEncoder for label embeddings per task
6. **Processor delegation** — main processor delegates prompt/label work to per-task processors
7. **Registry-based polymorphism** — configurable components use `__init_subclass__` self-registration into `_registry` + `from_config()` factory. No `if/elif` branching. To add a variant: subclass with type keyword (e.g. `class MyPooling(Pooling, pooling_type="mine")`) and implement `forward()`. Used by: `Pooling`, `EmbeddingLoss`, `AnchorLayer`, `AnchorModeling`
8. **ABC hierarchy** — `TaskHead`/`TaskProcessor`/`TaskDecoder` enforce contracts; `SpanProcessor`/`SpanDecoder` share extraction logic
9. **Dual scoring paths** — extraction heads support both token-level BIO and span-level scoring; decoders auto-select
10. **Batched label encoding** — all task label inputs batched into single `encode_labels()` pass, split back per-task