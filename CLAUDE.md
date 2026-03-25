# GLiNExT — Multi-task Information Extraction Framework

Multi-task information extraction model built on top of GLiNER, supporting:
- Named-entity recognition (NER)
- Relation extraction
- Joint NER + relation extraction
- Text classification
- Embedding (pairwise similarity)
- Grouping / JSON schema extraction (structuring)

## New Modular Architecture

Each task is a self-contained module with three components:
- **Processor** — data preparation, prompt contribution, label creation
- **Model (Head)** — task-specific neural layers, forward pass, loss computation
- **Decoder** — post-processing logits into structured predictions

### Task Modules

| Module | Description | Status |
|--------|-------------|--------|
| **NER** | Named entity recognition via span scoring | Implemented (`tasks/ner/`) |
| **Relation Extraction** | Extract (head, relation, tail) triples without linking to NER entities; GLiNER2-style | Implemented (`tasks/relations/`) |
| **Joint NER + Relex** | First NER, then relation extraction in latent space linking to extracted entities; GLiNER-relex style | It should be a separate task, but that utilizes tasks/ner components |
| **Classification** | Text classification with configurable scorers; GLiClass-style | Implemented (`tasks/classification/`) |
| **Embedding** | Text pair similarity with configurable pooling and loss functions | Implemented (`tasks/embedding/`) |
| **Structuring** | Group entities into clusters for JSON schema extraction; GLiNER2-style | Implemented (`tasks/structuring/`) |
| **Count** | Predict instance counts per parent group | Implemented (`tasks/count/`) |
| **Decoder** | Autoregressive label generation from span representations | Implemented (`tasks/decoder/`) |

### General Factory Modules

Configurable general modules act as factories that assemble task-specific components:

- **`GLiNextProcessor`** (`processor.py`) — orchestrates per-task processors, builds unified prompts, delegates label creation
- **`GLiNExTModel`** (`model.py`) — manages shared encoder, registers task heads via `nn.ModuleDict`, executes in dependency order
- **Decoder** (`decoder.py`) — currently empty, should be factory built based on model configuration, each task has it's own processor
### General Model Flow

1. **Encode text** — shared transformer backbone (`Encoder` or `BiEncoder` (it encodes text and labels, for each tasks labels are encoded separetly))
2. **Extract task-specific prompt embeddings** — [ENT], [CAT], [REL], [PARENT], [CHILD] tokens from encoder output
3. **Route to task-specific heads** — each head receives `SharedRepresentations` (word embeddings + prompt embeddings (task specific))
4. **Decode** — general method assembles outputs from all active heads into `GLiNExTOutput`

### Head Execution Order & Dependencies

```
NER (no deps) → Relations (depends on NER) → Decoder (depends on NER)
Classification, Count, Structuring, Embedding (no deps, run independently)
```

## Current File Structure

```
glinext/
├── config.py          — GLiNextConfig with per-task sub-configs (NERHeadConfig, etc.)
├── model.py           — GLiNExTModel orchestrator, SharedRepresentations, GLiNExTOutput
├── processor.py       — GLiNextProcessor orchestrator, per-task processor delegation
├── mappings.py        — BatchClassesMapping, CatClassMapping, ExtractionClassMapping, etc.
├── layers.py          — Shared primitives: AnchoredSpanScorer, PairRepLayer, group layers, attention, RoPE
├── decoder.py         — Empty (uses GLiNER's Decoder)
├── utils.py           — Utilities
├── tasks/
│   ├── ner/
│   │   ├── model.py       — NERHead
│   │   ├── processor.py   — NERProcessor
│   │   └── decoder.py     — NER decoder
│   ├── classification/
│   │   ├── model.py       — ClassificationHead
│   │   ├── processor.py   — ClassificationProcessor
│   │   └── decoder.py     — Classification decoder
│   ├── relations/
│   │   ├── model.py       — RelationsHead
│   │   ├── processor.py   — RelationsProcessor
│   │   └── decoder.py     — Relations decoder
│   ├── structuring/
│   │   ├── model.py       — StructuringHead
│   │   ├── processor.py   — StructuringProcessor
│   │   └── decoder.py     — Structuring decoder
│   ├── embedding/
│   │   ├── model.py       — EmbeddingHead
│   │   ├── processor.py   — EmbeddingProcessor
│   │   └── decoder.py     — Embedding decoder
│   ├── count/
│   │   ├── model.py       — CountHead
│   │   ├── processor.py   — CountProcessor
│   │   └── decoder.py     — Count decoder
│   └── decoder/
│       ├── model.py       — DecoderHead
│       ├── processor.py   — DecoderProcessor
│       └── decoder.py     — Decoder post-processing
```

## What Needs to Change / Be Updated

### Make each head more configurable
- Each head (classification, NER, relation extraction, structuring) should expose more configuration options
- Scorer types, layer types, representation strategies should all be selectable via config
- Current state: partially done — classification has dot/weighted-dot/mlp, joint relations has adjacency/prompt modes, structuring has LSTM/query variants

### Fix labels encoder input data preparation and modeling
- Labels encoding via BiEncoder needs review
- Task-specific label encoder inputs should be properly collected and batched

### Labels decoding should use span representations
- Decoder head currently takes span token embeddings
- Ensure decoding consistently uses span-level representations across all tasks

### Unified anchor paradigm
All extraction tasks follow: **anchor + child → spans**:
- **Classification:** anchor = parent embedding, child = class embeddings
- **NER:** anchor = parent embedding, child = entity types ([ENT]) → spans in text
- **Relation extraction:** anchor = source entity span, child = relation types ([REL]) → target entity spans
- **Structuring:** anchor = instance rep, child = field types ([CHILD]) → value spans

### Anchor acquisition strategies
Different tasks need anchors from different sources. The anchor layer should be a configurable abstraction:

| Strategy | Description | Used by |
|----------|-------------|---------|
| **Parent embedding** | Single [P] token embedding per extraction group; simplest case, no learned anchor generation | NER, Classification |
| **Extracted entities** | Span representations of entities detected by NER or standalone detection | Relation extraction, Structuring |
| **Fixed embeddings (GLiNER2-style)** | Learnable fixed-size embedding table for N anchor slots; number of anchors is a hyperparameter | Structuring (fixed schema instances) |
| **Rotary Groups (parent-conditioned)** | Learnable, scalable anchor generation conditioned on parent embedding via RotaryGroupLSTM; produces variable number of anchors | Structuring (open-ended grouping) |
| **Query Groups** | Cross-attention over text with learnable query embeddings (QueryGroupLSTM / QueryGroupTransformer) | Structuring |

The anchor layer should be universal and selectable via config — any task that needs multiple anchors (relations, structuring) should be able to pick its strategy independently.

### Anchor modeling layers
Once anchors are obtained, they are combined with child representations before scoring:
- Linear projection of parent + child representations
- LSTM (GLiNER2-style recurrent modeling)
- MLP layer
 
## Data Format

### Prompt Structure

**General format:**
```
[TEXT]Prompt[SEP][P][CLS][CLS][CLS][SEP]<extraction groups>[SEP]Text...
```

**NER only:**
```
...[SEP][P]name [ENT]type1 [ENT]type2[SEP]Text...
```

**Joint NER + Relation Extraction (GLiNER-relex style):**
NER and relations share the same extraction group; [REL] tokens follow [ENT] tokens:
```
...[SEP][P]name [ENT]type1 [ENT]type2 [REL]rel1 [REL]rel2[SEP]Text...
```
NER runs first → extracted entity representations are paired → scored against [REL] embeddings.

**Standalone Relation Extraction (GLiNER2-style):**
Uses Anchor layers directly — anchor = source entity span, child = [REL] type embeddings → target spans in text. No explicit NER dependency; entity spans are detected as part of the relation scoring.
```
...[SEP][P]name [ENT]type1 [ENT]type2[SEP][P]name [REL]rel1 [REL]rel2[SEP]Text...
```
Relation types live in their own extraction group with separate [P] parent.

### Input Data Format
- **Text** — raw input text
- **Output:**
  - Classification: `[{name, all_labels, true_labels}]`
  - NER: `[{name, "ner": [text, (optional: start, end) label]}]`
  - Relation extraction: `[{name, ner: List, "relations": [(head_id, relation, tail_id)]}]`
  - Embedding: `[(text1, text2, score)]`
  - Structuring: `{schema_name: [{field: value_text}, ...]}`

### Label Tensor Shapes

**Discriminative mode:**
- Classification: `(B*N, C)` — B batch, N parent classes, C child classes
- NER: `(B*N, L, C+1, 3)` — L sequence length, C+1 includes parent class at index 0, 3 = start/inside/end
- Joint NER + Relex: `(B*N, E, E, C)` — E max entities, C relation classes; entity pair matrix scored against [REL] embeddings
- Standalone Relex (GLiNER2): `(B*N, L, X*C, 2, 3)` — same shape as NER; anchor = entity span, child = [REL] types → target spans via anchor layers, where 2 means head and tail 
- Embedding: `(B, 1)`
- Structuring spans: `(B*N, L, X*C, 3)` — X max groups, anchors: `(B*N, L, 3)`

**Generative mode:**
- Classification: `(B, N, T)` — T max tokens
- NER: `(B, N, E, T)`
- Joint NER + Relex: `(B, N, R, T)` — R max relations
- Standalone Relex (GLiNER2): `(B, N, E, T)` — per-entity relation decoding
- Embedding: `(B, 1)`
- Structuring: `(B, N, X, T)`

## Architecture Components

- **Encoder:** input tokens `(B, L)` → features `(B, L, D)`
- **Labels Encoder**: input labels tokens `(B*N, L)` -> features `(B*N, L, D)`
- **Decoder**: input span tokens `(B*S, L)` -> probs `(B*S, L, Vocab)`
- **Span rep layer:** token reps `(B, L)` + span_idx `(B, S)` → span features `(B, S, D)`
- **PairRepLayer:** entity pair representations (concat_proj/bilinear/additive/mlp) (for joint NER & Relation Extraction)
- **Group layers (Anchor layers):** parent reps (or anchors, for structuring) + child reps → instance anchors `(B*N, X*C, D)`
- **CountModule:** parent reps → count `(B, N, 1)` (regression or classification)
- **Pooling layers** - pooling text or labels representations;
- **Scoring layers** - for classification, see GLiClass for example;

## Key Design Patterns

1. **Modular heads** — each task is independent and optional; toggle via config (set to `None` to disable)
2. **Shared encoder** — single transformer backbone serving all tasks
3. **Unified scoring** — anchor detection, groups layers for generalization in NER, relations, and structuring
4. **Dependency ordering** — heads execute in declared order; relations/decoder depend on NER
5. **Task-specific label encoding** — optional BiEncoder for label embeddings per task
6. **Backward compatibility** — flat config params auto-migrate to per-task sub-configs
7. **Processor delegation** — main processor delegates prompt/label work to per-task processors
