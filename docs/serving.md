# GLiFormer serving

GLiFormer provides a Ray Serve deployment for heterogeneous text inference,
text embeddings, and optional per-request PolyLoRA routing. A physical batch
may contain different task combinations, schemas, decoding controls, and LoRA
adapter IDs while sharing the model forward.

## Installation

Install GLiFormer with the serving dependencies:

```bash
pip install -e ".[serve]"
```

Install PolyLoRA support when adapters are required:

```bash
pip install -e ".[serve,polylora]"
```

## Starting the server

Run the published base checkpoint on one GPU:

```bash
python -m gliformer.serve \
  --model knowledgator/gliformer-base-v1 \
  --device cuda \
  --dtype float16 \
  --max-batch-size 32
```

The default endpoint is `http://localhost:8000/gliformer`. CPU inference is
also supported:

```bash
python -m gliformer.serve \
  --model knowledgator/gliformer-base-v1 \
  --device cpu \
  --dtype float32 \
  --no-compile \
  --no-memory-calibration
```

Use `python -m gliformer.serve --help` for the complete CLI reference.

## HTTP API

### Multitask inference

Send `POST /gliformer` with a string `text` and at least one task schema:

```bash
curl -X POST http://localhost:8000/gliformer \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works for Acme in Paris.",
    "entities": ["person", "organization", "location"],
    "classes": ["employment", "geography"],
    "threshold": 0.3
  }'
```

Supported task fields are:

| Request field | Response field | Task |
| --- | --- | --- |
| `entities` | `ner` | Named entity recognition |
| `classes` | `classification` | Text classification |
| `relations` | `open_relex` | Open relation extraction |
| `joint_relations` | `joint_relex` | Joint entity and relation extraction |
| `structures` | `structuring` | Structured record extraction |

Any compatible combination can be included in one request. Available heads
depend on the loaded checkpoint.

The endpoint also accepts these decoding and structuring controls:

- `threshold`
- `flat_ner`
- `multi_label`
- `objectness_threshold`
- `preserve_empty_records`
- `manual_structuring_count`
- `structuring_dedup`
- `decoder_kwargs`
- `adapter_id`

Unknown fields, invalid option types, oversized requests, unknown adapters, and
requests without a task schema are rejected before inference.

### Embeddings

Embeddings use a separate dynamically batched route:

```bash
curl -X POST http://localhost:8000/gliformer/embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": "A scientist works in a laboratory."}'
```

The response contains an `embedding` array. `adapter_id` is supported on this
route as well.

### Operational endpoints

- `GET /gliformer/health` returns model and serving health information.
- `GET /gliformer/ready` is an alias for the readiness response.
- `GET /gliformer/metadata` reports loaded heads, limits, model variant,
  packing state, and adapter state.
- `GET /gliformer/adapter-cache` reports PolyLoRA CPU, disk, and GPU residency.
- `GET /gliformer/adapter-cache?adapter_id=legal` reports one adapter.

## Python client

The synchronous client mirrors the model's text APIs:

```python
from gliformer.serve import GLiFormerClient

client = GLiFormerClient()

entities = client.predict_entities(
    "Alice works at Acme.",
    ["person", "organization"],
    threshold=0.3,
)

result = client.predict(
    "Alice works at Acme.",
    entities=["person", "organization"],
    classes=["employment", "other"],
)

embedding = client.embed_text("Alice works at Acme.")
```

`predict_async()` and `predict_requests_async()` provide asynchronous facades.
`health()`, `metadata()`, and `adapter_cache_status()` expose the operational
endpoints.

### Batched texts and schemas

Passing a list of texts sends concurrent single-text HTTP requests. Ray Serve
can accumulate these requests, including requests from other clients, into one
physical model batch:

```python
results = client.predict(
    ["Alice joined Acme.", "Bob joined Globex."],
    entities=["person", "organization"],
    classes=["employment", "other"],
    structures={"employee": ["name", "company"]},
)
```

Here the same entity labels, class labels, and structuring schema are applied to
every input text. By default, a value supplied to `entities`, `classes`,
`relations`, `joint_relations`, or `structures` is shared across the entire
text list.

Simple per-text label sets can also be passed as a nested list:

```python
results = client.predict(
    ["Alice joined Acme.", "Paris is in France."],
    entities=[["person", "organization"], ["location", "country"]],
)
```

For list-shaped complex schemas, use the explicit `PerText` wrapper. This
avoids guessing whether a list is one shared schema or one schema per text:

```python
from gliformer.serve import GLiFormerClient, PerText

client = GLiFormerClient()
results = client.predict(
    ["Alice works at Acme.", "Paris is in France."],
    joint_relations=PerText([
        {
            "employment": {
                "entities": ["person", "organization"],
                "relations": ["works_at"],
            }
        },
        None,
    ]),
    structures=PerText([
        None,
        {"place": ["name", "country"]},
    ]),
)
```

An ordinary `joint_relations` or `structures` value is shared by every text.
Within `PerText`, `None` means that the corresponding task is not requested for
that row.

For completely independent payloads, use `predict_requests()`:

```python
results = client.predict_requests([
    {
        "text": "Alice works at Acme.",
        "entities": ["person", "organization"],
        "threshold": 0.3,
    },
    {
        "text": "Paris is in France.",
        "structures": {"place": ["name", "country"]},
        "objectness_threshold": 0.6,
    },
])
```

`per_text_tasks` remains available as a backwards-compatible override when
using `predict()`.

## Dynamic batching

Ray accumulates HTTP or deployment-handle calls for up to
`--batch-wait-timeout-ms`, bounded by `--max-batch-size`. Rows in a batch may
have different:

- task combinations;
- labels and extraction schemas;
- PolyLoRA adapter IDs;
- `threshold`, `flat_ner`, `multi_label`, and supported decoder controls.

GLiFormer performs one compatible model forward using a candidate threshold
that preserves the rows' required predictions, then applies decoding controls
independently per row. Uniform controls retain the scalar decoding path.

The server estimates effective sequence length from the text and schemas,
selects a calibrated batch size, and retries with a smaller internal minibatch
after a CUDA out-of-memory error. A validation or schema error in one request
is isolated from other requests in the same Ray batch. Embeddings have their
own adaptive batcher and OOM retry path.

### Memory calibration and compilation

At GPU startup the server can warm configured batch shapes and calibrate peak
memory over increasing sequence lengths. Relevant options include:

```text
--precompiled-batch-sizes 1,2,4,8,16,32
--warmup-iterations 3
--target-memory-fraction 0.8
--memory-overhead-factor 1.3
--calibration-min-seq-len 64
--calibration-probe-batch-size 2
```

Disable compilation or calibration independently with `--no-compile` and
`--no-memory-calibration`. Full-model Inductor compilation can take a long time
for multitask checkpoints. Use `--compile-backend aot_eager` for bounded
startup time or `--no-compile` for eager serving.

Before starting local Ray, the CLI checks available host memory. Set
`--min-available-host-memory-gb 0` only when an external scheduler already
enforces the limit.

## Sequence packing

Packing is opt-in:

```bash
python -m gliformer.serve \
  --model knowledgator/gliformer-base-v1 \
  --device cuda \
  --dtype float16 \
  --enable-sequence-packing
```

Compatible short text rows are packed into fewer token streams with a
block-diagonal attention mask and restored to their original batch layout
before task heads run. Packing is enabled only for verified text encoder paths.
Layout inputs containing aligned layout/media tensors use the ordinary
unpacked path.

Packing is currently disabled when PolyLoRA is enabled because independently
routed row adapters cannot yet be mapped safely onto packed token streams.

## PolyLoRA

PolyLoRA applies a different PEFT LoRA adapter to each row while sharing the
base text backbone and GLiFormer task heads. It is an inference runtime; adapter
training remains a separate PEFT workflow.

Start the server with adapters preloaded from trusted local directories:

```bash
python -m gliformer.serve \
  --model knowledgator/gliformer-base-v1 \
  --device cuda \
  --dtype float16 \
  --enable-polylora \
  --polylora-adapter legal=/models/legal \
  --polylora-adapter finance=/models/finance \
  --polylora-adapter-weight-modules query_proj,value_proj
```

Adapter directories must use the standard PEFT layout with
`adapter_config.json` and `adapter_model.safetensors` or `adapter_model.bin`.
Adapters may also be present in PolyLoRA's configured disk cache.

Requests select an adapter by ID:

```python
result = client.predict(
    "Alice works for Acme.",
    entities=["person", "organization"],
    adapter_id="legal",
)
```

The supported modes are:

- PolyLoRA disabled: the unchanged base model is used;
- PolyLoRA enabled without `adapter_id`: the reserved `__base__` slot is used;
- `adapter_id` supplied: the registered adapter is selected for that row.

Different requests in one physical batch may use the base slot and different
adapters. The number of simultaneously required non-base adapters must not
exceed `--polylora-max-gpu-adapters`, and adapter rank must not exceed
`--polylora-max-rank`.

Inference requests cannot upload adapters or supply filesystem paths. Unknown
IDs are rejected. This keeps adapter I/O outside latency-sensitive batches and
prevents arbitrary model loading through HTTP.

PolyLoRA requires a variant exposing a text backbone. Vision-only and
audio-only variants are rejected. In bi-encoder models, the document encoder
is adapted while the label encoder remains shared.

## In-process Ray usage

`GLiFormerFactory` starts and owns a Ray Serve application and provides sync and
async prediction methods:

```python
from gliformer.serve import GLiFormerFactory, PerText

with GLiFormerFactory(
    model="knowledgator/gliformer-base-v1",
    device="cuda",
    dtype="float16",
) as server:
    results = server.predict(
        ["Alice works at Acme.", "Paris is in France."],
        structures=PerText([None, {"place": ["name", "country"]}]),
        entities=PerText([["person", "organization"], None]),
    )
```

Use `predict_requests()` when each item should provide its entire payload.

## Container deployment

The included Compose service reserves one NVIDIA GPU and persists Hugging Face
and PolyLoRA caches:

```bash
docker compose up --build
curl http://localhost:8000/gliformer/health
```

Common environment variables are:

- `GLIFORMER_MODEL`
- `GLIFORMER_DEVICE`
- `GLIFORMER_DTYPE`
- `GLIFORMER_MAX_BATCH_SIZE`
- `GLIFORMER_BATCH_WAIT_MS`
- `GLIFORMER_MEMORY_FRACTION`
- `GLIFORMER_DISABLE_COMPILE`
- `GLIFORMER_COMPILE_BACKEND`
- `GLIFORMER_DISABLE_MEMORY_CALIBRATION`
- `GLIFORMER_ENABLE_SEQUENCE_PACKING`
- `GLIFORMER_ENABLE_POLYLORA`
- `GLIFORMER_POLYLORA_ADAPTERS`
- `GLIFORMER_POLYLORA_DISK_CACHE_DIR`

`GLIFORMER_POLYLORA_ADAPTERS` is a comma-separated list of `ID=PATH` entries.
Additional CLI options can be appended to the container command.

## Scope and limitations

The HTTP deployment serves text-backed task heads and text embeddings. The
Python model API continues to support its configured layout, image, audio, and
PDF workflows, but the HTTP server does not accept server-side document or
media paths. A production media endpoint requires an explicit upload or object
storage policy.

Serving exposes only heads available in the selected checkpoint. PolyLoRA does
not adapt task heads, and packing cannot currently be combined with PolyLoRA.
