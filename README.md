<div align="center">

# GLiFormer: Multi-Task Information Extraction

**Named Entity Recognition | Text Classification | Relation Extraction | Multi-Level Structuring | Embeddings**

**Text · Documents · Images · Audio**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](pyproject.toml)
[![Built on GLiNER](https://img.shields.io/badge/Built%20on-GLiNER-8A2BE2)](https://github.com/urchade/GLiNER)

[Quick Start](#quick-start) • [Structuring](#structured-extraction) • [Usage](#usage) • [Training](#training) • [Architectures](#architectures) • [Evaluation](#evaluation)

</div>

GLiFormer is a framework for training and running models that turn unstructured inputs into labeled spans, relations, classifications, and structured records. Built on [GLiNER](https://github.com/urchade/GLiNER), it combines a shared encoder with configurable task heads and lets you specify entity types, class labels, relation types, and extraction schemas at inference time.

Alongside text extraction, the codebase includes model variants for document layout, vision, audio, and combined modalities. Available tasks depend on the heads and modalities configured and trained in your checkpoint.

## Why GLiFormer?

<table>
  <tr>
    <td width="33%"><strong>Multiple Tasks, One Encoder</strong><br>Run entity recognition, classification, relations, and structuring together through a shared backbone.</td>
    <td width="33%"><strong>Labels at Inference Time</strong><br>Describe the entities, classes, and relations you need with readable labels.</td>
    <td width="33%"><strong>Multi-Level Structuring</strong><br>Extract nested records and connect children to their parents: companies → departments → employees, all in one schema.</td>
  </tr>
  <tr>
    <td><strong>Beyond Plain Text</strong><br>Use dedicated text, layout, vision, audio, and omni model variants.</td>
    <td><strong>Flexible Task Heads</strong><br>Configure individual tasks, shared components, and anchor representations for your workload.</td>
    <td><strong>Fine-Tune on Your Data</strong><br>Train with task-specific annotations, freeze selected components, or update only task heads.</td>
  </tr>
</table>

## Quick Start

### Installation

Requires **Python 3.10 or newer**. Install from the repository:

```bash
git clone https://github.com/Bionity/GLiFormer.git
cd GLiFormer
pip install -e .
```

With uv:

```bash
uv pip install -e .
```

Optional dependencies can be installed for the features you use:

| Extra | Purpose |
| --- | --- |
| `demo` | Gradio demos and comparison models |
| `data` | Dataset loading with Hugging Face Datasets |
| `pdf` | PDF processing with PyMuPDF and pdfplumber |
| `vision` | Torchvision utilities |
| `audio` | Torchaudio utilities |
| `flash` | FlashDeBERTa support |
| `dev` | pytest and Ruff |
| `all` | All extras listed above |

For example:

```bash
pip install -e ".[demo,pdf]"
```

### Basic Usage

Load a **trained GLiFormer checkpoint** containing an NER head. Replace `path/to/checkpoint` throughout these examples with your checkpoint directory. `from_pretrained` also accepts a Hugging Face repository ID for a compatible GLiFormer checkpoint.

```python
from gliformer import GLiFormer

model = GLiFormer.from_pretrained("path/to/checkpoint", load_tokenizer=True)

text = "Marie Curie worked at the University of Paris in France."
labels = ["person", "organization", "location"]

entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

Entity results contain `text`, `label`, `start`, `end`, and `score`. Character offsets use an exclusive `end`, so `text[start:end]` gives the extracted mention.

Pass a list of texts for batched inference:

```python
entities = model.predict_entities(
    ["Alice works at Acme.", "Bob lives in Berlin."],
    ["person", "organization", "location"],
    batch_size=8,
)
```

Per-task methods return one result for a string input and a list of results for a batch. The examples below reuse `model`; each requires a checkpoint trained for the corresponding task.

## Usage

### Text Classification

Provide candidate labels to classify text:

```python
predictions = model.classify(
    "The new search feature is fast and easy to use.",
    ["positive", "negative", "neutral"],
    threshold=0.5,
)
print(predictions)
```

Use a dictionary of label lists to request named classification groups, such as `{"sentiment": ["positive", "negative"], "topic": ["product", "support"]}`.

### Relation Extraction

The open relation head extracts relation endpoints directly from text using the supplied relation labels:

```python
relations = model.predict_relations(
    "Alice works at Acme and lives in London.",
    ["works_at", "lives_in"],
    threshold=0.5,
)

for relation in relations:
    print(
        relation["head"]["text"],
        "=>", relation["relation"], "=>",
        relation["tail"]["text"],
    )
```

For a checkpoint with a **joint relation head**, supply both entity and relation types through `inference`:

```python
results = model.inference(
    "Alice works at Acme.",
    joint_relations={
        "employment": {
            "entities": ["person", "organization"],
            "relations": ["works_at"],
        }
    },
)
```

### Structured Extraction

Define nested Pydantic models to extract **company → departments → employees** in one call. This example requires a checkpoint trained with multi-level structuring enabled (`structuring_config.multi_level: true`).

```python
import json
from pydantic import BaseModel


class Employee(BaseModel):
    name: str
    role: str


class Department(BaseModel):
    name: str
    employees: list[Employee]


class Company(BaseModel):
    name: str
    departments: list[Department]


text = (
    "At Acme, Engineering includes Alice, a software engineer, and Bob, "
    "a designer. Sales includes Carol, an account manager."
)
records = model.structure(text, {"company": Company}, validate_output=True)
print(json.dumps(records, indent=2))
```

Illustrative output:

```json
{
  "company": [{
    "name": "Acme",
    "departments": [
      {
        "name": "Engineering",
        "employees": [
          {"name": "Alice", "role": "software engineer"},
          {"name": "Bob", "role": "designer"}
        ]
      },
      {
        "name": "Sales",
        "employees": [{"name": "Carol", "role": "account manager"}]
      }
    ]
  }]
}
```

Each employee stays under its department. Results are dictionaries and lists validated against the Pydantic schema; actual predictions depend on the checkpoint and threshold.

### Multi-Task Inference

Construct a reusable schema independently of the model:

```python
from gliformer import GLiFormerSchema

schema = GLiFormerSchema(
    entities=["person", "organization"],
    classes=["business", "sports", "technology"],
    structures={"employee": ["name", "company"]},
)

results = model.inference_from_schema(
    ["Alice joined Acme as a software engineer."],
    schema,
    threshold=0.5,
)

print(results["ner"][0])
print(results["classification"][0])
print(results["structuring"][0])
```

`inference` and `inference_from_schema` return a dictionary keyed by task, with one entry per input text under each key. You can also pass `entities`, `classes`, `relations`, and `structures` directly to `model.inference(...)`.

### Text Embeddings

A checkpoint with an embedding head can produce vectors for similarity and retrieval:

```python
import torch.nn.functional as F

embeddings = model.embed_text([
    "A scientist is working in a laboratory.",
    "A researcher is conducting an experiment.",
])
similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
print(similarity.item())
```

`embed_text` returns a CPU tensor of shape `(number_of_texts, embedding_dimension)`. Bi-encoder configurations also expose `model.embed_labels(labels)`.

## Training

Fine-tune a trained checkpoint with annotated examples using `train_model`. For NER, records use an `extraction` list containing entity mentions and their labels:

```python
from gliformer import GLiFormer

model = GLiFormer.from_pretrained("path/to/checkpoint", load_tokenizer=True)

train_data = [
    {
        "text": "Alice works at Acme.",
        "extraction": [
            {
                "name": "entities",
                "all_labels": ["person", "organization", "location"],
                "ner": [["Alice", "person"], ["Acme", "organization"]],
            }
        ],
    },
    {
        "text": "Bob lives in Berlin.",
        "extraction": [
            {
                "name": "entities",
                "all_labels": ["person", "organization", "location"],
                "ner": [["Bob", "person"], ["Berlin", "location"]],
            }
        ],
    },
]

trainer = model.train_model(
    train_dataset=train_data,
    output_dir="outputs/ner",
    max_steps=100,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
)
model.save_pretrained("outputs/ner/final")
```

This small dataset illustrates the format; use a representative training set and pass held-out examples as `eval_dataset` for your own task. Training records can also carry `classification`, `open_relex`, `structuring`, and `embedding` annotations for the enabled heads. See the [task processors](gliformer/tasks) and [sample records in the test fixtures](tests/conftest.py) for their formats.

Useful training options include `freeze_components=["text_encoder"]`, `train_head_only=True`, and `resume_from_checkpoint="path/to/training-checkpoint"`.

To initialize a new model with a pretrained backbone and new task heads, use `GLiFormer.load_from_config(...)`. It accepts a configuration object, a model configuration dictionary, or a JSON configuration path. The [YAML configurations](configs) contain separate `model`, `data`, and `training` sections; when using them programmatically, parse the YAML and pass its `model` section to `load_from_config`. Newly initialized heads require training before extraction.

## Architectures

`GLiFormer` selects the appropriate wrapper from the checkpoint configuration:

| Variant | Wrapper | Inputs | Example configuration |
| --- | --- | --- | --- |
| Text | `GLiFormerText` | Text and task labels | [NER](configs/ner.yaml) |
| Layout | `GLiFormerLayout` | Text with document geometry | [Layout](configs/layout.yaml) |
| Vision | `GLiFormerVision` | Images with label prompts | [Vision](configs/vision_only.yaml) |
| Audio | `GLiFormerAudio` | Audio with label prompts | [Audio](configs/audio_only.yaml) |
| Omni | `GLiFormerOmni` | Combined text, image, and audio features | [Omni](configs/omni.yaml) |

Text models support joint text/label encoding and separate label encoders. Task heads share encoder representations, while each task has its own processor, neural head, and decoder. Configurable anchors represent entities, relation pairs, or record instances for the corresponding prediction tasks.

The framework also includes instance counting, image and audio classification, object detection, image segmentation, and audio segmentation heads. Use configurations and checkpoints trained for the required modality and task; the text examples above do not activate additional heads in an existing checkpoint.

For implementation details, see the [configuration classes](gliformer/config.py), [model variants](gliformer/model.py), [task modules](gliformer/tasks), and [shared layers](gliformer/layers).

## Demo

Launch the Gradio interface with your checkpoint:

```bash
pip install -e ".[demo,pdf]"
GLIFORMER_MODEL_ID=path/to/checkpoint python demo.py
```

The demo includes annotated examples for NER, classification, relations, structured extraction, and embeddings. Select a checkpoint with the heads needed by the tabs you want to use.

## Evaluation

Run task-specific evaluators from the repository root. For example, evaluate NER on a prepared CrossNER dataset:

```bash
python gliformer_eval/eval_ner.py \
  --model path/to/checkpoint \
  --data path/to/NER \
  --datasets CrossNER_AI CrossNER_literature CrossNER_music CrossNER_politics CrossNER_science \
  --output eval_results/ner.json
```

The NER data directory must contain one subdirectory per dataset with `labels.json` and `test.json`. The evaluator reports strict entity-level precision, recall, and F1 using character spans and entity types.

See the [evaluation guide](gliformer_eval/README.md) for classification, relation extraction, structuring, and similarity. The [benchmark guide](benchmarks/README.md) documents classification and structuring latency, throughput, memory, and profiling measurements.

## Contributing

Bug reports, task examples, and contributions are welcome. For development, install the test and lint tools and run the checks relevant to your changes:

```bash
pip install -e ".[dev]"
python -m pytest tests/processing/test_schema.py tests/processing/test_formatting.py
```

New task implementations follow the processor → head → decoder organization in [gliformer/tasks](gliformer/tasks).

## Acknowledgements

GLiFormer builds on [GLiNER](https://github.com/urchade/GLiNER). Its task implementations also draw on ideas from [GLiNER2](https://github.com/fastino-ai/GLiNER2) and [GLiClass](https://github.com/Knowledgator/GLiClass).
