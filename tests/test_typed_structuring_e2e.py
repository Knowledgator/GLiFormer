"""End-to-end typed structuring: collate, train a few steps, infer.

Uses a real tokenizer and backbone, so it is skipped unless
``microsoft/deberta-v3-small`` is already in the local Hugging Face cache.
"""

import pytest
import torch

BACKBONE = "microsoft/deberta-v3-small"


def _cached(repo_id):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    weights = [
        try_to_load_from_cache(repo_id, name)
        for name in ("pytorch_model.bin", "model.safetensors")
    ]
    config = try_to_load_from_cache(repo_id, "config.json")
    return isinstance(config, str) and any(isinstance(path, str) for path in weights)


pytestmark = pytest.mark.skipif(not _cached(BACKBONE), reason=f"{BACKBONE} not cached")

TYPES = {
    "title": {"$type": "text", "$description": "item name"},
    "category": {"$type": "class", "$options": ["electronics", "furniture", "toys"]},
    "price": {"$type": "number", "$unit": "USD"},
    "posted": {"$type": "date"},
}
TEXTS = [
    "Listing posted on March 3, 2024: Sony headphones for $250. Pickup within 2 weeks.",
    "Listing posted on May 10, 2023: oak dining table for $900. Pickup within 2 weeks.",
]
OUTPUTS = [
    {"title": "Sony headphones", "category": ["electronics"], "price": 250, "posted": "2024-03-03"},
    {"title": "oak dining table", "category": ["furniture"], "price": 900, "posted": "2023-05-10"},
]


def _model(multi_level, **structuring):
    from gliformer import GLiFormer

    torch.manual_seed(0)
    return GLiFormer.load_from_config({
        "model_name": BACKBONE,
        "name": "glinext",
        "hidden_size": 128,
        "dropout": 0.0,
        "max_len": 128,
        "max_types": 30,
        "ner_config": None,
        "structuring_config": {
            "multi_level": multi_level,
            "typed_readout_heads": 4,
            "anchor_layer": {"type": "fixed_transformer", "params": {"num_slots": 2}},
            "anchor_objectness": True,
            "represent_spans": False,
            **structuring,
        },
    }).float()


def _batch(model, multi_level):
    from gliformer.processing.collator import resolve_gliformer_collator_class

    name = "$root" if multi_level else "listing"
    rows = [
        {
            "text": text,
            "structuring": {name: dict(output) if multi_level else [dict(output)]},
            "structuring_types": {name: TYPES},
        }
        for text, output in zip(TEXTS, OUTPUTS, strict=True)
    ]
    collator_cls = resolve_gliformer_collator_class(model.config)
    collator = collator_cls(model.config, data_processor=model.data_processor, prepare_labels=True)
    return collator(rows)


@pytest.mark.parametrize("multi_level", [True, False])
def test_typed_structuring_trains_and_decodes(multi_level):
    model = _model(multi_level, typed_fields=True)
    assert model.config.structuring_config.option_token_index >= 0
    batch = _batch(model, multi_level)
    assert batch["structuring_typed_option_count"].tolist() == [3, 3]
    assert not any("quantity" in key for key in batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    losses = []
    for _ in range(4):
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]

    model.eval()
    schema = {key: value for key, value in TYPES.items()}
    schema["title"] = "str"
    structures = {"$root": schema} if multi_level else {"listing": schema}
    result = model.inference(TEXTS, structures=structures)
    for row in result["structuring"]:
        record = row if multi_level else (row["listing"][0] if row["listing"] else {})
        if not record:
            continue
        assert record["category"] is None or isinstance(record["category"], list)
        assert set(record["category"] or []) <= set(TYPES["category"]["$options"])
        # Number/date fields return their span text until the normalization
        # head supplies values.
        for label in ("price", "posted"):
            assert record.get(label) is None or isinstance(record[label], str)


def test_enable_typed_structuring_adds_the_option_scorer():
    model = _model(True)
    head = model.model.heads["structuring"]
    assert not hasattr(head, "option_modeling")
    assert model.enable_typed_structuring(typed_readout_heads=4) is True
    assert model.enable_typed_structuring() is False  # already enabled
    assert hasattr(head, "option_modeling")
    with pytest.raises(ValueError, match="Unknown structuring option"):
        model.enable_typed_structuring(typed_quantity_head="bins")
    batch = _batch(model, True)
    model.train()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("normalization", [
    {"span_encoder": "marker", "token_queries": False},
    {"number": {"heads": ["bins", "sci_digits"]}},
])
def test_normalization_head_trains_and_reads_spans(normalization):
    from gliformer import GLiFormer

    torch.manual_seed(0)
    model = GLiFormer.load_from_config({
        "model_name": BACKBONE,
        "name": "glinext",
        "hidden_size": 128,
        "dropout": 0.0,
        "max_len": 128,
        "max_types": 30,
        "ner_config": None,
        "structuring_config": {
            "multi_level": True,
            "typed_fields": True,
            "typed_readout_heads": 4,
            "anchor_layer": {"type": "fixed_transformer", "params": {"num_slots": 2}},
            "anchor_objectness": True,
            "represent_spans": False,
        },
        "normalization_config": {**normalization, "parser": {"enabled": True}},
    }).float()
    batch = _batch(model, True)
    assert batch["normalization_span_mask"].any()
    assert batch["normalization_year"].max() == 2024

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    losses = []
    for _ in range(6):
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

    model.eval()
    with torch.no_grad():
        output = model(**{key: value for key, value in batch.items() if not key.startswith((
            "normalization_", "structuring_labels", "structuring_span", "structuring_typed_class_t",
            "structuring_typed_class_unit", "structuring_count",
        ))})
    # Without labels the head reads the spans the structuring head picked.
    assert output.normalization_span_idx is not None
    assert output.normalization_span_idx.shape[0] == len(TEXTS)
    if output.normalization_span_idx.shape[1]:
        assert output.normalization_number["bins"].dtype == torch.float64


def test_declared_value_fields_take_head_values_at_inference():
    import re

    from gliformer import GLiFormer

    torch.manual_seed(0)
    model = GLiFormer.load_from_config({
        "model_name": BACKBONE,
        "name": "glinext",
        "hidden_size": 128,
        "dropout": 0.0,
        "max_len": 128,
        "max_types": 30,
        "ner_config": None,
        "structuring_config": {
            "multi_level": True,
            "typed_fields": True,
            "typed_readout_heads": 4,
            "anchor_layer": {"type": "fixed_transformer", "params": {"num_slots": 2}},
            "anchor_objectness": True,
            "represent_spans": False,
        },
        "normalization_config": {"number": {"heads": ["sci_digits"]}},
    }).float()
    model.eval()
    schema = dict(TYPES)
    schema["title"] = "str"
    # An untrained model with a near-zero threshold always picks spans: this
    # checks the decoding path, not the quality of the values.
    result = model.inference(TEXTS, structures={"$root": schema}, threshold=0.001)
    seen = 0
    for record in result["structuring"]:
        if record.get("price") is not None:
            seen += 1
            assert isinstance(record["price"], (int, float))
        if record.get("posted") is not None:
            seen += 1
            assert re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", record["posted"])
        for title in record.get("title") or []:
            assert isinstance(title, str)  # undeclared type: span text
    assert seen, result["structuring"]
