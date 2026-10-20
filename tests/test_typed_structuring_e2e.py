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
    "due": {"$type": "date", "$stats": ["value", "p10", "p90"]},
    "fair_price": {"$type": "estimate", "$min": 1, "$max": 5000, "$scale": "log"},
}
TEXTS = [
    "Listing posted on March 3, 2024: Sony headphones for $250. Pickup within 2 weeks.",
    "Listing posted on May 10, 2023: oak dining table for $900. Pickup within 2 weeks.",
]
OUTPUTS = [
    {"title": "Sony headphones", "category": ["electronics"], "price": 250,
     "posted": "2024-03-03", "due": {"value": "2024-03-17", "p10": "2024-03-10", "p90": "2024-03-24"},
     "fair_price": {"value": 230, "median": 230, "p10": 180, "p90": 300}},
    {"title": "oak dining table", "category": ["furniture"], "price": 900,
     "posted": "2023-05-10", "due": {"value": "2023-05-24", "p10": "2023-05-17", "p90": "2023-05-31"},
     "fair_price": {"value": 850, "median": 850, "p10": 700, "p90": 1000}},
]


@pytest.mark.parametrize("multi_level", [True, False])
def test_typed_structuring_trains_and_decodes(multi_level):
    from gliformer import GLiFormer
    from gliformer.processing.collator import resolve_gliformer_collator_class

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
            "multi_level": multi_level,
            "typed_fields": True,
            "typed_num_bins": 32,
            "typed_readout_heads": 4,
            "anchor_layer": {"type": "fixed_transformer", "params": {"num_slots": 2}},
            "anchor_objectness": True,
            "represent_spans": False,
        },
    }).float()
    assert model.config.structuring_config.option_token_index >= 0
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
    batch = collator(rows)
    assert batch["structuring_typed_option_count"].tolist() == [3, 3]

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
    result = model.inference(TEXTS, structures=structures, reference_date="2024-01-01")
    for row in result["structuring"]:
        record = row if multi_level else (row["listing"][0] if row["listing"] else {})
        if not record:
            continue
        assert record["category"] is None or isinstance(record["category"], list)
        assert set(record["category"] or []) <= set(TYPES["category"]["$options"])
        if record.get("fair_price") is not None:
            assert set(record["fair_price"]) == {"value", "median", "p10", "p90"}
        if record.get("due") is not None:
            assert set(record["due"]) == {"value", "p10", "p90"}
