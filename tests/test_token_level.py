"""Token-level text units (``text_unit="token"``).

Pure unit tests always run; tests with a real tokenizer or backbone are
skipped unless ``microsoft/deberta-v3-small`` is in the local cache.
"""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import GLiFormerConfig, NERHeadConfig
from gliformer.processing.token_units import (
    TOKEN_UNIT_MARKER,
    WordIndexConverter,
    build_token_units,
    word_snapped_maps,
)
from tests.conftest import make_config

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


needs_backbone = pytest.mark.skipif(not _cached(BACKBONE), reason=f"{BACKBONE} not cached")


# ── Pure helpers ───────────────────────────────────────────────────────


def test_units_trim_whitespace_and_merge_byte_pieces():
    text = "New  York é"
    # ids: "New", " ", " York", two byte pieces of "é" sharing its offsets.
    offsets = [(0, 3), (3, 4), (4, 9), (9, 11), (10, 11)]
    encoding = build_token_units(text, [1, 2, 3, 4, 5], offsets)
    assert encoding.units == (("New", 0, 3), ("York", 5, 9), ("é", 10, 11))
    assert encoding.id_units == (0, -1, 1, 2, 2)
    assert encoding.unit_first_id == (0, 2, 3)
    assert encoding.id_count(1) == 2
    assert encoding.id_count(3) == 5


def test_word_snapped_maps_widen_units_to_words():
    units = [("Bar", 0, 3), ("ack", 3, 6), ("Obama", 7, 12)]
    words = [("Barack", 0, 6), ("Obama", 7, 12)]
    starts, ends = word_snapped_maps(units, words)
    assert starts == [0, 0, 7]
    assert ends == [6, 6, 12]


def test_word_index_converter_rewrites_ner_and_span_dicts():
    grid = [("Angela", 0, 6), ("Merkel", 7, 13), ("met", 14, 17)]
    converter = WordIndexConverter(grid)
    assert converter.ner_entity([0, 1, "person"]) == [0, 13, "person"]
    assert converter.ner_entity(["Merkel", 1, 1, "person"]) == ["Merkel", 7, 13, "person"]
    assert converter.ner_entity([5, 6, "person"]) == [-1, -1, "person"]
    payload = {"g": [{"name": {"text": "Angela Merkel", "start": 0, "end": 1}}]}
    assert converter.span_dicts(payload)["g"][0]["name"] == {"text": "Angela Merkel", "start": 0, "end": 13}


def test_config_validates_and_serializes_text_unit():
    config = make_config(text_unit="token", span_boundary="word")
    assert config.text_unit == "token"
    data = config.to_dict()
    assert data["text_unit"] == "token" and data["span_boundary"] == "word"
    assert GLiFormerConfig(**data).text_unit == "token"
    assert make_config().text_unit == "word"
    with pytest.raises(ValueError, match="text_unit"):
        make_config(text_unit="char")
    with pytest.raises(ValueError, match="span_boundary"):
        make_config(span_boundary="sentence")


# ── Processor with a real tokenizer ────────────────────────────────────


def _processor(max_len=512, **overrides):
    from transformers import AutoTokenizer

    from gliformer.processing.collator import GLiFormerTextDataCollator
    from gliformer.processing.processor import GLiFormerTextProcessor

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    overrides.setdefault("ner_config", asdict(NERHeadConfig()))
    config = make_config(text_unit="token", max_len=max_len, **overrides)
    processor = GLiFormerTextProcessor(config, tokenizer, None)
    collator = GLiFormerTextDataCollator(
        config, data_processor=processor, prepare_labels=True, return_tokens=True,
    )
    return tokenizer, processor, collator


def _ner_row(text, ner, **extra):
    return {
        "text": text,
        "extraction": [{"name": "e", "all_labels": ["person", "city", "money"], "ner": ner}],
        **extra,
    }


def _label_spans(batch, row):
    labels = batch["ner_labels"][row]
    starts = torch.nonzero(labels[:, :, 0]).tolist()
    ends = torch.nonzero(labels[:, :, 1]).tolist()
    return starts, ends


@needs_backbone
def test_text_is_encoded_from_raw_string_with_one_position_per_unit():
    tokenizer, processor, collator = _processor()
    text = "Ünïcödé café in Tokyo costs $1,000.50"
    batch = collator([_ner_row(text, [["Tokyo", "city"], ["$1,000.50", "money"]])])
    ids = batch["input_ids"][0].tolist()
    raw = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert any(ids[k:k + len(raw)] == raw for k in range(len(ids)))

    units = batch["tokens"][0]
    assert len(units) > len(text.split())
    words_mask = batch["words_mask"][0]
    assert sorted(words_mask[words_mask > 0].tolist()) == list(range(1, len(units) + 1))
    assert batch["text_lengths"].view(-1).tolist() == [len(units)]

    starts, ends = _label_spans(batch, 0)
    money = [index for index, (_s, label) in enumerate(starts) if label == 2]
    assert money, starts
    assert units.span_text(starts[money[0]][0], ends[money[0]][0]) == "$1,000.50"


@needs_backbone
def test_word_index_ner_is_converted_on_the_word_grid():
    _tokenizer, _processor_, collator = _processor()
    row = _ner_row(
        "Angela Merkel met Macronovich.",
        [[0, 1, "person"], [3, 3, "person"]],
        tokenized_text=["Angela", "Merkel", "met", "Macronovich", "."],
    )
    batch = collator([row])
    units = batch["tokens"][0]
    starts, ends = _label_spans(batch, 0)
    spans = sorted(units.span_text(s[0], e[0]) for s, e in zip(starts, ends, strict=True))
    assert spans == ["Angela Merkel", "Macronovich"]
    assert row[TOKEN_UNIT_MARKER] == "token"


@needs_backbone
def test_resolved_word_spans_without_a_word_grid_are_rejected():
    from gliformer.config import StructuringHeadConfig

    _tokenizer, _processor_, collator = _processor(
        ner_config=None, structuring_config=asdict(StructuringHeadConfig()),
    )
    row = {
        "text": "Alice works at Acme",
        "tokenized_text": ["Alicia", "works"],
        "structuring": {"person": [{"name": {"text": "Alice", "start": 0, "end": 0}}]},
        "_gliformer_structuring_spans_resolved": True,
    }
    with pytest.raises(ValueError, match="word indices"):
        collator([row])


@needs_backbone
def test_budget_truncates_text_at_a_unit_boundary():
    _tokenizer, _processor_, collator = _processor(max_len=24)
    batch = collator([_ner_row("Barack Obama visited Tokyo and Osaka last spring with friends.", [])])
    assert batch["input_ids"].shape[1] <= 24
    kept = int(batch["text_lengths"].item())
    assert 0 < kept < len(batch["tokens"][0])
    assert int(batch["words_mask"].max()) == kept


# ── End to end ─────────────────────────────────────────────────────────


def _model(**overrides):
    from gliformer import GLiFormer

    torch.manual_seed(0)
    config = {
        "model_name": BACKBONE,
        "name": "glinext",
        "hidden_size": 128,
        "dropout": 0.0,
        "max_len": 128,
        "max_types": 30,
        "max_width": 30,
        "text_unit": "token",
        "ner_config": {},
    }
    config.update(overrides)
    return GLiFormer.load_from_config(config).float()


def _train(model, rows, steps=4, lr=1e-4):
    from gliformer.processing.collator import resolve_gliformer_collator_class

    collator = resolve_gliformer_collator_class(model.config)(
        model.config, data_processor=model.data_processor, prepare_labels=True,
    )
    batch = collator([dict(row) for row in rows])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return batch, losses


@needs_backbone
def test_ner_trains_and_returns_character_offsets():
    model = _model()
    rows = [
        _ner_row("Barack Obama visited Tokyo.", [["Barack Obama", "person"], ["Tokyo", "city"]]),
        _ner_row("Angela Merkel met Macron.", [["Angela Merkel", "person"], ["Macron", "person"]]),
    ]
    _batch, losses = _train(model, rows)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]

    texts = ["Ünïcödé met Barack Obama.", "Macron visited Tokyo."]
    result = model.inference(texts, entities=["person", "city"], threshold=0.0)
    seen = 0
    for text, entities in zip(texts, result["ner"], strict=True):
        for entity in entities:
            seen += 1
            assert text[entity["start"]:entity["end"]] == entity["text"]
    assert seen


@needs_backbone
def test_word_boundary_mode_returns_whole_words():
    model = _model(span_boundary="word")
    text = "Ünïcödé met Barack Obama."
    words = {(start, start + len(word)) for start, word in _word_offsets(text)}
    result = model.inference([text], entities=["person"], threshold=0.0)
    entities = result["ner"][0]
    assert entities
    for entity in entities:
        assert text[entity["start"]:entity["end"]] == entity["text"]
        assert any(start == entity["start"] for start, _ in words)
        assert any(end == entity["end"] for _, end in words)


def _word_offsets(text):
    import re

    return [(match.start(), match.group()) for match in re.finditer(r"\w+(?:[-_]\w+)*|\S", text)]


@needs_backbone
def test_typed_structuring_with_normalization_trains_in_token_mode():
    model = _model(
        ner_config=None,
        structuring_config={
            "multi_level": True,
            "typed_fields": True,
            "typed_readout_heads": 4,
            "anchor_layer": {"type": "fixed_transformer", "params": {"num_slots": 2}},
            "anchor_objectness": True,
            "represent_spans": False,
        },
        normalization_config={
            "span_encoder": "tokens",
            "token_queries": True,
            "number": {"heads": ["sci_digits"]},
            "parser": {"enabled": True},
        },
    )
    types = {
        "title": {"$type": "text"},
        "price": {"$type": "number", "$unit": "USD"},
        "posted": {"$type": "date"},
    }
    rows = [
        {
            "text": "Listing posted on March 3, 2024: Sony headphones for $1,250.",
            "structuring": {"$root": {"title": "Sony headphones", "price": 1250, "posted": "2024-03-03"}},
            "structuring_types": {"$root": types},
        },
    ]
    batch, losses = _train(model, rows, steps=4, lr=3e-4)
    assert batch["normalization_span_mask"].any()
    # Every subword of a unit is mapped, prompt and special tokens are not.
    token_words = batch["token_words_mask"]
    assert token_words.shape == batch["input_ids"].shape
    assert int(token_words.max()) == int(batch["text_lengths"].max())
    assert (token_words > 0).sum() >= (batch["words_mask"] > 0).sum()
    assert all(torch.isfinite(torch.tensor(losses)))

    schema = dict(types)
    schema["title"] = "str"
    result = model.inference([rows[0]["text"]], structures={"$root": schema}, threshold=0.001)
    record = result["structuring"][0]
    for title in record.get("title") or []:
        assert title in rows[0]["text"]
