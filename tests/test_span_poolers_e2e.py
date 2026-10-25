"""Span poolers in real heads: collate, train a few steps, infer.

Skipped unless ``microsoft/deberta-v3-small`` is in the local Hugging Face cache.
"""

import pytest
import torch

from tests.test_typed_structuring_e2e import BACKBONE, TEXTS, TYPES, _batch, _cached

pytestmark = pytest.mark.skipif(not _cached(BACKBONE), reason=f"{BACKBONE} not cached")


def _model(pooler, *, ner=False):
    from gliformer import GLiFormer

    torch.manual_seed(0)
    config = {
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
            "span_pooler": pooler,
            "span_max_tokens": 8,
        },
    }
    if ner:
        config["ner_config"] = {"represent_spans": True, "span_pooler": pooler}
    return GLiFormer.load_from_config(config).float()


def test_marker_keeps_the_historical_parameter_names():
    head = _model("marker").model.heads["structuring"]
    keys = {key for key in head.state_dict() if key.startswith("entity_span_rep_layer.")}
    assert keys and all(key.startswith("entity_span_rep_layer.span_rep_layer.") for key in keys)


def test_token_map_marks_every_subword():
    model = _model("tokens")
    text = "Sony headphones cost 1234.5678 dollars"
    rows = [{
        "text": text,
        "structuring": {"$root": {"title": "Sony headphones", "price": 1234.5678}},
        "structuring_types": {"$root": TYPES},
    }]
    from gliformer.processing.collator import resolve_gliformer_collator_class

    collator = resolve_gliformer_collator_class(model.config)(
        model.config, data_processor=model.data_processor, prepare_labels=True,
    )
    batch = collator(rows)
    full, first = batch["token_words_mask"][0], batch["words_mask"][0]
    assert torch.equal(full.amax(), first.amax())
    # Each word is marked once by ``first`` pooling and on all its subwords here.
    assert (first > 0).sum() == first.amax()
    number_word = text.split().index("1234.5678") + 1
    assert (full == number_word).sum() > 1
    assert (first == number_word).sum() == 1


@pytest.mark.parametrize("pooler", ["marker", "mean", "tokens"])
def test_structuring_trains_and_decodes_with_each_pooler(pooler):
    model = _model(pooler)
    batch = _batch(model, True)
    if pooler == "marker":
        assert "token_words_mask" not in batch
    else:
        assert torch.equal(batch["token_words_mask"].amax(-1), batch["words_mask"].amax(-1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    losses = []
    for _ in range(5):
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    if pooler == "tokens":
        # The token part starts gated off and is learned.
        assert model.model.heads["structuring"].entity_span_rep_layer.token_gate.item() != 0

    model.eval()
    result = model.inference(TEXTS, structures={"$root": dict(TYPES, title="str")})
    assert len(result["structuring"]) == len(TEXTS)


def test_ner_span_loss_with_tokens_pooler():
    model = _model("tokens", ner=True)
    assert type(model.model.heads["ner"].span_rep_layer).__name__ == "TokensSpanPooler"
    rows = [
        {"text": text, "extraction": [{"name": "entities", "ner": [["Listing", "event"], ["$", "currency"]]}]}
        for text in TEXTS
    ]
    from gliformer.processing.collator import resolve_gliformer_collator_class

    collator = resolve_gliformer_collator_class(model.config)(
        model.config, data_processor=model.data_processor, prepare_labels=True,
    )
    batch = collator(rows)
    model.train()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.model.heads["ner"].span_rep_layer.token_gate.grad is not None
