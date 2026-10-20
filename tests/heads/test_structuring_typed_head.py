"""Typed (class/quantity) outputs and losses of the structuring head."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from gliformer.config import StructuringHeadConfig
from gliformer.processing.mappings import (
    BaseClassMapping,
    StructuringItemMapping,
)
from gliformer.processing.structuring_decoder import StructuringDecoder
from gliformer.processing.typed_fields import ORIGIN_ANCHOR, TypedFieldSpec
from gliformer.tasks import SharedRepresentations, TaskFlatInputs
from gliformer.tasks.structuring.model import StructuringHead
from gliformer.tasks.structuring.typed import (
    TypedFieldModule,
    gold_anchor_entities,
    gold_record_anchors,
    select_inference_anchors,
)
from tests.heads.conftest import D, make_config

OPTION_ID = 7
W = 8
C = 3  # fields: 0 span, 1 class, 2 quantity
K_BINS = 16


def _head(**overrides):
    config = make_config(
        structuring_config=asdict(
            StructuringHeadConfig(
                anchor_mode="fixed_transformer",
                num_fixed_slots=2,
                max_count=4,
                anchor_num_heads=4,
                anchor_num_layers=1,
                anchor_objectness=True,
                typed_fields=True,
                typed_num_bins=K_BINS,
                typed_readout_heads=4,
                typed_max_horizon=4,
                **overrides,
            )
        )
    )
    # Normally assigned by set_class_indices once the tokenizer grows.
    config.structuring_config.option_token_index = OPTION_ID
    return StructuringHead.from_config(config)


def _shared():
    input_ids = torch.ones(1, 12, dtype=torch.long)
    input_ids[0, 2:5] = OPTION_ID  # three options
    return SharedRepresentations(
        token_embeds=torch.randn(1, 12, D),
        input_ids=input_ids,
        attention_mask=torch.ones(1, 12, dtype=torch.long),
        words_embedding=torch.randn(1, W, D),
        mask=torch.ones(1, W, dtype=torch.long),
        prompts_embedding=torch.randn(1, C, D),
        prompts_embedding_mask=torch.ones(1, C, dtype=torch.long),
    )


def _flat():
    return TaskFlatInputs(
        words_embedding=torch.randn(1, W, D),
        mask=torch.ones(1, W, dtype=torch.long),
        parent_embedding=torch.randn(1, D),
        child_embedding=torch.randn(1, C, D),
        child_mask=torch.ones(1, C, dtype=torch.long),
        batch_origin=torch.zeros(1, dtype=torch.long),
    )


def _typed_inputs():
    return {
        "structuring_typed_class_field_idx": torch.tensor([[1]]),
        "structuring_typed_class_field_mask": torch.tensor([[True]]),
        "structuring_typed_class_multi": torch.tensor([[True]]),
        "structuring_typed_option_index": torch.tensor([[[0, 1, 2]]]),
        "structuring_typed_option_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "structuring_typed_option_count": torch.tensor([3]),
        "structuring_typed_quantity_field_idx": torch.tensor([[2]]),
        "structuring_typed_quantity_field_mask": torch.tensor([[True]]),
        "structuring_typed_quantity_horizon": torch.tensor([[1]]),
        "structuring_typed_quantity_bounded": torch.tensor([[False]]),
        "structuring_typed_quantity_reference": torch.tensor([[False]]),
    }


def _typed_labels():
    labels = torch.zeros(1, 1, W, C, 3)
    labels[0, 0, 1, 0, 0] = labels[0, 0, 2, 0, 1] = 1  # span field
    labels[0, 0, 1:3, 0, 2] = 1
    labels[0, 0, 5, 2, :] = 1  # quantity anchor span
    span_idx = torch.tensor([[[1, 2], [5, 5]]])
    span_labels = torch.zeros(1, 2, 1, C)
    span_labels[0, 0, 0, 0] = 1
    span_labels[0, 1, 0, 2] = 1
    return {
        "structuring_labels": labels,
        "structuring_count": torch.tensor([1]),
        "structuring_span_idx": span_idx,
        "structuring_span_mask": torch.ones(1, 2, dtype=torch.bool),
        "structuring_span_labels": span_labels,
        "structuring_typed_bio_field_mask": torch.ones(1, C, dtype=torch.bool),
        "structuring_typed_class_targets": torch.tensor([[[[0.0, 1.0, 0.0]]]]),
        "structuring_typed_class_unit_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "structuring_typed_quantity_levels": torch.tensor([[[[[0.5]]]]]),
        "structuring_typed_quantity_values": torch.tensor([[[[[0.5]]]]]),
        "structuring_typed_quantity_point_mask": torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
        "structuring_typed_quantity_point": torch.tensor([[[[0.5]]]]),
        "structuring_typed_quantity_point_value_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "structuring_typed_quantity_presence": torch.ones(1, 1, 1),
        "structuring_typed_quantity_presence_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "structuring_typed_quantity_origin": torch.tensor([[[ORIGIN_ANCHOR]]]),
    }


def _bce(predictions, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, labels.to(predictions.dtype), reduction="none",
    )


def test_supervised_typed_losses_backpropagate():
    head = _head()
    out = head(
        _shared(), {}, flat_inputs=_flat(), base_loss_fn=_bce,
        **_typed_inputs(), **_typed_labels(),
    )
    for key in ("typed_class_loss", "typed_quantity_loss", "typed_point_loss", "typed_presence_loss"):
        assert torch.isfinite(out.extra[key])
    out.loss.backward()
    grads = [
        parameter.grad
        for name, parameter in head.typed_head.named_parameters()
        if name.startswith(("class_", "quantity_head", "anchor_proj", "presence_head"))
    ]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_inference_outputs_have_slot_shapes():
    head = _head().eval()
    with torch.no_grad():
        out = head(_shared(), {}, flat_inputs=_flat(), **_typed_inputs())
    extra = out.extra
    slots = extra["anchor_mask"].shape[1]
    assert extra["typed_class_logits"].shape == (1, slots, 1, 3)
    assert extra["typed_quantity_probs"].shape == (1, slots, 1, 1, K_BINS)
    assert torch.allclose(extra["typed_quantity_probs"].sum(-1), torch.ones(1, slots, 1, 1))
    assert extra["typed_quantity_origin"].shape == (1, slots, 1)


def test_truncated_options_are_not_scored():
    head = _head().eval()
    shared = _shared()
    shared.input_ids[0, 4] = 1  # the prompt lost the third option
    with torch.no_grad():
        out = head(shared, {}, flat_inputs=_flat(), **_typed_inputs())
    logits = out.extra["typed_class_logits"]
    assert torch.isinf(logits[..., 2]).all() and torch.isfinite(logits[..., :2]).all()


def test_typed_disabled_head_has_no_typed_parameters():
    config = make_config(structuring_config=asdict(StructuringHeadConfig(num_fixed_slots=2)))
    head = StructuringHead.from_config(config)
    assert not any("typed_head" in name for name, _ in head.named_parameters())


def test_class_loss_ignores_padding_and_unlabelled_units():
    module = TypedFieldModule(D, 0.0, num_bins=8, num_heads=4, max_horizon=2)
    logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
    none = torch.zeros(1, 2, 2)
    targets = torch.zeros(1, 2, 2, 3)
    targets[0, 0, 0, 1] = 1
    option_mask = torch.tensor([[[True, True, True], [True, True, False]]])
    unit = torch.tensor([[[True, False], [False, False]]])
    loss = module.class_loss(logits, none, targets, option_mask, unit, torch.tensor([[True, False]]), _bce)
    loss.backward()
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[0, 0, 1].abs().sum() == 0
    expected = _bce(torch.zeros(3), torch.tensor([0.0, 1.0, 0.0])).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_record_and_anchor_gathering():
    anchors = torch.arange(6, dtype=torch.float).reshape(1, 3, 2)
    gathered, mask = gold_record_anchors(anchors, torch.ones(1, 3, dtype=torch.bool), [[(2, 0)]], 2, None)
    assert gathered[0, 0].tolist() == [4.0, 5.0] and mask.tolist() == [[True, False]]
    span_labels = torch.zeros(1, 3, 2, 4)
    span_labels[0, 2, 1, 3] = 1
    index, present = gold_anchor_entities(span_labels, torch.tensor([[3, 0]]), 2)
    assert present.tolist() == [[[False, False], [True, False]]]
    assert index[0, 1, 0].item() == 2

    field_logits = torch.full((1, 2, 4), -5.0)
    field_logits[0, 1, 3] = 5.0
    membership = torch.full((1, 2, 2), -5.0)
    membership[0, 0, 1] = 5.0
    entity, ok = select_inference_anchors(
        field_logits, membership, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[3]]), 0.5,
    )
    assert entity.tolist() == [[[1], [-1]]] and ok.tolist() == [[[True], [False]]]


def test_decoder_attaches_typed_values():
    config = make_config(structuring_config=asdict(StructuringHeadConfig()))
    decoder = StructuringDecoder(config)
    field_types = {
        "category": {"$type": "class", "$options": ["a", "b", "c"]},
        "kind": {"$type": "class", "$options": ["x", "y"], "$multi": False},
        "posted": {"$type": "date"},
        "price": {"$type": "estimate", "$min": 0, "$max": 100},
    }
    item = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(class_to_id={
            "title": 0, "category": 1, "kind": 2, "posted": 3, "price": 4,
        }),
        field_types=field_types,
    )
    bins = 32
    date = TypedFieldSpec(type="date")
    origin = date.encode("2024-03-03")
    u = date.to_unit(origin + 14, origin)
    date_probs = torch.zeros(bins)
    date_probs[int(u * bins)] = 1
    date_offsets = torch.full((bins,), 0.5)
    date_offsets[int(u * bins)] = u * bins - int(u * bins)
    price_probs = torch.zeros(bins)
    price_probs[15:17] = 0.5
    output = SimpleNamespace(
        structuring_typed_class_logits=torch.tensor([[[[5.0, -5.0, 5.0], [-5.0, 5.0, float("-inf")]]]]),
        structuring_typed_class_none_logits=torch.tensor([[[0.0, -9.0]]]),
        structuring_typed_quantity_probs=torch.stack([date_probs, price_probs]).reshape(1, 1, 2, 1, bins),
        structuring_typed_quantity_offsets=torch.stack(
            [date_offsets, torch.full((bins,), 0.5)]
        ).reshape(1, 1, 2, 1, bins),
        structuring_typed_quantity_presence_logits=torch.tensor([[[5.0, 5.0]]]),
        structuring_typed_quantity_anchor_entity=torch.tensor([[[0, -1]]]),
        structuring_typed_quantity_origin=torch.tensor([[[ORIGIN_ANCHOR, 0]]]),
    )
    entries = [{
        "anchor_index": 0,
        "fields": [
            {"field": "title", "start": 0, "end": 0, "text": "Lamp", "score": 0.9},
            {"field": "posted", "start": 1, "end": 3, "text": "March 3 , 2024", "score": 0.9},
        ],
        "presence_is_reliable": True,
    }]
    decoder.resolve_span_text = lambda texts, index, start, end: "March 3 , 2024"
    entries = decoder._attach_typed_fields(
        entries, output, group_idx=0, text_idx=0, struct_item=item,
        reference_date=None, texts=None, span_idx=torch.tensor([[[1, 3]]]),
        objectness_logits=None,
    )
    values = {field["field"]: field.get("value", field.get("text")) for field in entries[0]["fields"]}
    assert values["title"] == "Lamp"
    assert values["category"] == ["a", "c"]
    assert values["kind"] == "y"
    assert values["posted"] == "2024-03-17"
    assert 45 <= values["price"]["median"] <= 55
