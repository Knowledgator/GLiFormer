"""Class-field (option) outputs and losses of the structuring head."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from gliformer.config import GLiFormerConfig, StructuringHeadConfig
from gliformer.processing.mappings import (
    BaseClassMapping,
    StructuringItemMapping,
)
from gliformer.processing.structuring_decoder import StructuringDecoder
from gliformer.tasks import SharedRepresentations, TaskFlatInputs
from gliformer.tasks.structuring.model import (
    StructuringHead,
    gold_record_anchors,
    option_loss,
)
from tests.heads.conftest import D, make_config

OPTION_ID = 7
W = 8
C = 3  # fields: 0 span, 1 class, 2 number (a span field)


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
                typed_readout_heads=4,
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
    }


def _typed_labels():
    labels = torch.zeros(1, 1, W, C, 3)
    labels[0, 0, 1, 0, 0] = labels[0, 0, 2, 0, 1] = 1  # span field
    labels[0, 0, 1:3, 0, 2] = 1
    labels[0, 0, 5, 2, :] = 1  # number field span
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
    }


def _bce(predictions, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, labels.to(predictions.dtype), reduction="none",
    )


def test_supervised_option_loss_backpropagates():
    head = _head()
    out = head(
        _shared(), {}, flat_inputs=_flat(), base_loss_fn=_bce,
        **_typed_inputs(), **_typed_labels(),
    )
    assert torch.isfinite(out.extra["typed_class_loss"])
    out.loss.backward()
    grads = [
        parameter.grad
        for module in head._option_modules()
        for parameter in module.parameters()
    ]
    assert grads and all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_inference_outputs_have_slot_shapes():
    head = _head().eval()
    with torch.no_grad():
        out = head(_shared(), {}, flat_inputs=_flat(), **_typed_inputs())
    extra = out.extra
    slots = extra["anchor_mask"].shape[1]
    assert extra["typed_class_logits"].shape == (1, slots, 1, 3)
    assert extra["typed_class_none_logits"].shape == (1, slots, 1)
    assert not any(key.startswith("typed_quantity") for key in extra)


def test_truncated_options_are_not_scored():
    head = _head().eval()
    shared = _shared()
    shared.input_ids[0, 4] = 1  # the prompt lost the third option
    with torch.no_grad():
        out = head(shared, {}, flat_inputs=_flat(), **_typed_inputs())
    logits = out.extra["typed_class_logits"]
    assert torch.isinf(logits[..., 2]).all() and torch.isfinite(logits[..., :2]).all()


def test_typed_disabled_head_has_no_option_parameters():
    config = make_config(structuring_config=asdict(StructuringHeadConfig(num_fixed_slots=2)))
    head = StructuringHead.from_config(config)
    assert not any(name.startswith("option_") for name, _ in head.named_parameters())


def test_typed_head_has_only_option_parameters():
    names = {name.split(".")[0] for name, _ in _head().named_parameters()}
    assert {"option_modeling", "option_none"} <= names
    assert not names & {"typed_head", "typed_regression_head", "typed_bin_head"}


def test_option_loss_ignores_padding_and_unlabelled_units():
    logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
    none = torch.zeros(1, 2, 2)
    targets = torch.zeros(1, 2, 2, 3)
    targets[0, 0, 0, 1] = 1
    option_mask = torch.tensor([[[True, True, True], [True, True, False]]])
    unit = torch.tensor([[[True, False], [False, False]]])
    loss = option_loss(logits, none, targets, option_mask, unit, torch.tensor([[True, False]]), _bce)
    loss.backward()
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[0, 0, 1].abs().sum() == 0
    expected = _bce(torch.zeros(3), torch.tensor([0.0, 1.0, 0.0])).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_record_gathering():
    anchors = torch.arange(6, dtype=torch.float).reshape(1, 3, 2)
    gathered, mask = gold_record_anchors(anchors, torch.ones(1, 3, dtype=torch.bool), [[(2, 0)]], 2, None)
    assert gathered[0, 0].tolist() == [4.0, 5.0] and mask.tolist() == [[True, False]]


@pytest.mark.parametrize("key", ["typed_quantity_head", "typed_num_bins", "typed_year_head"])
def test_removed_quantity_options_are_rejected(key):
    with pytest.raises(TypeError, match=key):
        GLiFormerConfig(structuring_config={"typed_fields": True, key: 1})


def test_decoder_attaches_class_values_and_keeps_value_spans():
    config = make_config(structuring_config=asdict(StructuringHeadConfig()))
    decoder = StructuringDecoder(config)
    field_types = {
        "category": {"$type": "class", "$options": ["a", "b", "c"]},
        "kind": {"$type": "class", "$options": ["x", "y"], "$multi": False},
        "posted": {"$type": "date"},
    }
    item = StructuringItemMapping(
        field_class_to_id=BaseClassMapping(class_to_id={
            "title": 0, "category": 1, "kind": 2, "posted": 3,
        }),
        field_types=field_types,
    )
    output = SimpleNamespace(
        structuring_typed_class_logits=torch.tensor([[[[5.0, -5.0, 5.0], [-5.0, 5.0, float("-inf")]]]]),
        structuring_typed_class_none_logits=torch.tensor([[[0.0, -9.0]]]),
    )
    entries = [{
        "anchor_index": 0,
        "fields": [
            {"field": "title", "start": 0, "end": 0, "text": "Lamp", "score": 0.9},
            {"field": "posted", "start": 1, "end": 3, "text": "March 3 , 2024", "score": 0.9},
        ],
        "presence_is_reliable": True,
    }]
    entries = decoder._attach_typed_fields(
        entries, output, group_idx=0, struct_item=item, objectness_logits=None,
    )
    fields = {field["field"]: field for field in entries[0]["fields"]}
    assert fields["title"]["text"] == "Lamp"
    assert fields["category"]["value"] == ["a", "c"]
    assert fields["kind"]["value"] == "y"
    # Date fields stay spans: their value comes from the normalization head.
    assert fields["posted"]["text"] == "March 3 , 2024" and "value" not in fields["posted"]
