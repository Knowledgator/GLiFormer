from dataclasses import asdict

import pytest
import torch
from unittest.mock import Mock

from gliformer.config import StructuringHeadConfig
from gliformer.tasks.structuring.model import StructuringHead
from tests.heads.conftest import B, make_config


@pytest.mark.parametrize(
    "anchor_mode",
    [
        "position_buckets",
        "topk_norm",
        "topk_distinct",
        "topk_parent",
        "topk_density_distinct",
    ],
)
def test_parameter_free_anchor_modes_run_in_structuring(
    anchor_mode, shared, flat_inputs,
):
    structuring_config = StructuringHeadConfig(
        anchor_mode=anchor_mode,
        num_fixed_slots=3,
    )
    config = make_config(structuring_config=asdict(structuring_config))
    head = StructuringHead.from_config(config)

    output = head(shared, {}, flat_inputs=flat_inputs)

    assert head._record_uses_fixed_slots
    assert list(head.record_anchor_layer.parameters()) == []
    assert output.extra["membership_logits"].shape[:2] == (B, 3)
    assert output.extra["anchor_mask"].shape == (B, 3)
    assert output.extra["anchor_mask"].all()


def test_position_bucket_stabilization_reaches_anchor_and_refinement_layers(
    shared,
    flat_inputs,
):
    structuring_config = StructuringHeadConfig(
        anchor_mode="position_buckets",
        num_fixed_slots=3,
        position_bucket_normalization="center_rms",
        anchor_refine_layers=2,
        anchor_refine_heads=4,
        position_bucket_attention_bias_type="gaussian",
        position_bucket_attention_sigma=0.5,
    )
    config = make_config(structuring_config=asdict(structuring_config))
    head = StructuringHead.from_config(config).eval()
    for block in head.record_anchor_refine.layers:
        block.cross_attn.forward = Mock(wraps=block.cross_attn.forward)

    output = head(shared, {}, flat_inputs=flat_inputs)

    assert output.logits is not None
    assert head.record_anchor_layer.normalization == "center_rms"
    expected = head.record_anchor_refine_positions.position_bucket_attention_bias(
        flat_inputs.words_embedding.new_zeros(B, 3, flat_inputs.words_embedding.shape[-1]),
        flat_inputs.words_embedding,
        memory_mask=flat_inputs.mask,
        sigma=0.5,
    )
    for block in head.record_anchor_refine.layers:
        attention_mask = block.cross_attn.forward.call_args.kwargs["attn_mask"]
        attention_mask = attention_mask.reshape(B, 4, 3, -1)
        torch.testing.assert_close(
            attention_mask,
            expected[:, None].expand_as(attention_mask),
        )
