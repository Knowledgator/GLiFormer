"""Tests for multi-level structuring anchor adjacency."""

from dataclasses import asdict

import pytest
import torch
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from torch import nn

from gliformer.config import StructuringHeadConfig
from gliformer.layers.structuring_relations import (
    anchor_relation_loss,
    remap_anchor_relation_targets,
)
from gliformer.tasks.losses import binary_focal_or_bce
from gliformer.tasks.structuring.model import StructuringHead
from tests.heads.conftest import make_config


def test_structuring_head_uses_gliner_relations_rep_layer():
    config = make_config(structuring_config=asdict(
        StructuringHeadConfig(
            multi_level=True,
            anchor_mode="fixed_transformer",
            num_fixed_slots=2,
            anchor_num_heads=4,
            anchor_num_layers=1,
        )
    ))

    head = StructuringHead.from_config(config)

    assert isinstance(
        head.anchor_relations_rep_layer,
        RelationsRepLayer,
    )


def test_structuring_head_emits_scores_and_trains_relation_loss(
    shared, flat_inputs,
):
    config = make_config(structuring_config=asdict(StructuringHeadConfig(
        multi_level=True,
        anchor_mode="fixed_transformer",
        num_fixed_slots=2,
        anchor_num_heads=4,
        anchor_num_layers=1,
    )))
    head = StructuringHead.from_config(config)
    batch_size, sequence_length = flat_inputs.words_embedding.shape[:2]
    fields = flat_inputs.child_embedding.shape[1]
    anchors = 2
    spans = torch.tensor(
        [[[0, 0], [1, 1]], [[0, 0], [1, 1]]],
        dtype=torch.long,
    )
    span_mask = torch.ones(batch_size, 2, dtype=torch.bool)
    labels = torch.zeros(
        batch_size,
        anchors,
        sequence_length,
        fields,
        3,
    )
    labels[:, 0, 0, 0] = 1.0
    labels[:, 1, 1, 0] = 1.0
    span_labels = torch.zeros(batch_size, 2, anchors, fields)
    span_labels[:, 0, 0, 0] = 1.0
    span_labels[:, 1, 1, 0] = 1.0
    relation_labels = torch.zeros(batch_size, anchors, anchors)
    relation_labels[:, 0, 1] = 1.0

    output = head(
        shared,
        {},
        flat_inputs=flat_inputs,
        structuring_labels=labels,
        structuring_count=torch.full((batch_size,), 2),
        structuring_span_idx=spans,
        structuring_span_mask=span_mask,
        structuring_span_labels=span_labels,
        structuring_relation_labels=relation_labels,
        structuring_relation_group_mask=torch.ones(
            batch_size, dtype=torch.bool
        ),
        base_loss_fn=binary_focal_or_bce,
    )

    assert output.extra["anchor_relation_scores"].shape == (
        batch_size,
        anchors,
        anchors,
    )
    assert output.extra["anchor_relation_loss"] is not None
    assert torch.isfinite(output.loss)


def test_structuring_head_emits_anchor_relation_scores(shared, flat_inputs):
    config = make_config(
        structuring_config=asdict(StructuringHeadConfig(
            multi_level=True,
            anchor_mode="fixed_transformer",
            num_fixed_slots=2,
            anchor_num_heads=4,
            anchor_num_layers=1,
        )),
    )
    head = StructuringHead.from_config(config)

    output = head(shared, {}, flat_inputs=flat_inputs)

    assert output.extra["anchor_relation_scores"].shape == (
        flat_inputs.words_embedding.shape[0],
        2,
        2,
    )


def test_structuring_head_compacts_relation_scoring_to_active_anchors():
    config = make_config(
        structuring_config=asdict(StructuringHeadConfig(
            multi_level=True,
            anchor_mode="fixed_transformer",
            num_fixed_slots=2,
            anchor_num_heads=4,
            anchor_num_layers=1,
        )),
    )
    head = StructuringHead.from_config(config)

    class RecordingRelations(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_shape = None

        def forward(self, anchors, mask):
            self.input_shape = anchors.shape
            return anchors.new_ones(
                anchors.shape[0],
                anchors.shape[1],
                anchors.shape[1],
            )

    scorer = RecordingRelations()
    head.anchor_relations_rep_layer = scorer
    anchors = torch.randn(2, 5, config.hidden_size)
    mask = torch.tensor([
        [True, False, True, False, False],
        [False, False, False, False, False],
    ])

    scores = head._score_anchor_relations(anchors, mask)

    assert scorer.input_shape == (1, 2, config.hidden_size)
    assert scores.shape == (2, 5, 5)
    assert scores[0].nonzero().tolist() == [
        [0, 0], [0, 2], [2, 0], [2, 2],
    ]
    assert not scores[1].any()


def test_relation_targets_apply_hungarian_assignment_to_both_axes():
    scores = torch.zeros(1, 3, 3)
    labels = torch.zeros(1, 2, 2)
    labels[0, 0, 1] = 1.0

    targets, pair_mask = remap_anchor_relation_targets(
        scores,
        labels,
        torch.ones(1, 3, dtype=torch.bool),
        relation_group_mask=torch.tensor([True]),
        anchor_matches=[[(2, 0), (0, 1)]],
    )

    assert targets[0, 2, 0] == 1.0
    assert targets.sum() == 1.0
    assert pair_mask.sum() == 2
    assert pair_mask.nonzero().tolist() == [[0, 0, 2], [0, 2, 0]]
    assert not pair_mask[0].diagonal().any()


def test_relation_targets_without_assignment_keep_all_active_endpoints():
    scores = torch.zeros(1, 3, 3)
    labels = torch.zeros(1, 2, 2)
    labels[0, 0, 1] = 1.0

    targets, pair_mask = remap_anchor_relation_targets(
        scores,
        labels,
        torch.ones(1, 3, dtype=torch.bool),
        relation_group_mask=torch.tensor([True]),
        label_count=torch.tensor([2]),
    )

    assert targets[0, 0, 1] == 1.0
    assert pair_mask.sum() == 6


def test_relation_group_and_padding_masks_remove_all_invalid_pairs():
    scores = torch.zeros(2, 3, 3)
    labels = torch.zeros(2, 3, 3)
    _, pair_mask = remap_anchor_relation_targets(
        scores,
        labels,
        torch.tensor([[True, True, False], [True, True, True]]),
        relation_group_mask=torch.tensor([True, False]),
        anchor_matches=[[(0, 0), (1, 1)], [(0, 0)]],
    )

    assert pair_mask[0].sum() == 2
    assert pair_mask[1].sum() == 0


@pytest.mark.parametrize("anchor_count", [2, 3, 5])
def test_relation_loss_is_normalized_by_directed_non_self_pairs(anchor_count):
    observed = {}

    def unit_loss(scores, targets, *, normalize_prob):
        del targets
        observed["normalize_prob"] = normalize_prob
        return torch.ones_like(scores)

    scores = torch.zeros(1, anchor_count, anchor_count)
    loss = anchor_relation_loss(
        scores,
        torch.zeros_like(scores),
        torch.ones(1, anchor_count, dtype=torch.bool),
        base_loss_fn=unit_loss,
    )
    _, pair_mask = remap_anchor_relation_targets(
        scores,
        torch.zeros_like(scores),
        torch.ones(1, anchor_count, dtype=torch.bool),
    )

    assert pair_mask.sum().item() == anchor_count * (anchor_count - 1)
    assert loss.item() == pytest.approx(1.0)
    assert observed == {"normalize_prob": False}


def test_directed_probability_loss_prefers_the_correct_edge_direction():
    labels = torch.zeros(1, 2, 2)
    labels[0, 0, 1] = 1.0
    correct = torch.tensor([[[0.0, 0.9], [0.1, 0.0]]])
    reversed_scores = torch.tensor([[[0.0, 0.1], [0.9, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    kwargs = dict(
        relation_labels=labels,
        anchor_mask=mask,
        base_loss_fn=binary_focal_or_bce,
        relation_group_mask=torch.tensor([True]),
        anchor_matches=[[(0, 0), (1, 1)]],
    )

    correct_loss = anchor_relation_loss(correct, **kwargs)
    reversed_loss = anchor_relation_loss(reversed_scores, **kwargs)

    assert correct_loss < reversed_loss


def test_single_anchor_zero_edge_loss_is_finite_and_differentiable():
    scores = torch.tensor([[[0.5]]], requires_grad=True)
    loss = anchor_relation_loss(
        scores,
        torch.zeros_like(scores),
        torch.ones(1, 1, dtype=torch.bool),
        relation_group_mask=torch.tensor([True]),
        anchor_matches=[[(0, 0)]],
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert scores.grad is not None


def test_zero_anchor_relation_loss_is_finite_and_differentiable():
    scores = torch.empty(1, 0, 0, requires_grad=True)
    loss = anchor_relation_loss(
        scores,
        torch.empty_like(scores),
        torch.empty(1, 0, dtype=torch.bool),
        relation_group_mask=torch.tensor([True]),
        anchor_matches=[[]],
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert scores.grad is not None


@pytest.mark.parametrize(
    ("labels", "mask", "matches", "message"),
    [
        (
            torch.zeros(2, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
            None,
            "same batch size",
        ),
        (
            torch.zeros(1, 2, 2),
            torch.ones(1, 3, dtype=torch.bool),
            None,
            "anchor_mask",
        ),
        (
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
            [],
            "anchor_matches",
        ),
    ],
)
def test_relation_target_validation_fails_with_actionable_errors(
    labels, mask, matches, message,
):
    with pytest.raises(ValueError, match=message):
        remap_anchor_relation_targets(
            torch.zeros(1, 2, 2),
            labels,
            mask,
            anchor_matches=matches,
        )


def test_relation_loss_does_not_hide_internal_type_errors():
    def broken_loss(scores, targets, *, normalize_prob):
        del scores, targets, normalize_prob
        raise TypeError("loss implementation failed")

    with pytest.raises(TypeError, match="loss implementation failed"):
        anchor_relation_loss(
            torch.full((1, 2, 2), 0.5),
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
            base_loss_fn=broken_loss,
        )


def test_relation_loss_rejects_reduced_custom_loss():
    def reduced_loss(scores, targets, *, normalize_prob):
        del targets, normalize_prob
        return scores.mean()

    with pytest.raises(ValueError, match="reduction='none'"):
        anchor_relation_loss(
            torch.full((1, 2, 2), 0.5),
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
            base_loss_fn=reduced_loss,
        )


def test_relation_loss_declares_scores_are_already_probabilities():
    observed = {}

    def recording_loss(scores, targets, *, normalize_prob):
        observed["normalize_prob"] = normalize_prob
        return (scores - targets).square()

    anchor_relation_loss(
        torch.full((1, 2, 2), 0.5),
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
        base_loss_fn=recording_loss,
    )

    assert observed == {"normalize_prob": False}


def test_relation_loss_accepts_independent_focal_overrides():
    observed = {}

    def recording_loss(
        scores,
        targets,
        *,
        normalize_prob,
        focal_loss_alpha,
        focal_loss_gamma,
        focal_loss_prob_margin,
    ):
        observed.update(
            normalize_prob=normalize_prob,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_prob_margin=focal_loss_prob_margin,
        )
        return (scores - targets).square()

    anchor_relation_loss(
        torch.full((1, 2, 2), 0.5),
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
        base_loss_fn=recording_loss,
        focal_loss_alpha=0.4,
        focal_loss_gamma=1.5,
        focal_loss_prob_margin=0.1,
    )

    assert observed == {
        "normalize_prob": False,
        "focal_loss_alpha": 0.4,
        "focal_loss_gamma": 1.5,
        "focal_loss_prob_margin": 0.1,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("anchor_relations_focal_loss_alpha", 1.1),
        ("anchor_relations_focal_loss_gamma", float("nan")),
        ("anchor_relations_focal_loss_prob_margin", float("inf")),
    ],
)
def test_relation_focal_configuration_is_validated(name, value):
    with pytest.raises(ValueError, match=name):
        StructuringHeadConfig(**{name: value})
