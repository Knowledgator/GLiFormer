"""Tests for per-objective focal-control overrides."""

import pytest
import torch

from gliformer.tasks.losses import (
    binary_focal_or_bce,
    focal_override_kwargs,
    with_focal_overrides,
)


class _Config:
    focal_loss_alpha = 0.75
    focal_loss_gamma = -1
    anchor_matching_focal_loss_alpha = 0.9
    anchor_matching_focal_loss_gamma = None
    anchor_matching_focal_loss_prob_margin = None
    anchor_relations_focal_loss_alpha = None
    anchor_relations_focal_loss_gamma = None
    anchor_relations_focal_loss_prob_margin = None
    assignment_focal_loss_alpha = 0.9
    assignment_focal_loss_gamma = None
    assignment_focal_loss_prob_margin = None


def _task_loss_fn(logits, labels, **kwargs):
    """Stand-in for `GLiFormerModel._make_task_loss_fn`'s merge order."""

    merged = {
        "focal_loss_alpha": 0.75,
        "focal_loss_gamma": -1,
        "focal_loss_prob_margin": 0.0,
    }
    merged.update({k: v for k, v in kwargs.items() if v is not None})
    merged["reduction"] = "none"
    return binary_focal_or_bce(logits, labels, **merged)


class TestFocalOverrideKwargs:
    def test_collects_only_set_fields(self):
        assert focal_override_kwargs(_Config, "anchor_matching") == {
            "focal_loss_alpha": 0.9
        }

    def test_unset_objective_is_empty(self):
        assert focal_override_kwargs(_Config, "anchor_relations") == {}

    def test_absent_prefix_is_empty(self):
        assert focal_override_kwargs(_Config, "objectness") == {}

    def test_prefixes_are_independent(self):
        # The three structuring objectives read disjoint fields, so setting one
        # cannot leak into another.
        assert focal_override_kwargs(_Config, "assignment") == {
            "focal_loss_alpha": 0.9
        }
        assert focal_override_kwargs(_Config, "anchor_relations") == {}

    def test_prefix_does_not_match_the_task_level_fields(self):
        # `focal_loss_alpha` itself must never be picked up as an override, or
        # every objective would silently re-apply the task-level value.
        assert "focal_loss_alpha" not in focal_override_kwargs(_Config, "")


class TestWithFocalOverrides:
    def test_empty_overrides_return_the_original_callable(self):
        assert with_focal_overrides(_task_loss_fn, {}) is _task_loss_fn

    def test_none_loss_fn_passes_through(self):
        assert with_focal_overrides(None, {"focal_loss_alpha": 0.9}) is None

    def test_override_is_applied(self):
        logits, labels = torch.randn(6, 4), (torch.rand(6, 4) > 0.7).float()
        wrapped = with_focal_overrides(
            _task_loss_fn, {"focal_loss_alpha": 0.9}
        )
        expected = binary_focal_or_bce(
            logits,
            labels,
            focal_loss_alpha=0.9,
            focal_loss_gamma=-1,
            focal_loss_prob_margin=0.0,
            reduction="none",
        )
        assert torch.allclose(wrapped(logits, labels), expected, atol=1e-7)
        assert not torch.allclose(
            wrapped(logits, labels), _task_loss_fn(logits, labels), atol=1e-6
        )

    def test_unset_controls_still_inherit(self):
        # gamma is not overridden, so the task-level -1 survives and focal
        # modulation stays disabled.
        logits, labels = torch.randn(5, 3), (torch.rand(5, 3) > 0.5).float()
        wrapped = with_focal_overrides(
            _task_loss_fn, {"focal_loss_alpha": 0.9}
        )
        assert torch.allclose(
            wrapped(logits, labels),
            binary_focal_or_bce(
                logits,
                labels,
                focal_loss_alpha=0.9,
                focal_loss_gamma=-1,
                reduction="none",
            ),
            atol=1e-7,
        )

    def test_explicit_call_keyword_wins(self):
        logits, labels = torch.randn(4, 2), (torch.rand(4, 2) > 0.5).float()
        wrapped = with_focal_overrides(
            _task_loss_fn, {"focal_loss_alpha": 0.9}
        )
        assert torch.allclose(
            wrapped(logits, labels, focal_loss_alpha=0.5),
            _task_loss_fn(logits, labels, focal_loss_alpha=0.5),
            atol=1e-7,
        )

    def test_positive_to_negative_weight_ratio_moves(self):
        logit = torch.zeros(1, 1)
        one, zero = torch.ones(1, 1), torch.zeros(1, 1)
        wrapped = with_focal_overrides(
            _task_loss_fn, {"focal_loss_alpha": 0.9}
        )
        base_ratio = _task_loss_fn(logit, one) / _task_loss_fn(logit, zero)
        wrapped_ratio = wrapped(logit, one) / wrapped(logit, zero)
        assert base_ratio.item() == pytest.approx(3.0)
        assert wrapped_ratio.item() == pytest.approx(9.0)
