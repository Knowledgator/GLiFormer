"""Tests for entity-conditioned anchor refinement."""

import pytest
import torch

from gliformer.layers.anchor_entity_refinement import (
    AnchorEntityRefinement,
    MLPAnchorEntityRefinement,
    NoAnchorEntityRefinement,
    membership_selection_weights,
    pool_claimed_entities,
)

D = 8
B = 2
A = 4
E = 5


def _inputs(seed=0):
    torch.manual_seed(seed)
    anchors = torch.randn(B, A, D)
    entities = torch.randn(B, E, D)
    anchor_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    entity_mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool
    )
    # Mirror ``MembershipScorer.forward``, which returns ``logits * valid``:
    # a padded cell arrives as exactly 0.0, i.e. a probability of 0.5.
    valid = anchor_mask[:, :, None] & entity_mask[:, None, :]
    logits = (torch.randn(B, A, E) * 3) * valid
    return anchors, anchor_mask, entities, entity_mask, logits, valid


def _layer(spec, **defaults):
    layer = AnchorEntityRefinement.from_config(spec, D, dropout=0.0, **defaults)
    layer.eval()
    return layer


class TestAnchorEntityRefinementFactory:
    def test_none_by_default(self):
        assert isinstance(
            AnchorEntityRefinement.from_config(None, D),
            NoAnchorEntityRefinement,
        )

    def test_named_type(self):
        assert isinstance(
            AnchorEntityRefinement.from_config("mlp", D),
            MLPAnchorEntityRefinement,
        )

    def test_mapping_spec(self):
        layer = AnchorEntityRefinement.from_config(
            {"type": "mlp", "params": {"weighting": "softmax"}}, D
        )
        assert layer.weighting == "softmax"

    def test_concise_mapping_spec(self):
        layer = AnchorEntityRefinement.from_config(
            {"type": "mlp", "include_mass": True}, D
        )
        assert layer.include_mass is True

    def test_defaults_fill_missing_params(self):
        layer = AnchorEntityRefinement.from_config("mlp", D, dropout=0.0)
        assert all(
            module.p == 0.0
            for module in layer.mlp
            if isinstance(module, torch.nn.Dropout)
        )

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown anchor entity refinement"):
            AnchorEntityRefinement.from_config("cross_attention", D)

    def test_unknown_option_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            AnchorEntityRefinement.from_config(
                {"type": "mlp", "params": {"num_heads": 4}}, D
            )

    def test_invalid_weighting_raises(self):
        with pytest.raises(ValueError, match="weighting must be one of"):
            AnchorEntityRefinement.from_config(
                {"type": "mlp", "weighting": "bogus"}, D
            )

    def test_layer_scale_requires_residual(self):
        with pytest.raises(ValueError, match="requires residual"):
            AnchorEntityRefinement.from_config(
                {"type": "mlp", "residual": False, "layer_scale_init": 0.1}, D
            )


class TestSelectionWeights:
    def test_padded_cells_never_selected(self):
        _, anchor_mask, _, entity_mask, logits, valid = _inputs()
        weights = membership_selection_weights(
            logits, anchor_mask, entity_mask, threshold=0.5
        )
        # A padded cell sits at p == 0.5 exactly; validity comes from the
        # masks, never from the logit value.
        assert float(weights[~valid].abs().max()) == 0.0

    def test_uniform_is_the_thresholded_mask(self):
        _, anchor_mask, _, entity_mask, logits, valid = _inputs()
        weights = membership_selection_weights(
            logits, anchor_mask, entity_mask, threshold=0.5
        )
        expected = ((logits.sigmoid() > 0.5) & valid).to(weights.dtype)
        assert torch.equal(weights, expected)

    def test_probability_weights_within_the_mask(self):
        _, anchor_mask, _, entity_mask, logits, valid = _inputs()
        weights = membership_selection_weights(
            logits, anchor_mask, entity_mask, weighting="probability"
        )
        selected = (logits.sigmoid() > 0.5) & valid
        assert torch.allclose(weights[selected], logits.sigmoid()[selected])
        assert float(weights[~selected].abs().max()) == 0.0

    def test_softmax_makes_slots_compete_for_each_entity(self):
        _, anchor_mask, _, entity_mask, logits, valid = _inputs()
        weights = membership_selection_weights(
            logits, anchor_mask, entity_mask, weighting="softmax"
        )
        per_entity = weights.sum(dim=1)
        assert torch.allclose(
            per_entity[entity_mask],
            torch.ones(int(entity_mask.sum())),
            atol=1e-5,
        )
        assert float(weights[~valid].abs().max()) == 0.0

    def test_softmax_row_without_valid_cells_is_finite(self):
        _, anchor_mask, _, entity_mask, logits, _ = _inputs()
        weights = membership_selection_weights(
            logits,
            torch.zeros_like(anchor_mask),
            entity_mask,
            weighting="softmax",
        )
        assert torch.isfinite(weights).all()
        assert float(weights.abs().max()) == 0.0


class TestPooling:
    def test_pooled_content_is_the_mean_of_claimed_entities(self):
        _, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        weights = membership_selection_weights(logits, anchor_mask, entity_mask)
        pooled, mass = pool_claimed_entities(weights, entities)
        for batch_idx in range(B):
            for anchor_idx in range(A):
                claimed = torch.where(weights[batch_idx, anchor_idx] > 0)[0]
                expected = (
                    entities[batch_idx, claimed].mean(dim=0)
                    if claimed.numel()
                    else torch.zeros(D)
                )
                assert torch.allclose(
                    pooled[batch_idx, anchor_idx], expected, atol=1e-5
                )
                assert float(mass[batch_idx, anchor_idx, 0]) == claimed.numel()

    def test_anchor_claiming_nothing_pools_to_zero(self):
        _, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        weights = torch.zeros(B, A, E)
        pooled, mass = pool_claimed_entities(weights, entities)
        assert float(pooled.abs().max()) == 0.0
        assert float(mass.abs().max()) == 0.0


class TestNoAnchorEntityRefinement:
    def test_returns_anchors_unchanged(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        layer = _layer(None)
        out = layer(anchors, anchor_mask, entities, entity_mask, logits)
        assert torch.equal(out, anchors)

    def test_holds_no_parameters(self):
        assert list(_layer("none").parameters()) == []

    def test_validates_shapes(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        with pytest.raises(ValueError, match="membership logits must have shape"):
            _layer(None)(
                anchors, anchor_mask, entities, entity_mask, logits[:, :, :2]
            )


class TestMLPAnchorEntityRefinement:
    def test_identity_at_zero_layer_scale(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        layer = _layer({"type": "mlp", "layer_scale_init": 0.0})
        out = layer(anchors, anchor_mask, entities, entity_mask, logits)
        assert torch.allclose(out[anchor_mask], anchors[anchor_mask], atol=1e-6)

    def test_refines_active_anchors(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        layer = _layer({"type": "mlp", "layer_scale_init": 0.1})
        out = layer(anchors, anchor_mask, entities, entity_mask, logits)
        assert out.shape == anchors.shape
        assert not torch.allclose(out[anchor_mask], anchors[anchor_mask])

    def test_padded_slots_are_zeroed(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        layer = _layer("mlp")
        out = layer(anchors, anchor_mask, entities, entity_mask, logits)
        assert float(out[~anchor_mask].abs().max()) == 0.0

    def test_mass_feature_widens_the_input(self):
        without = _layer({"type": "mlp", "include_mass": False})
        with_mass = _layer({"type": "mlp", "include_mass": True})
        assert with_mass.mlp[0].in_features == without.mlp[0].in_features + 1

    def test_empty_entity_axis_is_a_no_op(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        layer = _layer("mlp")
        out = layer(
            anchors,
            anchor_mask,
            entities[:, :0],
            entity_mask[:, :0],
            logits[:, :, :0],
        )
        assert torch.equal(out, anchors)

    def test_hard_selection_does_not_train_the_membership_scorer(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        logits = logits.clone().requires_grad_(True)
        layer = _layer({"type": "mlp", "weighting": "uniform"})
        layer(anchors, anchor_mask, entities, entity_mask, logits).sum().backward()
        assert logits.grad is None or float(logits.grad.abs().sum()) == 0.0

    def test_probability_weighting_trains_the_membership_scorer(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        logits = logits.clone().requires_grad_(True)
        layer = _layer({"type": "mlp", "weighting": "probability"})
        layer(anchors, anchor_mask, entities, entity_mask, logits).sum().backward()
        assert float(logits.grad.abs().sum()) > 0.0

    def test_entities_receive_gradient(self):
        anchors, anchor_mask, entities, entity_mask, logits, _ = _inputs()
        entities = entities.clone().requires_grad_(True)
        layer = _layer("mlp")
        layer(anchors, anchor_mask, entities, entity_mask, logits).sum().backward()
        assert float(entities.grad.abs().sum()) > 0.0
