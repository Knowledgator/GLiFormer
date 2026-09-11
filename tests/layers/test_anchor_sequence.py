"""Tests for the optional post-refinement anchor sequence layers."""

import pytest
import torch

from gliformer.layers.anchor_sequence import (
    AnchorSequenceLayer,
    GRUAnchorSequence,
    LSTMAnchorSequence,
    NoAnchorSequence,
    VanillaRNNAnchorSequence,
)

D = 16
B = 3
A = 6


def _anchors(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, A, D)


def _layer(spec, **defaults):
    layer = AnchorSequenceLayer.from_config(spec, D, **defaults)
    layer.eval()
    return layer


class TestAnchorSequenceFactory:
    def test_none_by_default(self):
        assert isinstance(AnchorSequenceLayer.from_config(None, D), NoAnchorSequence)

    def test_named_types(self):
        assert isinstance(AnchorSequenceLayer.from_config("gru", D), GRUAnchorSequence)
        assert isinstance(AnchorSequenceLayer.from_config("lstm", D), LSTMAnchorSequence)
        assert isinstance(
            AnchorSequenceLayer.from_config("rnn", D), VanillaRNNAnchorSequence
        )

    def test_mapping_spec(self):
        layer = AnchorSequenceLayer.from_config(
            {"type": "gru", "params": {"num_layers": 2}}, D
        )
        assert layer.rnn.num_layers == 2

    def test_concise_mapping_spec(self):
        layer = AnchorSequenceLayer.from_config({"type": "gru", "bidirectional": True}, D)
        assert layer.bidirectional is True

    def test_defaults_fill_missing_params(self):
        layer = AnchorSequenceLayer.from_config("gru", D, dropout=0.0)
        assert layer.dropout.p == 0.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown anchor sequence layer"):
            AnchorSequenceLayer.from_config("transformer", D)

    def test_unknown_option_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            AnchorSequenceLayer.from_config({"type": "gru", "nonsense": 1}, D)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers must be positive"):
            AnchorSequenceLayer.from_config({"type": "gru", "num_layers": 0}, D)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout must be finite"):
            AnchorSequenceLayer.from_config({"type": "gru", "dropout": 1.5}, D)

    def test_layer_scale_requires_residual(self):
        with pytest.raises(ValueError, match="requires residual"):
            AnchorSequenceLayer.from_config(
                {"type": "gru", "residual": False, "layer_scale_init": 0.1}, D
            )


class TestNoAnchorSequence:
    def test_is_identity(self):
        anchors = _anchors()
        assert torch.equal(_layer("none")(anchors), anchors)

    def test_owns_no_parameters(self):
        assert list(_layer("none").parameters()) == []

    def test_rejects_mismatched_mask(self):
        with pytest.raises(ValueError, match="anchor mask must match"):
            _layer("none")(_anchors(), torch.ones(B, A + 1, dtype=torch.bool))


class TestRecurrentAnchorSequence:
    def test_preserves_shape(self):
        for name in ("gru", "lstm", "rnn"):
            out = _layer(name)(_anchors())
            assert out.shape == (B, A, D)

    def test_zero_layer_scale_is_identity(self):
        anchors = _anchors()
        layer = _layer({"type": "gru", "layer_scale_init": 0.0})
        assert torch.allclose(layer(anchors), anchors, atol=0, rtol=0)

    def test_padded_anchors_are_zeroed(self):
        anchors = _anchors()
        mask = torch.ones(B, A, dtype=torch.bool)
        mask[0, 3:] = False
        out = _layer({"type": "gru", "dropout": 0.0})(anchors, mask)
        assert bool((out[~mask] == 0).all())

    def test_padded_content_cannot_change_valid_anchors(self):
        """Padded slots may sit anywhere in the anchor axis, not just at the end."""
        anchors = _anchors()
        mask = torch.tensor(
            [[1, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        polluted = anchors.clone()
        polluted[~mask] = torch.randn_like(polluted[~mask]) * 50
        for name in ("gru", "lstm", "rnn"):
            for bidirectional in (False, True):
                layer = _layer(
                    {"type": name, "bidirectional": bidirectional, "dropout": 0.0}
                )
                assert torch.allclose(
                    layer(anchors, mask)[mask],
                    layer(polluted, mask)[mask],
                    atol=1e-5,
                )

    def test_unidirectional_pass_is_causal(self):
        anchors = _anchors()
        layer = _layer({"type": "gru", "dropout": 0.0})
        later = anchors.clone()
        later[:, 3:] = torch.randn(B, A - 3, D)
        assert torch.allclose(layer(anchors)[:, :3], layer(later)[:, :3], atol=1e-6)

    def test_bidirectional_pass_sees_later_anchors(self):
        anchors = _anchors()
        layer = _layer({"type": "gru", "bidirectional": True, "dropout": 0.0})
        later = anchors.clone()
        later[:, 3:] = torch.randn(B, A - 3, D)
        assert not torch.allclose(layer(anchors)[:, :3], layer(later)[:, :3], atol=1e-6)

    def test_empty_anchor_axis(self):
        empty = torch.randn(B, 0, D)
        out = _layer("gru")(empty, torch.ones(B, 0, dtype=torch.bool))
        assert out.shape == (B, 0, D)

    def test_optional_output_norm(self):
        layer = _layer({"type": "gru", "norm": "layer_norm"})
        assert layer.norm is not None
        assert torch.isfinite(layer(_anchors())).all()

    def test_gradients_reach_the_recurrence(self):
        layer = AnchorSequenceLayer.from_config("gru", D)
        layer(_anchors()).sum().backward()
        assert all(p.grad is not None for p in layer.parameters())
