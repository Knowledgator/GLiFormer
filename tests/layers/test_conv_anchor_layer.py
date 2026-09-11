"""Tests for the convolutional record-proposal anchor layer."""

import math

import pytest
import torch

from gliformer.layers.anchor_layer import (
    AnchorLayer,
    ConvAnchorLayer,
    PositionBucketAnchorLayer,
)

D = 16
B = 2


def _layer(**kwargs):
    defaults = dict(
        kernel_size=5, stride=4, num_layers=3, dilation=2, max_slots=128,
        dropout=0.0,
    )
    defaults.update(kwargs)
    layer = ConvAnchorLayer(D, **defaults)
    layer.eval()
    return layer


def _call(layer, valid_words, padded_length=None):
    """Run the layer over a document of ``valid_words`` real words."""

    padded_length = padded_length or valid_words
    features = torch.randn(B, padded_length, D)
    mask = torch.zeros(B, padded_length, dtype=torch.bool)
    mask[:, :valid_words] = True
    return layer(torch.randn(B, D), features, feature_mask=mask)


class TestConstruction:
    def test_registered_under_conv(self):
        assert AnchorLayer.strategy_class("conv") is ConvAnchorLayer

    def test_from_config_strict_mapping(self):
        layer = AnchorLayer.from_config(
            {"type": "conv", "params": {"stride": 8, "max_slots": 64}},
            D,
            dropout=0.1,
        )
        assert isinstance(layer, ConvAnchorLayer)
        assert layer.stride == 8 and layer.max_slots == 64

    def test_declares_static_slots(self):
        # The head reads this to decide it must not pass the gold record
        # count into the layer.
        assert ConvAnchorLayer.static_slots is True

    def test_rejects_even_kernel(self):
        # An even kernel has no centre tap, so anchor k would drift off the
        # k*stride coordinate the positional hook promises.
        with pytest.raises(ValueError, match="odd"):
            _layer(kernel_size=4)

    @pytest.mark.parametrize("field", ["stride", "num_layers", "max_slots"])
    def test_rejects_non_positive(self, field):
        with pytest.raises(ValueError, match="positive integer"):
            _layer(**{field: 0})

    def test_receptive_field_grows_with_depth(self):
        assert _layer(num_layers=1).receptive_field < _layer(
            num_layers=3
        ).receptive_field
        # One strided layer alone sees exactly its kernel.
        assert _layer(num_layers=1, kernel_size=5).receptive_field == 5


class TestAnchorGeometry:
    @pytest.mark.parametrize("valid_words", [1, 7, 10, 61, 99, 100, 137, 512])
    def test_anchor_count_tracks_document_length(self, valid_words):
        layer = _layer()
        anchors, mask = _call(layer, valid_words)
        expected = math.ceil(valid_words / layer.stride)
        assert int(mask[0].sum()) == expected
        assert anchors.shape[-1] == D

    def test_no_gaps_below_the_document_end(self):
        # The defect this layer exists to remove: quantile buckets leave
        # arbitrary interior slots empty whenever the document is shorter
        # than the slot count, and *which* slots depends on the length.
        features = torch.randn(B, 61, D)
        mask = torch.ones(B, 61, dtype=torch.bool)
        conv_mask = _layer()(torch.randn(B, D), features, feature_mask=mask)[1]
        active = torch.where(conv_mask[0])[0]
        assert active.tolist() == list(range(len(active)))

        buckets = PositionBucketAnchorLayer(D, num_slots=100)
        bucket_mask = buckets(torch.randn(B, D), features, feature_mask=mask)[1]
        bucket_active = set(torch.where(bucket_mask[0])[0].tolist())
        assert 2 not in bucket_active and 17 not in bucket_active

    def test_padding_does_not_create_anchors(self):
        layer = _layer()
        anchors, mask = _call(layer, 12, padded_length=200)
        assert int(mask[0].sum()) == math.ceil(12 / layer.stride)
        assert torch.equal(anchors[~mask], torch.zeros_like(anchors[~mask]))

    def test_masked_out_anchors_are_zeroed(self):
        anchors, mask = _call(_layer(), 9, padded_length=64)
        assert bool((anchors[mask].norm(dim=-1) > 0).all())
        assert float(anchors[~mask].abs().max()) == 0.0

    def test_respects_max_slots_cap(self):
        layer = _layer(stride=2, max_slots=16)
        anchors, mask = _call(layer, 200)
        assert anchors.shape[1] == 16 and mask.shape[1] == 16

    def test_empty_sequence(self):
        layer = _layer()
        anchors, mask = layer(torch.randn(B, D), torch.randn(B, 0, D))
        assert anchors.shape == (B, 0, D) and mask.shape == (B, 0)

    def test_per_sample_lengths_are_independent(self):
        features = torch.randn(B, 40, D)
        mask = torch.zeros(B, 40, dtype=torch.bool)
        mask[0, :40] = True
        mask[1, :8] = True
        _, anchor_mask = _layer()(torch.randn(B, D), features, feature_mask=mask)
        assert int(anchor_mask[0].sum()) == 10
        assert int(anchor_mask[1].sum()) == 2  # ceil(8 / 4)


class TestAnchorWordPositions:
    def test_positions_are_stride_multiples(self):
        layer = _layer(stride=4)
        positions = layer.anchor_word_positions(5, 200, torch.device("cpu"))
        assert positions.tolist() == [0.0, 4.0, 8.0, 12.0, 16.0]

    def test_positions_do_not_depend_on_document_length(self):
        # The property that makes a slot index mean the same thing in every
        # sample, unlike a quantile bucket.
        layer = _layer()
        short = layer.anchor_word_positions(6, 30, torch.device("cpu"))
        long = layer.anchor_word_positions(6, 500, torch.device("cpu"))
        assert torch.equal(short, long)

    def test_position_buckets_positions_do_depend_on_length(self):
        buckets = PositionBucketAnchorLayer(D, num_slots=100)
        short = buckets.anchor_word_positions(6, 30, torch.device("cpu"))
        long = buckets.anchor_word_positions(6, 500, torch.device("cpu"))
        assert not torch.equal(short, long)

    def test_non_positional_layers_return_none(self):
        layer = AnchorLayer.from_config("fixed", D, num_slots=4)
        assert layer.anchor_word_positions(4, 100, torch.device("cpu")) is None


class TestGradients:
    def test_backward_reaches_the_stack(self):
        layer = _layer()
        layer.train()
        anchors, _ = _call(layer, 40)
        anchors.sum().backward()
        assert layer.convolutions[0].weight.grad is not None
        assert float(layer.convolutions[0].weight.grad.abs().sum()) > 0
