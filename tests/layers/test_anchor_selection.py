import pytest
import torch
from torch.nn import functional as F

from glinext.layers.anchor_layer import (
    AnchorLayer,
    PositionBucketAnchorLayer,
    TopKDensityDistinctAnchorLayer,
    TopKDistinctAnchorLayer,
    TopKNormAnchorLayer,
    TopKParentAnchorLayer,
)

HIDDEN_SIZE = 2
MODES_AND_TYPES = (
    ("position_buckets", PositionBucketAnchorLayer),
    ("topk_norm", TopKNormAnchorLayer),
    ("topk_distinct", TopKDistinctAnchorLayer),
    ("topk_parent", TopKParentAnchorLayer),
    ("topk_density_distinct", TopKDensityDistinctAnchorLayer),
)


@pytest.mark.parametrize(("mode", "layer_type"), MODES_AND_TYPES)
def test_selection_modes_are_registered_and_parameter_free(mode, layer_type):
    layer = AnchorLayer.from_config(mode, HIDDEN_SIZE, num_slots=3)

    assert isinstance(layer, layer_type)
    assert layer.num_slots == 3
    assert list(layer.parameters()) == []


def test_position_buckets_mean_pool_relative_regions():
    layer = PositionBucketAnchorLayer(HIDDEN_SIZE, num_slots=2)
    context = torch.zeros(1, HIDDEN_SIZE)
    features = torch.tensor(
        [[[0.0, 10.0], [2.0, 12.0], [4.0, 14.0], [6.0, 16.0]]]
    )

    anchors, mask = layer(context, features)

    torch.testing.assert_close(
        anchors, torch.tensor([[[1.0, 11.0], [5.0, 15.0]]])
    )
    assert mask.tolist() == [[True, True]]


def test_position_buckets_center_rms_removes_shared_document_component():
    layer = PositionBucketAnchorLayer(
        hidden_size=4,
        num_slots=2,
        position_bucket_normalization="center_rms",
    )
    context = torch.zeros(1, 4)
    common = torch.tensor([100.0, -50.0, 25.0, 10.0])
    local = torch.tensor([1.0, -1.0, 0.0, 0.0])
    features = torch.stack(
        [common + local, common + local, common - local, common - local]
    ).unsqueeze(0)

    anchors, mask = layer(context, features)

    assert mask.all()
    torch.testing.assert_close(
        anchors.mean(dim=1),
        torch.zeros_like(anchors[:, 0]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        anchors.float().square().mean(dim=-1).sqrt(),
        torch.ones(1, 2),
        atol=2e-6,
        rtol=0.0,
    )
    assert F.cosine_similarity(anchors[:, 0], anchors[:, 1]).item() < -0.99


def test_position_buckets_center_rms_keeps_single_bucket_finite():
    layer = PositionBucketAnchorLayer(
        HIDDEN_SIZE,
        num_slots=3,
        position_bucket_normalization="center_rms",
    )
    context = torch.zeros(1, HIDDEN_SIZE)
    features = torch.tensor([[[2.0, -1.0]]])

    anchors, mask = layer(context, features)

    assert mask.tolist() == [[True, False, False]]
    assert torch.isfinite(anchors).all()
    assert anchors[0, 0].square().mean().sqrt().item() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_position_buckets_reject_unknown_normalization():
    with pytest.raises(ValueError, match="position_bucket_normalization"):
        PositionBucketAnchorLayer(
            HIDDEN_SIZE,
            num_slots=2,
            position_bucket_normalization="whiten_everything",
        )


def test_topk_norm_selects_largest_embedding_magnitudes():
    layer = TopKNormAnchorLayer(HIDDEN_SIZE, num_slots=2)
    context = torch.zeros(1, HIDDEN_SIZE)
    features = torch.tensor([[[1.0, 0.0], [0.0, 3.0], [2.0, 0.0]]])

    anchors, mask = layer(context, features)

    torch.testing.assert_close(
        anchors, torch.tensor([[[0.0, 3.0], [2.0, 0.0]]])
    )
    assert mask.all()


def test_topk_distinct_avoids_duplicate_directions():
    layer = TopKDistinctAnchorLayer(HIDDEN_SIZE, num_slots=3)
    context = torch.zeros(1, HIDDEN_SIZE)
    features = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
    )

    anchors, mask = layer(context, features)

    torch.testing.assert_close(
        anchors,
        torch.tensor([[[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
    )
    assert mask.all()


def test_topk_parent_selects_by_cosine_similarity():
    layer = TopKParentAnchorLayer(HIDDEN_SIZE, num_slots=2)
    context = torch.tensor([[1.0, 0.0]])
    features = torch.tensor([[[0.0, 1.0], [3.0, 0.0], [0.8, 0.6]]])

    anchors, mask = layer(context, features)

    torch.testing.assert_close(
        anchors, torch.tensor([[[3.0, 0.0], [0.8, 0.6]]])
    )
    assert mask.all()


def test_topk_density_distinct_selects_different_dense_groups():
    layer = TopKDensityDistinctAnchorLayer(HIDDEN_SIZE, num_slots=2)
    context = torch.zeros(1, HIDDEN_SIZE)
    features = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ]
    )

    anchors, mask = layer(context, features)

    torch.testing.assert_close(
        anchors, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    )
    assert mask.all()


def test_topk_density_distinct_falls_back_to_diversity_for_zero_density():
    layer = TopKDensityDistinctAnchorLayer(HIDDEN_SIZE, num_slots=2)
    context = torch.zeros(1, HIDDEN_SIZE)
    features = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]]
    )

    anchors, mask = layer(context, features)

    assert torch.equal(anchors[0, 0], -anchors[0, 1])
    assert mask.all()


@pytest.mark.parametrize(("mode", "_"), MODES_AND_TYPES)
def test_selection_modes_respect_feature_mask_and_pad_short_inputs(mode, _):
    layer = AnchorLayer.from_config(mode, HIDDEN_SIZE, num_slots=4)
    context = torch.tensor([[1.0, 0.0]])
    features = torch.tensor(
        [[[1.0, 0.0], [1000.0, 1000.0], [0.0, 1.0]]]
    )
    feature_mask = torch.tensor([[True, False, True]])

    anchors, anchor_mask = layer(
        context, features, feature_mask=feature_mask
    )

    assert anchors.shape == (1, 4, HIDDEN_SIZE)
    assert anchor_mask.shape == (1, 4)
    assert anchor_mask.sum().item() == 2
    assert torch.equal(
        anchors.masked_select(~anchor_mask.unsqueeze(-1)).reshape(-1),
        torch.zeros((~anchor_mask).sum().item() * HIDDEN_SIZE),
    )
    assert not torch.any(anchors == 1000.0)


@pytest.mark.parametrize(("mode", "_"), MODES_AND_TYPES)
def test_selection_modes_keep_gradient_path_to_selected_features(mode, _):
    layer = AnchorLayer.from_config(mode, HIDDEN_SIZE, num_slots=2)
    context = torch.tensor([[1.0, 0.0]])
    features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        requires_grad=True,
    )

    anchors, anchor_mask = layer(context, features)
    anchors.masked_select(anchor_mask.unsqueeze(-1)).sum().backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0


@pytest.mark.parametrize(("mode", "_"), MODES_AND_TYPES)
def test_selection_modes_return_masked_fixed_width_when_features_are_missing(
    mode, _,
):
    layer = AnchorLayer.from_config(mode, HIDDEN_SIZE, num_slots=3)

    anchors, mask = layer(torch.zeros(2, HIDDEN_SIZE))

    assert anchors.shape == (2, 3, HIDDEN_SIZE)
    assert mask.shape == (2, 3)
    assert not anchors.any()
    assert not mask.any()


@pytest.mark.parametrize(("mode", "_"), MODES_AND_TYPES)
def test_selection_modes_handle_empty_feature_sequences(mode, _):
    layer = AnchorLayer.from_config(mode, HIDDEN_SIZE, num_slots=3)
    context = torch.zeros(2, HIDDEN_SIZE)
    features = torch.empty(2, 0, HIDDEN_SIZE)

    anchors, mask = layer(context, features)

    assert anchors.shape == (2, 3, HIDDEN_SIZE)
    assert mask.shape == (2, 3)
    assert not anchors.any()
    assert not mask.any()
