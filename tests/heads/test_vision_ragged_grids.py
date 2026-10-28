"""Detection and segmentation over rows whose images have different patch grids.

Dynamic-resolution backbones (e.g. Qwen3.5 with ``vision_encoder_type='native'``)
give each row its own ``(h, w)`` grid, flattened into the front of a padded
token sequence. A row in such a batch must predict exactly what it predicts
when encoded alone.
"""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import ObjectDetectionHeadConfig, SegmentationHeadConfig
from gliformer.layers import normalized_grid_2d, ragged_normalized_grid_2d
from gliformer.tasks import TaskFlatInputs
from gliformer.tasks.vision.model import ObjectDetectionHead, SegmentationHead
from tests.heads.conftest import C, D, make_config

SHAPES = [(2, 3), (3, 3), (1, 4)]


def _flat_inputs(features, mask, shapes, parent, child):
    rows = features.shape[0]
    flat_inputs = TaskFlatInputs(
        words_embedding=features,
        mask=mask,
        parent_embedding=parent,
        child_embedding=child,
        child_mask=torch.ones(rows, C, dtype=torch.long),
        batch_origin=torch.arange(rows),
    )
    flat_inputs.feature_spatial_shape = torch.tensor(shapes, dtype=torch.long)
    flat_inputs.feature_prefix_tokens = torch.zeros(rows, dtype=torch.long)
    return flat_inputs


def _ragged_batch(shapes=SHAPES, seed=0):
    generator = torch.Generator().manual_seed(seed)
    count = max(h * w for h, w in shapes)
    features = torch.randn(len(shapes), count, D, generator=generator)
    mask = torch.zeros(len(shapes), count, dtype=torch.long)
    for row, (h, w) in enumerate(shapes):
        mask[row, : h * w] = 1
        # Padding content must not matter.
        features[row, h * w :] = 100.0
    parent = torch.randn(len(shapes), D, generator=generator)
    child = torch.randn(len(shapes), C, D, generator=generator)
    return _flat_inputs(features, mask, shapes, parent, child)


def _single_row(batch, row):
    h, w = batch.feature_spatial_shape[row].tolist()
    return _flat_inputs(
        batch.words_embedding[row : row + 1, : h * w],
        batch.mask[row : row + 1, : h * w],
        [(h, w)],
        batch.parent_embedding[row : row + 1],
        batch.child_embedding[row : row + 1],
    )


def _iterative_config(**overrides):
    values = dict(
        num_fixed_slots=4,
        anchor_refine_layers=2,
        query_position_embedding_type="sine_bbox2d",
        reference_box_initialization="random",
        iterative_box_refinement=True,
        bbox_head_zero_init=False,
        spatial_attention_bias_type="gaussian",
    )
    values.update(overrides)
    return ObjectDetectionHeadConfig(**values)


DETECTION_CONFIGS = {
    "one_shot": ObjectDetectionHeadConfig(num_fixed_slots=4),
    "iterative_gaussian_bias": _iterative_config(),
    "learned_grid_memory": ObjectDetectionHeadConfig(
        num_fixed_slots=4,
        memory_position_embedding_type="learned_grid2d",
        memory_position_embedding_kwargs={"grid_size": [4, 4]},
    ),
}


def test_ragged_grid_coordinates_match_each_rows_own_grid():
    coordinates, valid = ragged_normalized_grid_2d(torch.tensor(SHAPES + [(0, 0)]), 9)

    assert valid.sum(1).tolist() == [6, 9, 4, 0]
    for row, (h, w) in enumerate(SHAPES):
        expected = normalized_grid_2d(h, w, device=torch.device("cpu"))
        assert torch.allclose(coordinates[row, : h * w], expected)
        assert coordinates[row, h * w :].abs().sum() == 0
    assert coordinates[3].abs().sum() == 0


@pytest.mark.parametrize("name", sorted(DETECTION_CONFIGS))
def test_detection_rows_match_their_unbatched_predictions(name):
    torch.manual_seed(0)
    config = make_config(object_detection_config=asdict(DETECTION_CONFIGS[name]))
    head = ObjectDetectionHead.from_config(config).eval()
    batch = _ragged_batch()

    with torch.no_grad():
        batched = head._compute_detection(batch)
        assert isinstance(batched.spatial_shape, torch.Tensor)
        for row in range(len(SHAPES)):
            single = head._compute_detection(_single_row(batch, row))
            assert single.spatial_shape == SHAPES[row]
            for field in ("class_logits", "boxes_xyxy", "objectness_logits"):
                assert torch.allclose(
                    getattr(batched, field)[row],
                    getattr(single, field)[0],
                    atol=1e-5,
                ), field


def test_detection_ignores_padding_past_each_rows_grid():
    torch.manual_seed(0)
    config = make_config(object_detection_config=asdict(DETECTION_CONFIGS["one_shot"]))
    head = ObjectDetectionHead.from_config(config).eval()
    batch = _ragged_batch()
    _, mask, _, _ = head._dense_features(batch)
    assert mask.sum(1).tolist() == [6, 9, 4]

    # Even a row whose encoder mask leaves padding "valid" is cut at its grid.
    batch.mask = torch.ones_like(batch.mask)
    _, mask, _, _ = head._dense_features(batch)
    assert mask.sum(1).tolist() == [6, 9, 4]


def test_detection_trains_on_a_ragged_batch_with_an_imageless_row():
    torch.manual_seed(0)
    config = make_config(object_detection_config=asdict(_iterative_config(auxiliary_detection_loss_coef=1.0)))
    head = ObjectDetectionHead.from_config(config)
    batch = _ragged_batch(SHAPES + [(0, 0)])
    batch.words_embedding.requires_grad_(True)
    rows = len(SHAPES) + 1

    out = head(
        None,
        {},
        flat_inputs=batch,
        object_detection_class_labels=torch.zeros(rows, 1, dtype=torch.long),
        object_detection_bbox_labels=torch.tensor([[[0.1, 0.2, 0.5, 0.7]]]).expand(rows, -1, -1),
        object_detection_object_mask=torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.extra["bbox_preds"]).all()
    assert batch.words_embedding.grad[:3].abs().sum() > 0


def test_spatial_attention_bias_uses_each_rows_grid():
    config = make_config(object_detection_config=asdict(_iterative_config()))
    head = ObjectDetectionHead.from_config(config)
    boxes = torch.tensor([[[0.15, 0.15, 0.1, 0.1]], [[0.15, 0.15, 0.1, 0.1]]])
    shapes = torch.tensor([(1, 4), (3, 3)])

    bias = head._spatial_attention_bias(boxes, shapes, dtype=torch.float32, device=boxes.device)
    single = head._spatial_attention_bias(boxes[1:], (3, 3), dtype=torch.float32, device=boxes.device)

    assert bias.shape == (2, 1, 9)
    assert torch.allclose(bias[1], single[0])
    assert bias[0, 0, :4].argmax().item() == 0


def test_segmentation_rows_match_their_unbatched_masks():
    torch.manual_seed(0)
    config = make_config(
        segmentation_config=asdict(SegmentationHeadConfig(num_fixed_slots=4, mask_size=16, num_prototypes=8)),
    )
    head = SegmentationHead.from_config(config).eval()
    batch = _ragged_batch()

    with torch.no_grad():
        batched = head(None, {}, flat_inputs=batch)
        assert batched.extra["mask_validity"].all()
        for row in range(len(SHAPES)):
            single = head(None, {}, flat_inputs=_single_row(batch, row))
            assert torch.allclose(
                batched.extra["mask_logits"][row],
                single.extra["mask_logits"][0],
                atol=1e-5,
            )
            assert torch.allclose(batched.extra["prototypes"][row], single.extra["prototypes"][0], atol=1e-5)


def test_segmentation_trains_on_a_ragged_batch_with_an_imageless_row():
    torch.manual_seed(0)
    config = make_config(
        segmentation_config=asdict(SegmentationHeadConfig(num_fixed_slots=4, mask_size=16, num_prototypes=8)),
    )
    head = SegmentationHead.from_config(config)
    batch = _ragged_batch(SHAPES + [(0, 0)])
    rows = len(SHAPES) + 1
    mask_labels = torch.zeros(rows, 1, 16, 16)
    mask_labels[:, 0, 4:10, 2:8] = 1.0

    out = head(
        None,
        {},
        flat_inputs=batch,
        segmentation_class_labels=torch.zeros(rows, 1, dtype=torch.long),
        segmentation_bbox_labels=torch.tensor([[[0.1, 0.25, 0.5, 0.6]]]).expand(rows, -1, -1),
        segmentation_object_mask=torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
        segmentation_mask_labels=mask_labels,
    )
    out.loss.backward()

    assert torch.isfinite(out.loss)
    validity = out.extra["mask_validity"]
    assert validity[:3].all() and not validity[3].any()
    assert head.proto_proj.weight.grad.abs().sum() > 0
