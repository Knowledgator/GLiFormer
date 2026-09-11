"""Tests for multimodal vision heads."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from gliner.modeling.loss_functions import focal_loss_with_logits

from gliformer.config import (
    ImageClassificationHeadConfig,
    ObjectDetectionHeadConfig,
    SegmentationHeadConfig,
)
from gliformer.tasks import TaskFlatInputs
from gliformer.tasks.box_ops import box_xyxy_to_cxcywh
from gliformer.tasks.losses import binary_focal_or_bce
from gliformer.tasks.media import matched_mask_loss, normalized_objectness_loss
from gliformer.tasks.vision.decoder import ObjectDetectionDecoder, SegmentationDecoder
from gliformer.tasks.vision.model import (
    ImageClassificationHead,
    ObjectDetectionHead,
    SegmentationHead,
    _reference_box_grid,
    _xyxy_from_raw,
)
from tests.heads.conftest import B, C, D, W, make_config


def _vision_flat_inputs(num_slots=5):
    flat_inputs = TaskFlatInputs(
        words_embedding=torch.randn(B, W, D),
        mask=torch.ones(B, W, dtype=torch.long),
        parent_embedding=torch.randn(B, D),
        child_embedding=torch.randn(B, C, D),
        child_mask=torch.ones(B, C, dtype=torch.long),
        batch_origin=torch.arange(B),
    )
    # W=10: one prefix/CLS token followed by a 3x3 dense patch grid.
    flat_inputs.feature_spatial_shape = torch.tensor([[3, 3]]).expand(B, -1).clone()
    flat_inputs.feature_prefix_tokens = torch.ones(B, dtype=torch.long)
    return flat_inputs


def test_image_classification_head_forward(shared):
    config = make_config(
        image_classification_config=asdict(ImageClassificationHeadConfig()),
    )
    head = ImageClassificationHead.from_config(config)
    flat_inputs = _vision_flat_inputs()
    labels = torch.zeros(B, C)
    labels[:, 0] = 1.0

    out = head(shared, {}, flat_inputs=flat_inputs, image_classification_labels=labels)

    assert out.logits.shape == (B, C)
    assert out.loss is not None
    assert out.loss.item() >= 0


def test_image_classification_uses_feature_embedding_not_words(shared):
    config = make_config(
        image_classification_config=asdict(ImageClassificationHeadConfig()),
    )
    head = ImageClassificationHead.from_config(config)
    head.eval()
    flat_inputs = _vision_flat_inputs()
    flat_inputs.feature_embedding = torch.randn_like(flat_inputs.words_embedding)
    flat_inputs.feature_mask = torch.ones_like(flat_inputs.mask)
    flat_inputs.words_embedding = torch.zeros_like(flat_inputs.words_embedding)
    flat_inputs.mask = torch.zeros_like(flat_inputs.mask)

    first = head(shared, {}, flat_inputs=flat_inputs).logits
    flat_inputs.feature_embedding = flat_inputs.feature_embedding + 1.0
    second = head(shared, {}, flat_inputs=flat_inputs).logits

    assert not torch.allclose(first, second)


def test_object_detection_head_forward_with_labels(shared):
    config = make_config(
        object_detection_config=asdict(ObjectDetectionHeadConfig(num_fixed_slots=4)),
    )
    head = ObjectDetectionHead.from_config(config)
    flat_inputs = _vision_flat_inputs()
    class_labels = torch.tensor([[0, 1], [2, -1]])
    bbox_labels = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5], [0.4, 0.4, 0.8, 0.8]],
        [[0.2, 0.2, 0.6, 0.7], [0.0, 0.0, 0.0, 0.0]],
    ], dtype=torch.float)
    object_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float)

    out = head(
        shared, {}, flat_inputs=flat_inputs,
        object_detection_class_labels=class_labels,
        object_detection_bbox_labels=bbox_labels,
        object_detection_object_mask=object_mask,
    )

    assert out.logits.shape == (B, 4, C)
    assert out.extra["bbox_preds"].shape == (B, 4, 4)
    assert out.extra["objectness_logits"].shape == (B, 4)
    assert out.loss is not None


def test_object_detection_routes_fixed_anchor_context_gate_config():
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=4,
                anchor_context_gate_init=0.0,
                anchor_context_gate_trainable=False,
            )
        ),
    )

    head = ObjectDetectionHead.from_config(config)

    assert head.anchor_layer.context_gate.item() == 0.0
    assert not head.anchor_layer.context_gate.requires_grad


@pytest.mark.parametrize(
    ("query_position_type", "reference_box_mode"),
    [("sine2d", "learned"), ("learned", "none")],
)
def test_object_detection_refinement_breaks_slot_symmetry(
    query_position_type,
    reference_box_mode,
):
    # Both coordinate-backed and learned-index query strategies provide stable,
    # distinct slot identities through the same PositionEmbedding hierarchy.
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=6,
                anchor_refine_layers=3,
                query_position_embedding_type=query_position_type,
                reference_box_mode=reference_box_mode,
            ),
        ),
    )
    head = ObjectDetectionHead.from_config(config).eval()
    assert (head.reference_boxes is not None) == (reference_box_mode == "learned")
    num_slots = 6

    img = torch.randn(1, W, D)
    mask = torch.ones(1, W)
    identical = torch.randn(1, 1, D).expand(1, num_slots, D).contiguous()
    pos = head._slot_query_pos(identical)
    assert pos is not None

    with torch.no_grad():
        without = head.anchor_refine(identical.clone(), img, token_mask=mask, query_pos_emb=None)
        with_pos = head.anchor_refine(identical.clone(), img, token_mask=mask, query_pos_emb=pos)

    # Without positions the slots remain collapsed; with positions they diverge.
    assert torch.allclose(without[:, 0], without[:, 1])
    assert not torch.allclose(with_pos[:, 0], with_pos[:, 1])


def test_object_detection_supports_learned_grid_query_positions():
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=5,
                query_position_embedding_type="learned_grid2d",
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)

    positions = head._slot_query_pos(torch.randn(B, 5, D))

    assert positions.shape == (1, 5, D)


def test_object_detection_loads_legacy_spatial_parameter_names():
    reference_config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(num_fixed_slots=3)
        ),
    )
    reference_head = ObjectDetectionHead.from_config(reference_config)
    reference_state = reference_head.state_dict()
    expected_reference = reference_state["reference_boxes"].clone()
    reference_state["reference_points"] = reference_state.pop("reference_boxes")

    loaded_reference = ObjectDetectionHead.from_config(reference_config)
    result = loaded_reference.load_state_dict(reference_state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert torch.equal(loaded_reference.reference_boxes, expected_reference)

    learned_config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=3,
                reference_box_mode="none",
                query_position_embedding_type="learned",
                memory_position_embedding_type="linear2d",
            )
        ),
    )
    learned_head = ObjectDetectionHead.from_config(learned_config)
    learned_state = learned_head.state_dict()
    learned_state["slot_pos_emb.weight"] = learned_state.pop(
        "query_position_embedding.embedding.weight"
    )
    learned_state["coord_proj.weight"] = learned_state.pop(
        "memory_position_embedding.projection.weight"
    )

    loaded_learned = ObjectDetectionHead.from_config(learned_config)
    result = loaded_learned.load_state_dict(learned_state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys


def test_object_detection_class_loss_is_multilabel_focal_over_matched_anchors(shared):
    # Each matched query receives an independent one-hot binary target across
    # open-vocabulary labels; unmatched queries are supervised by objectness.
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=2,
                bbox_loss_coef=0.0,
                iou_loss_coef=0.0,
                objectness_loss_coef=0.0,
            ),
        ),
    )
    head = ObjectDetectionHead.from_config(config)

    class_logits = torch.zeros(1, 2, 2, requires_grad=True)
    bbox_preds = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.9, 0.9, 1.0, 1.0]]])
    objectness_logits = torch.zeros(1, 2, requires_grad=True)
    anchor_mask = torch.ones(1, 2, dtype=torch.bool)
    child_mask = torch.ones(1, 2)
    class_labels = torch.tensor([[0]])
    bbox_labels = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
    object_mask = torch.ones(1, 1)

    loss, matches = head._detection_loss(
        class_logits,
        bbox_preds,
        box_xyxy_to_cxcywh(bbox_preds),
        objectness_logits,
        anchor_mask,
        child_mask,
        class_labels,
        bbox_labels,
        object_mask,
    )
    loss.backward()

    assert matches == {0: [(0, 0)]}
    # Only the matched anchor (0) receives class gradient.
    assert class_logits.grad[0, 0].abs().sum().item() > 0
    assert class_logits.grad[0, 1].abs().sum().item() == 0
    # Sigmoid focal raises the positive class and lowers independent negatives.
    assert class_logits.grad[0, 0, 0].item() < 0
    assert class_logits.grad[0, 0, 1].item() > 0


def test_sigmoid_training_is_independent_of_decoder_cardinality(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                multi_label=False,
                class_probability="sigmoid",
                num_fixed_slots=1,
                bbox_loss_coef=0.0,
                iou_loss_coef=0.0,
                objectness_loss_coef=0.0,
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    class_logits = torch.zeros(1, 1, 2)
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])

    loss, _ = head._detection_loss(
        class_logits,
        boxes,
        box_xyxy_to_cxcywh(boxes),
        torch.zeros(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 2),
        torch.tensor([[0]]),
        boxes,
        torch.ones(1, 1),
    )

    expected = focal_loss_with_logits(
        torch.zeros(1, 2),
        torch.tensor([[1.0, 0.0]]),
    ).mean()
    assert torch.allclose(loss, expected)


def test_object_detection_class_scoring_uses_configured_anchor_modeling(shared):
    config = make_config(
        object_detection_config=asdict(ObjectDetectionHeadConfig(num_fixed_slots=2)),
    )
    head = ObjectDetectionHead.from_config(config)
    anchors = torch.randn(1, 2, D, requires_grad=True)
    label_reps = torch.randn(1, 3, D)
    label_mask = torch.tensor([[1, 1, 0]])

    logits = head._score_anchor_labels(anchors, label_reps, label_mask)
    logits[..., :2].sum().backward()

    assert head.anchor_modeling.__class__.__name__ == "LinearAnchorModeling"
    assert head.anchor_modeling.proj.weight.grad is not None
    assert head.anchor_modeling.proj.weight.grad.abs().sum() > 0
    assert torch.all(logits[..., 2] == -1e4)


def test_object_detection_fixed_anchors_without_refine_are_image_independent(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(num_fixed_slots=2, anchor_refine_layers=0),
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    head.eval()
    flat_inputs = _vision_flat_inputs()

    assert head.memory_position_embedding is None
    assert head.query_position_embedding is None

    first_boxes = head._compute_detection(flat_inputs).boxes_xyxy
    flat_inputs.words_embedding = flat_inputs.words_embedding + 100.0
    second_boxes = head._compute_detection(flat_inputs).boxes_xyxy

    assert torch.allclose(first_boxes, second_boxes)


def test_reference_box_grid_uses_cxcywh_with_bounded_logits():
    bias = _reference_box_grid(num_slots=100, margin=0.1)
    assert bias.shape == (100, 4)
    assert torch.isfinite(bias).all()
    decoded = bias.sigmoid()
    assert torch.all((decoded > 0) & (decoded < 1))


def _iterative_detection_config(**overrides):
    values = dict(
        num_fixed_slots=4,
        anchor_modeling="identity",
        anchor_refine_layers=3,
        anchor_refine_norm="pre_norm",
        anchor_refine_layer_scale_init=0.1,
        anchor_context_gate_init=0.0,
        anchor_context_gate_trainable=False,
        query_position_embedding_type="sine_bbox2d",
        reference_box_initialization="random",
        iterative_box_refinement=True,
        bbox_head_zero_init=True,
        auxiliary_detection_loss_coef=1.0,
        spatial_attention_bias_type="gaussian",
    )
    values.update(overrides)
    return ObjectDetectionHeadConfig(**values)


def test_iterative_detector_emits_one_prediction_per_decoder_layer():
    config = make_config(
        object_detection_config=asdict(_iterative_detection_config()),
    )
    head = ObjectDetectionHead.from_config(config).eval()

    predictions = head._compute_detection(_vision_flat_inputs())

    assert len(predictions.auxiliary_predictions) == 2
    stages = (*predictions.auxiliary_predictions, predictions)
    assert all(stage.boxes_cxcywh.shape == (B, 4, 4) for stage in stages)
    assert len(head.auxiliary_bbox_heads) == 2
    # Zero-initialized bbox heads make every decoder stage an identity update,
    # so training starts from valid references rather than arbitrary deltas.
    expected = head.reference_boxes.sigmoid().unsqueeze(0).expand(B, -1, -1)
    assert all(torch.allclose(stage.boxes_cxcywh, expected) for stage in stages)


def test_full_box_query_positions_condition_on_width_and_height():
    config = make_config(
        object_detection_config=asdict(_iterative_detection_config()),
    )
    head = ObjectDetectionHead.from_config(config)
    anchors = torch.randn(1, 4, D)
    boxes = torch.tensor(
        [[
            [0.5, 0.5, 0.1, 0.1],
            [0.5, 0.5, 0.4, 0.1],
            [0.5, 0.5, 0.1, 0.4],
            [0.5, 0.5, 0.4, 0.4],
        ]]
    )

    positions = head._slot_query_pos(anchors, boxes)

    assert positions.shape == anchors.shape
    assert not torch.allclose(positions[:, 0], positions[:, 1])
    assert not torch.allclose(positions[:, 0], positions[:, 2])


def test_spatial_attention_bias_assigns_distinct_patch_neighborhoods():
    config = make_config(
        object_detection_config=asdict(_iterative_detection_config()),
    )
    head = ObjectDetectionHead.from_config(config)
    boxes = torch.tensor(
        [[[0.15, 0.15, 0.1, 0.1], [0.85, 0.85, 0.1, 0.1]]]
    )

    bias = head._spatial_attention_bias(
        boxes,
        (3, 3),
        dtype=torch.float32,
        device=boxes.device,
    )

    assert bias.shape == (1, 2, 9)
    assert bias[0, 0].argmax().item() == 0
    assert bias[0, 1].argmax().item() == 8


def test_auxiliary_detection_loss_trains_every_bbox_refinement_head(shared):
    config = make_config(
        object_detection_config=asdict(
            _iterative_detection_config(
                class_loss_coef=0.0,
                objectness_loss_coef=0.0,
                bbox_loss_coef=1.0,
                iou_loss_coef=0.0,
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    out = head(
        shared,
        {},
        flat_inputs=_vision_flat_inputs(),
        object_detection_class_labels=torch.tensor([[0], [1]]),
        object_detection_bbox_labels=torch.tensor(
            [[[0.05, 0.1, 0.45, 0.6]], [[0.3, 0.2, 0.8, 0.9]]]
        ),
        object_detection_object_mask=torch.ones(B, 1),
    )

    out.loss.backward()

    bbox_heads = (*head.auxiliary_bbox_heads, head.bbox_head)
    for bbox_head in bbox_heads:
        gradient = sum(
            parameter.grad.abs().sum()
            for parameter in bbox_head.parameters()
            if parameter.grad is not None
        )
        assert gradient.item() > 0


def test_xyxy_from_raw_decodes_cxcywh_correctly():
    # Slot at center of image: cx=0.5, cy=0.5, w=0.4, h=0.3
    # Expected xyxy: [0.3, 0.35, 0.7, 0.65]
    cx, cy, w, h = 0.5, 0.5, 0.4, 0.3
    raw = torch.tensor([[
        cx / (1 - cx),  # logit(0.5)=0
        cy / (1 - cy),
        w / (1 - w),
        h / (1 - h),
    ]]).log()
    xyxy = _xyxy_from_raw(raw)[0]
    assert abs(xyxy[0].item() - 0.3) < 1e-5  # x1 = cx - w/2
    assert abs(xyxy[1].item() - 0.35) < 1e-5  # y1 = cy - h/2
    assert abs(xyxy[2].item() - 0.7) < 1e-5  # x2 = cx + w/2
    assert abs(xyxy[3].item() - 0.65) < 1e-5  # y2 = cy + h/2


def test_xyxy_from_raw_keeps_edge_crossing_corners_differentiable():
    # cx=0.1 and w=0.6 puts x1 at -0.2. Training must retain that value rather
    # than clamp it to zero, otherwise neither center nor width receives an x1
    # loss gradient and predictions become stuck on image boundaries.
    values = torch.tensor([[0.1, 0.5, 0.6, 0.4]])
    raw = torch.logit(values).requires_grad_(True)

    box = _xyxy_from_raw(raw)[0]
    box[0].backward()

    assert box[0].item() < 0.0
    assert raw.grad[0, 0].abs().item() > 0.0
    assert raw.grad[0, 2].abs().item() > 0.0


def test_object_detection_uses_explicit_prefix_and_spatial_metadata(shared):
    config = make_config(
        object_detection_config=asdict(
                ObjectDetectionHeadConfig(
                    anchor_mode="features",
                    reference_box_mode="none",
                    query_position_embedding_type="none",
            ),
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    flat_inputs = _vision_flat_inputs()
    # W=10 in this test module: one CLS token plus a 3x3 patch grid.

    out = head(shared, {}, flat_inputs=flat_inputs)

    assert out.logits.shape[1] == W - 1
    assert out.extra["bbox_preds"].shape[1] == W - 1


def test_object_detection_keeps_content_and_position_features_separate(shared):
    config = make_config(
        object_detection_config=asdict(
                ObjectDetectionHeadConfig(
                    anchor_mode="features",
                    reference_box_mode="none",
                    query_position_embedding_type="none",
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    flat_inputs = _vision_flat_inputs()
    flat_inputs.words_embedding = torch.zeros_like(flat_inputs.words_embedding)

    dense_features, _, memory_positions, spatial_shape = head._dense_features(flat_inputs)

    assert dense_features.shape[1] == W - 1
    assert torch.allclose(dense_features, torch.zeros_like(dense_features))
    assert spatial_shape == (3, 3)
    assert not torch.allclose(memory_positions[:, 0], memory_positions[:, -1])


@pytest.mark.parametrize("memory_position_in_values", [True, False])
def test_object_detection_forwards_configured_memory_value_positions(
    monkeypatch,
    memory_position_in_values,
):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=4,
                memory_position_in_values=memory_position_in_values,
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config).eval()
    forwarded = {}
    original_forward = head.anchor_refine.forward

    def capture_forward(*args, **kwargs):
        forwarded.update(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(head.anchor_refine, "forward", capture_forward)
    head._compute_detection(_vision_flat_inputs())

    assert forwarded["memory_position_in_values"] is memory_position_in_values


def test_object_detection_bbox_loss_only_updates_matched_anchors(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=2,
                class_loss_coef=0.0,
                objectness_loss_coef=0.0,
                bbox_loss_coef=1.0,
                bbox_l1_format="xyxy",
                iou_loss_coef=0.0,
            ),
        ),
    )
    head = ObjectDetectionHead.from_config(config)

    class_logits = torch.tensor([[[5.0, 0.0], [0.0, 0.0]]])
    bbox_preds = torch.tensor(
        [[[0.1, 0.0, 0.6, 0.5], [0.9, 0.9, 1.0, 1.0]]],
        requires_grad=True,
    )
    objectness_logits = torch.zeros(1, 2)
    anchor_mask = torch.ones(1, 2, dtype=torch.bool)
    child_mask = torch.ones(1, 2)
    class_labels = torch.tensor([[0]])
    bbox_labels = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
    object_mask = torch.ones(1, 1)

    loss, matches = head._detection_loss(
        class_logits,
        bbox_preds,
        box_xyxy_to_cxcywh(bbox_preds.detach()),
        objectness_logits,
        anchor_mask,
        child_mask,
        class_labels,
        bbox_labels,
        object_mask,
    )
    loss.backward()

    assert matches == {0: [(0, 0)]}
    assert bbox_preds.grad[0, 0].abs().sum().item() > 0
    assert bbox_preds.grad[0, 1].abs().sum().item() == 0


def test_object_detection_bbox_head_receives_gradients_through_detection_loss(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=2,
                class_loss_coef=0.0,
                objectness_loss_coef=0.0,
                bbox_loss_coef=1.0,
                iou_loss_coef=0.0,
            ),
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    flat_inputs = _vision_flat_inputs()
    flat_inputs.words_embedding = flat_inputs.words_embedding.detach().requires_grad_(True)
    class_labels = torch.tensor([[0], [1]])
    bbox_labels = torch.tensor([
        [[0.05, 0.10, 0.45, 0.60]],
        [[0.30, 0.20, 0.80, 0.90]],
    ], dtype=torch.float)
    object_mask = torch.ones(B, 1)

    out = head(
        shared, {}, flat_inputs=flat_inputs,
        object_detection_class_labels=class_labels,
        object_detection_bbox_labels=bbox_labels,
        object_detection_object_mask=object_mask,
    )
    out.loss.backward()

    bbox_grad = sum(
        param.grad.abs().sum()
        for param in head.bbox_head.parameters()
        if param.grad is not None
    )
    assert bbox_grad.item() > 0
    assert flat_inputs.words_embedding.grad.abs().sum().item() > 0


def test_object_detection_objectness_normalizes_positive_and_negative_terms(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=3,
                bbox_loss_coef=0.0,
                iou_loss_coef=0.0,
                class_loss_coef=0.0,
                objectness_loss_coef=1.0,
                objectness_positive_weight=3.0,
                objectness_negative_weight=2.0,
                focal_loss_alpha=0.75,
                focal_loss_gamma=1.0,
                objectness_focal_loss_alpha=0.25,
                objectness_focal_loss_gamma=2.0,
            ),
        ),
    )
    head = ObjectDetectionHead.from_config(config)

    class_logits = torch.zeros(1, 3, 1)
    bbox_preds = torch.tensor([[
        [0.0, 0.0, 0.5, 0.5],
        [0.9, 0.9, 1.0, 1.0],
        [0.7, 0.7, 0.8, 0.8],
    ]])
    objectness_logits = torch.tensor([[0.0, 1.0, -1.0]])
    anchor_mask = torch.ones(1, 3, dtype=torch.bool)
    child_mask = torch.ones(1, 1)
    class_labels = torch.tensor([[0]])
    bbox_labels = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
    object_mask = torch.ones(1, 1)

    loss, matches = head._detection_loss(
        class_logits,
        bbox_preds,
        box_xyxy_to_cxcywh(bbox_preds),
        objectness_logits,
        anchor_mask,
        child_mask,
        class_labels,
        bbox_labels,
        object_mask,
    )

    assert matches == {0: [(0, 0)]}
    target = torch.tensor([[1.0, 0.0, 0.0]])
    per_anchor = binary_focal_or_bce(
        objectness_logits,
        target,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
    )
    pos_weight = head.det_cfg.objectness_positive_weight
    neg_weight = head.det_cfg.objectness_negative_weight
    expected = pos_weight * per_anchor[0, 0] + neg_weight * per_anchor[0, 1:].mean()
    assert torch.allclose(loss, expected)


def test_objectness_can_disable_focal_without_disabling_class_focal(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                focal_loss_alpha=0.75,
                focal_loss_gamma=2.0,
                objectness_focal_loss_alpha=-1.0,
                objectness_focal_loss_gamma=0.0,
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])

    loss = normalized_objectness_loss(
        logits,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        head._default_binary_loss,
        loss_kwargs=head._objectness_loss_kwargs(),
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    assert torch.allclose(loss, bce[0, 0] + bce[0, 1:].mean())


def test_objectness_focal_none_inherits_the_task_loss_policy(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                focal_loss_alpha=0.6,
                focal_loss_gamma=1.5,
                objectness_focal_loss_alpha=None,
                objectness_focal_loss_gamma=None,
                objectness_focal_loss_prob_margin=None,
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])

    loss = normalized_objectness_loss(
        logits,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        head._default_binary_loss,
        loss_kwargs=head._objectness_loss_kwargs(),
    )
    expected = binary_focal_or_bce(
        logits,
        targets,
        focal_loss_alpha=0.6,
        focal_loss_gamma=1.5,
    )

    assert head._objectness_loss_kwargs() == {}
    assert torch.allclose(loss, expected[0, 0] + expected[0, 1:].mean())


def test_objectness_scale_is_invariant_to_background_query_count(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                bbox_loss_coef=0.0,
                iou_loss_coef=0.0,
                class_loss_coef=0.0,
                objectness_loss_coef=1.0,
                focal_loss_alpha=-1.0,
                focal_loss_gamma=0.0,
                objectness_focal_loss_alpha=-1.0,
                objectness_focal_loss_gamma=0.0,
            )
        ),
    )
    head = ObjectDetectionHead.from_config(config)

    def objectness_loss(query_count):
        boxes = torch.full((1, query_count, 4), 0.9)
        boxes[:, 0] = torch.tensor([0.0, 0.0, 0.5, 0.5])
        loss, _ = head._detection_loss(
            torch.zeros(1, query_count, 1),
            boxes,
            box_xyxy_to_cxcywh(boxes),
            torch.zeros(1, query_count),
            torch.ones(1, query_count, dtype=torch.bool),
            torch.ones(1, 1),
            torch.tensor([[0]]),
            torch.tensor([[[0.0, 0.0, 0.5, 0.5]]]),
            torch.ones(1, 1),
        )
        return loss

    assert torch.allclose(objectness_loss(3), objectness_loss(12))


def test_detection_uses_stable_bce_only_when_focal_is_fully_disabled(shared):
    bce_config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(focal_loss_alpha=-1.0, focal_loss_gamma=0.0)
        ),
    )
    focal_config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(focal_loss_alpha=0.25, focal_loss_gamma=2.0)
        ),
    )
    bce_head = ObjectDetectionHead.from_config(bce_config)
    focal_head = ObjectDetectionHead.from_config(focal_config)
    logits = torch.tensor([-100.0, 100.0])
    targets = torch.tensor([1.0, 0.0])

    bce_loss = bce_head._default_binary_loss(logits, targets)
    focal_loss = focal_head._default_binary_loss(logits, targets)

    assert torch.allclose(
        bce_loss,
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        ),
    )
    # Focal keeps its weighting/modulation while using BCEWithLogits as the
    # stable base term. The upstream probability/log implementation clamps at
    # this magnitude and loses the gradient needed to recover hard examples.
    assert torch.allclose(focal_loss, torch.tensor([25.0, 75.0]))

    focal_logits = logits.clone().requires_grad_(True)
    focal_head._default_binary_loss(focal_logits, targets).sum().backward()
    assert focal_logits.grad is not None
    assert focal_logits.grad[0].item() < 0.0
    assert focal_logits.grad[1].item() > 0.0


def test_set_decoder_rejects_multilabel_override_for_softmax():
    decoder = ObjectDetectionDecoder(
        SimpleNamespace(
            object_detection_config=ObjectDetectionHeadConfig(
                multi_label=False,
                class_probability="softmax",
            )
        )
    )
    model_output = SimpleNamespace(
        object_detection_logits=torch.zeros(1, 1, 2),
        object_detection_boxes=torch.tensor([[[0.1, 0.1, 0.5, 0.5]]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.zeros(1, 1),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="multi-label"):
        decoder.decode(model_output, multi_label=True)


def test_object_detection_decoder_returns_class_and_objectness_scores():
    # Independent sigmoid class confidence is multiplied by sigmoid objectness.
    decoder = ObjectDetectionDecoder(config=None)
    model_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[2.0, -10.0], [-10.0, -10.0]]]),
        object_detection_boxes=torch.tensor([[
            [0.1, 0.1, 0.5, 0.5],
            [0.2, 0.2, 0.6, 0.6],
        ]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[5.0, -5.0]]),
        object_detection_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        batch_size=1,
    )

    decoded = decoder.decode(model_output, threshold=0.1)

    pred = decoded[0][0][0]
    assert len(decoded[0][0]) == 1
    expected_cls = torch.sigmoid(torch.tensor(2.0))
    assert torch.isclose(torch.tensor(pred["class_score"]), expected_cls)
    assert torch.isclose(torch.tensor(pred["objectness_score"]), torch.sigmoid(torch.tensor(5.0)))
    assert torch.isclose(
        torch.tensor(pred["score"]),
        torch.tensor(pred["class_score"] * pred["objectness_score"]),
    )


def test_object_detection_decoder_is_multilabel_by_default():
    decoder = ObjectDetectionDecoder(config=None)
    model_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[2.0, 1.0]]]),
        object_detection_boxes=torch.tensor([[[0.1, 0.1, 0.5, 0.5]]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[5.0]]),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    decoded = decoder.decode(model_output, threshold=0.5)

    assert [prediction["label"] for prediction in decoded[0][0]] == ["0", "1"]


def test_object_detection_decoder_honors_softmax_single_label_config():
    config = SimpleNamespace(
        object_detection_config=ObjectDetectionHeadConfig(
            multi_label=False,
            class_probability="softmax",
        )
    )
    decoder = ObjectDetectionDecoder(config=config)
    model_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[2.0, 1.0]]]),
        object_detection_boxes=torch.tensor([[[0.1, 0.1, 0.5, 0.5]]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[5.0]]),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    decoded = decoder.decode(model_output, threshold=0.5)

    assert len(decoded[0][0]) == 1
    assert decoded[0][0][0]["label"] == "0"


def test_vision_decoders_preserve_tiny_fp16_scores_at_zero_threshold():
    from gliformer.tasks.vision.decoder import ImageClassificationDecoder

    image_decoder = ImageClassificationDecoder(config=None)
    cls_output = SimpleNamespace(
        image_classification_logits=torch.tensor([[-30.0]], dtype=torch.float16),
        image_classification_batch_origin=torch.tensor([0]),
        batch_size=1,
    )

    cls_decoded = image_decoder.decode(cls_output, threshold=0.0)

    assert len(cls_decoded[0][0]) == 1
    assert cls_decoded[0][0][0]["score"] > 0.0

    det_decoder = ObjectDetectionDecoder(config=None)
    det_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[-30.0]]], dtype=torch.float16),
        object_detection_boxes=torch.tensor(
            [[[0.1, 0.1, 0.5, 0.5]]],
            dtype=torch.float16,
        ),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[0.0]], dtype=torch.float16),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    det_decoded = det_decoder.decode(det_output, threshold=0.0)

    assert len(det_decoded[0][0]) == 1
    assert det_decoded[0][0][0]["score"] > 0.0


def test_object_detection_decoder_uses_winning_class_bbox():
    decoder = ObjectDetectionDecoder(config=None)
    model_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[0.0, 5.0]]]),
        object_detection_boxes=torch.tensor([[[0.2, 0.3, 0.4, 0.5]]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[5.0]]),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    decoded = decoder.decode(model_output, threshold=0.1, multi_label=False)

    assert decoded[0][0][0]["label"] == "1"
    assert torch.allclose(
        torch.tensor(decoded[0][0][0]["bbox"]),
        torch.tensor([0.2, 0.3, 0.4, 0.5]),
    )


def test_object_detection_decoder_clips_boxes_to_normalized_image_space():
    decoder = ObjectDetectionDecoder(config=None)
    model_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[5.0]]]),
        object_detection_boxes=torch.tensor([[[-0.2, 0.3, 1.2, 0.8]]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[5.0]]),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    decoded = decoder.decode(model_output, threshold=0.1)

    assert torch.allclose(
        torch.tensor(decoded[0][0][0]["bbox"]),
        torch.tensor([0.0, 0.3, 1.0, 0.8]),
    )


def test_object_detection_decoder_rejects_nonfinite_boxes():
    decoder = ObjectDetectionDecoder(config=None)
    model_output = SimpleNamespace(
        object_detection_logits=torch.tensor([[[5.0]]]),
        object_detection_boxes=torch.tensor([[[float("nan"), 0.1, 0.5, 0.5]]]),
        object_detection_batch_origin=torch.tensor([0]),
        object_detection_objectness_logits=torch.tensor([[5.0]]),
        object_detection_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        batch_size=1,
    )

    assert decoder.decode(model_output, threshold=0.1) == [[[]]]


def test_segmentation_decoder_excludes_invalid_pixels_at_any_threshold():
    decoder = SegmentationDecoder(config=None)
    model_output = SimpleNamespace(
        segmentation_logits=torch.tensor([[[5.0]]]),
        segmentation_boxes=torch.tensor([[[0.1, 0.1, 0.5, 0.5]]]),
        segmentation_batch_origin=torch.tensor([0]),
        segmentation_objectness_logits=torch.tensor([[5.0]]),
        segmentation_anchor_mask=torch.ones(1, 1, dtype=torch.bool),
        segmentation_mask_logits=torch.full((1, 1, 2, 2), 5.0),
        segmentation_mask_validity=torch.tensor(
            [[[True, False], [True, False]]]
        ),
        batch_size=1,
    )

    decoded = decoder.decode(
        model_output,
        threshold=0.1,
        mask_threshold=0.0,
        return_masks=True,
    )

    assert torch.equal(
        decoded[0][0][0]["mask"],
        model_output.segmentation_mask_validity[0].cpu(),
    )


def test_segmentation_head_is_detection_child_and_outputs_masks(shared):
    config = make_config(
        segmentation_config=asdict(
            SegmentationHeadConfig(num_fixed_slots=3, num_prototypes=8, mask_size=16),
        ),
    )
    head = SegmentationHead.from_config(config)
    flat_inputs = _vision_flat_inputs()
    class_labels = torch.tensor([[0], [1]])
    bbox_labels = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5]],
        [[0.2, 0.2, 0.6, 0.7]],
    ], dtype=torch.float)
    object_mask = torch.ones(B, 1)
    mask_labels = torch.zeros(B, 1, 16, 16)
    mask_labels[:, :, 4:12, 4:12] = 1.0

    assert isinstance(head, ObjectDetectionHead)
    out = head(
        shared, {}, flat_inputs=flat_inputs,
        segmentation_class_labels=class_labels,
        segmentation_bbox_labels=bbox_labels,
        segmentation_object_mask=object_mask,
        segmentation_mask_labels=mask_labels,
    )

    assert out.logits.shape == (B, 3, C)
    assert out.extra["mask_logits"].shape == (B, 3, 16, 16)
    assert out.extra["prototypes"].shape == (B, 8, 16, 16)
    assert out.loss is not None


def test_segmentation_reuses_compatible_detection_parameters_and_predictions(shared):
    detection_config = ObjectDetectionHeadConfig(num_fixed_slots=3)
    segmentation_config = SegmentationHeadConfig(
        num_fixed_slots=3,
        num_prototypes=8,
        mask_size=16,
    )
    config = make_config(
        object_detection_config=asdict(detection_config),
        segmentation_config=asdict(segmentation_config),
    )
    detection_head = ObjectDetectionHead.from_config(config)
    segmentation_head = SegmentationHead.from_config(
        config,
        detection_head=detection_head,
    )
    flat_inputs = _vision_flat_inputs()
    calls = 0
    original_compute = detection_head._compute_detection

    def counted_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_compute(*args, **kwargs)

    detection_head._compute_detection = counted_compute
    detection_output = detection_head(shared, {}, flat_inputs=flat_inputs)
    segmentation_output = segmentation_head(
        shared,
        {"object_detection": detection_output},
        flat_inputs=flat_inputs,
    )

    assert segmentation_head._uses_shared_detection_head
    assert not hasattr(segmentation_head, "bbox_head")
    assert calls == 1
    assert segmentation_output.extra["mask_logits"].shape == (B, 3, 16, 16)


def test_mask_only_segmentation_config_inherits_detector_architecture(shared):
    config = make_config(
        object_detection_config=asdict(
            ObjectDetectionHeadConfig(
                num_fixed_slots=3,
                anchor_refine_layers=4,
            )
        ),
        segmentation_config={
            "num_prototypes": 8,
            "mask_size": 16,
            # These options are not consumed by a fixed anchor layer and must
            # not prevent otherwise identical modules from being shared.
            "max_count": 7,
            "anchor_num_heads": 2,
            "anchor_num_layers": 9,
        },
    )
    detection_head = ObjectDetectionHead.from_config(config)
    segmentation_head = SegmentationHead.from_config(
        config,
        detection_head=detection_head,
    )

    assert config.segmentation_config.num_fixed_slots == 3
    assert config.segmentation_config.anchor_refine_layers == 4
    assert segmentation_head._uses_shared_detection_head


def test_dataclass_segmentation_config_inherits_detector_architecture(shared):
    config = make_config(
        object_detection_config=ObjectDetectionHeadConfig(
            num_fixed_slots=3,
            anchor_refine_layers=4,
        ),
        segmentation_config=SegmentationHeadConfig(
            num_prototypes=8,
            mask_size=16,
        ),
    )
    detection_head = ObjectDetectionHead.from_config(config)
    segmentation_head = SegmentationHead.from_config(
        config,
        detection_head=detection_head,
    )

    assert config.segmentation_config.num_fixed_slots == 3
    assert config.segmentation_config.anchor_refine_layers == 4
    assert segmentation_head._uses_shared_detection_head


def test_independent_segmentation_preserves_its_detector_architecture(shared):
    config = make_config(
        object_detection_config=ObjectDetectionHeadConfig(num_fixed_slots=3),
        segmentation_config=SegmentationHeadConfig(
            num_fixed_slots=5,
            num_prototypes=8,
            mask_size=16,
            reuse_detection_head=False,
        ),
    )
    detection_head = ObjectDetectionHead.from_config(config)
    segmentation_head = SegmentationHead.from_config(
        config,
        detection_head=detection_head,
    )

    assert config.segmentation_config.num_fixed_slots == 5
    assert not segmentation_head._uses_shared_detection_head


def test_segmentation_shares_modules_but_keeps_its_own_loss_policy(shared):
    detection_config = ObjectDetectionHeadConfig(
        num_fixed_slots=3,
        matcher_bbox_cost=5.0,
        class_loss_coef=1.0,
    )
    segmentation_config = SegmentationHeadConfig(
        num_fixed_slots=3,
        num_prototypes=8,
        mask_size=16,
        matcher_bbox_cost=1.25,
        class_loss_coef=3.0,
        focal_loss_alpha=0.4,
        focal_loss_gamma=1.5,
    )
    config = make_config(
        object_detection_config=asdict(detection_config),
        segmentation_config=asdict(segmentation_config),
    )
    detection_head = ObjectDetectionHead.from_config(config)
    segmentation_head = SegmentationHead.from_config(
        config,
        detection_head=detection_head,
    )
    flat_inputs = _vision_flat_inputs()
    detection_output = detection_head(shared, {}, flat_inputs=flat_inputs)

    assert segmentation_head._uses_shared_detection_head
    assert segmentation_head.matcher.cost_geometry == pytest.approx(1.25)
    assert segmentation_head.matcher is not detection_head.matcher

    def delegated_loss_would_be_wrong(*args, **kwargs):
        raise AssertionError("segmentation delegated its loss to detection")

    detection_head._detection_loss = delegated_loss_would_be_wrong
    output = segmentation_head(
        shared,
        {"object_detection": detection_output},
        flat_inputs=flat_inputs,
        segmentation_class_labels=torch.tensor([[0], [1]]),
        segmentation_bbox_labels=torch.tensor(
            [
                [[0.1, 0.1, 0.5, 0.5]],
                [[0.2, 0.2, 0.6, 0.7]],
            ],
            dtype=torch.float,
        ),
        segmentation_object_mask=torch.ones(B, 1),
    )

    assert output.loss is not None


def test_segmentation_recomputes_shared_detector_when_task_counts_differ(shared):
    detection_config = ObjectDetectionHeadConfig(num_fixed_slots=3)
    segmentation_config = SegmentationHeadConfig(
        num_fixed_slots=3,
        num_prototypes=8,
        mask_size=16,
    )
    config = make_config(
        object_detection_config=asdict(detection_config),
        segmentation_config=asdict(segmentation_config),
    )
    detection_head = ObjectDetectionHead.from_config(config)
    segmentation_head = SegmentationHead.from_config(
        config,
        detection_head=detection_head,
    )
    flat_inputs = _vision_flat_inputs()
    calls = 0
    original_compute = detection_head._compute_detection

    def counted_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_compute(*args, **kwargs)

    detection_head._compute_detection = counted_compute
    detection_output = detection_head(
        shared,
        {},
        flat_inputs=flat_inputs,
        object_detection_count=torch.ones(B, dtype=torch.long),
    )
    segmentation_output = segmentation_head(
        shared,
        {"object_detection": detection_output},
        flat_inputs=flat_inputs,
        segmentation_count=torch.full((B,), 3, dtype=torch.long),
    )

    assert calls == 2
    assert segmentation_output.extra["anchor_mask"].sum(dim=1).tolist() == [3] * B


def test_segmentation_mask_loss_is_normalized_per_mask_pixel():
    matches = {0: [(0, 0), (1, 1)]}
    small_logits = torch.zeros(1, 2, 8, 8)
    small_labels = torch.zeros(1, 2, 8, 8)
    large_logits = torch.zeros(1, 2, 32, 32)
    large_labels = torch.zeros(1, 2, 32, 32)

    small_loss = matched_mask_loss(small_logits, small_labels, matches)
    large_loss = matched_mask_loss(large_logits, large_labels, matches)

    assert torch.allclose(small_loss, large_loss)


def test_segmentation_prototypes_ignore_masked_dense_tokens(shared):
    config = make_config(
        segmentation_config=asdict(
            SegmentationHeadConfig(
                num_fixed_slots=3,
                num_prototypes=8,
                mask_size=16,
            )
        ),
    )
    head = SegmentationHead.from_config(config).eval()
    tokens = torch.randn(B, 9, D)
    dense_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0]] * B,
        dtype=torch.long,
    )
    perturbed = tokens.clone()
    perturbed[~dense_mask.bool()] = 10_000

    first = head._prototype_masks(tokens, (3, 3), dense_mask)
    second = head._prototype_masks(perturbed, (3, 3), dense_mask)

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_segmentation_keeps_resized_padded_pixels_invalid(shared):
    config = make_config(
        segmentation_config=asdict(
            SegmentationHeadConfig(
                num_fixed_slots=3,
                num_prototypes=8,
                mask_size=16,
            )
        ),
    )
    head = SegmentationHead.from_config(config).eval()
    dense_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0]] * B,
        dtype=torch.long,
    )

    prototypes, validity = head._prototype_masks(
        torch.randn(B, 9, D),
        (3, 3),
        dense_mask,
        return_validity=True,
    )

    assert validity.shape == (B, 16, 16)
    assert not validity[:, -1].any()
    assert torch.count_nonzero(prototypes.masked_select(~validity[:, None])) == 0


def test_segmentation_mask_loss_excludes_invalid_pixels():
    logits = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    validity = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.bool)

    masked = matched_mask_loss(
        logits,
        labels,
        {0: [(0, 0)]},
        validity_mask=validity,
    )
    reference = matched_mask_loss(
        logits[:, :, :1],
        labels[:, :, :1],
        {0: [(0, 0)]},
    )

    assert torch.allclose(masked, reference)


def test_segmentation_mask_loss_is_normalized_per_matched_instance():
    logits = torch.zeros(1, 2, 8, 8)
    labels = torch.zeros(1, 2, 8, 8)

    one_match = matched_mask_loss(logits, labels, {0: [(0, 0)]})
    two_matches = matched_mask_loss(logits, labels, {0: [(0, 0), (1, 1)]})

    assert torch.allclose(one_match, two_matches)
