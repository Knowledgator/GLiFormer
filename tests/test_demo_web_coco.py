from PIL import Image

from demo_web_coco import (
    bbox_to_pixels,
    draw_objects,
    evaluate_predictions,
    latest_checkpoint,
)


def _row():
    return {
        "id": "sample",
        "width": 100,
        "height": 100,
        "image_classification": [
            {"all_labels": ["male", "female"], "true_labels": ["male"]}
        ],
        "object_detection": [
            {
                "all_labels": ["male", "female"],
                "objects": [{"label": "male", "bbox": [10, 10, 50, 50]}],
            }
        ],
    }


def test_evaluate_predictions_matches_classification_and_normalized_box():
    metrics = evaluate_predictions(
        _row(),
        [{"label": "male", "score": 0.9}],
        [{"label": "male", "score": 0.8, "bbox": [0.1, 0.1, 0.5, 0.5]}],
    )

    assert metrics["classification"]["f1"] == 1.0
    assert metrics["object_detection"]["f1"] == 1.0
    assert metrics["object_detection"]["mean_matched_iou"] == 1.0


def test_bbox_and_rendering_helpers():
    assert bbox_to_pixels([0.1, 0.2, 0.4, 0.8], (100, 50), normalized=True) == [
        10.0,
        10.0,
        40.0,
        40.0,
    ]
    rendered = draw_objects(
        Image.new("RGB", (100, 50), "white"),
        [{"label": "male", "score": 0.9, "bbox": [0.1, 0.2, 0.4, 0.8]}],
        ["male"],
        normalized_boxes=True,
        include_scores=True,
    )
    assert rendered.size == (100, 50)


def test_latest_checkpoint_uses_highest_numeric_step(tmp_path):
    for step in (20, 3, 100):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "gliner_config.json").write_text("{}", encoding="utf-8")

    assert latest_checkpoint(tmp_path).name == "checkpoint-100"
