import json

from scripts.convert_lvis_to_vision import convert_annotations


def test_lvis_conversion_keeps_true_labels_as_image_positives_with_global_label_scope(tmp_path):
    annotations = {
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "images": [
            {"id": 10, "file_name": "sample.jpg", "height": 8, "width": 8},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 1, 2, 2]},
        ],
    }
    output = tmp_path / "lvis.json"

    convert_annotations(
        annotation_data=annotations,
        image_root=tmp_path,
        output=output,
        masks_dir=tmp_path / "masks",
        split="validation",
        task="object_detection",
        label_scope="all",
        rasterize_masks=False,
        mask_size=16,
        max_examples=None,
        allow_missing_images=True,
        absolute_image_paths=True,
        include_polygons=False,
        pretty=False,
    )

    row = json.loads(output.read_text())[0]

    assert row["image_classification"][0]["all_labels"] == ["cat", "dog"]
    assert row["image_classification"][0]["true_labels"] == ["cat"]


def test_lvis_conversion_can_add_classification_negatives_without_expanding_detection_scope(tmp_path):
    annotations = {
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "horse"},
        ],
        "images": [
            {"id": 10, "file_name": "sample.jpg", "height": 8, "width": 8},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 1, 2, 2]},
        ],
    }
    output = tmp_path / "lvis.json"

    convert_annotations(
        annotation_data=annotations,
        image_root=tmp_path,
        output=output,
        masks_dir=tmp_path / "masks",
        split="validation",
        task="object_detection",
        label_scope="image",
        rasterize_masks=False,
        mask_size=16,
        max_examples=None,
        allow_missing_images=True,
        absolute_image_paths=True,
        include_polygons=False,
        pretty=False,
        classification_negative_labels=1,
    )

    row = json.loads(output.read_text())[0]

    assert row["image_classification"][0]["true_labels"] == ["cat"]
    assert row["image_classification"][0]["all_labels"][0] == "cat"
    assert len(row["image_classification"][0]["all_labels"]) == 2
    assert row["object_detection"][0]["all_labels"] == ["cat"]
