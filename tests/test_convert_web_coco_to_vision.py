import json

from PIL import Image

from scripts.convert_web_coco_to_vision import convert_split, load_class_names


def _write_dataset(tmp_path):
    dataset = tmp_path / "web_coco"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "config.yaml").write_text(
        "names:\n- male\n- female\n- human skin\nnc: 3\n",
        encoding="utf-8",
    )

    Image.new("RGB", (100, 50)).save(dataset / "images" / "train" / "positive.jpg")
    (dataset / "labels" / "train" / "positive.txt").write_text(
        "0 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n"
        "2 0.5 0.1 0.9 0.1 0.9 0.8 0.5 0.8\n"
        "0 0.2 0.3 0.3 0.3 0.3 0.5 0.2 0.5\n",
        encoding="utf-8",
    )

    Image.new("RGB", (20, 10)).save(dataset / "images" / "train" / "empty.jpg")
    (dataset / "labels" / "train" / "empty.txt").write_text("", encoding="utf-8")
    return dataset


def test_conversion_creates_detection_boxes_and_classification_true_labels(tmp_path):
    dataset = _write_dataset(tmp_path)
    output = tmp_path / "web_coco_train.json"
    class_names = load_class_names(dataset / "config.yaml")

    stats = convert_split(
        dataset_dir=dataset,
        split="train",
        output=output,
        class_names=class_names,
    )

    rows = {row["id"]: row for row in json.loads(output.read_text(encoding="utf-8"))}
    positive = rows["positive"]
    classification = positive["image_classification"][0]
    detection = positive["object_detection"][0]

    assert classification["all_labels"] == ["male", "female", "human skin"]
    assert classification["true_labels"] == ["male", "human skin"]
    assert detection["objects"][0]["bbox"] == [10.0, 10.0, 40.0, 30.0]
    assert {obj["label"] for obj in detection["objects"]} <= set(
        classification["true_labels"]
    )
    assert rows["empty"]["image_classification"][0]["true_labels"] == []
    assert rows["empty"]["object_detection"][0]["objects"] == []
    assert stats == {
        "rows": 2,
        "objects": 3,
        "positive_rows": 1,
        "empty_rows": 1,
        "labels": 3,
        "clamped_polygons": 0,
    }
