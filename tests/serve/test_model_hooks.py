"""Model-side hooks used by the analyze API."""

from gliformer.gliformer import BaseGLiFormer
from gliformer.processing.decoder import apply_task_controls
from gliformer.tasks.ner.decoder import NERDecoder
from gliformer.tasks.open_relex.decoder import OpenRelexDecoder
from gliformer.tasks.span_decoder import Span


def test_task_controls_override_one_decoder():
    base = {"threshold": 0.5, "multi_label": False, "flat_ner": True}
    controls = {"classification": {"threshold": -1.0, "multi_label": True}}
    assert apply_task_controls(base, controls, "classification") == {
        "threshold": -1.0, "multi_label": True, "flat_ner": True,
    }
    assert apply_task_controls(base, controls, "ner") is base
    assert apply_task_controls(base, None, "ner") is base


def test_per_row_task_controls_fall_back_to_row_values():
    base = {"threshold": [0.3, 0.4, 0.5]}
    controls = [{"ner": {"threshold": 0.1}}, None, {"classification": {"threshold": 0.9}}]
    assert apply_task_controls(base, controls, "ner")["threshold"] == [0.1, 0.4, 0.5]
    assert apply_task_controls({"threshold": 0.5}, controls, "classification")["threshold"] == [0.5, 0.5, 0.9]


def test_ner_results_can_carry_group_index():
    groups = [[[Span(0, 0, "person", 0.9)], [Span(2, 2, "city", 0.8)]]]
    common = dict(
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 6, 9]],
        all_end_maps=[[5, 8, 15]],
        valid_texts=["Alice in London"],
        num_original=1,
    )
    decoder = NERDecoder.__new__(NERDecoder)
    plain = decoder.map_results(groups, **common)
    assert "group_index" not in plain[0][0]
    tagged = decoder.map_results(groups, return_group_index=True, **common)
    assert [(entity["text"], entity["group_index"]) for entity in tagged[0]] == [("Alice", 0), ("London", 1)]


def test_relation_results_can_carry_group_index():
    triple = {
        "head": {"start": 0, "end": 0, "type": "person"},
        "tail": {"start": 2, "end": 2, "type": "city"},
        "relation": "lives in",
        "score": 0.9,
    }
    decoder = OpenRelexDecoder.__new__(OpenRelexDecoder)
    mapped = decoder.map_results(
        [[[], [triple]]],
        valid_to_orig_idx=[0],
        all_start_maps=[[0, 6, 9]],
        all_end_maps=[[5, 8, 15]],
        valid_texts=["Alice in London"],
        num_original=1,
        return_group_index=True,
    )
    assert mapped[0][0]["group_index"] == 1
    assert mapped[0][0]["tail"]["text"] == "London"


def test_request_prompts_set_group_names_and_descriptions():
    item = {
        "classification": [{"name": "c0", "all_labels": ["a"]}],
        "extraction": [{"name": "e0", "all_labels": ["person"]}, {"name": "e1", "all_labels": ["x"]}],
    }
    BaseGLiFormer._apply_request_prompts(item, {
        "group_prompts": {
            "c0": {"name": None, "description": "Intent?"},
            "e0": {"name": None},
        },
    })
    assert item["classification"][0] == {"name": None, "all_labels": ["a"], "description": "Intent?"}
    assert item["extraction"][0]["name"] is None and "description" not in item["extraction"][0]
    assert item["extraction"][1]["name"] == "e1"


def test_request_prompt_applies_to_groups_and_structuring():
    item = {"classification": [{"name": None, "all_labels": ["a"]}]}
    BaseGLiFormer._apply_request_prompts(item, {"prompt": "Extract.", "structures": {"$root": {"a": "str"}}})
    assert item["classification"][0]["description"] == "Extract."
    assert item["prompt"] == "Extract."
