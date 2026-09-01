import json
import re
from types import SimpleNamespace

import pytest

import demo


@pytest.mark.parametrize("parent", [None, ""])
def test_parse_label_groups_keeps_unnamed_singleton_as_list(parent):
    group = {"labels": ["positive", "negative"]}
    if parent is not None:
        group["parent"] = parent

    assert demo.parse_label_groups(json.dumps([group])) == [
        "positive",
        "negative",
    ]


def test_parse_label_groups_preserves_named_singleton():
    groups = json.dumps([{
        "parent": "sentiment",
        "labels": ["positive", "negative"],
    }])

    assert demo.parse_label_groups(groups) == {
        "sentiment": ["positive", "negative"],
    }


def test_parse_joint_relex_groups_accepts_gradio_decoded_value():
    groups = [{
        "parent": "general",
        "entities": ["person", "organization"],
        "relations": ["works at"],
    }]

    assert demo.parse_joint_relex_groups(groups) == {
        "general": {
            "entities": ["person", "organization"],
            "relations": ["works at"],
        }
    }


class _Schema:
    def __init__(self):
        self.structures = {}

    def add_structure(self, name, fields):
        self.structures[name] = fields


class _StructuringModel:
    def __init__(self):
        self.config = SimpleNamespace(structuring_config=object())
        self.schema = _Schema()

    def create_schema(self):
        return self.schema

    def inference_from_schema(self, *, schema, texts, **kwargs):
        assert schema is self.schema
        assert texts == "Alice works at Acme."
        return {
            "structuring": [
                {"person": [{"name": "Alice", "employer": "Acme"}]}
            ]
        }

    def inference(self, texts, **kwargs):
        assert texts == "Alice works at Acme."
        return {
            "structuring": [
                {"person": [{"name": "Alice", "employer": "Acme"}]}
            ]
        }


class _MultiLevelStructuringModel:
    def __init__(self, *, result=None, diagnostics=None):
        multi_level_config = SimpleNamespace(multi_level=True)
        self.config = SimpleNamespace(structuring_config=multi_level_config)
        self.result = result
        self.diagnostics = diagnostics
        self.calls = []

    def structure(self, text, structures, **kwargs):
        self.calls.append((text, structures, kwargs))
        if kwargs.get("return_anchor_diagnostics"):
            return self.result, self.diagnostics
        return self.result


class _OpenRelexModel:
    def __init__(self):
        self.config = SimpleNamespace(open_relex_config=object())
        self.calls = []

    def inference(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "open_relex": [[
                {
                    "relation": "works at",
                    "source": {"text": "Alice"},
                    "target": {"text": "Acme"},
                }
            ]]
        }


def test_joint_relation_runner_prefers_joint_head_when_available(monkeypatch):
    class JointModel:
        config = SimpleNamespace(joint_relex_config=object())

        def __init__(self):
            self.calls = []

        def inference(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return {
                "ner": [[{"text": "Alice", "label": "person"}]],
                "joint_relex": [[{"relation": "works at"}]],
            }

    model = JointModel()
    monkeypatch.setattr(demo, "get_model", lambda: model)
    groups = json.dumps([{
        "parent": "general",
        "entities": ["person", "organization"],
        "relations": ["works at"],
    }])

    output = demo.run_joint_relex(
        "Alice works at Acme.",
        groups,
        threshold=0.42,
        flat_ner=True,
        objectness_threshold=0.77,
    )

    assert json.loads(output) == {
        "ner": [{"text": "Alice", "label": "person"}],
        "joint_relex": [{"relation": "works at"}],
    }
    assert model.calls == [(
        "Alice works at Acme.",
        {
            "joint_relations": {
                "general": {
                    "entities": ["person", "organization"],
                    "relations": ["works at"],
                }
            },
            "threshold": 0.42,
            "flat_ner": True,
        },
    )]


def test_regular_relation_tab_uses_open_relex_head(monkeypatch):
    model = _OpenRelexModel()
    monkeypatch.setattr(demo, "get_model", lambda: model)

    output = demo.run_open_relex(
        "Alice works at Acme.",
        json.dumps([{"parent": "general", "labels": ["works at"]}]),
        threshold=0.4,
        flat_ner=True,
    )

    assert json.loads(output)[0]["relation"] == "works at"
    assert model.calls[0][1] == {
        "relations": {"general": ["works at"]},
        "threshold": 0.4,
        "flat_ner": True,
    }


def test_structuring_tab_accepts_structuring_result(monkeypatch):
    model = _StructuringModel()
    monkeypatch.setattr(demo, "get_model", lambda: model)

    output = demo.run_structuring(
        "Alice works at Acme.",
        json.dumps(
            [{"parent": "person", "labels": ["name:str", "employer"]}]
        ),
        threshold=0.5,
        flat_ner=True,
    )

    assert json.loads(output) == {
        "person": [{"name": "Alice", "employer": "Acme"}]
    }
    assert set(model.schema.structures["person"]) == {"name", "employer"}


def test_bundled_checkpoint_splitter_matches_its_synthetic_training_tokens():
    splitter = demo._LegacySyntheticWordsSplitter()

    assert [token for token, _, _ in splitter("CLM-789456")] == [
        "CLM",
        "-",
        "789456",
    ]


def test_default_model_selection_prefers_stable_best_alias(tmp_path):
    best = tmp_path / "best"
    latest = tmp_path / "checkpoint-1000"
    best.mkdir()
    latest.mkdir()
    (best / "gliner_config.json").write_text("{}", encoding="utf-8")
    (latest / "gliner_config.json").write_text("{}", encoding="utf-8")

    assert demo._default_model_id(str(tmp_path)) == str(best)


def test_multitask_tab_normalizes_structuring_result(monkeypatch):
    model = _StructuringModel()
    monkeypatch.setattr(demo, "get_model", lambda: model)

    output = demo.run_multitask(
        "Alice works at Acme.",
        ner_json="[]",
        cls_json="[]",
        relex_json="[]",
        joint_json="[]",
        struct_json=json.dumps(
            [{"parent": "person", "labels": ["name", "employer"]}]
        ),
        threshold=0.5,
        flat_ner=True,
        multi_label=False,
    )

    assert json.loads(output) == {
        "structuring": {
            "person": [{"name": "Alice", "employer": "Acme"}]
        }
    }


def test_multi_level_structuring_capability_accepts_canonical_head():
    model = _MultiLevelStructuringModel()

    assert demo._model_supports_task(model, "multi_level_structuring")
    assert demo._unsupported_tasks_message(
        model,
        ["multi_level_structuring"],
    ) is None


def test_multi_level_structuring_capability_rejects_flat_checkpoint():
    model = SimpleNamespace(
        config=SimpleNamespace(
            structuring_config=SimpleNamespace(multi_level=False),
        )
    )

    assert not demo._model_supports_task(
        model,
        "multi_level_structuring",
    )
    assert "multi-level structuring" in demo._unsupported_tasks_message(
        model,
        ["multi_level_structuring"],
    )


@pytest.mark.parametrize(
    "structures",
    [
        {"catalog": {"orders": [{"id": ""}]}},
        {"$root": {"orders": [{"id": ""}]}},
        [{"name": "", "children": [{"name": ""}]}],
    ],
)
def test_parse_multi_level_structures_preserves_root_shape(structures):
    encoded = json.dumps(structures)

    assert demo.parse_multi_level_structures(encoded) == structures


@pytest.mark.parametrize(
    "schema_json, message",
    [
        ("", "Please enter"),
        ("{broken", "Invalid schema JSON"),
        ('"field"', "object or an array"),
        ("{}", "must not be empty"),
        ("[]", "must not be empty"),
    ],
)
def test_parse_multi_level_structures_rejects_invalid_schema(
    schema_json,
    message,
):
    with pytest.raises(ValueError, match=message):
        demo.parse_multi_level_structures(schema_json)


@pytest.mark.parametrize(
    "structures, result",
    [
        (
            {"$root": {"orders": [{"id": "", "items": [{"sku": ""}]}]}},
            {"orders": [{"id": "A1", "items": [{"sku": "X"}]}]},
        ),
        (
            [{"name": "", "children": [{"name": ""}]}],
            [{"name": "root", "children": [{"name": "leaf"}]}],
        ),
    ],
)
def test_multi_level_runner_forwards_schema_and_preserves_output(
    monkeypatch,
    structures,
    result,
):
    model = _MultiLevelStructuringModel(result=result)
    monkeypatch.setattr(demo, "get_model", lambda: model)

    output = demo.run_multi_level_structuring(
        "Nested example text",
        json.dumps(structures),
        threshold=0.37,
        flat_ner=False,
    )

    assert json.loads(output) == result
    assert model.calls == [
        (
            "Nested example text",
            structures,
            {"threshold": 0.37, "flat_ner": False},
        )
    ]


def test_multi_level_demo_returns_anchor_diagnostics_separately(
    monkeypatch,
):
    structures = [{"name": "", "children": [{"name": ""}]}]
    result = [{"name": "root", "children": [{"name": "leaf"}]}]
    diagnostics = {
        "summary": {
            "activated_anchor_count": 2,
            "selected_connection_count": 1,
        },
        "groups": [{
            "active_anchor_ids": [4, 9],
            "connections": [{
                "parent_anchor_id": 4,
                "child_anchor_id": 9,
            }],
        }],
    }
    model = _MultiLevelStructuringModel(
        result=result,
        diagnostics=diagnostics,
    )
    monkeypatch.setattr(demo, "get_model", lambda: model)

    output, returned_diagnostics = (
        demo.run_multi_level_structuring_with_diagnostics(
            "Nested example text",
            json.dumps(structures),
            threshold=0.37,
            flat_ner=False,
        )
    )

    assert json.loads(output) == result
    assert returned_diagnostics == diagnostics
    assert json.loads(json.dumps(returned_diagnostics)) == diagnostics
    assert model.calls == [(
        "Nested example text",
        structures,
        {
            "threshold": 0.37,
            "flat_ner": False,
            "return_anchor_diagnostics": True,
        },
    )]


def test_multi_level_demo_clears_diagnostics_on_invalid_input(monkeypatch):
    model = _MultiLevelStructuringModel(result=[])
    monkeypatch.setattr(demo, "get_model", lambda: model)

    output, diagnostics = demo.run_multi_level_structuring_with_diagnostics(
        "",
        "[]",
        threshold=0.5,
        flat_ner=True,
    )

    assert output == "Please enter some text."
    assert diagnostics is None
    assert model.calls == []


def test_multi_level_runner_rejects_flat_checkpoint_before_inference(
    monkeypatch,
):
    class FlatModel:
        config = SimpleNamespace(
            structuring_config=SimpleNamespace(multi_level=False),
        )

        def structure(self, *args, **kwargs):
            raise AssertionError("inference must not run")

    monkeypatch.setattr(demo, "get_model", FlatModel)

    output = demo.run_multi_level_structuring(
        "Nested example text",
        '{"$root": {"children": [{"name": ""}]}}',
        threshold=0.5,
        flat_ner=True,
    )

    assert "multi_level: true" in output


def test_multi_level_demo_examples_keep_nested_schema_json():
    rows = demo._multi_level_structuring_example_rows()

    assert len(rows) == len(demo.MULTI_LEVEL_STRUCTURING_EXAMPLES)
    for row, example in zip(
        rows,
        demo.MULTI_LEVEL_STRUCTURING_EXAMPLES,
        strict=True,
    ):
        assert row[0] == example["text"]
        assert json.loads(row[1]) == example["structures"]
        assert row[2] == example["threshold"]
        assert row[3] is True
    assert any(
        isinstance(example["structures"], list)
        for example in demo.MULTI_LEVEL_STRUCTURING_EXAMPLES
    )
    assert any(
        "$root" in example["structures"]
        for example in demo.MULTI_LEVEL_STRUCTURING_EXAMPLES
        if isinstance(example["structures"], dict)
    )


def test_multi_level_demo_examples_reach_five_hierarchy_levels():
    def record_count(value):
        if isinstance(value, dict):
            return 1 + sum(record_count(child) for child in value.values())
        if isinstance(value, list):
            return sum(record_count(child) for child in value)
        return 0

    def mapping_depth(mapping, current_depth):
        deepest = current_depth
        for value in mapping.values():
            if isinstance(value, dict):
                # Inline objects remain fields on the current record anchor.
                deepest = max(
                    deepest,
                    mapping_depth(value, current_depth),
                )
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        deepest = max(
                            deepest,
                            mapping_depth(child, current_depth + 1),
                        )
        return deepest

    def example_depth(structures):
        if isinstance(structures, list):
            roots = [value for value in structures if isinstance(value, dict)]
        elif set(structures) == {"$root"}:
            root = structures["$root"]
            roots = (
                [value for value in root if isinstance(value, dict)]
                if isinstance(root, list)
                else [root]
            )
        else:
            roots = [value for value in structures.values() if isinstance(value, dict)]
        return max(mapping_depth(root, 1) for root in roots)

    depths = {
        example["label"]: example_depth(example["structures"])
        for example in demo.MULTI_LEVEL_STRUCTURING_EXAMPLES
    }

    assert depths["Project portfolio — recursive raw list"] == 3
    assert depths["Clinical study roster — 4 hierarchy levels"] == 4
    assert depths["Museum catalog — 5 hierarchy levels"] == 5
    assert max(depths.values()) == 5

    examples = {
        example["label"]: example
        for example in demo.MULTI_LEVEL_STRUCTURING_EXAMPLES
    }
    assert record_count(
        examples["Clinical study roster — 4 hierarchy levels"]["expected"]
    ) == 15
    assert record_count(
        examples["Museum catalog — 5 hierarchy levels"]["expected"]
    ) == 25


def test_multi_level_demo_expected_values_are_present_in_realistic_text():
    def scalar_values(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from scalar_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from scalar_values(child)
        elif value is not None:
            yield value

    for example in demo.MULTI_LEVEL_STRUCTURING_EXAMPLES:
        normalized_text = example["text"].casefold()
        assert len(example["text"].split()) >= 20
        for value in scalar_values(example["expected"]):
            if isinstance(value, bool):
                source_forms = (
                    ("true", "yes") if value else ("false", "no")
                )
                present = any(
                    re.search(rf"\b{source_form}\b", normalized_text)
                    for source_form in source_forms
                )
            else:
                present = str(value).casefold() in normalized_text
            assert present, (
                f"{example['label']} is missing extractable value {value!r}"
            )


def test_demo_renders_multi_level_structuring_tab_and_action():
    rendered_config = repr(demo.demo.get_config_file())

    assert "Multi-level Structuring" in rendered_config
    assert "Run Multi-level Structuring" in rendered_config
    assert "Anchor diagnostics" in rendered_config
    assert "Activated anchors and connections" in rendered_config
    assert "Joint NER + Relations" in rendered_config
    assert "Run NER + Relation Extraction" in rendered_config
    assert "Set Open Relation Extraction" not in rendered_config
    assert "Run Set Relation Extraction" not in rendered_config
    assert "Set-head objectness threshold" in rendered_config
    assert "Try a joint or set-relation example" in rendered_config


def test_embedding_examples_are_diverse_and_well_formed():
    assert len(demo.EMBEDDING_EXAMPLES) == 30

    labels = [label for label, _, _ in demo.EMBEDDING_EXAMPLES]
    assert len(labels) == len(set(labels))

    candidate_sets = [
        [candidate for candidate in candidates.splitlines() if candidate.strip()]
        for _, _, candidates in demo.EMBEDDING_EXAMPLES
    ]
    assert all(query.strip() for _, query, _ in demo.EMBEDDING_EXAMPLES)
    assert all(len(candidates) == 4 for candidates in candidate_sets)
    assert any(len(query.split()) <= 5 for _, query, _ in demo.EMBEDDING_EXAMPLES)
    assert any(len(query.split()) >= 30 for _, query, _ in demo.EMBEDDING_EXAMPLES)
    assert any(
        len(candidate.split()) >= 25
        for candidates in candidate_sets
        for candidate in candidates
    )


def test_joint_group_store_is_a_browser_backed_gradio5_input():
    config = demo.demo.get_config_file()
    components = {
        component["id"]: component
        for component in config["components"]
    }
    dependency = next(
        dependency
        for dependency in config["dependencies"]
        if dependency.get("api_name") == "run_joint_relex"
    )
    group_store = components[dependency["inputs"][1]]

    assert group_store["type"] == "textbox"
    assert group_store["props"]["label"] == "joint_relex groups JSON"
    assert group_store["props"]["visible"] is False
