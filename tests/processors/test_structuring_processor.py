"""Tests for structuring processor."""

import pytest
import torch

from glinext.model import GLiNExTModel
from glinext.tasks.structuring.processor import StructuringProcessor
from tests.conftest import FakeWordsSplitter, make_batch_classes_mapping, make_config


@pytest.fixture
def struct_proc():
    config = make_config(structuring_config={})
    return StructuringProcessor(config, words_splitter=FakeWordsSplitter())


class TestGetClassesMapping:
    def test_basic(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 1
        field_map = mapping[0].items[0].field_class_to_id
        assert "name" in field_map.class_to_id
        assert "location" in field_map.class_to_id
        assert mapping[0].items[0].name == "person"

    def test_multiple_schemas(self, struct_proc):
        item = {
            "text": "test",
            "structuring": {
                "person": [{"name": "John"}],
                "address": [{"city": "NYC", "zip": "10001"}],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        assert len(mapping[0].items) == 2

    def test_empty(self, struct_proc):
        mapping = struct_proc.get_classes_mapping([{"text": "hello"}])
        assert len(mapping) == 1
        assert len(mapping[0].items) == 0

    def test_multiple_instances_merge_fields(self, struct_proc):
        item = {
            "text": "test",
            "structuring": {
                "person": [
                    {"name": "John"},
                    {"name": "Jane", "age": "30"},
                ],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        field_map = mapping[0].items[0].field_class_to_id
        assert "name" in field_map.class_to_id
        assert "age" in field_map.class_to_id

    def test_explicit_schema_preserves_fields_without_placeholder_values(
        self, struct_proc,
    ):
        item = {
            "structuring": {"person": [{}]},
            "structuring_schema": {"person": ["name", "age"]},
        }

        mapping = struct_proc.get_classes_mapping([item])

        assert list(
            mapping[0].items[0].field_class_to_id.class_to_id
        ) == ["name", "age"]

    def test_exposes_flat_field_augmentation_groups(
        self,
        struct_proc,
        structuring_item,
    ):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping,
        )

        groups = struct_proc.get_augmentable_label_groups(
            [structuring_item], classes_mapping,
        )

        assert len(groups) == 1
        assert groups[0].task == "structuring"
        assert set(groups[0].positive_labels) == {"name", "location"}
        groups[0].replace_labels(["age", "location", "name"])
        assert list(
            mapping[0].items[0].field_class_to_id.class_to_id
        ) == ["age", "location", "name"]

class TestContributePrompt:
    def test_basic(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        prompt = struct_proc.contribute_prompt(classes_mapping, batch_idx=0)
        assert "[P]" in prompt
        assert "[SEP]" in prompt
        assert any("[CHILD]" in tok for tok in prompt)
        assert "person" in prompt

    def test_with_labels_encoder(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        prompt = struct_proc.contribute_prompt(classes_mapping, batch_idx=0, use_labels_encoder=True)
        assert not any("[CHILD]" in tok for tok in prompt)


class TestContributeInferenceInput:
    def test_nested_schema_requires_multi_level_mode(self, struct_proc):
        with pytest.raises(ValueError, match="multi_level=True"):
            struct_proc.contribute_inference_input(
                {}, structures={"catalog": {"seller": {"name": ""}}},
            )

    def test_root_list_requires_multi_level_mode(self, struct_proc):
        with pytest.raises(ValueError, match="multi_level=True"):
            struct_proc.contribute_inference_input(
                {}, structures=[{"name": ""}],
            )

    def test_empty_result_uses_canonical_task_key_and_root_shape(self):
        processor = StructuringProcessor(
            make_config(
                default_ner_config=False,
                structuring_config={"multi_level": True},
            ),
            words_splitter=FakeWordsSplitter(),
        )

        assert processor.empty_inference_result(
            2, structures=[{"name": ""}],
        ) == {"structuring": [[], []]}

    def test_multi_level_structure_mode_accepts_descriptor_schema(self):
        config = make_config(
            default_ner_config=False,
            structuring_config={"structure_mode": "multi_level"},
        )
        processor = StructuringProcessor(
            config,
            words_splitter=FakeWordsSplitter(),
        )
        item = {}

        processor.contribute_inference_input(
            item,
            structures={
                "catalog": {
                    "fields": {"title": ""},
                    "children": {
                        "items": {"fields": ["sku"]},
                    },
                },
            },
        )

        assert processor.multi_level is True
        assert "structuring" in item
        mapping = processor.get_classes_mapping([item])[0]
        assert mapping.multi_level is True
        assert list(
            mapping.items[0].field_class_to_id.class_to_id
        ) == ["title", "items.sku"]


class TestResolveSpans:
    def test_raw_values(self, struct_proc):
        item = {
            "text": "John lives in New York",
            "structuring": {
                "person": [{"name": "John", "location": "New York"}]
            },
        }
        struct_proc.resolve_spans(item)
        inst = item["structuring"]["person"][0]
        assert isinstance(inst["name"], dict)
        assert inst["name"]["start"] == 0
        assert inst["name"]["end"] == 0
        assert inst["location"]["start"] == 3
        assert inst["location"]["end"] == 4

    def test_dict_without_indices(self, struct_proc):
        item = {
            "text": "John lives in New York",
            "structuring": {
                "person": [{"name": {"text": "John"}}]
            },
        }
        struct_proc.resolve_spans(item)
        assert item["structuring"]["person"][0]["name"]["start"] == 0

    def test_list_values(self, struct_proc):
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [{"items": ["A", "C"]}]
            },
        }
        struct_proc.resolve_spans(item)
        items = item["structuring"]["schema"][0]["items"]
        assert isinstance(items, list)
        assert all(isinstance(v, dict) for v in items)
        assert items[0]["start"] == 0
        assert items[1]["start"] == 2

    def test_repeated_values_are_assigned_to_successive_instances(
        self, struct_proc,
    ):
        item = {
            "text": "approved then approved",
            "structuring": {
                "claim": [
                    {"status": "approved"},
                    {"status": "approved"},
                ],
            },
        }

        struct_proc.resolve_spans(item)

        instances = item["structuring"]["claim"]
        assert instances[0]["status"]["start"] == 0
        assert instances[1]["status"]["start"] == 2
        assert all(not isinstance(instance["status"], list) for instance in instances)

    def test_unresolvable_values_are_pruned(self, struct_proc):
        item = {
            "text": "John",
            "structuring": {
                "person": [{"name": "John", "employer": "Missing Corp"}],
            },
        }

        struct_proc.resolve_spans(item)

        assert item["structuring"]["person"] == [
            {"name": {"text": "John", "start": 0, "end": 0}}
        ]


class TestCreateLabels:
    def test_basic(self, struct_proc, structuring_item):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_labels([structuring_item], classes_mapping, max_seq_len=10)

        assert result is not None
        assert "structuring_labels" in result
        assert "structuring_mask" in result
        assert "structuring_batch_idx" in result
        assert "structuring_count" in result
        assert "structuring_relation_labels" not in result
        assert "structuring_relation_group_mask" not in result

        labels = result["structuring_labels"]
        # Shape: (total_groups, max_instances, max_seq_len, max_fields, 3)
        assert labels.dim() == 5
        assert labels.shape[0] == 1
        assert labels.shape[4] == 3  # start/end/inside

        assert result["structuring_mask"][0]
        assert result["structuring_count"][0] == 1

        # Check "name" field ("John" at token 0)
        field_map = mapping[0].items[0].field_class_to_id
        name_id = field_map.class_to_id["name"]
        assert labels[0, 0, 0, name_id, 0] == 1.0  # start
        assert labels[0, 0, 0, name_id, 1] == 1.0  # end
        assert labels[0, 0, 0, name_id, 2] == 1.0  # inside

    def test_multiple_instances(self, struct_proc):
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"f": {"text": "A", "start": 0, "end": 0}},
                    {"f": {"text": "C", "start": 2, "end": 2}},
                ],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_labels([item], classes_mapping, max_seq_len=5)
        assert result["structuring_count"][0] == 2
        assert result["structuring_labels"].shape[1] == 2  # 2 instances

    def test_structuring_slots_do_not_pad_gold_record_axes(
        self,
        structuring_item,
    ):
        config = make_config(
            structuring_config={
                "anchor_layer": {
                    "type": "position_buckets",
                    "params": {"num_slots": 150},
                },
                "neg_spans_ratio": 0.0,
            },
        )
        processor = StructuringProcessor(
            config,
            words_splitter=FakeWordsSplitter(),
        )
        mapping = processor.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping
        )

        dense = processor.create_labels(
            [structuring_item],
            classes_mapping,
            max_seq_len=10,
        )
        spans = processor.create_span_labels(
            [structuring_item],
            classes_mapping,
            max_seq_len=10,
        )

        assert processor._fixed_slot_pad == 0
        assert dense["structuring_labels"].shape[1] == 1
        assert spans["structuring_span_labels"].shape[2] == 1

    def test_empty(self, struct_proc):
        classes_mapping = make_batch_classes_mapping()
        result = struct_proc.create_labels([{"text": "hello"}], classes_mapping, max_seq_len=5)
        assert result is None

    def test_list_values_create_multiple_positive_spans(self, struct_proc):
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"items": [
                        {"text": "A", "start": 0, "end": 0},
                        {"text": "C", "start": 2, "end": 2},
                    ]},
                ],
            },
        }
        mapping = struct_proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_labels([item], classes_mapping, max_seq_len=5)

        field_id = mapping[0].items[0].field_class_to_id.class_to_id["items"]
        labels = result["structuring_labels"]
        assert labels[0, 0, 0, field_id, 0] == 1.0
        assert labels[0, 0, 2, field_id, 0] == 1.0
        assert labels[0, 0, 0, field_id, 1] == 1.0
        assert labels[0, 0, 2, field_id, 1] == 1.0


class TestCreateSpanLabels:
    def test_entity_first_span_targets_are_enabled_by_default(
        self,
        struct_proc,
        structuring_item,
    ):
        mapping = struct_proc.get_classes_mapping([structuring_item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = struct_proc.create_span_labels([structuring_item], classes_mapping, max_seq_len=10)
        assert result is not None
        assert result["structuring_span_mask"].any()

    def test_enabled(self):
        config = make_config(structuring_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = StructuringProcessor(config, words_splitter=FakeWordsSplitter())
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"f": {"text": "A", "start": 0, "end": 0}},
                ]
            },
        }
        mapping = proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = proc.create_span_labels([item], classes_mapping, max_seq_len=5)
        assert result is not None
        assert "structuring_span_idx" in result
        assert "structuring_span_labels" in result
        assert "structuring_span_mask" in result

    def test_list_values_create_multiple_positive_span_labels(self):
        config = make_config(structuring_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = StructuringProcessor(config, words_splitter=FakeWordsSplitter())
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [
                    {"items": [
                        {"text": "A", "start": 0, "end": 0},
                        {"text": "C", "start": 2, "end": 2},
                    ]},
                ]
            },
        }
        mapping = proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(structuring_mappings=mapping)
        result = proc.create_span_labels([item], classes_mapping, max_seq_len=5)

        field_id = mapping[0].items[0].field_class_to_id.class_to_id["items"]
        assert result is not None
        assert result["structuring_span_mask"][0, 0]
        assert result["structuring_span_mask"][0, 1]
        assert torch.equal(result["structuring_span_idx"][0, :2], torch.tensor([[0, 0], [2, 2]]))
        assert result["structuring_span_labels"][0, 0, 0, field_id] == 1.0
        assert result["structuring_span_labels"][0, 1, 0, field_id] == 1.0

    def test_same_entity_boundary_is_pooled_once_with_all_targets(self):
        config = make_config(
            structuring_config={"neg_spans_ratio": 0.0},
        )
        proc = StructuringProcessor(
            config,
            words_splitter=FakeWordsSplitter(),
        )
        item = {
            "text": "A B C",
            "structuring": {
                "schema": [{
                    "first": {"text": "A", "start": 0, "end": 0},
                    "alias": {"text": "A", "start": 0, "end": 0},
                }],
            },
        }
        mapping = proc.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping
        )

        result = proc.create_span_labels(
            [item],
            classes_mapping,
            max_seq_len=5,
        )

        field_ids = mapping[0].items[0].field_class_to_id.class_to_id
        assert result is not None
        assert result["structuring_span_mask"].sum().item() == 1
        assert result["structuring_span_idx"][0, 0].tolist() == [0, 0]
        assert (
            result["structuring_span_labels"][0, 0, 0, field_ids["first"]]
            == 1.0
        )
        assert (
            result["structuring_span_labels"][0, 0, 0, field_ids["alias"]]
            == 1.0
        )


class TestMultiLevelStructuring:
    @staticmethod
    def _processor():
        config = make_config(structuring_config={"multi_level": True})
        return StructuringProcessor(
            config,
            words_splitter=FakeWordsSplitter(),
        )

    @staticmethod
    def _nested_item():
        return {
            "text": "Seller A Paris X S1 Y B Z",
            "structuring": {
                "catalog": [{
                    "seller": {"name": "Seller"},
                    "orders": [
                        {
                            "id": "A",
                            "shipping": {"city": "Paris"},
                            "items": [
                                {
                                    "sku": "X",
                                    "suppliers": [{"name": "S1"}],
                                },
                                {"sku": "Y"},
                            ],
                        },
                        {"id": "B", "items": [{"sku": "Z"}]},
                    ],
                }],
            },
        }

    def test_dot_flattens_dicts_and_escapes_literal_dot_collision(self):
        processor = self._processor()
        item = {
            "text": "X Y Z",
            "structuring": {
                "plain": "Z",
                "a": {"b": "X"},
                "a.b": "Y",
            },
        }

        mapping = processor.get_classes_mapping([item])[0]

        assert mapping.multi_level is True
        assert mapping.output_mode == "object"
        assert len(mapping.items) == 1
        assert list(mapping.items[0].field_class_to_id.class_to_id) == [
            "plain",
            "a.b",
            r"a\.b",
        ]
        normalized = next(iter(item["structuring"].values()))
        assert len(normalized) == 1
        assert normalized[0]["a.b"] == "X"
        assert normalized[0][r"a\.b"] == "Y"

    def test_augmentation_keeps_hierarchy_prompt_and_mapping_aligned(self):
        processor = self._processor()
        item = self._nested_item()
        mapping = processor.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping,
        )
        struct_item = mapping[0].items[0]
        original = list(struct_item.field_class_to_id.class_to_id)
        kept = original[-1]

        groups = processor.get_augmentable_label_groups(
            [item], classes_mapping,
        )
        groups[0].replace_labels([kept, "batch_negative"])

        assert list(struct_item.field_class_to_id.class_to_id) == [
            "batch_negative", kept,
        ]
        root = next(
            node
            for node in struct_item.hierarchy
            if tuple(node.get("path") or ()) == ()
        )
        assert root["fields"][0]["label"] == "batch_negative"
        hierarchy_labels = [
            field["label"] if isinstance(field, dict) else field
            for node in struct_item.hierarchy
            for field in node.get("fields", [])
        ]
        assert set(hierarchy_labels) == {"batch_negative", kept}

        prompt = processor.contribute_prompt(classes_mapping, batch_idx=0)
        prompted = [
            token.removeprefix("[CHILD] ")
            for token in prompt
            if token.startswith("[CHILD] ")
        ]
        assert prompted == ["batch_negative", kept]

    def test_start_end_domain_object_is_not_mistaken_for_span_annotation(self):
        processor = self._processor()
        item = {
            "text": "Organization 1938 1940",
            "structuring": {
                "organization": {
                    "founding_date": {"start": 1938, "end": 1940},
                },
                "name": {"text": "Organization", "start": 0, "end": 0},
            },
        }

        mapping = processor.get_classes_mapping([item])[0]

        assert list(mapping.items[0].field_class_to_id.class_to_id) == [
            "organization.founding_date.start",
            "organization.founding_date.end",
            "name",
        ]
        normalized = next(iter(item["structuring"].values()))
        assert normalized[0]["organization.founding_date.start"] == 1938
        assert normalized[0]["organization.founding_date.end"] == 1940
        assert normalized[0]["name"] == {
            "text": "Organization",
            "start": 0,
            "end": 0,
        }

    def test_truncation_keeps_visible_nodes_and_their_ancestors_only(self):
        config = make_config(
            structuring_config={
                "multi_level": True,
                "neg_spans_ratio": 0.0,
            },
        )
        processor = StructuringProcessor(
            config,
            words_splitter=FakeWordsSplitter(),
        )
        item = {
            "text": "Visible Hidden",
            "structuring": {
                "$root": {
                    "children": [
                        {"name": "Visible"},
                        {"name": "Hidden"},
                    ],
                },
            },
        }
        mapping = processor.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping
        )

        dense = processor.create_labels(
            [item],
            classes_mapping,
            max_seq_len=1,
            sequence_lengths=[1],
            source_sequence_lengths=[2],
        )
        spans = processor.create_span_labels(
            [item],
            classes_mapping,
            max_seq_len=1,
            sequence_lengths=[1],
            source_sequence_lengths=[2],
        )

        # The visible child and its structural-only root remain; the hidden
        # sibling and its relation are removed and record ids are compacted.
        assert dense["structuring_count"].tolist() == [2]
        assert dense["structuring_relation_labels"].sum().item() == 1.0
        assert dense["structuring_labels"].shape[2] == 1
        assert spans["structuring_span_mask"].sum().item() == 1
        assert spans["structuring_span_labels"].shape[2] == 2

        full = processor.create_labels(
            [item],
            classes_mapping,
            max_seq_len=2,
            sequence_lengths=[2],
            source_sequence_lengths=[2],
        )
        assert full["structuring_count"].tolist() == [3]
        assert full["structuring_relation_labels"].sum().item() == 2.0

    def test_array_and_empty_container_shapes_are_retained_in_metadata(self):
        processor = self._processor()
        item = {
            "text": "Root classical Latin CellOne CellTwo CellThree",
            "structuring": {
                "name": "Root",
                "styles": ["classical Latin"],
                "tags": [],
                "meta": {},
                "matrix": [["CellOne", "CellTwo"], ["CellThree"]],
            },
        }

        mapping = processor.get_classes_mapping([item])[0].items[0]
        root = mapping.hierarchy[0]
        fields = {field["label"]: field for field in root["fields"]}
        containers = {
            tuple(container["local_path"]): container
            for container in root["containers"]
        }

        assert fields["name"]["shape"] == {"kind": "scalar"}
        assert fields["styles"]["shape"] == {"kind": "array", "rank": 1}
        assert fields["tags"]["shape"] == {"kind": "array", "rank": 1}
        assert fields["matrix"]["shape"] == {"kind": "array", "rank": 2}
        assert containers[("meta",)]["kind"] == "object"
        assert containers[("tags",)] == {
            "local_path": ["tags"], "kind": "array", "rank": 1,
        }

        labels = processor.create_labels(
            [item],
            make_batch_classes_mapping(structuring_mappings=[
                processor.get_classes_mapping([item])[0],
            ]),
            max_seq_len=6,
        )
        matrix_id = mapping.field_class_to_id.class_to_id["matrix"]
        assert labels["structuring_labels"][
            0, 0, :, matrix_id, 0
        ].nonzero().flatten().tolist() == [3, 4, 5]

    def test_same_leaf_path_on_inline_and_child_objects_gets_distinct_labels(self):
        processor = self._processor()
        item = {
            "text": "Direct Child",
            "structuring": {
                "catalog": [{
                    "events": [
                        {"related": {"title": "Direct"}},
                        {"related": [{"title": "Child"}]},
                    ],
                }],
            },
        }

        mapping = processor.get_classes_mapping([item])[0].items[0]
        field_owners = {
            field["label"]: tuple(node["path"])
            for node in mapping.hierarchy
            for field in node["fields"]
        }

        assert field_owners == {
            "events.related.title": ("events",),
            "events.related.title#2": ("events", "related"),
        }

    def test_null_observation_does_not_erase_array_cardinality(self):
        processor = self._processor()
        item = {
            "text": "Alias",
            "structuring": {
                "people": [
                    {"aliases": None},
                    {"aliases": ["Alias"]},
                ],
            },
        }

        mapping = processor.get_classes_mapping([item])[0].items[0]
        field = mapping.hierarchy[0]["fields"][0]

        assert field["shape"] == {"kind": "array", "rank": 1}

    def test_public_root_wrapper_disambiguates_raw_object_from_schema_groups(self):
        processor = self._processor()
        item = {}

        processor.contribute_inference_input(item, {
            "$root": {
                "seller": {"name": ""},
                "orders": [{"id": ""}],
            },
        })
        mapping = processor.get_classes_mapping([item])[0]

        assert mapping.output_mode == "object"
        assert len(mapping.items) == 1
        assert list(mapping.items[0].field_class_to_id.class_to_id) == [
            "seller.name", "orders.id",
        ]

    def test_literal_children_and_fields_keys_are_valid_json_exemplar_fields(self):
        processor = self._processor()
        item = {}

        processor.contribute_inference_input(item, {
            "catalog": {
                "children": [{"value": ""}],
                "fields": "",
                "name": "",
            },
        })
        mapping_list = processor.get_classes_mapping([item])
        mapping = mapping_list[0].items[0]

        assert list(mapping.field_class_to_id.class_to_id) == [
            "fields", "name", "children.value",
        ]
        prompt = processor.contribute_prompt(
            make_batch_classes_mapping(structuring_mappings=mapping_list),
            batch_idx=0,
        )
        assert [
            token.removeprefix("[CHILD] ")
            for token in prompt
            if token.startswith("[CHILD] ")
        ] == list(mapping.field_class_to_id.class_to_id)

    def test_descriptor_children_must_be_a_mapping(self):
        processor = self._processor()

        with pytest.raises(TypeError, match="'children' must be a dictionary"):
            processor.contribute_inference_input({}, {
                "catalog": {
                    "fields": ["name"],
                    "children": [{"value": ""}],
                },
            })

    def test_recursive_anchor_targets_and_balanced_prompt(self):
        processor = self._processor()
        item = self._nested_item()
        mapping = processor.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping
        )

        prompt = processor.contribute_prompt(classes_mapping, batch_idx=0)
        assert prompt == [
            "[P]",
            "catalog",
            "[CHILD] seller.name",
            "<<CHILD>> orders",
            "[CHILD] orders.id",
            "[CHILD] orders.shipping.city",
            "<<CHILD>> items",
            "[CHILD] orders.items.sku",
            "<<CHILD>> suppliers",
            "[CHILD] orders.items.suppliers.name",
            "<<END>>",
            "<<END>>",
            "<<END>>",
            "[SEP]",
        ]

        labels = processor.create_labels(
            [item],
            classes_mapping,
            max_seq_len=8,
        )
        assert labels["structuring_count"].tolist() == [7]
        assert labels["structuring_relation_group_mask"].tolist() == [True]
        relation = labels["structuring_relation_labels"][0, :7, :7]
        expected = torch.zeros(7, 7)
        for parent, child in (
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 4),
            (0, 5),
            (5, 6),
        ):
            expected[parent, child] = 1.0
        assert torch.equal(relation, expected)

    def test_root_only_repeated_records_skip_relation_supervision(self):
        processor = self._processor()
        item = {
            "text": "A B",
            "structuring": {
                "records": [
                    {"name": "A"},
                    {"name": "B"},
                ],
            },
        }
        mapping = processor.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping
        )

        labels = processor.create_labels(
            [item],
            classes_mapping,
            max_seq_len=2,
        )

        hierarchy = mapping[0].items[0].hierarchy
        assert [node["path"] for node in hierarchy] == [[]]
        assert labels["structuring_count"].tolist() == [2]
        assert labels["structuring_relation_group_mask"].tolist() == [False]
        assert labels["structuring_relation_labels"].sum().item() == 0.0

    def test_labels_encoder_keeps_hierarchy_markers(self):
        processor = self._processor()
        item = self._nested_item()
        mapping = processor.get_classes_mapping([item])
        classes_mapping = make_batch_classes_mapping(
            structuring_mappings=mapping
        )

        prompt = processor.contribute_prompt(
            classes_mapping,
            batch_idx=0,
            use_labels_encoder=True,
        )

        assert prompt.count("<<CHILD>> orders") == 1
        assert prompt.count("<<END>>") == 3
        assert not any(token.startswith("[CHILD]") for token in prompt)


class TestModelStructuringCountBridge:
    def test_uses_tail_count_predictions_for_structuring_groups_regression(self):
        model = object.__new__(GLiNExTModel)
        model.config = make_config(count_config={"mode": "regression"})

        flat_inputs_map = {
            "count": type("Flat", (), {"batch_origin": torch.tensor([0, 0, 0, 1])})(),
            "structuring": type("Flat", (), {"batch_origin": torch.tensor([0, 1])})(),
        }
        count_logits = torch.tensor([[1.0], [1.0], [2.6], [0.4]])

        result = model._predict_structuring_counts_from_count_head(count_logits, flat_inputs_map)
        assert torch.equal(result, torch.tensor([3, 0]))

    def test_uses_tail_count_predictions_for_structuring_groups_classification(self):
        model = object.__new__(GLiNExTModel)
        model.config = make_config(count_config={"mode": "classification", "max_count": 5})

        flat_inputs_map = {
            "count": type("Flat", (), {"batch_origin": torch.tensor([0, 0, 0])})(),
            "structuring": type("Flat", (), {"batch_origin": torch.tensor([0])})(),
        }
        count_logits = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ])

        result = model._predict_structuring_counts_from_count_head(count_logits, flat_inputs_map)
        assert torch.equal(result, torch.tensor([2]))
