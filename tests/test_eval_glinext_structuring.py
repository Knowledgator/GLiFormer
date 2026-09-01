import json

import pytest

from glinext_eval.eval_structuring import (
    canonicalize_json,
    compute_json_f1,
    evaluate,
    evaluate_single,
    infer_structures,
    json_accuracy,
    json_structure_f1,
    prepare_test_data,
    values_match,
)


def test_infer_structures_uses_only_present_non_null_fields():
    target = {
        "person": [
            {"name": "Alice", "employer": "Acme", "optional": None},
            {"name": "Bob", "employer": "null"},
        ]
    }

    assert infer_structures(target) == {"person": ["name", "employer"]}


def test_infer_structures_builds_recursive_union_for_nested_named_schema():
    target = {
        "catalog": [
            {
                "id": "A",
                "seller": {"name": "Acme"},
                "items": [
                    {"sku": "X", "details": {"color": "red"}},
                ],
            },
            {
                "id": "B",
                "seller": {"email": "sales@example.com"},
                "items": [
                    {"price": 12},
                    {"sku": "Y", "suppliers": [{"name": "Supply Co"}]},
                ],
            },
        ],
    }

    assert infer_structures(target) == {
        "catalog": [{
            "id": "",
            "seller": {"name": "", "email": ""},
            "items": [{
                "sku": "",
                "details": {"color": ""},
                "price": "",
                "suppliers": [{"name": ""}],
            }],
        }],
    }


def test_infer_structures_wraps_arbitrary_object_root():
    target = {
        "document": {
            "title": "Example",
            "authors": [
                {"name": "Alice"},
                {"name": "Bob", "role": "editor"},
            ],
        },
        "tags": ["one", "two"],
    }

    assert infer_structures(target) == {
        "$root": {
            "document": {
                "title": "",
                "authors": [{"name": "", "role": ""}],
            },
            "tags": [""],
        },
    }


def test_infer_structures_preserves_raw_list_root_and_unions_records():
    target = [
        {"name": "Alice", "address": {"city": "Rome"}},
        {
            "name": "Bob",
            "address": {"country": "Italy"},
            "phones": [{"number": "123"}],
        },
    ]

    assert infer_structures(target) == [{
        "name": "",
        "address": {"city": "", "country": ""},
        "phones": [{"number": ""}],
    }]


def test_json_metrics_compare_structure_and_values():
    solution = {"person": [{"name": "Alice", "age": "30"}]}
    prediction = {"person": [{"name": "alice", "age": 30}]}

    precision, recall, f1 = compute_json_f1(prediction, solution)

    assert precision == recall == f1 == 1.0
    assert json_structure_f1(prediction, solution) == 1.0
    assert json_accuracy(prediction, solution) == 1.0


def test_json_metrics_ignore_flat_record_order():
    solution = {
        "meter_reading": [
            {
                "meter_id": "MTR-010551",
                "site": "East Clinic",
                "status": "review",
            },
            {
                "meter_id": "MTR-010552",
                "site": "Market House",
                "status": "verified",
            },
            {
                "meter_id": "MTR-010553",
                "site": "Water Plant",
                "status": "estimated",
            },
        ]
    }
    prediction = {
        "meter_reading": [
            solution["meter_reading"][2],
            {**solution["meter_reading"][1], "status": "Verified"},
            solution["meter_reading"][0],
        ]
    }

    precision, recall, f1 = compute_json_f1(prediction, solution)

    assert precision == recall == f1 == 1.0
    assert json_structure_f1(prediction, solution) == 1.0
    assert json_accuracy(prediction, solution) == 1.0


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("50.5", "50.5 m"),
        (50.5, "50.5 m"),
        ("6.2", "6.2%"),
        ("3007.5", "3007.5 kWh"),
        ("37,173", "37,173 km"),
        ("24.1 °C", "24.1"),
    ],
)
def test_values_match_when_recognized_measurement_unit_is_omitted(
    prediction, gold
):
    assert values_match(prediction, gold)
    assert values_match(gold, prediction)


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("50.6", "50.5 m"),
        ("50.5 kg", "50.5 m"),
        ("2027", "2027-01-07"),
        ("10", "10.5555/zen.2026.14"),
        ("5", "5 Canal Street"),
        ("2", "2 tablets"),
    ],
)
def test_values_match_rejects_mismatches_and_non_measurement_suffixes(
    prediction, gold
):
    assert not values_match(prediction, gold)


def test_unitless_core_measurements_score_as_exact():
    solution = {
        "core_sample": [
            {
                "core_id": "CORE-010631",
                "depth": "50.5 m",
                "porosity": "6.2%",
            }
        ]
    }
    prediction = {
        "core_sample": [
            {
                "core_id": "CORE-010631",
                "depth": "50.5",
                "porosity": "6.2",
            }
        ]
    }

    metrics = evaluate_single(prediction, solution)

    assert metrics["json_f1"] == 1.0
    assert metrics["json_accuracy"] == 1.0


def test_record_matching_maximizes_correct_values():
    solution = {
        "record": [
            {"id": "A", "value": "red"},
            {"id": "B", "value": "blue"},
        ]
    }
    prediction = {
        "record": [
            {"id": "B", "value": "blue"},
            {"id": "A", "value": "wrong"},
        ]
    }

    precision, recall, f1 = compute_json_f1(prediction, solution)

    assert precision == recall == f1 == pytest.approx(0.75)
    assert json_accuracy(prediction, solution) == pytest.approx(0.75)


def test_scalar_list_order_remains_significant():
    solution = {"tags": ["first", "second"]}
    prediction = {"tags": ["second", "first"]}

    assert compute_json_f1(prediction, solution) == (0.0, 0.0, 0.0)
    assert json_accuracy(prediction, solution) == 0.0


def test_structure_f1_descends_into_instance_lists():
    solution = {"person": [{"name": "Alice", "age": "30"}]}
    prediction = {"person": [{"name": "Alice"}]}

    assert 0 < json_structure_f1(prediction, solution) < 1


def test_evaluate_single_rejects_non_mapping_native_output():
    metrics = evaluate_single("not a native object", {"person": [{"name": "A"}]})

    assert metrics["valid_json"] is False
    assert metrics["json_consistency"] == 0.0
    assert metrics["json_f1"] == 0.0


def test_evaluate_single_accepts_raw_list_root():
    solution = [{"name": "Alice"}, {"name": "Bob"}]

    metrics = evaluate_single(
        [{"name": "alice"}, {"name": "Bob"}],
        solution,
    )

    assert metrics["valid_json"] is True
    assert metrics["json_consistency"] == 1.0
    assert metrics["json_structure_f1"] == 1.0
    assert metrics["json_accuracy"] == 1.0
    assert metrics["json_f1"] == 1.0


def test_prepare_test_data_infers_schema_from_converted_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "Alice works at Acme",
                "structuring": {
                    "person": [{"name": "Alice", "employer": "Acme"}]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = prepare_test_data(path)

    assert prepared == [
        {
            "text": "Alice works at Acme",
            "structures": {"person": ["name", "employer"]},
            "solution": {
                "person": [{"name": "Alice", "employer": "Acme"}]
            },
        }
    ]


def test_prepare_test_data_handles_object_and_list_roots(tmp_path):
    path = tmp_path / "multi-level.jsonl"
    rows = [
        {
            "text": "Example by Alice and Bob",
            "structuring": {
                "document": {
                    "title": "Example",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                },
            },
        },
        {
            "text": "Rome Alice, Milan Bob",
            "structuring": [
                {"name": "Alice", "city": "Rome"},
                {"name": "Bob", "city": "Milan", "role": "editor"},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    prepared = prepare_test_data(path)

    assert prepared == [
        {
            "text": rows[0]["text"],
            "structures": {
                "$root": {
                    "document": {
                        "title": "",
                        "authors": [{"name": ""}],
                    },
                },
            },
            "solution": rows[0]["structuring"],
        },
        {
            "text": rows[1]["text"],
            "structures": [{"name": "", "city": "", "role": ""}],
            "solution": rows[1]["structuring"],
        },
    ]


class _FakeModel:
    def __init__(self):
        self.calls = []

    def structure(self, text, structures, **kwargs):
        self.calls.append((text, structures, kwargs))
        return {"person": [{"name": "Alice", "employer": None}]}


def test_evaluate_uses_native_structure_api_and_prunes_optional_nulls():
    model = _FakeModel()
    test_data = [
        {
            "text": "Alice",
            "structures": {"person": ["name"]},
            "solution": {"person": [{"name": "Alice"}]},
        }
    ]

    metrics = evaluate(model, test_data, threshold=0.4, batch_size=2)

    assert metrics.num_samples == 1
    assert metrics.num_valid_json == 1
    assert metrics.json_f1 == pytest.approx(1.0)
    assert model.calls[0][0:2] == ("Alice", {"person": ["name"]})
    assert model.calls[0][2]["threshold"] == 0.4
    assert model.calls[0][2]["batch_size"] == 2


def test_canonicalize_json_removes_nested_optional_values():
    assert canonicalize_json(
        {"a": None, "b": "null", "c": [{"d": "value", "e": ""}]}
    ) == {"c": [{"d": "value"}]}


def test_canonicalize_json_preserves_empty_list_root_type():
    assert canonicalize_json([]) == []
