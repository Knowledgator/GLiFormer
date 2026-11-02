"""Calculation head: program executor, [CALC] prompt, labels, train + infer."""

import copy
import math

import pytest
import torch

from gliformer.processing._structuring_hierarchy import render_field_prompt
from gliformer.processing.typed_fields import TypedFieldSpec
from gliformer.tasks.calculation.program import (
    ProgramError,
    canonical_port,
    execute_program,
    validate_program,
)

BACKBONE = "microsoft/deberta-v3-small"


def _span(text, written):
    start = text.index(written)
    return {"span": written, "start": start, "end": start + len(written), "value": float(written.replace(",", ""))}


# -- program -----------------------------------------------------------------


def test_execute_binary_and_refs():
    text = "9,362.2 then 9,244.9"
    steps = [
        {"op": "subtract", "args": [[_span(text, "9,362.2")], [_span(text, "9,244.9")]]},
        {"op": "divide", "args": [[{"ref": 0}], [_span(text, "9,244.9")]]},
        {"op": "multiply", "args": [[{"ref": 1}], [{"const": 100}]]},
    ]
    validate_program(steps)
    assert math.isclose(execute_program(steps), (9362.2 - 9244.9) / 9244.9 * 100)


def test_execute_set_ops():
    text = "2.80 4.20 4.80 5.40 limit 3.50"
    values = [_span(text, v) for v in ("2.80", "4.20", "4.80", "5.40")]
    steps = [
        {"op": "collect", "args": [values]},
        {"op": "filter_gt", "args": [[{"ref": 0}], [_span(text, "3.50")]]},
        {"op": "count", "args": [[{"ref": 1}]]},
    ]
    assert execute_program(steps) == 3.0
    assert execute_program([{"op": "mean", "args": [values]}]) == pytest.approx(4.3)
    assert execute_program([{"op": "median", "args": [values]}]) == pytest.approx(4.5)
    rng = [
        {"op": "collect", "args": [values]},
        {"op": "max", "args": [[{"ref": 0}]]},
        {"op": "min", "args": [[{"ref": 0}]]},
        {"op": "subtract", "args": [[{"ref": 1}], [{"ref": 2}]]},
    ]
    assert execute_program(rng) == pytest.approx(2.6)
    adjacent = [
        {"op": "diff_adjacent", "args": [values]},
        {"op": "max", "args": [[{"ref": 0}]]},
    ]
    assert execute_program(adjacent) == pytest.approx(1.4)


def test_ports_are_ordered_by_text_position():
    text = "a 1 b 2"
    port = [_span(text, "2"), {"const": 100}, {"ref": 0}, _span(text, "1")]
    assert [m.get("span", m.get("ref", m.get("const"))) for m in canonical_port(port)] == ["1", "2", 0, 100]


def test_elementwise_weighted_mean():
    text = "counts 2 3 means 10 20"
    counts = [_span(text, "2"), _span(text, "3")]
    means = [_span(text, "10"), _span(text, "20")]
    steps = [
        {"op": "collect", "args": [counts]},
        {"op": "collect", "args": [means]},
        {"op": "multiply", "args": [[{"ref": 0}], [{"ref": 1}]]},
        {"op": "sum", "args": [[{"ref": 2}]]},
        {"op": "sum", "args": [[{"ref": 0}]]},
        {"op": "divide", "args": [[{"ref": 3}], [{"ref": 4}]]},
    ]
    assert execute_program(steps) == pytest.approx(16.0)


@pytest.mark.parametrize("steps", [
    [],
    [{"op": "nope", "args": [[{"const": 1}]]}],
    [{"op": "add", "args": [[{"const": 1}]]}],
    [{"op": "copy", "args": [[{"ref": 0}]]}],
    [{"op": "copy", "args": [[{"const": 7.5}]]}],
])
def test_invalid_programs(steps):
    with pytest.raises(ProgramError):
        validate_program(steps)


def test_execution_errors():
    with pytest.raises(ProgramError):
        # 1 / (1 - 1): division by zero.
        execute_program([
            {"op": "subtract", "args": [[{"const": 1}], [{"const": 1}]]},
            {"op": "divide", "args": [[{"const": 1}], [{"ref": 0}]]},
        ])
    with pytest.raises(ProgramError):
        execute_program([{"op": "copy", "args": [[{"span": "x", "start": 0, "end": 1}]]}])


# -- prompt ------------------------------------------------------------------


def test_calc_marker_only_for_calculate_fields():
    plain = render_field_prompt("[FIELD]", "total", {"$type": "number"}, None, "[CALC]")
    marked = render_field_prompt("[FIELD]", "total", {"$type": "number", "$calculate": True}, None, "[CALC]")
    assert "[CALC]" not in plain
    assert marked[-1] == "[CALC]"
    spec = TypedFieldSpec.from_value({"$type": "number", "$calculate": True})
    assert spec.calculate and spec.to_dict()["$calculate"] is True


# -- model -------------------------------------------------------------------


def _cached(repo_id):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    weights = [try_to_load_from_cache(repo_id, name) for name in ("pytorch_model.bin", "model.safetensors")]
    config = try_to_load_from_cache(repo_id, "config.json")
    return isinstance(config, str) and any(isinstance(path, str) for path in weights)


needs_backbone = pytest.mark.skipif(not _cached(BACKBONE), reason=f"{BACKBONE} not cached")

TEXTS = [
    "Revenue was 9,362.2 in 2008 and 9,244.9 in 2007.",
    "Team A scored 17 points, team B scored 24 points and team C scored 31 points.",
]


def _rows():
    a, b = TEXTS
    return [
        {
            "text": a, "prompt": "What was the change in revenue?",
            "structuring": {"$root": {"answer": {"value": 117.3, "anchor": None}}},
            "structuring_types": {"$root": {"answer": {"$type": "number", "$calculate": True}}},
            "calculation": {"$root": {"answer": {"steps": [
                {"op": "subtract", "args": [[_span(a, "9,362.2")], [_span(a, "9,244.9")]]},
            ]}}},
        },
        {
            "text": b, "prompt": "What was the average score?",
            "structuring": {"$root": {"answer": {"value": 24, "anchor": None}}},
            "structuring_types": {"$root": {"answer": {"$type": "number", "$calculate": True}}},
            "calculation": {"$root": {"answer": {"steps": [
                {"op": "mean", "args": [[_span(b, "17"), _span(b, "24"), _span(b, "31")]]},
            ]}}},
        },
    ]


def _model(**calculation):
    from gliformer import GLiFormer

    torch.manual_seed(0)
    return GLiFormer.load_from_config({
        "model_name": BACKBONE,
        "name": "calc-test",
        "hidden_size": 128,
        "dropout": 0.0,
        "max_len": 128,
        "max_types": 30,
        "ner_config": None,
        "structuring_config": {
            "multi_level": True,
            "typed_fields": True,
            "typed_readout_heads": 4,
            "anchor_layer": {"type": "fixed_transformer", "params": {"num_slots": 2}},
            "anchor_objectness": True,
            "represent_spans": False,
        },
        "normalization_config": {"span_encoder": "marker", "token_queries": False},
        "calculation_config": {
            "num_slots": 3, "slot_heads": 4, "readout_heads": 4, "span_pooler": "marker",
            "span_pooler_heads": 4, **calculation,
        },
    }).float()


@needs_backbone
def test_labels_and_prompt():
    model = _model()
    batch = model.data_processor.collate_fn([copy.deepcopy(row) for row in _rows()])
    tokenizer = model.data_processor.transformer_tokenizer
    assert "[CALC]" in tokenizer.decode(batch["input_ids"][0])
    assert batch["calculation_marker_mask"].all()
    assert batch["calculation_query_group"].tolist() == [0, 1]
    assert batch["calculation_active"][0].tolist() == [1.0, 0.0, 0.0]
    assert batch["calculation_active"][1].tolist() == [1.0, 0.0, 0.0]
    # Two operands in the first program, three in the second.
    assert batch["calculation_span_mask"].sum(dim=1).tolist() == [2, 3]
    assert batch["calculation_port_spans"][1, 0, 0].sum() == 3
    # Operand values are number targets of the normalization head.
    assert batch["normalization_number_mask"].sum() >= 5


@needs_backbone
def test_train_and_infer():
    model = _model()
    rows = _rows()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-4)
    model.model.train()
    losses = []
    for _ in range(100):
        batch = model.data_processor.collate_fn([copy.deepcopy(row) for row in rows])
        output = model.model(**batch)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(output.loss))
    assert losses[-1] < losses[0]
    model.model.eval()
    with torch.no_grad():
        result = model.inference(
            TEXTS[1],
            structures={"$root": {"answer": {"$type": "number", "$calculate": True}}},
            prompt="What was the average score?",
        )
    assert "calculation" in result
    queries = result["calculation"][0]
    assert queries and queries[0]["steps"], queries
    assert queries[0]["steps"][0]["op"] == "mean"


@needs_backbone
@pytest.mark.parametrize(
    "modeling",
    ["mlp", {"type": "transformer", "num_layers": 1, "num_heads": 4}],
)
def test_query_modeling_is_configurable(modeling):
    from gliformer.layers import AnchorModeling

    model = _model(query_modeling=modeling)
    head = model.model.heads["calculation"]
    expected = modeling if isinstance(modeling, str) else modeling["type"]
    assert isinstance(head.query_modeling, AnchorModeling.strategy_class(expected))
    batch = model.data_processor.collate_fn([copy.deepcopy(row) for row in _rows()])
    model.model.train()
    output = model.model(**batch)
    output.loss.backward()
    assert any(p.grad is not None for p in head.query_modeling.parameters())


def test_query_modeling_rejects_unknown_options():
    from gliformer.config import CalculationHeadConfig

    with pytest.raises(ValueError, match="query_modeling"):
        CalculationHeadConfig(query_modeling={"type": "readout", "num_layers": 2})
    with pytest.raises(ValueError, match="Unknown modeling_type"):
        CalculationHeadConfig(query_modeling="nope")


def test_old_readout_keys_are_migrated():
    from gliformer.tasks.calculation.model import CalculationHead
    from gliformer.tasks.structuring.model import StructuringHead

    state = {"heads.calculation.query_readout.readout.in_proj_weight": 1, "heads.x.query_readout.a": 2}
    CalculationHead._migrate_state_dict(state, "heads.calculation.")
    assert state == {"heads.calculation.query_modeling.readout.in_proj_weight": 1, "heads.x.query_readout.a": 2}
    state = {"s.option_readout_norm.weight": 1, "s.option_readout_ffn.0.weight": 2, "s.option_query.0.bias": 3}
    StructuringHead._migrate_option_state_dict(state, "s.")
    assert state == {
        "s.option_modeling.readout_norm.weight": 1,
        "s.option_modeling.ffn.0.weight": 2,
        "s.option_modeling.query.0.bias": 3,
    }
