import json
import re
from collections import Counter

import pytest

from data_prod.generate_sts_gemini import (
    CONTENT_TYPES,
    RELATION_BANDS,
    GeminiSTSGenerator,
    GenerationResult,
    GenerationTask,
    InvalidExampleError,
    _load_existing_anchors,
    _write_json_line,
    build_tasks,
    classify_source_text,
    load_few_shots,
    main,
    make_prompt,
    make_schedule_example,
    run_generation,
    validate_generated_batch,
    validate_text_type,
)


def _paragraph(subject):
    return (
        f"{subject} volunteers reviewed the proposed neighborhood garden during a public "
        "meeting and compared several practical locations near the library. They selected "
        "the quiet east courtyard because it receives enough sunlight, has an accessible "
        "water line, and leaves the main walkway open for visitors."
    )


def _task(
    anchor_type="sentence",
    positive_type="sentence",
    partial_type="sentence",
    negative_type="sentence",
    count=1,
    index=0,
):
    return GenerationTask(
        index=index,
        anchor_type=anchor_type,
        positive_type=positive_type,
        partial_type=partial_type,
        negative_type=negative_type,
        example_count=count,
        topic_hint="community life",
    )


def _text(content_type, role, unique=0):
    if content_type == "short_query":
        choices = {
            "anchor": f"garden opening schedule district {unique}",
            "positive": f"public garden hours zone {unique}",
            "partial": f"library gardening workshops area {unique}",
            "negative": f"laptop battery replacement model {unique}",
        }
    elif content_type == "sentence":
        choices = {
            "anchor": f"The city opened public garden number {unique} beside the library today.",
            "positive": f"Municipal garden number {unique} opened next to the main library today.",
            "partial": f"Library branch number {unique} announced gardening workshops for local residents.",
            "negative": f"Laptop model number {unique} loses power after every operating system update.",
        }
    elif content_type == "paragraph":
        choices = {
            "anchor": _paragraph(f"Anchor group {unique}"),
            "positive": _paragraph(f"Positive group {unique}"),
            "partial": _paragraph(f"Partial group {unique}"),
            "negative": _paragraph(f"Negative group {unique}"),
        }
    else:
        raise AssertionError(content_type)
    return choices[role]


def _payload(task, unique=0):
    return {
        "anchor": _text(task.anchor_type, "anchor", unique),
        "positive": _text(task.positive_type, "positive", unique),
        "positive_score": 0.88,
        "partial": _text(task.partial_type, "partial", unique),
        "partial_score": 0.51,
        "negative": _text(task.negative_type, "negative", unique),
        "negative_score": 0.21,
    }


def _infer_type(text):
    words = len(text.split())
    sentence_count = len(re.findall(r"[.!?]+(?:\s|$)", text))
    if words >= 35 and sentence_count >= 2:
        return "paragraph"
    if words <= 8:
        return "short_query"
    return "sentence"


def test_classify_source_text_uses_all_length_groups():
    assert classify_source_text("weather tomorrow") == "short_query"
    assert classify_source_text("The train reaches the central station before noon today.") == "sentence"
    assert classify_source_text("word " * 30) == "paragraph"


def test_load_few_shots_stratifies_content_type_and_available_score_bands(tmp_path):
    source_records = []
    anchors = {
        "short_query": "cheap train tickets",
        "sentence": "The workshop begins in the main hall after lunch today.",
        "paragraph": "Long source " + "word " * 29,
    }
    for content_type, anchor in anchors.items():
        source_records.append(
            {
                "text": anchor,
                "embedding": [
                    [f"{content_type} positive comparison", 0.84],
                    [f"{content_type} partial comparison", 0.5],
                    [f"{content_type} negative comparison", 0.22],
                ],
            }
        )
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source_records), encoding="utf-8")

    samples = load_few_shots(source_path, per_content_type=3, seed=7)

    assert len(samples) == 9
    observed = Counter((sample["content_type"], sample["score_band"]) for sample in samples)
    assert observed == Counter(
        {
            (content_type, band): 1
            for content_type in CONTENT_TYPES
            for band in RELATION_BANDS
        }
    )


def test_load_few_shots_allows_a_source_band_missing_from_one_length_group(tmp_path):
    records = []
    for anchor in ("short search", "A complete sentence provides enough words for this test."):
        records.append(
            {"text": anchor, "embedding": [["positive", 0.8], ["partial", 0.5], ["negative", 0.2]]}
        )
    records.append(
        {
            "text": "Long paragraph source " + "word " * 28,
            "embedding": [["long positive", 0.8], ["long partial", 0.5], ["another partial", 0.6]],
        }
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(records), encoding="utf-8")

    samples = load_few_shots(source_path, per_content_type=3, seed=7)

    assert len(samples) == 9
    assert {sample["score_band"] for sample in samples} == set(RELATION_BANDS)


def test_build_tasks_guarantees_uniform_relation_and_ordered_type_pairs():
    tasks = build_tasks(num_examples=18, batch_size=2, seed=42)
    anchor_counts = Counter()
    cells = Counter()
    for task in tasks:
        anchor_counts[task.anchor_type] += task.example_count
        assert {task.positive_type, task.partial_type, task.negative_type} == set(CONTENT_TYPES)
        cells["positive", task.anchor_type, task.positive_type] += task.example_count
        cells["partial", task.anchor_type, task.partial_type] += task.example_count
        cells["negative", task.anchor_type, task.negative_type] += task.example_count

    assert len(tasks) == 9
    assert anchor_counts == {content_type: 6 for content_type in CONTENT_TYPES}
    assert len(cells) == 27
    assert set(cells.values()) == {2}


@pytest.mark.parametrize("num_examples", [1, 8, 10, 17])
def test_build_tasks_rejects_counts_that_cannot_be_exactly_uniform(num_examples):
    with pytest.raises(ValueError, match="uniform|divisible"):
        build_tasks(num_examples=num_examples, batch_size=2, seed=42)


def test_build_tasks_is_deterministic_for_a_fixed_seed():
    assert build_tasks(num_examples=90, batch_size=4, seed=123) == build_tasks(
        num_examples=90, batch_size=4, seed=123
    )


def test_validate_generated_batch_supports_heterogeneous_pairs():
    task = _task(
        anchor_type="short_query",
        positive_type="paragraph",
        partial_type="sentence",
        negative_type="short_query",
    )
    payload = _payload(task)

    examples = validate_generated_batch([payload], task=task)

    assert examples == [
        {
            "text": payload["anchor"],
            "embedding": [
                [payload["positive"], 0.88],
                [payload["partial"], 0.51],
                [payload["negative"], 0.21],
            ],
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "pair_index", "expected"),
    [
        ("positive_score", 0.69, 0, 0.7),
        ("positive_score", 1.01, 0, 1.0),
        ("partial_score", 0.34, 1, 0.35),
        ("partial_score", 0.66, 1, 0.65),
        ("negative_score", 0.35, 2, 0.34),
        ("negative_score", -0.01, 2, 0.0),
    ],
)
def test_validate_generated_batch_softly_clamps_score_bands(field, value, pair_index, expected):
    task = _task()
    payload = _payload(task)
    payload[field] = value
    result = validate_generated_batch([payload], task=task)

    assert result[0]["embedding"][pair_index][1] == expected


def test_validate_generated_batch_drops_unusable_item_for_top_up():
    task = _task()
    payload = _payload(task)
    payload["negative_score"] = True

    assert validate_generated_batch([payload], task=task, drop_invalid=True) == []


def test_runtime_text_type_constraints_are_soft():
    task = _task(positive_type="paragraph")
    payload = _payload(task)
    payload["positive"] = _text("sentence", "positive")

    result = validate_generated_batch([payload], task=task)

    assert result[0]["embedding"][0][0] == payload["positive"]
    with pytest.raises(InvalidExampleError, match="paragraph"):
        validate_text_type(payload["positive"], "paragraph", "positive", strict=True)


def test_prompt_provides_execution_order_quantity_and_matching_example():
    task = _task(
        anchor_type="short_query",
        positive_type="paragraph",
        partial_type="sentence",
        negative_type="short_query",
        count=4,
    )
    prompt = make_prompt(task, [])
    example = make_schedule_example(task)

    assert "QUANTITY: Return exactly 4" in prompt
    assert "SILENT EXECUTION ORDER" in prompt
    assert example["anchor"] in prompt
    validate_text_type(example["anchor"], task.anchor_type, "anchor", strict=True)
    validate_text_type(example["positive"], task.positive_type, "positive", strict=True)
    validate_text_type(example["partial"], task.partial_type, "partial", strict=True)
    validate_text_type(example["negative"], task.negative_type, "negative", strict=True)


def test_schedule_examples_match_every_planned_form():
    for task in build_tasks(num_examples=9, batch_size=1, seed=42):
        example = make_schedule_example(task)
        validate_text_type(example["anchor"], task.anchor_type, "anchor", strict=True)
        validate_text_type(example["positive"], task.positive_type, "positive", strict=True)
        validate_text_type(example["partial"], task.partial_type, "partial", strict=True)
        validate_text_type(example["negative"], task.negative_type, "negative", strict=True)


def test_process_top_up_requests_only_missing_quantity(monkeypatch):
    task = _task()
    generator = object.__new__(GeminiSTSGenerator)
    generator._validation_retries = 1
    responses = [[], [_payload(task)]]
    requested_counts = []

    def generate(requested_task):
        requested_counts.append(requested_task.example_count)
        return responses.pop(0)

    monkeypatch.setattr(generator, "generate", generate)

    result = generator.process(task)

    assert result.accepted
    assert responses == []
    assert requested_counts == [1, 1]


def test_process_keeps_valid_items_and_tops_up_only_dropped_items(monkeypatch):
    task = _task(count=2)
    generator = object.__new__(GeminiSTSGenerator)
    generator._validation_retries = 1
    invalid = _payload(task, 1)
    invalid["negative_score"] = "not a number"
    responses = [[_payload(task, 0), invalid], [_payload(task, 1)]]
    requested_counts = []

    def generate(requested_task):
        requested_counts.append(requested_task.example_count)
        return responses.pop(0)

    monkeypatch.setattr(generator, "generate", generate)

    result = generator.process(task)

    assert result.accepted
    assert len(result.examples) == 2
    assert requested_counts == [2, 1]


def test_run_generation_writes_uniform_mixed_pairs(tmp_path):
    tasks = build_tasks(num_examples=9, batch_size=1, seed=42)

    class FakeGenerator:
        def process(self, task):
            return GenerationResult(
                task=task,
                examples=validate_generated_batch([_payload(task, task.index)], task=task),
            )

    output_path = tmp_path / "generated.jsonl"
    accepted, rejected = run_generation(
        tasks,
        FakeGenerator(),
        output_path=output_path,
        append=False,
        workers=3,
        progress_every=2,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    observed_cells = Counter()
    for record in records:
        anchor_type = _infer_type(record["text"])
        assert len(record["embedding"]) == 3
        for band, pair in zip(RELATION_BANDS, record["embedding"], strict=True):
            observed_cells[band, anchor_type, _infer_type(pair[0])] += 1

    assert accepted == 9
    assert rejected == {}
    assert len(observed_cells) == 27
    assert set(observed_cells.values()) == {1}


def test_jsonl_writer_flushes_every_record():
    class TrackingHandle:
        def __init__(self):
            self.parts = []
            self.flush_count = 0

        def write(self, value):
            self.parts.append(value)

        def flush(self):
            self.flush_count += 1

    handle = TrackingHandle()
    _write_json_line(handle, {"text": "anchor", "embedding": []})

    assert handle.flush_count == 1
    assert "".join(handle.parts).endswith("\n")


def test_run_generation_keeps_completed_requests_after_one_failed_batch(tmp_path):
    tasks = build_tasks(num_examples=27, batch_size=2, seed=42)

    class PartiallyFailingGenerator:
        def process(self, task):
            if task.index == 0:
                return GenerationResult(task=task, rejection_reason="validation: forced")
            payload = [_payload(task, task.index * 10 + offset) for offset in range(task.example_count)]
            return GenerationResult(
                task=task,
                examples=validate_generated_batch(payload, task=task),
            )

    output_path = tmp_path / "streamed_partial.jsonl"
    accepted, rejected = run_generation(
        tasks,
        PartiallyFailingGenerator(),
        output_path=output_path,
        append=False,
        workers=4,
        progress_every=20,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    observed_cells = Counter()
    for record in records:
        anchor_type = _infer_type(record["text"])
        for band, pair in zip(RELATION_BANDS, record["embedding"], strict=True):
            observed_cells[band, anchor_type, _infer_type(pair[0])] += 1

    assert accepted == 25
    assert len(records) == 25
    assert len(observed_cells) == 27
    assert set(observed_cells.values()) == {1, 3}
    assert rejected["validation"] == 2
    assert "balance_trim" not in rejected


def test_append_rejects_legacy_two_comparison_output(tmp_path):
    output_path = tmp_path / "legacy.jsonl"
    output_path.write_text(
        json.dumps({"text": "anchor", "embedding": [["positive", 0.8], ["negative", 0.2]]})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected exactly three"):
        _load_existing_anchors(output_path)


def test_main_dry_run_needs_no_api_key(tmp_path):
    source_records = []
    anchors = [
        "quiet coffee shop",
        "The small cafe opens early every weekday morning for local commuters.",
        "Source paragraph " + "word " * 29,
    ]
    for anchor in anchors:
        source_records.append(
            {
                "text": anchor,
                "embedding": [["positive", 0.8], ["partial", 0.5], ["negative", 0.2]],
            }
        )
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source_records), encoding="utf-8")

    assert main(["--source", str(source_path), "--num-examples", "9", "--dry-run"]) == 0
