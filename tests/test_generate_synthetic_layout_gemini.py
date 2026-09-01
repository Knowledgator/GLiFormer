import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "data_prod" / "generate_synthetic_layout_gemini.py"
)
SPEC = importlib.util.spec_from_file_location("generate_synthetic_layout_gemini", SCRIPT_PATH)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


def _task(*block_names, seed=0, document_type="Synthetic test document"):
    blocks = tuple(generator.BlockSpec(name, True) for name in block_names)
    schema = generator.DocumentSchema("Test category", document_type, blocks)
    return generator.GenerationTask(
        index=0,
        sample_index=0,
        seed=seed,
        schema=schema,
        selected_blocks=blocks,
        locale_hint="fictional test locale",
        date_hint="contemporary dates",
        density_hint="moderately detailed",
    )


def _style(*, size=12, family="sans", border=True):
    return generator.BlockStyle(
        requested_font_size=size,
        word_spacing=4,
        line_spacing=1.12,
        paragraph_spacing=3,
        padding=8,
        alignment="left",
        font_family=family,
        bold=False,
        border=border,
    )


def _build_record(
    task,
    placements,
    content_by_name,
    *,
    styles=None,
    page_count=1,
    images_dir=None,
    token_order_strategy="spatial",
):
    if styles is None:
        styles = {placement.name: _style() for placement in placements}
    return generator.build_training_record(
        task,
        model="test-model",
        page_count=page_count,
        placements=placements,
        styles=styles,
        content_by_name=content_by_name,
        category_labels=[task.schema.category],
        document_type_labels=[task.schema.document_type],
        used_layout_fallback=False,
        images_dir=images_dir,
        token_order_strategy=token_order_strategy,
    )


def _indices(spans):
    return {index for start, end in spans for index in range(start, end + 1)}


def test_taxonomy_contains_all_categories_and_document_types():
    schemas = generator.load_document_schemas(generator.DEFAULT_SCHEMA_PATH)

    assert len(schemas) == 100
    assert len({schema.category for schema in schemas}) == 10
    assert all(any(block.required for block in schema.blocks) for schema in schemas)


def test_required_blocks_are_never_sampled_out():
    schemas = generator.load_document_schemas(generator.DEFAULT_SCHEMA_PATH)
    tasks = generator.build_tasks(
        schemas,
        samples_per_document=1,
        optional_probability=0.0,
        base_seed=42,
    )

    assert len(tasks) == 100
    for task in tasks:
        assert task.selected_blocks
        assert all(block.required for block in task.selected_blocks)


def test_api_key_can_be_read_from_dotenv_without_exposing_other_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UNRELATED=do-not-use\nexport GEMINI_API_KEY='test-secret'\n",
        encoding="utf-8",
    )

    assert generator.read_api_key_from_env_file(env_file, ["GEMINI_API_KEY"]) == "test-secret"
    assert generator.read_api_key_from_env_file(env_file, ["GOOGLE_API_KEY"]) is None


def test_optional_blocks_are_balanced_across_multi_sample_runs():
    schema = generator.DocumentSchema(
        "Test category",
        "Balanced optional fixture",
        (
            generator.BlockSpec("required_field", True),
            generator.BlockSpec("optional_a", False),
            generator.BlockSpec("optional_b", False),
        ),
    )
    tasks = generator.build_tasks(
        [schema],
        samples_per_document=5,
        optional_probability=0.4,
        base_seed=42,
    )

    assert all(
        any(block.name == "required_field" for block in task.selected_blocks) for task in tasks
    )
    for optional_name in ("optional_a", "optional_b"):
        assert (
            sum(
                any(block.name == optional_name for block in task.selected_blocks) for task in tasks
            )
            == 2
        )


def test_seeded_layout_jitter_breaks_coordinate_grid_without_overlap():
    placements = [
        generator.BlockPlacement("field_a", 0, 50, 50, 400, 300),
        generator.BlockPlacement("field_b", 0, 520, 50, 400, 300),
        generator.BlockPlacement("field_c", 0, 50, 400, 870, 300),
    ]
    adjusted = generator.jitter_layout_plan(placements, rng=generator.random.Random(7))

    assert any(coordinate % 5 for block in adjusted for coordinate in block.bbox)
    payload = {
        "page_count": 1,
        "blocks": [
            {
                "name": block.name,
                "page": block.page,
                "x": block.x,
                "y": block.y,
                "width": block.width,
                "height": block.height,
            }
            for block in adjusted
        ],
    }
    _, validated = generator.validate_layout_plan(
        payload,
        expected_names=["field_a", "field_b", "field_c"],
        max_pages=1,
    )
    assert validated == adjusted


def test_local_layout_renderer_builds_aligned_glinext_record():
    schemas = generator.load_document_schemas(generator.DEFAULT_SCHEMA_PATH)
    invoice = next(schema for schema in schemas if schema.document_type == "Invoice")
    task = generator.build_tasks(
        [invoice],
        samples_per_document=1,
        optional_probability=1.0,
        base_seed=7,
    )[0]
    page_count, placements = generator.fallback_layout_plan(
        [block.name for block in task.selected_blocks],
        max_pages=3,
        rng=generator.random.Random(task.seed),
    )
    styles = generator.sample_styles(placements, rng=generator.random.Random(task.seed ^ 123))
    content = {
        placement.name: f"{placement.name.replace('_', ' ').title()} REF-1234"
        for placement in placements
    }
    category_labels = list(dict.fromkeys(schema.category for schema in schemas))
    document_type_labels = [schema.document_type for schema in schemas]

    record = generator.build_training_record(
        task,
        model="test-model",
        page_count=page_count,
        placements=placements,
        styles=styles,
        content_by_name=content,
        category_labels=category_labels,
        document_type_labels=document_type_labels,
        used_layout_fallback=True,
    )

    assert len(record["tokenized_text"]) == len(record["bboxes"])
    assert len(record["tokenized_text"]) == len(record["layout"])
    assert len(record["blocks"]) == len(task.selected_blocks)
    assert len(record["extraction"][0]["ner"]) == len(task.selected_blocks)
    assert record["classification"][1]["true_labels"] == ["Invoice"]
    assert all(0 <= coordinate <= 1000 for box in record["bboxes"] for coordinate in box)


def test_two_stage_generation_calls_layout_then_content():
    schemas = generator.load_document_schemas(generator.DEFAULT_SCHEMA_PATH)
    invoice = next(schema for schema in schemas if schema.document_type == "Invoice")
    task = generator.build_tasks(
        [invoice],
        samples_per_document=1,
        optional_probability=0.0,
        base_seed=19,
    )[0]
    _, placements = generator.fallback_layout_plan(
        [block.name for block in task.selected_blocks],
        max_pages=3,
        rng=generator.random.Random(task.seed),
    )
    layout_payload = {
        "page_count": max(placement.page for placement in placements) + 1,
        "blocks": [
            {
                "name": placement.name,
                "page": placement.page,
                "x": placement.x,
                "y": placement.y,
                "width": placement.width,
                "height": placement.height,
            }
            for placement in placements
        ],
    }
    content_payload = {
        "blocks": [
            {"name": placement.name, "content": "Fictional value REF-1234"}
            for placement in placements
        ]
    }
    responses = [json.dumps(layout_payload), json.dumps(content_payload)]
    calls = []

    instance = object.__new__(generator.GeminiSyntheticLayoutGenerator)
    instance._model = "fake-gemini"
    instance._category_labels = list(dict.fromkeys(schema.category for schema in schemas))
    instance._document_type_labels = [schema.document_type for schema in schemas]
    instance._max_pages = 3
    instance._layout_temperature = 0.8
    instance._content_temperature = 1.0
    instance._layout_max_output_tokens = 4096
    instance._content_max_output_tokens = 8192
    instance._validation_retries = 0

    def fake_call(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return responses[len(calls) - 1]

    instance._call = fake_call
    result = instance.process(task)

    assert result.accepted, result.rejection_reason
    assert len(calls) == 2
    assert calls[0][0].startswith("Plan the geometry")
    assert calls[1][0].startswith("Write the complete block content")
    assert result.record["_source"]["layout_fallback"] is False


def test_process_repairs_newline_overflow_instead_of_rejecting_document():
    task = _task("notes", seed=81, document_type="Overflow recovery fixture")
    layout_payload = {
        "page_count": 1,
        "blocks": [{"name": "notes", "page": 0, "x": 100, "y": 100, "width": 800, "height": 60}],
    }
    verbose_content = {"blocks": [{"name": "notes", "content": "A\nB\nC\nD\nE\nF\nG\nH"}]}
    responses = [
        json.dumps(layout_payload),
        json.dumps(verbose_content),
        json.dumps(verbose_content),
    ]
    calls = []
    instance = object.__new__(generator.GeminiSyntheticLayoutGenerator)
    instance._model = "fake-gemini"
    instance._category_labels = [task.schema.category]
    instance._document_type_labels = [task.schema.document_type]
    instance._max_pages = 1
    instance._layout_temperature = 0.8
    instance._content_temperature = 1.0
    instance._layout_max_output_tokens = 4096
    instance._content_max_output_tokens = 8192
    instance._validation_retries = 1

    def fake_call(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return responses[len(calls) - 1]

    instance._call = fake_call
    result = instance.process(task)

    assert result.accepted, result.rejection_reason
    assert len(calls) == 3
    assert "previous response failed exact local font-metric validation" in calls[-1][0]
    assert result.record["_source"]["content_attempts"] == 2
    assert result.record["_source"]["compacted_blocks"] == ["notes"]


def test_table_anchors_and_headers_are_o_tokens_and_cells_are_aligned():
    task = _task("line_items_table")
    placements = [generator.BlockPlacement("line_items_table", 0, 80, 100, 840, 500)]
    record = _build_record(
        task,
        placements,
        {
            "line_items_table": (
                "Item | Qty | Amount\nCopper washer | 2 | $20.00\nSteel bracket | 1 | $15.00"
            )
        },
    )

    block = record["blocks"][0]
    labeled = _indices([[start, end] for start, end, _ in record["extraction"][0]["ner"]])
    anchor_indices = _indices(block["anchor_token_spans"])
    header_indices = _indices(block["table_header_token_spans"])
    value_indices = _indices(block["value_token_spans"])

    assert anchor_indices
    assert header_indices
    assert value_indices
    assert anchor_indices.isdisjoint(labeled)
    assert header_indices.isdisjoint(labeled)
    assert value_indices == labeled
    assert "|" not in record["tokenized_text"]
    assert all(
        entry["page"] == record["page_ids"][index] for index, entry in enumerate(record["layout"])
    )

    header_spans = block["table_header_token_spans"]
    value_spans = block["value_token_spans"]

    def minimum_x(span):
        start, end = span
        return min(record["bboxes"][index][0] for index in range(start, end + 1))

    assert len(header_spans) == 3
    assert len(value_spans) == 6
    assert [minimum_x(span) for span in header_spans] == sorted(
        minimum_x(span) for span in header_spans
    )
    assert [minimum_x(span) for span in value_spans[:3]] == sorted(
        minimum_x(span) for span in value_spans[:3]
    )
    assert [minimum_x(span) for span in value_spans[3:]] == sorted(
        minimum_x(span) for span in value_spans[3:]
    )


def test_exact_png_output_uses_single_and_multipage_conventions(tmp_path):
    single_task = _task("account_number", document_type="Single page fixture")
    single_record = _build_record(
        single_task,
        [generator.BlockPlacement("account_number", 0, 100, 120, 800, 180)],
        {"account_number": "AC-10492"},
        images_dir=tmp_path / "single",
    )

    assert isinstance(single_record["image"], str)
    assert single_record["image_page_ids"] == [0]
    single_path = Path(single_record["image"])
    assert single_path.is_file()
    with generator.Image.open(single_path) as image:
        assert image.format == "PNG"
        assert image.size == (generator.PAGE_PIXEL_WIDTH, generator.PAGE_PIXEL_HEIGHT)

    multi_task = _task(
        "account_number",
        "amount_due",
        document_type="Multipage fixture",
    )
    multi_record = _build_record(
        multi_task,
        [
            generator.BlockPlacement("account_number", 0, 100, 120, 800, 180),
            generator.BlockPlacement("amount_due", 1, 100, 120, 800, 180),
        ],
        {"account_number": "AC-10492", "amount_due": "$481.27"},
        page_count=2,
        images_dir=tmp_path / "multi",
    )

    assert isinstance(multi_record["image"], list)
    assert len(multi_record["image"]) == 2
    assert multi_record["image_page_ids"] == [0, 1]
    assert set(multi_record["page_ids"]) == {0, 1}
    for expected_page, image_name in enumerate(multi_record["image"]):
        image_path = Path(image_name)
        assert image_path.is_file()
        assert image_path.name.endswith(f"-page-{expected_page}.png")
        with generator.Image.open(image_path) as image:
            assert image.format == "PNG"
            assert image.size == (generator.PAGE_PIXEL_WIDTH, generator.PAGE_PIXEL_HEIGHT)


def test_overflow_is_compacted_at_realistic_font_floors():
    task = _task("notes", "items_table")
    placements = [
        generator.BlockPlacement("notes", 0, 80, 100, 840, 100),
        generator.BlockPlacement("items_table", 0, 80, 240, 840, 180),
    ]
    long_notes = " ".join(f"detail-{index:03d}" for index in range(180))
    long_table = (
        "Code | Units | Amount\n"
        + "\n".join(f"ITEM-{index:03d} | {index + 1} | ${index + 1}.00" for index in range(30))
        + "\nTotal |  | $465.00"
    )
    record = _build_record(
        task,
        placements,
        {"notes": long_notes, "items_table": long_table},
        styles={"notes": _style(size=16), "items_table": _style(size=12)},
    )

    assert set(record["_source"]["compacted_blocks"]) == {"notes", "items_table"}
    blocks = {block["name"]: block for block in record["blocks"]}
    assert blocks["notes"]["style"]["font_size"] >= generator.MIN_RENDER_FONT_SIZE
    assert blocks["items_table"]["style"]["font_size"] >= generator.MIN_TABLE_FONT_SIZE
    assert blocks["notes"]["style"]["font_size"] / 16 >= 0.75
    assert blocks["items_table"]["style"]["font_size"] / 12 >= 0.75
    assert blocks["notes"]["dropped_tokens_or_rows"] > 0
    assert blocks["items_table"]["dropped_tokens_or_rows"] > 0
    assert "Total" in blocks["items_table"]["content"]
    assert all(start <= end for start, end, _ in record["extraction"][0]["ner"])


def test_long_unbroken_identifier_is_split_without_clipped_boxes():
    task = _task("reference_code")
    placement = generator.BlockPlacement("reference_code", 0, 100, 100, 800, 700)
    identifier = "REF/" + "A" * 600
    record = _build_record(
        task,
        [placement],
        {"reference_code": identifier},
        styles={"reference_code": _style(size=12, family="mono")},
    )

    block = record["blocks"][0]
    value_indices = sorted(_indices(block["value_token_spans"]))
    value_tokens = [record["tokenized_text"][index] for index in value_indices]
    assert len(value_tokens) > 1
    assert "".join(value_tokens) == identifier
    assert block["compacted"] is False
    left, top, right, bottom = block["value_bbox"]
    for index in value_indices:
        x0, y0, x1, y1 = record["bboxes"][index]
        assert left <= x0 < x1 <= right
        assert top <= y0 < y1 <= bottom
        assert all(0 <= coordinate <= 1000 for coordinate in (x0, y0, x1, y1))


def test_layout_dependent_order_separates_context_and_reports_quality():
    task = _task("customer_name", "policy_number", "effective_date", seed=0)
    placements = [
        generator.BlockPlacement("customer_name", 0, 100, 100, 800, 120),
        generator.BlockPlacement("policy_number", 0, 100, 300, 800, 120),
        generator.BlockPlacement("effective_date", 0, 100, 500, 800, 120),
    ]
    record = _build_record(
        task,
        placements,
        {
            "customer_name": "Morgan Lee",
            "policy_number": "PX-009471",
            "effective_date": "2026-08-30",
        },
        token_order_strategy="layout-dependent",
    )

    assert record["_source"]["token_order_mode"] == "anchors_first"
    anchor_indices = sorted(
        index for block in record["blocks"] for index in _indices(block["anchor_token_spans"])
    )
    value_indices = sorted(
        index for block in record["blocks"] for index in _indices(block["value_token_spans"])
    )
    assert max(anchor_indices) < min(value_indices)

    quality = record["_source"]["layout_dependency"]
    assert set(quality) == {
        "score",
        "o_token_ratio",
        "exact_label_leakage_ratio",
        "component_label_leakage_ratio",
        "median_anchor_token_distance",
        "sequence_nearest_anchor_accuracy",
        "geometric_nearest_anchor_accuracy",
        "repeated_surface_kind_count",
        "max_entity_tokens",
        "passes_heuristic_gate",
    }
    assert 0.0 <= quality["score"] <= 1.0
    assert quality["o_token_ratio"] >= generator.TARGET_MIN_O_TOKEN_RATIO
    assert quality["median_anchor_token_distance"] > 0
    assert quality["max_entity_tokens"] <= generator.MAX_ENTITY_TOKENS


def test_resolved_rejections_are_pruned_and_latest_failure_is_retained(tmp_path):
    rejection_path = tmp_path / "records.jsonl.rejections.jsonl"
    rejection_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "accepted", "reason": "old failure"},
                {"id": "unresolved", "reason": "first failure"},
                {"id": "unresolved", "reason": "latest failure"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    remaining = generator.prune_resolved_rejections(
        rejection_path,
        accepted_ids={"accepted"},
    )
    rows = [json.loads(line) for line in rejection_path.read_text(encoding="utf-8").splitlines()]

    assert remaining == 1
    assert rows == [{"id": "unresolved", "reason": "latest failure"}]
