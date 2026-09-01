from scripts.generate_structuring_data import (
    load_existing_text_fingerprints,
    text_fingerprint,
    tokenize_text,
    validate_and_clean_extraction,
)


def test_generated_tokens_match_default_inference_word_boundaries():
    assert tokenize_text("Claim #CLM-789456 costs $1,000.") == [
        "Claim",
        "#",
        "CLM-789456",
        "costs",
        "$",
        "1",
        ",",
        "000",
        ".",
    ]


def test_extraction_validation_drops_unknown_and_ungrounded_fields():
    fields = {
        "claim_number": ("str", [], ""),
        "status": ("str", [], ""),
        "damage": ("list", [], ""),
    }
    items = [
        {
            "claim_number": "clm-1",
            "status": "approved",
            "damage": ["window", "invented roof damage"],
            "hallucinated_key": "value",
        },
        {
            "claim_number": "CLM-2",
            "status": "not in the passage",
            "damage": ["paint"],
        },
    ]
    text = "CLM-1 was APPROVED for window damage. CLM-2 had paint damage."

    result = validate_and_clean_extraction(
        items,
        fields,
        min_instances=2,
        min_fields_per_item=2,
        text=text,
    )

    assert result == [
        {
            "claim_number": "clm-1",
            "status": "approved",
            "damage": ["window"],
        },
        {
            "claim_number": "CLM-2",
            "damage": ["paint"],
        },
    ]


def test_extraction_validation_rejects_more_than_bucket_capacity():
    items = [
        {"identifier": f"ID-{index}", "status": "open"}
        for index in range(4)
    ]
    text = " ".join(
        f"ID-{index} is open." for index in range(4)
    )

    assert validate_and_clean_extraction(
        items,
        {
            "identifier": ("str", [], ""),
            "status": ("str", [], ""),
        },
        text=text,
        max_instances=3,
    ) is None


def test_existing_output_fingerprints_prevent_append_duplicates(tmp_path):
    output = tmp_path / "generated.jsonl"
    output.write_text(
        '{"text": "same passage"}\n{"text": "another passage"}\n',
        encoding="utf-8",
    )

    fingerprints = load_existing_text_fingerprints(output)

    assert fingerprints == {
        text_fingerprint("same passage"),
        text_fingerprint("another passage"),
    }
