from scripts.remove_missing_structuring_fields import clean_row


def test_clean_row_removes_only_the_invalid_record_field_value():
    row = {
        "text": "Alice works at Acme. Bob is mentioned.",
        "structuring": {
            "person": [
                {"name": "Alice", "employer": "Acme"},
                {"name": "Bob", "employer": "Invented Corp"},
            ],
        },
    }

    cleaned, stats = clean_row(
        row,
        drop_row_threshold=0.5,
        match_mode="exact",
    )

    assert cleaned["structuring"]["person"] == [
        {"name": "Alice", "employer": "Acme"},
        {"name": "Bob"},
    ]
    assert stats["field_values_marked_for_removal"] == 1


def test_clean_row_removes_only_the_ungrounded_list_item():
    row = {
        "text": "Damage included a window and keyed paint.",
        "structuring": {
            "claim": [
                {
                    "damage": ["window", "invented roof", "keyed paint"],
                    "status": "Damage",
                },
            ],
        },
    }

    cleaned, _ = clean_row(
        row,
        drop_row_threshold=0.5,
        match_mode="exact",
    )

    assert cleaned["structuring"]["claim"][0]["damage"] == [
        "window",
        "keyed paint",
    ]


def test_clean_row_updates_instance_metadata_after_pruning():
    row = {
        "text": "Alice",
        "n_objects_extracted": 2,
        "structuring": {
            "person": [
                {"name": "Alice"},
                {"name": "Invented"},
            ],
        },
    }

    cleaned, stats = clean_row(
        row,
        drop_row_threshold=0.5,
        match_mode="exact",
    )

    assert cleaned["structuring"]["person"] == [{"name": "Alice"}]
    assert cleaned["n_objects_extracted"] == 1
    assert stats["metadata_counts_updated"] == 1


def test_clean_row_can_drop_samples_above_model_capacity():
    row = {
        "text": "A B C",
        "structuring": {
            "record": [
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
            ],
        },
    }

    cleaned, stats = clean_row(
        row,
        drop_row_threshold=0.5,
        match_mode="exact",
        max_instances=2,
    )

    assert cleaned is None
    assert stats["rows_dropped_over_capacity"] == 1
