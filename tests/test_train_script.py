import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from gliner.utils import load_config_as_namespace as read_config_namespace

from train import (
    apply_checkpoint_config_overrides,
    build_training_kwargs,
    load_json_data,
    main,
    prepare_training_records,
    resolve_checkpoint_mode,
    validate_checkpoint_architecture,
    validate_resume_training_args,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_load_jsonl_dataset(tmp_path):
    dataset_path = tmp_path / "train.jsonl"
    rows = [
        {"text": "Apple is a company."},
        {"tokenized_text": ["Paris", "is", "beautiful"]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )

    assert load_json_data(dataset_path) == rows


def test_load_ndjson_dataset(tmp_path):
    dataset_path = tmp_path / "train.ndjson"
    dataset_path.write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")

    assert load_json_data(dataset_path) == [{"text": "one"}, {"text": "two"}]


def test_load_json_array_dataset(tmp_path):
    dataset_path = tmp_path / "train.json"
    rows = [{"text": "one"}, {"text": "two"}]
    dataset_path.write_text(json.dumps(rows), encoding="utf-8")

    assert load_json_data(dataset_path) == rows


def test_load_multiple_dataset_sources_in_order(tmp_path):
    jsonl_path = tmp_path / "nested.jsonl"
    json_path = tmp_path / "repeated.json"
    jsonl_path.write_text('{"text": "nested"}\n', encoding="utf-8")
    json_path.write_text(
        json.dumps([{"text": "first"}, {"text": "second"}]),
        encoding="utf-8",
    )

    assert load_json_data([jsonl_path, json_path]) == [
        {"text": "nested"},
        {"text": "first"},
        {"text": "second"},
    ]


def test_load_multiple_dataset_sources_with_repeat(tmp_path):
    hierarchical_path = tmp_path / "hierarchical.jsonl"
    flat_path = tmp_path / "flat.jsonl"
    hierarchical_path.write_text(
        '{"text": "hierarchical-one"}\n{"text": "hierarchical-two"}\n',
        encoding="utf-8",
    )
    flat_path.write_text('{"text": "flat"}\n', encoding="utf-8")

    assert load_json_data(
        [
            {"path": hierarchical_path, "repeat": 2},
            flat_path,
        ]
    ) == [
        {"text": "hierarchical-one"},
        {"text": "hierarchical-two"},
        {"text": "hierarchical-one"},
        {"text": "hierarchical-two"},
        {"text": "flat"},
    ]


def test_load_dataset_source_namespace_from_yaml_config(tmp_path):
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"text": "example"}\n', encoding="utf-8")

    source = Namespace(path=str(dataset_path), repeat=2)

    assert load_json_data(source) == [
        {"text": "example"},
        {"text": "example"},
    ]


def test_multitask_label_augmentation_example_is_disabled_by_default():
    payload = yaml.safe_load(
        (CONFIG_DIR / "multitask.yaml").read_text(encoding="utf-8")
    )

    augmentation = payload["training"]["label_augmentation"]

    assert augmentation["enabled"] is False
    assert set(augmentation["tasks"]) == {
        "classification",
        "ner",
        "joint_relex",
        "structuring",
    }
    assert augmentation["defaults"]["pool_scope"] == "task"


def test_train_script_forwards_nested_label_augmentation(monkeypatch, tmp_path):
    supplied_augmentation = Namespace(
        enabled=True,
        seed=19,
        defaults=Namespace(
            enabled=True,
            shuffle_probability=0.2,
            drop_probability=0.1,
            add_probability=0.3,
            preserve_positive_labels=True,
            min_labels_per_group=1,
            max_added_labels=4,
            pool_scope="task",
        ),
        tasks=Namespace(joint_relex=Namespace(add_probability=1.0)),
    )
    training = Namespace(
        prev_path=None,
        resume_from_checkpoint=None,
        num_steps=1,
        scheduler_type="linear",
        warmup_ratio=0.0,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        lr_encoder=1e-5,
        lr_others=3e-5,
        weight_decay_encoder=0.0,
        weight_decay_other=0.0,
        max_grad_norm=1.0,
        focal_loss_alpha=0.9,
        focal_loss_gamma=2.0,
        focal_loss_prob_margin=0.0,
        loss_reduction="sum",
        negatives=1.0,
        masking="none",
        eval_every=1,
        save_total_limit=1,
        train_head_only=False,
        freeze_components=None,
        classification_parent_name_dropout=0.0,
        label_augmentation=supplied_augmentation,
        bf16=False,
        compile_model=False,
    )
    config = Namespace(
        model=Namespace(model_variant="text"),
        data=Namespace(
            root_dir=str(tmp_path / "output"),
            train_data="train.json",
            val_data="none",
        ),
        training=training,
    )
    calls = {}

    class _Model:
        model = SimpleNamespace(heads={"classification": object()})

        def to(self, **kwargs):
            calls["dtype"] = kwargs["dtype"]
            return self

        def train_model(self, **kwargs):
            calls["train_model"] = kwargs

    monkeypatch.setattr("train.load_config_as_namespace", lambda path: config)
    monkeypatch.setattr("train.load_json_data", lambda path: [{"text": "x"}])
    monkeypatch.setattr("train.build_model", lambda model_cfg, train_cfg: _Model())

    main("unused.yaml")

    assert calls["train_model"]["label_augmentation"] == {
        "enabled": True,
        "seed": 19,
        "defaults": {
            "enabled": True,
            "shuffle_probability": 0.2,
            "drop_probability": 0.1,
            "add_probability": 0.3,
            "preserve_positive_labels": True,
            "min_labels_per_group": 1,
            "max_added_labels": 4,
            "pool_scope": "task",
        },
        "tasks": {"joint_relex": {"add_probability": 1.0}},
    }


def test_training_kwargs_forward_runtime_controls_and_nested_arguments():
    kwargs = build_training_kwargs(
        {
            "num_steps": 0,
            "train_batch_size": 2,
            "lr_encoder": 0.0,
            "use_cpu": True,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "eval_every": 7,
            "trainer_args": {
                "seed": 19,
                "fp16": False,
            },
        },
        has_eval=True,
    )

    assert kwargs["max_steps"] == 0
    assert kwargs["learning_rate"] == 0.0
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["per_device_eval_batch_size"] == 2
    assert kwargs["use_cpu"] is True
    assert kwargs["gradient_checkpointing"] is False
    assert kwargs["dataloader_num_workers"] == 0
    assert kwargs["seed"] == 19
    assert kwargs["fp16"] is False
    assert kwargs["save_steps"] == 7
    assert kwargs["logging_steps"] == 7
    assert kwargs["eval_steps"] == 7
    assert kwargs["eval_strategy"] == "steps"


def test_training_kwargs_reject_conflicting_alias_and_nested_value():
    with pytest.raises(ValueError, match="Conflicting values.*max_steps"):
        build_training_kwargs(
            {
                "num_steps": 10,
                "trainer_args": {"max_steps": 20},
            },
            has_eval=False,
        )


def test_training_kwargs_reject_unknown_top_level_option():
    with pytest.raises(ValueError, match="Unknown training option.*lern_rate"):
        build_training_kwargs(
            {"lern_rate": 1e-5},
            has_eval=False,
        )


def test_prev_path_and_full_resume_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_checkpoint_mode(
            {
                "prev_path": "weights",
                "resume_from_checkpoint": "checkpoint-10",
            }
        )


def test_weight_only_retrain_rejects_parameterized_head_changes():
    saved_config = SimpleNamespace(
        classification_config=SimpleNamespace(scorer_type="dot")
    )
    model = SimpleNamespace(
        config=saved_config,
        model=SimpleNamespace(heads={"classification": object()}),
    )

    with pytest.raises(ValueError, match="classification_config.scorer_type"):
        validate_checkpoint_architecture(
            model,
            {"classification_config": {"scorer_type": "mlp"}},
        )


def test_full_resume_rejects_optimizer_parameter_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    torch.save(
        SimpleNamespace(learning_rate=1e-5, max_steps=100),
        checkpoint / "training_args.bin",
    )

    with pytest.raises(ValueError, match="learning_rate.*prev_path"):
        validate_resume_training_args(
            checkpoint,
            tmp_path,
            {"learning_rate": 2e-5, "max_steps": 100},
            {"learning_rate", "max_steps"},
        )


def test_train_script_forwards_relation_focal_settings(monkeypatch, tmp_path):
    config = read_config_namespace(CONFIG_DIR / "multitask.yaml")
    config.data.root_dir = str(tmp_path / "output")
    calls = {}

    class _Model:
        model = SimpleNamespace(heads={"classification": object()})

        def to(self, **kwargs):
            return self

        def train_model(self, **kwargs):
            calls["train_model"] = kwargs

    monkeypatch.setattr("train.load_config_as_namespace", lambda path: config)
    monkeypatch.setattr("train.load_json_data", lambda path: [{"text": "x"}])
    monkeypatch.setattr("train.build_model", lambda model_cfg, train_cfg: _Model())

    main("unused.yaml")

    assert calls["train_model"]["rel_focal_loss_alpha"] == pytest.approx(0.8)
    assert calls["train_model"]["rel_focal_loss_gamma"] == pytest.approx(2.0)


@pytest.mark.parametrize("repeat", [True, 0, -1, 1.5, "2", None])
def test_load_dataset_source_rejects_invalid_repeat(tmp_path, repeat):
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"text": "example"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="repeat.*positive integer"):
        load_json_data({"path": dataset_path, "repeat": repeat})


def test_load_dataset_source_requires_path():
    with pytest.raises(ValueError, match="must contain 'path'"):
        load_json_data({"repeat": 2})


def test_jsonl_error_reports_line_number(tmp_path):
    dataset_path = tmp_path / "broken.jsonl"
    dataset_path.write_text('{"text": "valid"}\n{"text": }\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl at line 2"):
        load_json_data(dataset_path)


def test_jsonl_requires_object_records(tmp_path):
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"at line 1, got list"):
        load_json_data(dataset_path)


def test_open_relex_only_training_drops_relation_free_extraction_rows():
    processor = SimpleNamespace(
        has_training_annotations=lambda row: bool(row.get("has_relations"))
    )
    model = SimpleNamespace(
        model=SimpleNamespace(heads={"open_relex": object()}),
        data_processor=SimpleNamespace(
            task_processors={"open_relex": processor}
        ),
    )
    records = [
        {"text": "NER only", "has_relations": False},
        {"text": "relation", "has_relations": True},
    ]

    filtered, dropped = prepare_training_records(records, model)

    assert filtered == [records[1]]
    assert dropped == 1


def test_mixed_task_training_keeps_relation_free_rows():
    model = SimpleNamespace(
        model=SimpleNamespace(
            heads={"ner": object(), "open_relex": object()}
        )
    )
    records = [{"text": "NER only"}, {"text": "relation"}]

    filtered, dropped = prepare_training_records(records, model)

    assert filtered is records
    assert dropped == 0


def test_open_relex_only_training_rejects_dataset_without_relations():
    processor = SimpleNamespace(has_training_annotations=lambda row: False)
    model = SimpleNamespace(
        model=SimpleNamespace(heads={"open_relex": object()}),
        data_processor=SimpleNamespace(
            task_processors={"open_relex": processor}
        ),
    )

    with pytest.raises(ValueError, match="no usable relation annotations"):
        prepare_training_records([{"text": "NER only"}], model)


def test_checkpoint_structuring_policy_is_overridden_from_active_config():
    anchor_layer = SimpleNamespace(normalization="none")
    head = SimpleNamespace(
        anchor_layer=anchor_layer,
        negatives=1.0,
        masking_mode="none",
        span_loss_reduction="sum",
    )
    checkpoint_config = SimpleNamespace(
        negatives=1.0,
        masking="none",
        span_loss_reduction="sum",
    )
    model = SimpleNamespace(
        config=SimpleNamespace(structuring_config=checkpoint_config),
        model=SimpleNamespace(heads={"structuring": head}),
    )

    applied = apply_checkpoint_config_overrides(
        model,
        {
            "structuring_config": {
                "negatives": 0.005,
                "masking": "global_weighted",
                "span_loss_reduction": "mean",
                "position_bucket_normalization": "center_rms",
            }
        },
    )

    assert set(applied) == {
        "negatives",
        "masking",
        "span_loss_reduction",
        "position_bucket_normalization",
    }
    assert checkpoint_config.negatives == 0.005
    assert checkpoint_config.masking == "global_weighted"
    assert head.negatives == 0.005
    assert head.masking_mode == "global_weighted"
    assert head.span_loss_reduction == "mean"
    assert head.anchor_layer.normalization == "center_rms"


def test_checkpoint_structuring_loss_weights_are_overridden():
    head = SimpleNamespace(
        assignment_loss_coef=1.0,
        entity_loss_coef=1.0,
        anchor_objectness_loss_coef=0.001,
        anchor_relations_loss_coef=0.1,
        record_anchor_layer=SimpleNamespace(normalization="none"),
    )
    checkpoint_config = SimpleNamespace(
        span_loss_coef=1.0,
        entity_loss_coef=1.0,
        anchor_objectness_loss_coef=0.001,
        anchor_relations_loss_coef=0.1,
        position_bucket_normalization="none",
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            structuring_config=checkpoint_config,
        ),
        model=SimpleNamespace(heads={"structuring": head}),
    )

    applied = apply_checkpoint_config_overrides(
        model,
        {
            "structuring_config": {
                "span_loss_coef": 4.0,
                "entity_loss_coef": 0.5,
                "anchor_objectness_loss_coef": 1.0,
                "anchor_relations_loss_coef": 1.0,
                "position_bucket_normalization": "center_rms",
            }
        },
    )

    assert set(applied) == {
        "span_loss_coef",
        "entity_loss_coef",
        "anchor_objectness_loss_coef",
        "anchor_relations_loss_coef",
        "position_bucket_normalization",
    }
    assert head.assignment_loss_coef == 4.0
    assert head.entity_loss_coef == 0.5
    assert head.anchor_objectness_loss_coef == 1.0
    assert head.anchor_relations_loss_coef == 1.0
    assert head.record_anchor_layer.normalization == "center_rms"


def test_checkpoint_task_focal_policy_updates_config_and_live_head():
    head = SimpleNamespace(loss_coef=1.0)
    checkpoint_config = SimpleNamespace(
        loss_coef=1.0,
        focal_loss_alpha=None,
        focal_loss_gamma=None,
        focal_loss_prob_margin=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(classification_config=checkpoint_config),
        model=SimpleNamespace(heads={"classification": head}),
    )

    applied = apply_checkpoint_config_overrides(
        model,
        {
            "classification_config": {
                "loss_coef": 0.25,
                "focal_loss_alpha": 0.8,
                "focal_loss_gamma": 1.5,
                "focal_loss_prob_margin": 0.1,
            }
        },
    )

    assert set(applied) == {
        "loss_coef",
        "focal_loss_alpha",
        "focal_loss_gamma",
        "focal_loss_prob_margin",
    }
    assert head.loss_coef == pytest.approx(0.25)
    assert checkpoint_config.loss_coef == pytest.approx(0.25)
    assert checkpoint_config.focal_loss_alpha == pytest.approx(0.8)
    assert checkpoint_config.focal_loss_gamma == pytest.approx(1.5)
    assert checkpoint_config.focal_loss_prob_margin == pytest.approx(0.1)


def test_checkpoint_structuring_component_focal_policy_is_overridden():
    head = SimpleNamespace(
        assignment_loss_coef=1.0,
        ner_focal_loss_alpha=None,
        matching_focal_loss_gamma=None,
        objectness_focal_loss_prob_margin=None,
        anchor_relations_focal_loss_alpha=None,
    )
    checkpoint_config = SimpleNamespace(
        span_loss_coef=1.0,
        assignment_loss_coef=1.0,
        ner_focal_loss_alpha=None,
        matching_focal_loss_gamma=None,
        objectness_focal_loss_prob_margin=None,
        anchor_relations_focal_loss_alpha=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(structuring_config=checkpoint_config),
        model=SimpleNamespace(heads={"structuring": head}),
    )

    apply_checkpoint_config_overrides(
        model,
        {
            "structuring_config": {
                "span_loss_coef": 2.0,
                "ner_focal_loss_alpha": 0.7,
                "matching_focal_loss_gamma": 1.0,
                "objectness_focal_loss_prob_margin": 0.2,
                "anchor_relations_focal_loss_alpha": 0.6,
            }
        },
    )

    assert checkpoint_config.assignment_loss_coef == pytest.approx(2.0)
    assert head.assignment_loss_coef == pytest.approx(2.0)
    assert head.ner_focal_loss_alpha == pytest.approx(0.7)
    assert head.matching_focal_loss_gamma == pytest.approx(1.0)
    assert head.objectness_focal_loss_prob_margin == pytest.approx(0.2)
    assert head.anchor_relations_focal_loss_alpha == pytest.approx(0.6)


def test_checkpoint_embedding_loss_is_overridden_without_changing_architecture():
    from glinext.tasks.embedding.model import CosineMarginLoss

    head = SimpleNamespace(
        loss_coef=0.5,
        similarity_fn="dot",
        loss=object(),
    )
    checkpoint_config = SimpleNamespace(
        loss_coef=0.5,
        similarity_fn="dot",
        loss_fn="mse",
        margin=None,
        projection_dim=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            embedding_config=checkpoint_config,
            embedding_loss_coef=0.5,
        ),
        model=SimpleNamespace(heads={"embedding": head}),
    )

    applied = apply_checkpoint_config_overrides(
        model,
        {
            "embedding_config": {
                "loss_coef": 1.0,
                "similarity_fn": "cosine",
                "loss_fn": "cosine_margin",
                "margin": 0.2,
                "projection_dim": 768,
            }
        },
    )

    assert set(applied) == {"loss_coef", "similarity_fn", "loss_fn", "margin"}
    assert head.loss_coef == 1.0
    assert head.similarity_fn == "cosine"
    assert isinstance(head.loss, CosineMarginLoss)
    assert head.loss.margin == pytest.approx(0.2)
    assert model.config.embedding_loss_coef == 1.0
    assert checkpoint_config.loss_fn == "cosine_margin"
    assert checkpoint_config.margin == pytest.approx(0.2)
    assert checkpoint_config.projection_dim is None


def test_checkpoint_embedding_projection_dropout_is_overridden_in_place():
    projection = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(4, 4),
    )
    head = SimpleNamespace(
        loss_coef=1.0,
        similarity_fn="cosine",
        projection_dropout=0.1,
        projection=projection,
    )
    checkpoint_config = SimpleNamespace(
        loss_coef=1.0,
        similarity_fn="cosine",
        loss_fn="mse",
        margin=None,
        projection_dim=4,
        projection_dropout=0.1,
        encoder_dropout=0.1,
    )
    backbone = torch.nn.Sequential(torch.nn.Dropout(0.1))
    backbone.config = SimpleNamespace(
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler_dropout=0.1,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            embedding_config=checkpoint_config,
            embedding_loss_coef=1.0,
        ),
        model=SimpleNamespace(
            heads={"embedding": head},
            token_rep_layer=SimpleNamespace(
                bert_layer=SimpleNamespace(model=backbone),
            ),
        ),
    )

    applied = apply_checkpoint_config_overrides(
        model,
        {
            "embedding_config": {
                "projection_dropout": 0.0,
                "encoder_dropout": 0.0,
            }
        },
    )

    assert applied == ("projection_dropout", "encoder_dropout")
    assert checkpoint_config.projection_dropout == 0.0
    assert checkpoint_config.encoder_dropout == 0.0
    assert head.projection_dropout == 0.0
    assert head.encoder_dropout == 0.0
    assert projection[2].p == 0.0
    assert backbone[0].p == 0.0
    assert backbone.config.hidden_dropout_prob == 0.0
    assert backbone.config.attention_probs_dropout_prob == 0.0
