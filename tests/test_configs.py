from pathlib import Path

import pytest
import yaml

from gliformer.config import GLiFormerConfig, resolve_gliformer_config_class

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = tuple(sorted(CONFIG_DIR.glob("*.yaml")))

LEGACY_ANCHOR_FIELDS = {
    "anchor_mode",
    "num_fixed_slots",
    "anchor_num_heads",
    "anchor_num_layers",
    "anchor_refine_layers",
    "anchor_refine_heads",
    "anchor_refine_norm",
    "anchor_refine_layer_scale_init",
    "anchor_context_gate_init",
    "anchor_context_gate_trainable",
    "memory_position_embedding_type",
    "query_position_embedding_type",
    "memory_position_embedding_kwargs",
    "query_position_embedding_kwargs",
    "memory_position_in_values",
    "position_bucket_normalization",
    "position_bucket_attention_bias_type",
    "position_bucket_attention_sigma",
    "position_bucket_attention_bias_weight",
    "spatial_attention_bias_type",
    "spatial_attention_sigma",
    "spatial_attention_bias_weight",
    "shared_anchor_refine_layers",
    "shared_anchor_refine_heads",
}


def _mapping_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_keys(child)


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_shipped_config_uses_component_anchor_schema_and_loads(config_path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = payload["model"]

    legacy_fields = LEGACY_ANCHOR_FIELDS.intersection(
        _mapping_keys(model_config)
    )
    assert not legacy_fields, (
        f"{config_path.name} uses legacy anchor fields: "
        f"{sorted(legacy_fields)}"
    )

    config_class = resolve_gliformer_config_class(model_config)
    config_class(**model_config)


def test_joint_relex_uses_gliner_parameter_names():
    config = GLiFormerConfig(
        joint_relex_config={
            "relations_layer": "dot",
            "triples_layer": None,
            "relation_loss_coef": 2.0,
        },
    )
    assert config.joint_relex_config.relations_layer == "dot"
    assert config.joint_relex_config.triples_layer is None
    assert config.joint_relex_config.relation_loss_coef == 2.0
    assert config.relation_loss_coef == 2.0


def test_cosine_margin_embedding_config_round_trips_and_validates_margin():
    config = GLiFormerConfig(
        embedding_config={"loss_fn": "cosine_margin", "margin": 0.25},
    )

    assert config.embedding_config.loss_fn == "cosine_margin"
    assert config.embedding_config.margin == pytest.approx(0.25)
    assert GLiFormerConfig(**config.to_dict()).embedding_config.margin == pytest.approx(0.25)

    with pytest.raises(ValueError, match=r"margin must be in \[-1, 1\]"):
        GLiFormerConfig(
            embedding_config={"loss_fn": "cosine_margin", "margin": 1.1},
        )


def test_embedding_projection_dropout_round_trips_and_validates():
    config = GLiFormerConfig(
        embedding_config={
            "projection_dim": 32,
            "projection_dropout": 0.0,
            "encoder_dropout": 0.0,
        },
    )

    assert config.embedding_config.projection_dropout == 0.0
    assert config.embedding_config.encoder_dropout == 0.0
    reloaded = GLiFormerConfig(**config.to_dict())
    assert reloaded.embedding_config.projection_dropout == 0.0
    assert reloaded.embedding_config.encoder_dropout == 0.0

    for invalid in (-0.1, 1.0, float("inf")):
        with pytest.raises(ValueError, match="projection_dropout"):
            GLiFormerConfig(
                embedding_config={"projection_dropout": invalid},
            )

    for invalid in (-0.1, 1.0, float("inf")):
        with pytest.raises(ValueError, match="encoder_dropout"):
            GLiFormerConfig(
                embedding_config={"encoder_dropout": invalid},
            )


def test_multitask_joint_relex_configures_anchor_refinement():
    model_config = yaml.safe_load(
        (CONFIG_DIR / "multitask.yaml").read_text(encoding="utf-8")
    )["model"]

    config = resolve_gliformer_config_class(model_config)(**model_config)
    joint_config = config.joint_relex_config

    assert joint_config.relations_layer is None
    assert joint_config.effective_anchor_refine_layers() == 2
    assert joint_config.anchor_refinement["params"]["num_heads"] == 8
    assert joint_config.anchor_cross_attention_bias["type"] == (
        "gaussian_distance"
    )
    assert joint_config.anchor_memory_position_usage == "keys_and_values"


def test_legacy_strategy_specific_structuring_matcher_cost_is_discarded():
    config = GLiFormerConfig(
        default_ner_config=False,
        structuring_config={"position_bucket_matcher_cost": 0.1},
    )

    assert not hasattr(config.structuring_config, "position_bucket_matcher_cost")


def test_open_relex_training_config_uses_entity_first_head_without_standalone_ner():
    model_config = yaml.safe_load(
        (CONFIG_DIR / "open_relex.yaml").read_text(encoding="utf-8")
    )["model"]

    config = resolve_gliformer_config_class(model_config)(**model_config)

    assert model_config["default_ner_config"] is False
    assert config.ner_config is None
    assert config.open_relex_config.head_type == "open_relex"
    assert config.open_relex_config.represent_spans is True
    assert config.open_relex_config.entity_loss_coef == pytest.approx(1.0)
    assert config.open_relex_config.assignment_loss_coef == pytest.approx(1.0)
    assert not hasattr(config, "set_open_relex_config")


def test_open_relex_training_config_uses_entity_first_loss_policy():
    payload = yaml.safe_load(
        (CONFIG_DIR / "open_relex.yaml").read_text(encoding="utf-8")
    )
    model_config = payload["model"]
    head_config = model_config["open_relex_config"]
    config = resolve_gliformer_config_class(model_config)(**model_config)

    assert model_config["default_ner_config"] is False
    assert config.ner_config is None
    assert head_config["head_type"] == "open_relex"
    assert head_config["bio_loss_reduction"] == "sum"
    assert head_config["neg_spans_ratio"] == 0.0
    assert head_config["anchor_objectness"] is True


def test_multitask_config_has_independent_normalized_text_task_losses():
    payload = yaml.safe_load(
        (CONFIG_DIR / "multitask.yaml").read_text(encoding="utf-8")
    )
    model_config = payload["model"]

    config = resolve_gliformer_config_class(model_config)(**model_config)

    for task_config in (
        config.ner_config,
        config.classification_config,
    ):
        assert task_config.focal_loss_alpha == pytest.approx(0.8)
        assert task_config.focal_loss_gamma == pytest.approx(2.0)
        assert task_config.focal_loss_prob_margin == pytest.approx(0.0)

    relation_config = config.joint_relex_config
    assert relation_config.relation_loss_reduction == "mean"
    assert relation_config.relation_focal_loss_alpha == pytest.approx(0.8)
    assert relation_config.relation_focal_loss_gamma == pytest.approx(2.0)
    assert relation_config.relation_focal_loss_prob_margin == pytest.approx(0.0)


def test_structuring_training_config_uses_entity_first_head():
    head_config = yaml.safe_load(
        (CONFIG_DIR / "structuring.yaml").read_text(encoding="utf-8")
    )["model"]["structuring_config"]

    assert head_config["head_type"] == "structuring"
    assert head_config["entity_loss_coef"] == pytest.approx(1.0)
    assert head_config["assignment_loss_coef"] == pytest.approx(1.0)
    assert head_config["bio_loss_reduction"] == "mean"
    for key in (
        "anchor_refinement",
        "anchor_cross_attention_bias",
        "anchor_memory_position",
        "anchor_query_position",
        "anchor_memory_position_usage",
    ):
        assert key in head_config


def test_multi_level_structuring_defaults_off_and_round_trips():
    config = GLiFormerConfig(
        default_ner_config=False,
        structuring_config={"multi_level": True},
        structuring_child_token="<child>",
        structuring_end_token="<end>",
    )

    assert config.structuring_config.multi_level is True
    assert config.structuring_config.anchor_relations_layer == "mlp"
    assert config.structuring_child_token == "<child>"
    assert config.structuring_end_token == "<end>"

    reloaded = GLiFormerConfig(**config.to_dict())
    assert reloaded.structuring_config.multi_level is True
    assert reloaded.structuring_child_token == "<child>"
    assert reloaded.structuring_end_token == "<end>"

    legacy = GLiFormerConfig(
        default_ner_config=False,
        structuring_config={},
    )
    assert legacy.structuring_config.multi_level is False


def test_structuring_mode_components_round_trip():
    config = GLiFormerConfig(
        default_ner_config=False,
        structuring_config={
            "structure_mode": {
                "type": "multi-level",
                "processor": {"type": "multi_level"},
                "decoder": {
                    "type": "multi_level",
                    "params": {"relation_threshold": 0.65},
                },
            },
        },
    )

    mode = config.structuring_config.effective_structure_mode()
    assert mode.type == "multi_level"
    assert config.structuring_config.multi_level is True
    assert mode.decoder["params"]["relation_threshold"] == 0.65

    reloaded = GLiFormerConfig(**config.to_dict())
    assert reloaded.structuring_config.effective_structure_mode().type == "multi_level"


def test_structuring_mode_explicit_options_and_assignment_coef_round_trip():
    config = GLiFormerConfig(
        default_ner_config=False,
        structuring_config={
            "span_loss_coef": 2.0,
            "assignment_loss_coef": 3.0,
            "structure_mode": {
                "type": "multi_level",
                "processor_options": {"processor_flag": True},
                "decoder_options": {"relation_threshold": 0.27},
            },
        },
    )

    mode = config.structuring_config.effective_structure_mode()
    assert mode.processor_spec()["params"]["processor_flag"] is True
    assert mode.decoder_spec()["params"]["relation_threshold"] == 0.27
    assert config.structuring_config.assignment_loss_coef == 3.0

    reloaded = GLiFormerConfig(**config.to_dict())
    assert reloaded.structuring_config.assignment_loss_coef == 3.0
    assert (
        reloaded.structuring_config.effective_structure_mode()
        .decoder_spec()["params"]["relation_threshold"]
        == 0.27
    )


def test_structuring_assignment_coef_migrates_from_span_coef():
    config = GLiFormerConfig(
        default_ner_config=False,
        structuring_config={"span_loss_coef": 2.5},
    )

    assert config.structuring_config.assignment_loss_coef == 2.5


def test_legacy_multi_level_flag_rejects_an_explicit_flat_mode():
    with pytest.raises(ValueError, match="conflicts"):
        GLiFormerConfig(
            default_ner_config=False,
            structuring_config={
                "multi_level": True,
                "structure_mode": "flat",
            },
        )


def test_multi_level_rejects_symmetric_relation_modes():
    with pytest.raises(ValueError, match="must be 'mlp'"):
        GLiFormerConfig(
            default_ner_config=False,
            structuring_config={
                "multi_level": True,
                "anchor_relations_layer": "dot",
            },
        )


def test_multi_level_training_config_targets_supplied_dataset():
    payload = yaml.safe_load(
        (CONFIG_DIR / "structuring_multi_level.yaml").read_text(
            encoding="utf-8"
        )
    )

    model = payload["model"]
    structuring = model["structuring_config"]
    assert structuring["multi_level"] is True
    assert structuring["head_type"] == "structuring"
    assert structuring["anchor_relations_layer"] == "mlp"
    assert structuring["anchor_layer"]["params"]["num_slots"] == (
        structuring["max_count"]
    )
    assert structuring["max_count"] >= 95
    assert structuring["anchor_normalization"] == "center_rms"
    assert structuring["bio_loss_reduction"] == "mean"
    assert structuring["neg_spans_ratio"] > 0
    assert structuring["anchor_objectness_loss_coef"] == 1.0
    assert structuring["anchor_relations_loss_coef"] == 1.0
    assert structuring["masking"] == "none"
    assert structuring["negatives"] == 1.0
    assert model["max_types"] > 0
    training = payload["training"]
    assert training["masking"] == "none"
    assert training["negatives"] == 1.0
    assert payload["data"]["train_data"] == [
        {
            "path": "data/structuring_multi_level.jsonl",
            "repeat": 8,
        },
        "data/structuring_synthetic.cleaned.jsonl",
    ]


def test_multitask_joint_and_structuring_reuse_enabled_ner_head():
    payload = yaml.safe_load(
        (CONFIG_DIR / "multitask.yaml").read_text(encoding="utf-8")
    )
    model_config = payload["model"]

    config = resolve_gliformer_config_class(model_config)(**model_config)

    assert config.ner_config is not None
    assert config.joint_relex_config is not None
    assert config.structuring_config.reuse_ner_head is True
    assert config.ner_config.effective_anchor_mode() == "parent"
    structuring = model_config["structuring_config"]
    for component in (
        "ner",
        "matching",
        "objectness",
        "anchor_relations",
    ):
        for suffix in ("alpha", "gamma", "prob_margin"):
            assert f"{component}_focal_loss_{suffix}" in structuring


@pytest.mark.parametrize(
    ("legacy_key", "legacy_head_type", "canonical_key", "canonical_head_type"),
    [
        (
            "set_open_relex_config",
            "set_open_relex",
            "open_relex_config",
            "open_relex",
        ),
        (
            "set_structuring_config",
            "set_structuring",
            "structuring_config",
            "structuring",
        ),
    ],
)
def test_legacy_set_config_keys_are_load_only(
    legacy_key,
    legacy_head_type,
    canonical_key,
    canonical_head_type,
):
    config = GLiFormerConfig(
        default_ner_config=False,
        **{legacy_key: {"head_type": legacy_head_type}},
    )

    canonical = getattr(config, canonical_key)
    assert canonical.head_type == canonical_head_type
    assert not hasattr(config, legacy_key)
    serialized = config.to_dict()
    assert legacy_key not in serialized
    assert serialized[canonical_key]["head_type"] == canonical_head_type


@pytest.mark.parametrize(
    ("canonical_key", "legacy_key"),
    [
        ("open_relex_config", "set_open_relex_config"),
        ("structuring_config", "set_structuring_config"),
    ],
)
def test_legacy_and_canonical_config_keys_conflict(canonical_key, legacy_key):
    with pytest.raises(ValueError, match="configured in both"):
        GLiFormerConfig(
            default_ner_config=False,
            **{canonical_key: {}, legacy_key: {}},
        )
