import argparse
import inspect
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

import torch
from gliner.training.trainer import TrainingArguments
from gliner.utils import load_config_as_namespace, namespace_to_dict
from transformers.trainer_utils import get_last_checkpoint

from gliformer import GLiFormer

_STRUCTURING_CHECKPOINT_OVERRIDE_FIELDS = (
    "loss_coef",
    "focal_loss_alpha",
    "focal_loss_gamma",
    "focal_loss_prob_margin",
    "bio_loss_reduction",
    "negatives",
    "masking",
    "use_anchor_matching",
    "log_loss_stats",
    "log_loss_stats_every",
    "anchor_normalization",
    "position_bucket_normalization",
    "position_bucket_attention_bias_type",
    "position_bucket_attention_sigma",
    "position_bucket_attention_bias_weight",
    "span_loss_coef",
    "entity_loss_coef",
    "assignment_loss_coef",
    "neg_spans_ratio",
    "matcher_membership_cost",
    "matcher_dice_cost",
    "matcher_objectness_cost",
    "matcher_membership_temperature",
    "matcher_objectness_temperature",
    "ner_focal_loss_alpha",
    "ner_focal_loss_gamma",
    "ner_focal_loss_prob_margin",
    "matching_focal_loss_alpha",
    "matching_focal_loss_gamma",
    "matching_focal_loss_prob_margin",
    "objectness_focal_loss_alpha",
    "objectness_focal_loss_gamma",
    "objectness_focal_loss_prob_margin",
    "anchor_objectness_loss_coef",
    "anchor_objectness_threshold",
    "anchor_relations_loss_coef",
    "anchor_relations_threshold",
    "anchor_relations_focal_loss_alpha",
    "anchor_relations_focal_loss_gamma",
    "anchor_relations_focal_loss_prob_margin",
)

_TRAINING_ALIASES = {
    "num_steps": "max_steps",
    "scheduler_type": "lr_scheduler_type",
    "train_batch_size": "per_device_train_batch_size",
    "lr_encoder": "learning_rate",
    "lr_others": "others_lr",
    "weight_decay_encoder": "weight_decay",
    "weight_decay_other": "others_weight_decay",
}

_TOP_LEVEL_TRAINING_ARGS = {
    "gradient_accumulation_steps",
    "warmup_ratio",
    "max_grad_norm",
    "focal_loss_alpha",
    "focal_loss_gamma",
    "focal_loss_prob_margin",
    "rel_focal_loss_alpha",
    "rel_focal_loss_gamma",
    "loss_reduction",
    "negatives",
    "masking",
    "save_total_limit",
    "load_best_model_at_end",
    "metric_for_best_model",
    "greater_is_better",
    "use_cpu",
    "bf16",
    "gradient_checkpointing",
    "dataloader_num_workers",
}

_TRAINING_CONTROL_FIELDS = {
    "prev_path",
    "resume_from_checkpoint",
    "compile_model",
    "train_head_only",
    "freeze_components",
    "classification_parent_name_dropout",
    "label_augmentation",
    "eval_every",
    "trainer_args",
}

# These values participate in optimizer/scheduler state. Changing them while
# loading full Trainer state produces a misleading mixture of the old and new
# run. A weight-only retrain (``prev_path``) intentionally has no such limit.
_STATEFUL_RESUME_FIELDS = {
    "learning_rate",
    "others_lr",
    "weight_decay",
    "others_weight_decay",
    "lr_scheduler_type",
    "lr_scheduler_kwargs",
    "warmup_ratio",
    "warmup_steps",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "optim",
    "optim_args",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
}

_TASK_CONFIG_ATTRS = {
    "ner": "ner_config",
    "classification": "classification_config",
    "image_classification": "image_classification_config",
    "audio_classification": "audio_classification_config",
    "object_detection": "object_detection_config",
    "segmentation": "segmentation_config",
    "audio_segmentation": "audio_segmentation_config",
    "joint_relex": "joint_relex_config",
    "open_relex": "open_relex_config",
    "set_open_relex": "set_open_relex_config",
    "structuring": "structuring_config",
    "set_structuring": "set_structuring_config",
    "count": "count_config",
    "embedding": "embedding_config",
}

_GLOBAL_ARCHITECTURE_FIELDS = {
    "model_variant",
    "model_name",
    "labels_encoder",
    "hidden_size",
    "dropout",
    "subtoken_pooling",
    "span_mode",
    "max_width",
    "backbone_type",
    "use_layout",
    "vision_model_name",
    "vision_encoder_type",
    "audio_encoder_type",
}

_TASK_ARCHITECTURE_FIELDS = {
    "head_type",
    "represent_spans",
    "anchor_mode",
    "anchor_layer",
    "anchor_modeling",
    "anchor_refinement",
    "anchor_memory_position",
    "anchor_query_position",
    "anchor_memory_position_usage",
    "anchor_self_attention_bias",
    "anchor_cross_attention_bias",
    "feature_anchor_mlp",
    "feature_anchor_mlp_hidden_multiplier",
    "anchor_refine_layers",
    "anchor_refine_heads",
    "anchor_refine_norm",
    "anchor_refine_layer_scale_init",
    "anchor_context_gate_trainable",
    "parent_token_index",
    "embed_parent_token",
    "cat_token_index",
    "embed_cat_token",
    "child_token_index",
    "embed_child_token",
    "rel_token_index",
    "embed_rel_token",
    "pooling_type",
    "scorer_type",
    "num_fixed_slots",
    "max_count",
    "anchor_num_heads",
    "anchor_num_layers",
    "relations_layer",
    "triples_layer",
    "anchor_objectness",
    "anchor_relations_layer",
    "structure_mode",
    "multi_level",
    "reuse_ner_head",
    "projection_dim",
    "mode",
    "class_probability",
    "multi_label",
    "memory_position_embedding_type",
    "query_position_embedding_type",
    "memory_position_embedding_kwargs",
    "query_position_embedding_kwargs",
}

_SAFE_TASK_POLICY_FIELDS = {
    "loss_coef",
    "focal_loss_alpha",
    "focal_loss_gamma",
    "focal_loss_prob_margin",
    "span_loss_coef",
    "neg_spans_ratio",
    "negatives",
    "masking",
    "bio_loss_reduction",
    "use_anchor_matching",
    "log_loss_stats",
    "log_loss_stats_every",
    "relation_loss_reduction",
    "objectness_positive_weight",
    "objectness_negative_weight",
    "similarity_fn",
    "loss_fn",
    "margin",
    "projection_dropout",
    "encoder_dropout",
}


def _is_safe_task_policy_field(name):
    return (
        name in _SAFE_TASK_POLICY_FIELDS
        or name.endswith("_loss_coef")
        or "_focal_loss_" in name
        or name.endswith("_threshold")
        or (
            name.startswith("matcher_")
            and (name.endswith("_cost") or name.endswith("_temperature"))
        )
    )


def _architecture_value(value):
    if isinstance(value, Mapping):
        return {
            key: _architecture_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list | tuple):
        return [_architecture_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return value.lower().replace("_", "-")
    if hasattr(value, "__dict__"):
        return _architecture_value(vars(value))
    return value


def validate_checkpoint_architecture(model, model_cfg: dict):
    """Reject active YAML changes that require differently shaped modules."""

    from gliformer.config import resolve_gliformer_config_class

    mismatches = []
    checkpoint_model_cfg = model.config
    active_config_class = resolve_gliformer_config_class(model_cfg)
    active_model_cfg = active_config_class(**model_cfg)
    for name in sorted(_GLOBAL_ARCHITECTURE_FIELDS & set(model_cfg)):
        if not hasattr(checkpoint_model_cfg, name) or not hasattr(
            active_model_cfg,
            name,
        ):
            continue
        active = _architecture_value(getattr(active_model_cfg, name))
        saved = _architecture_value(getattr(checkpoint_model_cfg, name))
        if active != saved:
            mismatches.append(f"model.{name} ({saved!r} -> {active!r})")

    heads = getattr(getattr(model, "model", None), "heads", {})
    for task_name, config_name in _TASK_CONFIG_ATTRS.items():
        if config_name not in model_cfg:
            continue
        configured_values = model_cfg[config_name]
        active_config = getattr(active_model_cfg, config_name, None)
        saved_config = getattr(checkpoint_model_cfg, config_name, None)
        active_enabled = active_config is not None
        saved_enabled = saved_config is not None and task_name in heads
        if active_enabled != saved_enabled:
            mismatches.append(
                f"model.{config_name} enabled "
                f"({saved_enabled!r} -> {active_enabled!r})"
            )
            continue
        if not active_enabled:
            continue
        if not isinstance(configured_values, Mapping):
            continue
        for name in configured_values:
            if name not in _TASK_ARCHITECTURE_FIELDS:
                continue
            if not hasattr(saved_config, name) or not hasattr(
                active_config,
                name,
            ):
                continue
            active_value = getattr(active_config, name)
            saved_value = getattr(saved_config, name)
            if _architecture_value(active_value) != _architecture_value(
                saved_value
            ):
                mismatches.append(
                    f"model.{config_name}.{name} "
                    f"({_architecture_value(saved_value)!r} -> "
                    f"{_architecture_value(active_value)!r})"
                )

    if mismatches:
        raise ValueError(
            "training.prev_path can only override checkpoint-safe loss and "
            "runtime policy fields; architecture changes were requested: "
            + ", ".join(mismatches)
            + ". Use a matching model config or initialize a new model."
        )


def _optional_checkpoint(value):
    if value is False:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "false",
    }:
        return None
    return value


def resolve_checkpoint_mode(
    train_cfg: dict,
    resume_from_checkpoint: str | bool | None = None,
):
    """Resolve mutually exclusive weight-only retraining and exact resume."""

    prev_path = _optional_checkpoint(train_cfg.get("prev_path"))
    configured_resume = _optional_checkpoint(
        train_cfg.get("resume_from_checkpoint")
    )
    resume = (
        configured_resume
        if resume_from_checkpoint is None
        else _optional_checkpoint(resume_from_checkpoint)
    )
    if prev_path is not None and resume is not None:
        raise ValueError(
            "training.prev_path and resume_from_checkpoint are mutually "
            "exclusive: use prev_path to start a new run with new training "
            "parameters, or resume_from_checkpoint for an exact continuation."
        )
    return prev_path, resume


def _set_training_arg(target, explicit, name, value, source):
    if name in target and target[name] != value:
        raise ValueError(
            f"Conflicting values for training argument {name!r}: "
            f"{target[name]!r} and {value!r} (from {source})."
        )
    target[name] = value
    explicit.add(name)


def _build_training_kwargs(train_cfg: dict, *, has_eval: bool):
    """Translate and validate YAML training settings for ``train_model``.

    Existing GLiFormer names remain supported at the top level. Arbitrary
    standard ``TrainingArguments`` values belong under ``trainer_args`` so a
    typo cannot silently become an ignored keyword.
    """

    if not isinstance(train_cfg, Mapping):
        raise TypeError("training configuration must be a mapping")

    allowed_trainer_args = set(
        inspect.signature(TrainingArguments.__init__).parameters
    ) - {"self"}
    unknown = set(train_cfg) - (
        set(_TRAINING_ALIASES)
        | _TOP_LEVEL_TRAINING_ARGS
        | _TRAINING_CONTROL_FIELDS
    )
    if unknown:
        formatted = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown training option(s): {formatted}. Put standard "
            "Transformers TrainingArguments options under training.trainer_args."
        )

    kwargs = {}
    explicit = set()
    for source, target in _TRAINING_ALIASES.items():
        if source in train_cfg:
            _set_training_arg(
                kwargs,
                explicit,
                target,
                train_cfg[source],
                f"training.{source}",
            )
    for name in _TOP_LEVEL_TRAINING_ARGS:
        if name in train_cfg:
            _set_training_arg(
                kwargs,
                explicit,
                name,
                train_cfg[name],
                f"training.{name}",
            )

    nested = train_cfg.get("trainer_args") or {}
    if not isinstance(nested, Mapping):
        raise TypeError("training.trainer_args must be a mapping")
    unsupported = set(nested) - allowed_trainer_args
    if unsupported:
        formatted = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Unknown training.trainer_args option(s): {formatted}."
        )
    for name, value in nested.items():
        if name == "output_dir":
            raise ValueError(
                "training.trainer_args.output_dir is managed by data.root_dir"
            )
        _set_training_arg(
            kwargs,
            explicit,
            name,
            value,
            f"training.trainer_args.{name}",
        )

    if "per_device_eval_batch_size" not in kwargs and (
        "per_device_train_batch_size" in kwargs
    ):
        kwargs["per_device_eval_batch_size"] = kwargs[
            "per_device_train_batch_size"
        ]

    if "eval_every" in train_cfg:
        interval = train_cfg["eval_every"]
        for name in ("save_steps", "logging_steps", "eval_steps"):
            if name not in kwargs:
                kwargs[name] = interval

    kwargs.setdefault("eval_strategy", "steps" if has_eval else "no")
    if not has_eval:
        if kwargs.get("load_best_model_at_end"):
            raise ValueError(
                "load_best_model_at_end requires data.val_data"
            )
        kwargs["load_best_model_at_end"] = False

    # Keep the explicit set private to the validation layer rather than
    # forwarding it into train_model.
    return kwargs, frozenset(explicit)


def build_training_kwargs(train_cfg: dict, *, has_eval: bool):
    """Return validated ``train_model`` arguments from a training config."""

    kwargs, _ = _build_training_kwargs(train_cfg, has_eval=has_eval)
    return kwargs


def _comparable_training_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def validate_resume_training_args(
    resume_from_checkpoint,
    output_dir: Path,
    training_kwargs: dict,
    explicit_fields,
):
    """Reject optimizer/scheduler changes during an exact Trainer resume."""

    if resume_from_checkpoint is None:
        return
    checkpoint_path = resume_from_checkpoint
    if resume_from_checkpoint is True:
        checkpoint_path = get_last_checkpoint(str(output_dir))
        if checkpoint_path is None:
            # Trainer owns the canonical error for a missing latest checkpoint.
            return
    checkpoint_path = Path(checkpoint_path)
    arguments_path = checkpoint_path / "training_args.bin"
    if not arguments_path.is_file():
        return

    saved_args = torch.load(
        arguments_path,
        map_location="cpu",
        weights_only=False,
    )
    conflicts = []
    for name in sorted(_STATEFUL_RESUME_FIELDS & set(explicit_fields)):
        if not hasattr(saved_args, name):
            continue
        active = _comparable_training_value(training_kwargs[name])
        saved = _comparable_training_value(getattr(saved_args, name))
        if active != saved:
            conflicts.append(f"{name} ({saved!r} -> {active!r})")
    if conflicts:
        raise ValueError(
            "Cannot change optimizer or schedule parameters during an exact "
            "resume: "
            + ", ".join(conflicts)
            + ". Use training.prev_path for a weight-only retrain with fresh "
            "optimizer, scheduler, RNG, and step state."
        )


def apply_checkpoint_config_overrides(model, model_cfg: dict):
    """Apply parameter-free task policy changes to a loaded checkpoint.

    Loading ``prev_path`` necessarily restores the checkpoint architecture,
    but loss, similarity, and position-bucket policies should come from the
    active training YAML. Restricting this overlay to fields that do not create
    parameters keeps checkpoint loading safe and makes continuation runs
    reproducible.
    """

    heads = getattr(getattr(model, "model", None), "heads", None)
    applied = []

    # Most heads keep only a few scalar loss weights on the module and retain
    # their task config object for the rest. Update both so the active policy is
    # effective immediately and is also serialized with the retrained model.
    for task_name, config_name in _TASK_CONFIG_ATTRS.items():
        if task_name in {"structuring", "set_structuring", "embedding"}:
            continue
        configured = model_cfg.get(config_name)
        checkpoint_config = getattr(model.config, config_name, None)
        head = (
            heads[task_name]
            if heads is not None and task_name in heads
            else None
        )
        if (
            not isinstance(configured, Mapping)
            or checkpoint_config is None
            or head is None
        ):
            continue
        for name, value in configured.items():
            if not _is_safe_task_policy_field(name):
                continue
            setattr(checkpoint_config, name, value)
            head_name = "masking_mode" if name == "masking" else name
            if hasattr(head, head_name):
                setattr(head, head_name, value)
            applied.append(name)

    for config_name, task_name in (
        ("structuring_config", "structuring"),
        ("set_structuring_config", "set_structuring"),
    ):
        configured = model_cfg.get(config_name)
        checkpoint_config = getattr(model.config, config_name, None)
        head = (
            heads[task_name]
            if heads is not None and task_name in heads
            else None
        )
        if (
            not isinstance(configured, dict)
            or checkpoint_config is None
            or head is None
        ):
            continue

        configured_values = dict(configured)
        anchor_spec = configured.get("anchor_layer")
        if isinstance(anchor_spec, dict):
            anchor_params = anchor_spec.get("params")
            if isinstance(anchor_params, dict) and (
                "position_bucket_normalization" in anchor_params
            ):
                configured_values.setdefault(
                    "position_bucket_normalization",
                    anchor_params["position_bucket_normalization"],
                )

        for name in _STRUCTURING_CHECKPOINT_OVERRIDE_FIELDS:
            if name not in configured_values:
                continue
            value = configured_values[name]
            setattr(checkpoint_config, name, value)
            applied.append(name)

        if task_name == "set_structuring" and (
            "span_loss_coef" in configured_values
        ):
            assignment_loss_coef = configured_values.get(
                "assignment_loss_coef",
                configured_values["span_loss_coef"],
            )
            checkpoint_config.assignment_loss_coef = assignment_loss_coef
            if hasattr(head, "assignment_loss_coef"):
                head.assignment_loss_coef = assignment_loss_coef

        if "loss_coef" in configured_values:
            model.config.structuring_loss_coef = configured_values[
                "loss_coef"
            ]

        for field_name, value in configured_values.items():
            head_name = {
                "masking": "masking_mode",
                "span_loss_coef": (
                    "assignment_loss_coef"
                    if task_name == "set_structuring"
                    else "span_loss_coef"
                ),
            }.get(field_name, field_name)
            if (
                field_name in _STRUCTURING_CHECKPOINT_OVERRIDE_FIELDS
                and hasattr(head, head_name)
            ):
                setattr(head, head_name, value)

        if "position_bucket_normalization" in configured_values:
            anchor_layer = getattr(
                head,
                "record_anchor_layer",
                getattr(head, "anchor_layer", None),
            )
            if anchor_layer is not None and hasattr(
                anchor_layer, "normalization"
            ):
                normalization = configured_values[
                    "position_bucket_normalization"
                ]
                anchor_layer.normalization = normalization
                if hasattr(anchor_layer, "normalizer"):
                    from gliformer.layers import AnchorNormalizer

                    anchor_layer.normalizer = AnchorNormalizer.from_config(
                        normalization,
                        int(model.config.hidden_size),
                    )
        if "anchor_normalization" in configured_values:
            from gliformer.layers import AnchorNormalizer

            normalizer_name = (
                "record_anchor_normalizer"
                if task_name == "set_structuring"
                else "anchor_normalizer"
            )
            if hasattr(head, normalizer_name):
                setattr(
                    head,
                    normalizer_name,
                    AnchorNormalizer.from_config(
                        configured_values["anchor_normalization"],
                        int(model.config.hidden_size),
                    ),
                )

    embedding_values = model_cfg.get("embedding_config")
    embedding_config = getattr(model.config, "embedding_config", None)
    embedding_head = (
        heads["embedding"]
        if heads is not None and "embedding" in heads
        else None
    )
    if (
        isinstance(embedding_values, dict)
        and embedding_config is not None
        and embedding_head is not None
    ):
        for name in ("loss_coef", "similarity_fn"):
            if name not in embedding_values:
                continue
            value = embedding_values[name]
            setattr(embedding_config, name, value)
            setattr(embedding_head, name, value)
            applied.append(name)

        if "loss_coef" in embedding_values:
            model.config.embedding_loss_coef = embedding_values["loss_coef"]

        if "projection_dropout" in embedding_values:
            projection_dropout = float(
                embedding_values["projection_dropout"]
            )
            if not math.isfinite(projection_dropout) or not (
                0.0 <= projection_dropout < 1.0
            ):
                raise ValueError(
                    "embedding projection_dropout must be finite and in [0, 1)"
                )
            embedding_config.projection_dropout = projection_dropout
            embedding_head.projection_dropout = projection_dropout
            projection = getattr(embedding_head, "projection", None)
            if projection is not None:
                for module in projection.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = projection_dropout
            applied.append("projection_dropout")

        if embedding_values.get("encoder_dropout") is not None:
            encoder_dropout = float(embedding_values["encoder_dropout"])
            if not math.isfinite(encoder_dropout) or not (
                0.0 <= encoder_dropout < 1.0
            ):
                raise ValueError(
                    "embedding encoder_dropout must be finite and in [0, 1)"
                )
            from gliformer.encoders.text import set_text_encoder_dropout

            embedding_config.encoder_dropout = encoder_dropout
            embedding_head.encoder_dropout = encoder_dropout
            token_rep_layer = getattr(
                getattr(model, "model", None),
                "token_rep_layer",
                None,
            )
            if token_rep_layer is not None:
                set_text_encoder_dropout(token_rep_layer, encoder_dropout)
            applied.append("encoder_dropout")

        if "loss_fn" in embedding_values or "margin" in embedding_values:
            from gliformer.tasks.embedding.model import EmbeddingLoss

            loss_fn = embedding_values.get(
                "loss_fn",
                getattr(embedding_config, "loss_fn", "mse"),
            )
            margin = embedding_values.get(
                "margin",
                getattr(embedding_config, "margin", None),
            )
            loss_kwargs = {} if margin is None else {"margin": margin}
            embedding_head.loss = EmbeddingLoss.from_config(
                loss_fn=loss_fn,
                **loss_kwargs,
            )
            embedding_config.loss_fn = loss_fn
            embedding_config.margin = margin
            if "loss_fn" in embedding_values:
                applied.append("loss_fn")
            if "margin" in embedding_values:
                applied.append("margin")
    return tuple(applied)


def load_json_data(path):
    """Load one or more JSON/JSONL datasets as a single record list.

    A multi-source entry may be either a path or ``{"path": ..., "repeat": N}``.
    Repetition is useful for balancing differently sized sources without
    materializing another JSONL file on disk.
    """

    if isinstance(path, list | tuple):
        records = []
        for source_path in path:
            records.extend(load_json_data(source_path))
        return records

    if isinstance(path, argparse.Namespace):
        path = vars(path)
    if isinstance(path, Mapping):
        unknown_keys = set(path).difference({"path", "repeat"})
        if unknown_keys:
            formatted = ", ".join(sorted(map(str, unknown_keys)))
            raise ValueError(
                f"Unknown dataset source option(s): {formatted}. "
                "Expected only 'path' and optional 'repeat'."
            )
        if "path" not in path:
            raise ValueError("Dataset source mapping must contain 'path'.")
        repeat = path.get("repeat", 1)
        if (
            isinstance(repeat, bool)
            or not isinstance(repeat, int)
            or repeat <= 0
        ):
            raise ValueError(
                "Dataset source 'repeat' must be a positive integer."
            )
        source_records = load_json_data(path["path"])
        return source_records * repeat

    dataset_path = Path(path)

    if dataset_path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with dataset_path.open("r", encoding="utf-8-sig") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {dataset_path} at line {line_number}: "
                        f"{exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object in {dataset_path} at line "
                        f"{line_number}, got {type(record).__name__}."
                    )
                records.append(record)
        return records

    with dataset_path.open("r", encoding="utf-8-sig") as data_file:
        records = json.load(data_file)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON array in {dataset_path}, got "
            f"{type(records).__name__}."
        )
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Expected a JSON object at index {index} in {dataset_path}, "
                f"got {type(record).__name__}."
            )
    return records


def prepare_training_records(records, model):
    """Drop relation-free rows for a standalone open-relation run.

    The canonical ``extraction`` format also permits NER-only examples. They
    remain valuable in mixed-task training, but a standalone relation head has
    no label space with which to supervise them. Keeping them can create an
    all-empty batch and make the trainer fail its label guard.
    """

    heads = set(getattr(getattr(model, "model", None), "heads", {}))
    supported_heads = {"open_relex", "set_open_relex"}
    if len(heads) != 1 or not heads.issubset(supported_heads):
        return records, 0
    task_name = next(iter(heads))

    task_processors = getattr(
        getattr(model, "data_processor", None),
        "task_processors",
        {},
    )
    processor = task_processors.get(task_name)
    if processor is None:
        return records, 0

    filtered = [
        record
        for record in records
        if processor.has_training_annotations(record)
    ]
    if not filtered:
        raise ValueError(
            "Open-relation training found no usable relation annotations. "
            "Use extraction groups containing ner plus indexed relations, or "
            "declare all_rel_labels for an all-negative group."
        )
    return filtered, len(records) - len(filtered)


def build_model(model_cfg: dict, train_cfg: dict):
    """Build or load GLiFormer model."""
    prev_path = train_cfg.get("prev_path")
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        print(f"Loading pretrained model from: {prev_path}")
        model = GLiFormer.from_pretrained(prev_path)
        validate_checkpoint_architecture(model, model_cfg)
        applied = apply_checkpoint_config_overrides(model, model_cfg)
        if applied:
            print(
                "Applied checkpoint-safe task overrides: "
                + ", ".join(applied)
            )
        return model
    print("Initializing model from config...")
    return GLiFormer.load_from_config(model_cfg)


def main(cfg_path: str, resume_from_checkpoint: str | bool | None = None):
    """Main training function."""
    cfg = load_config_as_namespace(cfg_path)

    model_cfg = namespace_to_dict(cfg.model)
    train_cfg = namespace_to_dict(cfg.training)
    prev_path, resume_from_checkpoint = resolve_checkpoint_mode(
        train_cfg,
        resume_from_checkpoint,
    )
    train_cfg["prev_path"] = prev_path

    output_dir = Path(cfg.data.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading training data from: {cfg.data.train_data}")
    train_dataset = load_json_data(cfg.data.train_data)
    print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    val_data = getattr(cfg.data, "val_data", "none")
    if val_data and str(val_data).lower() not in ("none", "null", ""):
        print(f"Loading validation data from: {val_data}")
        eval_dataset = load_json_data(val_data)
        print(f"Validation samples: {len(eval_dataset)}")

    training_kwargs, explicit_training_fields = _build_training_kwargs(
        train_cfg,
        has_eval=eval_dataset is not None,
    )
    validate_resume_training_args(
        resume_from_checkpoint,
        output_dir,
        training_kwargs,
        explicit_training_fields,
    )

    # Build model
    model = build_model(model_cfg, train_cfg).to(dtype=torch.float32)
    print(f"Model type: {model.__class__.__name__}")

    # Enabled task heads
    enabled = list(getattr(model.model, "heads", {}))
    print(f"Enabled tasks: {', '.join(enabled)}")

    train_dataset, dropped_records = prepare_training_records(
        train_dataset,
        model,
    )
    if dropped_records:
        print(
            "Skipped relation-free rows for standalone relation training: "
            f"{dropped_records} (remaining: {len(train_dataset)})"
        )

    # Freeze components
    freeze_components = train_cfg.get("freeze_components")
    train_head_only = train_cfg.get("train_head_only", False)
    if train_head_only:
        print("Training mode: head-only parameters")
    if freeze_components:
        print(f"Freezing components: {freeze_components}")

    # Train
    print("\nStarting training...")
    model.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir),
        resume_from_checkpoint=resume_from_checkpoint,
        compile_model=train_cfg.get("compile_model", False),
        freeze_components=freeze_components,
        train_head_only=train_head_only,
        classification_parent_name_dropout=train_cfg.get(
            "classification_parent_name_dropout",
            0.0,
        ),
        label_augmentation=train_cfg.get("label_augmentation"),
        **training_kwargs,
    )

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiFormer model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file (YAML or JSON)")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Resume full trainer state from this checkpoint; when supplied "
            "without a path, use the latest checkpoint in data.root_dir."
        ),
    )
    args = parser.parse_args()
    main(args.config, args.resume_from_checkpoint)
