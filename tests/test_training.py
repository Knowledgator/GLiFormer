from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from gliner.model import BaseGLiNER
from gliner.modeling.loss_functions import focal_loss_with_logits
from torch import nn
from transformers.trainer_pt_utils import nested_concat

from gliformer.config import ObjectDetectionHeadConfig, StructuringHeadConfig
from gliformer.gliformer import BaseGLiFormer
from gliformer.model import BaseGLiFormerModel
from gliformer.outputs import GLiFormerTextOutput, GLiFormerVisionOutput
from gliformer.tasks.losses import binary_focal_or_bce
from gliformer.training import (
    _LABEL_KEYS,
    ClassificationParentNameDropoutDataset,
    GLiFormerTrainer,
    LABEL_AUGMENTATION_INDEX_KEY,
    LABEL_AUGMENTATION_MARKER_KEY,
    TrainingLabelAugmentationDataset,
)


class _TaskLossModel(nn.Module):
    """Tiny kwargs-only model mirroring GLiFormer's forward contract."""

    def __init__(self, label_key: str):
        super().__init__()
        self.label_key = label_key
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.last_kwargs = None

    def forward(self, *args, **kwargs):
        del args
        self.last_kwargs = dict(kwargs)
        features = kwargs["features"].float()
        logits = self.scale * features
        target = kwargs.get(self.label_key)
        if target is None:
            return SimpleNamespace(logits=logits)
        loss = (logits - target.float()).square().mean()
        return SimpleNamespace(loss=loss, logits=logits)


class _RecordingAccelerator:
    def __init__(self):
        self.backward_loss = None

    def backward(self, loss):
        self.backward_loss = loss.detach().clone()
        loss.backward()


def _bare_trainer(*, gradient_accumulation_steps: int = 1) -> GLiFormerTrainer:
    """Build the unit under test without requiring the optional accelerate package."""
    trainer = object.__new__(GLiFormerTrainer)
    trainer.args = SimpleNamespace(
        device=torch.device("cpu"),
        n_gpu=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        focal_loss_alpha=-1,
        focal_loss_gamma=0,
        rel_focal_loss_alpha=None,
        rel_focal_loss_gamma=None,
        focal_loss_prob_margin=0,
        label_smoothing=0,
        loss_reduction="sum",
        negatives=1.0,
        masking="global",
    )
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = nullcontext
    return trainer


class _CheckpointingBackbone(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self, *, legacy_signature: bool = False):
        super().__init__()
        self.enabled = False
        self.received_kwargs = None
        if legacy_signature:
            self.gradient_checkpointing_enable = self._legacy_enable

    @property
    def is_gradient_checkpointing(self):
        return self.enabled

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enabled = True
        self.received_kwargs = gradient_checkpointing_kwargs

    def _legacy_enable(self):
        self.enabled = True

    def gradient_checkpointing_disable(self):
        self.enabled = False


def _bare_gliformer(*backbones: nn.Module) -> BaseGLiFormer:
    wrapper = object.__new__(BaseGLiFormer)
    nn.Module.__init__(wrapper)
    wrapper.model = nn.ModuleList(backbones)
    return wrapper


def test_gliformer_forwards_gradient_checkpointing_to_all_backbones():
    text_backbone = _CheckpointingBackbone()
    labels_backbone = _CheckpointingBackbone(legacy_signature=True)
    model = _bare_gliformer(text_backbone, labels_backbone)

    assert model.supports_gradient_checkpointing
    assert not model.is_gradient_checkpointing

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    assert text_backbone.received_kwargs == {"use_reentrant": True}
    assert labels_backbone.enabled
    assert model.is_gradient_checkpointing

    model.gradient_checkpointing_disable()

    assert not text_backbone.enabled
    assert not labels_backbone.enabled
    assert not model.is_gradient_checkpointing


def test_gliformer_rejects_gradient_checkpointing_without_capable_backbone():
    model = _bare_gliformer(nn.Linear(2, 2))

    assert not model.supports_gradient_checkpointing
    with pytest.raises(ValueError, match="no backbone that supports"):
        model.gradient_checkpointing_enable()


def test_train_model_forwards_full_checkpoint_resume(monkeypatch):
    calls = {}

    class _TrainerBase:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

    class _Trainer(_TrainerBase):
        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setattr("gliformer.training.GLiFormerTrainer", _Trainer)
    owner = SimpleNamespace(
        config=SimpleNamespace(),
        data_processor=SimpleNamespace(transformer_tokenizer="tokenizer"),
        _create_data_collator=lambda **kwargs: "collator",
    )
    training_args = SimpleNamespace()

    trainer = BaseGLiFormer.train_model(
        owner,
        train_dataset=[{"text": "example"}],
        training_args=training_args,
        resume_from_checkpoint="checkpoint-4000",
    )

    assert isinstance(trainer, _Trainer)
    assert calls["init"]["args"] is training_args
    assert calls["train"] == {
        "resume_from_checkpoint": "checkpoint-4000"
    }


def test_train_model_applies_parent_name_dropout_only_to_training_data(
    monkeypatch,
):
    calls = {}

    class _TrainerBase:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

    class _Trainer(_TrainerBase):
        def train(self, **kwargs):
            calls["train"] = kwargs

    monkeypatch.setattr("gliformer.training.GLiFormerTrainer", _Trainer)
    owner = SimpleNamespace(
        config=SimpleNamespace(),
        data_processor=SimpleNamespace(transformer_tokenizer="tokenizer"),
        _create_data_collator=lambda **kwargs: "collator",
    )
    train_dataset = [{
        "classification": [{
            "name": "sentiment",
            "all_labels": ["positive", "negative"],
            "true_labels": ["positive"],
        }],
    }]
    eval_dataset = [{
        "classification": [{
            "name": "sentiment",
            "all_labels": ["positive", "negative"],
            "true_labels": ["positive"],
        }],
    }]

    BaseGLiFormer.train_model(
        owner,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=SimpleNamespace(),
        classification_parent_name_dropout=1.0,
    )

    wrapped_train = calls["init"]["train_dataset"]
    assert isinstance(wrapped_train, ClassificationParentNameDropoutDataset)
    assert wrapped_train[0]["classification"][0]["name"] is None
    assert calls["init"]["eval_dataset"] is eval_dataset
    assert eval_dataset[0]["classification"][0]["name"] == "sentiment"


def test_training_label_augmentation_dataset_marks_a_shallow_copy():
    records = [{
        "text": "Apple is a company.",
        "extraction": [{"ner": [["Apple", "organization"]]}],
    }]
    dataset = TrainingLabelAugmentationDataset(records)

    marked = dataset[0]

    assert marked is not records[0]
    assert marked["extraction"] is records[0]["extraction"]
    assert marked[LABEL_AUGMENTATION_MARKER_KEY] is True
    assert marked[LABEL_AUGMENTATION_INDEX_KEY] == 0
    assert LABEL_AUGMENTATION_MARKER_KEY not in records[0]
    assert LABEL_AUGMENTATION_INDEX_KEY not in records[0]


def test_train_model_wires_label_augmentation_only_to_training_data(
    monkeypatch,
    caplog,
):
    from gliformer.processing.label_augmentation import LabelAugmentationConfig

    calls = {}

    class _TrainerBase:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

    class _Trainer(_TrainerBase):
        def train(self, **kwargs):
            calls["train"] = kwargs

    normalized = SimpleNamespace(
        enabled=True,
        is_active=True,
        tasks={
            "joint_relex": SimpleNamespace(
                enabled=True,
                add_probability=1.0,
            ),
        },
    )
    normalized.policy_for = lambda task_name: normalized.tasks.get(
        task_name,
        SimpleNamespace(enabled=False, add_probability=0.0),
    )

    def _from_value(value, default_seed=None):
        calls["normalization"] = (value, default_seed)
        return normalized

    monkeypatch.setattr(
        LabelAugmentationConfig,
        "from_value",
        staticmethod(_from_value),
    )
    monkeypatch.setattr("gliformer.training.GLiFormerTrainer", _Trainer)

    def _create_data_collator(**kwargs):
        calls["collator"] = kwargs
        return "collator"

    owner = SimpleNamespace(
        config=SimpleNamespace(),
        data_processor=SimpleNamespace(transformer_tokenizer="tokenizer"),
        _create_data_collator=_create_data_collator,
    )
    train_dataset = [{"text": "training"}]
    eval_dataset = [{"text": "validation"}]
    supplied = {"enabled": True, "tasks": {"joint_relex": {}}}
    training_args = SimpleNamespace(
        data_seed=17,
        seed=41,
        per_device_train_batch_size=1,
    )

    with caplog.at_level("WARNING"):
        BaseGLiFormer.train_model(
            owner,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            label_augmentation=supplied,
            classification_parent_name_dropout=1.0,
        )

    wrapped_train = calls["init"]["train_dataset"]
    assert isinstance(wrapped_train, TrainingLabelAugmentationDataset)
    assert isinstance(
        wrapped_train.dataset,
        ClassificationParentNameDropoutDataset,
    )
    assert wrapped_train.dataset.dataset is train_dataset
    assert calls["init"]["eval_dataset"] is eval_dataset
    assert LABEL_AUGMENTATION_MARKER_KEY not in eval_dataset[0]
    assert calls["normalization"] == (supplied, 17)
    assert calls["collator"] == {"label_augmentation": normalized}
    assert "gradient accumulation" in caplog.text


def test_classification_parent_name_dropout_creates_unnamed_singleton_view():
    records = [{
        "text": "The order arrived on time.",
        "classification": [{
            "name": "sentiment",
            "description": "Customer sentiment",
            "all_labels": ["positive", "negative"],
            "true_labels": ["positive"],
        }],
    }]

    dataset = ClassificationParentNameDropoutDataset(records, probability=1.0)
    augmented = dataset[0]

    assert augmented["classification"][0] == {
        "name": None,
        "description": "Customer sentiment",
        "all_labels": ["positive", "negative"],
        "true_labels": ["positive"],
    }
    assert records[0]["classification"][0]["name"] == "sentiment"


def test_classification_parent_name_dropout_preserves_named_view_and_groups():
    named_singleton = [{
        "classification": [{
            "name": "topic",
            "all_labels": ["technology", "sports"],
            "true_labels": ["technology"],
        }],
    }]
    multiple_groups = [{
        "classification": [
            {"name": "topic", "all_labels": ["technology"]},
            {"name": "sentiment", "all_labels": ["positive"]},
        ],
    }]

    assert (
        ClassificationParentNameDropoutDataset(
            named_singleton,
            probability=0.0,
        )[0]
        is named_singleton[0]
    )
    assert (
        ClassificationParentNameDropoutDataset(
            multiple_groups,
            probability=1.0,
        )[0]
        is multiple_groups[0]
    )


def test_classification_parent_name_dropout_samples_both_prompt_views(
    monkeypatch,
):
    records = [{
        "classification": [{
            "name": "sentiment",
            "all_labels": ["positive", "negative"],
            "true_labels": ["positive"],
        }],
    }]
    samples = iter([0.49, 0.50])
    monkeypatch.setattr(
        "gliformer.training.random.random",
        lambda: next(samples),
    )
    dataset = ClassificationParentNameDropoutDataset(records, probability=0.5)

    assert dataset[0]["classification"][0]["name"] is None
    assert dataset[0]["classification"][0]["name"] == "sentiment"


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_classification_parent_name_dropout_rejects_invalid_probability(
    probability,
):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ClassificationParentNameDropoutDataset([], probability)


def test_training_args_preserve_explicit_zero_optimizer_values_and_inherit_none(
    monkeypatch,
    tmp_path,
):
    def upstream_factory(
        cls,
        output_dir,
        learning_rate=5e-5,
        weight_decay=0.01,
        others_lr=None,
        others_weight_decay=None,
        **kwargs,
    ):
        del cls, output_dir, kwargs
        return SimpleNamespace(
            others_lr=others_lr or learning_rate,
            others_weight_decay=others_weight_decay or weight_decay,
        )

    monkeypatch.setattr(
        BaseGLiNER,
        "create_training_args",
        classmethod(upstream_factory),
    )
    explicit_zero = BaseGLiFormer.create_training_args(
        output_dir=tmp_path / "explicit-zero",
        learning_rate=3e-4,
        weight_decay=0.07,
        others_lr=0.0,
        others_weight_decay=0.0,
        use_cpu=True,
        report_to="none",
    )
    inherited = BaseGLiFormer.create_training_args(
        output_dir=tmp_path / "inherited",
        learning_rate=3e-4,
        weight_decay=0.07,
        others_lr=None,
        others_weight_decay=None,
        use_cpu=True,
        report_to="none",
    )

    assert explicit_zero.others_lr == 0.0
    assert explicit_zero.others_weight_decay == 0.0
    assert inherited.others_lr == pytest.approx(3e-4)
    assert inherited.others_weight_decay == pytest.approx(0.07)


@pytest.mark.parametrize("label_key", sorted(_LABEL_KEYS))
def test_prediction_step_collects_loss_for_every_task_label_key(label_key):
    trainer = _bare_trainer()
    model = _TaskLossModel(label_key)
    inputs = {
        "features": torch.tensor([1.0, 2.0]),
        label_key: torch.tensor([0.0, 0.0]),
    }

    loss, logits, labels = trainer.prediction_step(
        model,
        inputs,
        prediction_loss_only=False,
    )

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert torch.equal(logits, torch.tensor([2.0, 4.0]))
    assert set(labels) == {label_key}
    assert torch.equal(labels[label_key], inputs[label_key])


def test_prediction_step_returns_loss_in_loss_only_evaluation():
    trainer = _bare_trainer()
    model = _TaskLossModel("object_detection_class_labels")

    result = trainer.prediction_step(
        model,
        {
            "features": torch.tensor([1.0]),
            "object_detection_class_labels": torch.tensor([0.0]),
        },
        prediction_loss_only=True,
    )

    assert result[0] is not None
    assert result[0].item() == pytest.approx(4.0)
    assert result[1:] == (None, None)


def test_trainer_passes_config_style_focal_loss_names():
    trainer = _bare_trainer()
    trainer.args.rel_focal_loss_alpha = 0.6
    trainer.args.rel_focal_loss_gamma = 1.5
    model = _TaskLossModel("image_classification_labels")

    trainer.compute_loss(
        model,
        {
            "features": torch.tensor([1.0]),
            "image_classification_labels": torch.tensor([0.0]),
        },
    )

    assert model.last_kwargs["focal_loss_alpha"] == -1
    assert model.last_kwargs["focal_loss_gamma"] == 0
    assert model.last_kwargs["focal_loss_prob_margin"] == 0
    assert model.last_kwargs["rel_focal_loss_alpha"] == 0.6
    assert model.last_kwargs["rel_focal_loss_gamma"] == 1.5
    assert "alpha" not in model.last_kwargs
    assert "gamma" not in model.last_kwargs
    assert "prob_margin" not in model.last_kwargs


def test_training_step_skips_label_free_batch_without_forward(caplog):
    trainer = _bare_trainer()
    model = _TaskLossModel("ner_labels")

    with caplog.at_level("WARNING", logger="gliformer.training"):
        loss = trainer.training_step(
            model,
            {"features": torch.tensor([1.0])},
        )

    assert loss.item() == 0.0
    assert loss.device.type == "cpu"
    assert model.last_kwargs is None
    assert trainer._label_free_batches_skipped == 1
    assert "Skipping label-free training batch #1" in caplog.text


def test_prediction_step_allows_unlabelled_prediction():
    trainer = _bare_trainer()
    model = _TaskLossModel("image_classification_labels")

    loss, logits, labels = trainer.prediction_step(
        model,
        {"features": torch.tensor([1.0])},
        prediction_loss_only=False,
    )

    assert loss is None
    assert torch.equal(logits, torch.tensor([2.0]))
    assert labels is None


class _DetectionLossModel(nn.Module):
    def forward(self, **kwargs):
        class_logits = torch.tensor([[[1.0, -1.0]]])
        boxes = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        objectness = torch.tensor([[0.25]])
        return GLiFormerVisionOutput(
            loss=class_logits.sum() * 0.0,
            object_detection_logits=class_logits,
            object_detection_boxes=boxes,
            object_detection_objectness_logits=objectness,
            object_detection_anchor_mask=torch.tensor([[True]]),
        )


def test_prediction_step_extracts_task_specific_detection_outputs():
    trainer = _bare_trainer()
    predictions = trainer.prediction_step(
        _DetectionLossModel(),
        {"object_detection_class_labels": torch.tensor([[0]])},
        prediction_loss_only=False,
    )[1]

    assert isinstance(predictions, tuple)
    assert len(predictions) == 4
    assert predictions[0].shape == (1, 1, 2)
    assert predictions[1].shape == (1, 1, 4)
    assert predictions[2].shape == (1, 1)
    assert predictions[3].dtype == torch.bool


class _StructuringLossModel(nn.Module):
    def forward(self, **kwargs):
        del kwargs
        entity_logits = torch.zeros(1, 4, 2, 3)
        field_logits = torch.zeros(1, 3, 2)
        assignment_logits = torch.zeros(1, 3, 4)
        return GLiFormerTextOutput(
            loss=entity_logits.sum(),
            structuring_entity_logits=entity_logits,
            structuring_field_logits=field_logits,
            structuring_logits=assignment_logits.transpose(1, 2),
            structuring_assignment_logits=assignment_logits,
            structuring_span_idx=torch.zeros(
                1,
                3,
                2,
                dtype=torch.long,
            ),
            structuring_span_mask=torch.ones(1, 3, dtype=torch.bool),
            structuring_objectness_logits=torch.zeros(1, 4),
            structuring_anchor_mask=torch.ones(1, 4, dtype=torch.bool),
        )


def test_prediction_step_uses_entity_first_structuring_outputs():
    trainer = _bare_trainer()
    predictions = trainer.prediction_step(
        _StructuringLossModel(),
        {
            "structuring_labels": torch.zeros(1, 4, 4, 2, 3),
            "structuring_span_labels": torch.zeros(1, 3, 4, 2),
        },
        prediction_loss_only=False,
    )[1]

    assert isinstance(predictions, tuple)
    assert len(predictions) == 7
    assert predictions[0].shape == (1, 4, 2, 3)
    assert predictions[1].shape == (1, 3, 2)
    assert predictions[2].shape == (1, 3, 4)
    assert predictions[3].shape == (1, 3, 2)
    assert predictions[4].shape == (1, 3)
    assert predictions[5].shape == (1, 4)
    assert predictions[6].shape == (1, 4)


class _OpenRelexLossModel(nn.Module):
    def forward(self, **kwargs):
        labels = kwargs["open_rel_assignment_labels"]
        batch_size, _, entity_count, _ = labels.shape
        anchor_count = 2
        assignment_logits = torch.zeros(
            batch_size,
            anchor_count,
            entity_count,
            2,
        )
        return GLiFormerTextOutput(
            loss=assignment_logits.sum() * 0.0,
            open_rel_assignment_logits=assignment_logits,
            open_rel_span_idx=torch.zeros(
                batch_size,
                entity_count,
                2,
                dtype=torch.long,
            ),
            open_rel_span_mask=torch.ones(
                batch_size,
                entity_count,
                dtype=torch.bool,
            ),
        )


def test_prediction_step_flattens_variable_open_relation_assignments():
    trainer = _bare_trainer()
    model = _OpenRelexLossModel()
    first = trainer.prediction_step(
        model,
        {
            "open_rel_assignment_labels": torch.zeros(
                1, 1, 2, 2
            ),
        },
        prediction_loss_only=False,
    )
    second = trainer.prediction_step(
        model,
        {
            "open_rel_assignment_labels": torch.zeros(
                1, 3, 4, 2
            ),
        },
        prediction_loss_only=False,
    )

    predictions = nested_concat(first[1], second[1], padding_index=-100)
    labels = nested_concat(first[2], second[2], padding_index=-100)

    assert predictions[0].shape == (2, 8, 2)
    assert predictions[1].shape == (2, 4, 2)
    assert predictions[2].shape == (2, 4)
    assert labels["open_rel_assignment_labels"].shape == (2, 12, 2)


class _StructuringRelationLossModel(nn.Module):
    def __init__(self, score_field):
        super().__init__()
        self.score_field = score_field

    def forward(self, **kwargs):
        relation_labels = kwargs["structuring_relation_labels"]
        scores = relation_labels.float() + 0.25
        return GLiFormerTextOutput(
            loss=scores.sum() * 0.0,
            **{self.score_field: scores},
        )


def test_prediction_step_flattens_structuring_relations_for_nested_concat():
    trainer = _bare_trainer()
    model = _StructuringRelationLossModel(
        "structuring_anchor_relation_scores"
    )
    labels_a2 = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    labels_a3 = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])

    first = trainer.prediction_step(
        model,
        {
            "structuring_relation_labels": labels_a2,
            "structuring_relation_group_mask": torch.tensor([True]),
        },
        prediction_loss_only=False,
    )
    second = trainer.prediction_step(
        model,
        {
            "structuring_relation_labels": labels_a3,
            "structuring_relation_group_mask": torch.tensor([False]),
        },
        prediction_loss_only=False,
    )

    predictions = nested_concat(first[1], second[1], padding_index=-100)
    labels = nested_concat(first[2], second[2], padding_index=-100)

    assert predictions.shape == (2, 9)
    assert torch.equal(
        predictions,
        torch.tensor(
            [
                [0.25, 1.25, 0.25, 0.25, -100, -100, -100, -100, -100],
                [0.25, 1.25, 0.25, 0.25, 0.25, 1.25, 0.25, 0.25, 0.25],
            ]
        ),
    )
    assert set(labels) == {
        "structuring_relation_labels",
        "structuring_relation_group_mask",
    }
    assert labels["structuring_relation_labels"].shape == (2, 9)
    assert torch.equal(
        labels["structuring_relation_labels"],
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, -100, -100, -100, -100, -100],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        labels["structuring_relation_group_mask"],
        torch.tensor([True, False]),
    )
    assert "structuring_relation_group_mask" not in _LABEL_KEYS


def _loss_owner(task_config):
    return SimpleNamespace(
        config=SimpleNamespace(get_task_config=lambda _: task_config),
        _loss=focal_loss_with_logits,
    )


def test_model_orchestration_defaults_binary_tasks_to_focal_loss():
    owner = _loss_owner(ObjectDetectionHeadConfig())
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "object_detection",
        {},
    )
    logits = torch.tensor([0.25, -0.5])
    labels = torch.tensor([1.0, 0.0])

    assert torch.allclose(
        loss_fn(logits, labels),
        focal_loss_with_logits(logits, labels, alpha=0.25, gamma=2.0),
    )


def test_structuring_keeps_all_anchor_negatives_despite_runtime_sampler():
    task_config = StructuringHeadConfig(
        negatives=1.0,
        masking="none",
    )
    owner = _loss_owner(task_config)
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "structuring",
        {"negatives": 0.005, "masking": "global"},
    )
    logits = torch.zeros(100)
    labels = torch.zeros_like(logits)

    losses = loss_fn(logits, labels)

    assert torch.count_nonzero(losses).item() == labels.numel()
    assert torch.allclose(
        losses,
        binary_focal_or_bce(
            logits,
            labels,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0,
        ),
    )


def test_model_orchestration_uses_bce_when_both_focal_controls_are_disabled():
    owner = _loss_owner(
        ObjectDetectionHeadConfig(
            focal_loss_alpha=-1.0,
            focal_loss_gamma=0.0,
        )
    )
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "object_detection",
        {},
    )
    logits = torch.tensor([0.25, -0.5])
    labels = torch.tensor([1.0, 0.0])

    assert torch.allclose(
        loss_fn(logits, labels),
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        ),
    )


def test_model_orchestration_uses_config_style_runtime_focal_values():
    owner = _loss_owner(ObjectDetectionHeadConfig())
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "object_detection",
        {
            "focal_loss_alpha": -1.0,
            "focal_loss_gamma": 0.0,
        },
    )
    logits = torch.tensor([0.25, -0.5])
    labels = torch.tensor([1.0, 0.0])

    assert torch.allclose(
        loss_fn(logits, labels),
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        ),
    )


@pytest.mark.parametrize(
    ("alpha", "gamma"),
    [(-1.0, 2.0), (0.25, 0.0)],
)
def test_model_orchestration_keeps_focal_when_either_control_is_enabled(
    alpha,
    gamma,
):
    owner = _loss_owner(
        ObjectDetectionHeadConfig(
            focal_loss_alpha=alpha,
            focal_loss_gamma=gamma,
        )
    )
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "object_detection",
        {},
    )
    logits = torch.tensor([0.25, -0.5])
    labels = torch.tensor([1.0, 0.0])

    assert torch.allclose(
        loss_fn(logits, labels),
        focal_loss_with_logits(
            logits,
            labels,
            alpha=alpha if alpha > 0 else -1.0,
            gamma=gamma,
        ),
    )


def test_zero_focal_alpha_with_positive_gamma_keeps_positive_gradients():
    owner = _loss_owner(
        ObjectDetectionHeadConfig(
            focal_loss_alpha=0.0,
            focal_loss_gamma=2.0,
        )
    )
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "object_detection",
        {},
    )
    logits = torch.tensor([-1.0, 1.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])

    loss_fn(logits, labels).sum().backward()

    assert logits.grad is not None
    assert logits.grad[0].abs() > 0


def test_focal_loss_keeps_gradients_for_saturated_wrong_logits():
    logits = torch.tensor([-200.0, 200.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])

    losses = binary_focal_or_bce(
        logits,
        labels,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
    )
    losses.sum().backward()

    assert torch.isfinite(losses).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0].item() < 0.0
    assert logits.grad[1].item() > 0.0


def test_model_orchestration_uses_stable_focal_instead_of_upstream_loss():
    owner = _loss_owner(ObjectDetectionHeadConfig())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the upstream focal primitive must not be called")

    owner._loss = fail_if_called
    loss_fn = BaseGLiFormerModel._make_task_loss_fn(
        owner,
        "object_detection",
        {},
    )
    logits = torch.tensor([-200.0], requires_grad=True)

    loss_fn(logits, torch.ones_like(logits)).sum().backward()

    assert logits.grad is not None
    assert logits.grad.item() < 0.0


def test_training_step_scales_accumulated_loss_when_deepspeed_is_none():
    trainer = _bare_trainer(gradient_accumulation_steps=4)
    trainer.accelerator = _RecordingAccelerator()
    trainer.deepspeed = None
    model = _TaskLossModel("image_classification_labels")

    loss = trainer.training_step(
        model,
        {
            "features": torch.tensor([1.0]),
            "image_classification_labels": torch.tensor([0.0]),
        },
    )

    assert loss.item() == pytest.approx(1.0)
    assert trainer.accelerator.backward_loss.item() == pytest.approx(1.0)
    assert model.scale.grad.item() == pytest.approx(1.0)
