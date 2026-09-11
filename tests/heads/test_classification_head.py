"""Tests for ClassificationHead."""

from dataclasses import asdict

import pytest
import torch

from gliformer.config import ClassificationHeadConfig
from gliformer.tasks import TaskFlatInputs
from gliformer.tasks.classification.model import ClassificationHead
from gliformer.tasks.classification.scorer import ClassificationScorer
from gliformer.tasks.losses import binary_focal_or_bce
from tests.heads.conftest import B, C, D, W, make_config


def _make_head(**kwargs):
    config = make_config(classification_config=asdict(ClassificationHeadConfig(**kwargs)))
    return ClassificationHead.from_config(config)


# ── ClassificationScorer factory ─────────────────────────────────────────

class TestClassificationScorerFactory:
    def test_registry_has_all_types(self):
        expected = {"dot", "scaled-dot", "weighted-dot", "mlp", "hopfield"}
        assert expected.issubset(set(ClassificationScorer._registry))

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer type"):
            ClassificationScorer.from_config("nonexistent", hidden_size=D)

    @pytest.mark.parametrize(
        "scorer_type",
        ["dot", "scaled-dot", "weighted-dot", "mlp", "hopfield"],
    )
    def test_scorer_forward_shape(self, scorer_type):
        scorer = ClassificationScorer.from_config(scorer_type, hidden_size=D)
        text_rep = torch.randn(B, D)
        label_rep = torch.randn(B, C, D)
        scores = scorer(text_rep, label_rep)
        assert scores.shape == (B, C)

    def test_scaled_dot_divides_by_sqrt_hidden_size(self):
        scorer = ClassificationScorer.from_config("scaled-dot", hidden_size=4)
        text_rep = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        label_rep = torch.tensor(
            [[[4.0, 3.0, 2.0, 1.0], [1.0, -1.0, 1.0, -1.0]]]
        )

        expected = torch.einsum("bd,bcd->bc", text_rep, label_rep) / 2.0

        torch.testing.assert_close(scorer(text_rep, label_rep), expected)
        assert not tuple(scorer.parameters())


# ── ClassificationHead construction ──────────────────────────────────────

class TestClassificationHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.classification_config = None
        assert ClassificationHead.from_config(config) is None

    def test_from_config_default(self):
        head = _make_head()
        assert head is not None
        assert head.name == "classification"
        assert hasattr(head, "anchor_layer")
        assert hasattr(head, "anchor_modeling")
        assert hasattr(head, "pooling")
        assert hasattr(head, "scorer")

    def test_from_config_with_refine(self):
        head = _make_head(anchor_refine_layers=1)
        assert hasattr(head, "anchor_refine")

    @pytest.mark.parametrize(
        "scorer_type",
        ["dot", "scaled-dot", "weighted-dot", "mlp", "hopfield"],
    )
    def test_from_config_scorer_types(self, scorer_type):
        head = _make_head(scorer_type=scorer_type)
        assert head is not None
        assert isinstance(head.scorer, ClassificationScorer)


# ── ClassificationHead forward ───────────────────────────────────────────

class TestClassificationHeadForward:
    def test_inference_with_flat_inputs(self, shared, flat_inputs):
        head = _make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.loss is None
        assert out.logits is not None
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape == (BN, C)

    def test_training_with_labels(self, shared, flat_inputs):
        head = _make_head()
        BN = flat_inputs.words_embedding.shape[0]
        labels = torch.zeros(BN, C)
        labels[:, 0] = 1.0  # first class positive
        out = head(shared, {}, flat_inputs=flat_inputs, cat_labels=labels)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_loss_is_normalized_by_active_bn_class_cells(
        self,
        shared,
        flat_inputs,
    ):
        flat_inputs.child_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
        labels = torch.zeros(B, C)

        output = _make_head()(
            shared,
            {},
            flat_inputs=flat_inputs,
            cat_labels=labels,
            base_loss_fn=lambda logits, targets: torch.ones_like(logits),
        )

        # Four valid BN*C cells contribute one unit each.
        assert output.loss.item() == pytest.approx(1.0)

    def test_direct_head_uses_configured_focal_parameters(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head(
            focal_loss_alpha=0.4,
            focal_loss_gamma=1.5,
            focal_loss_prob_margin=0.1,
        )
        labels = torch.zeros(B, C)

        output = head(
            shared,
            {},
            flat_inputs=flat_inputs,
            cat_labels=labels,
        )

        expected = binary_focal_or_bce(
            output.logits,
            labels,
            focal_loss_alpha=0.4,
            focal_loss_gamma=1.5,
            focal_loss_prob_margin=0.1,
        ).mean()
        torch.testing.assert_close(output.loss, expected)

    def test_inference_with_different_bn(self, shared, flat_inputs):
        """Test with BN > B (multiple groups per batch item)."""
        head = _make_head()
        BN = 4
        fi = TaskFlatInputs(
            words_embedding=torch.randn(BN, W, D),
            mask=torch.ones(BN, W, dtype=torch.long),
            parent_embedding=torch.randn(BN, D),
            child_embedding=torch.randn(BN, C, D),
            child_mask=torch.ones(BN, C, dtype=torch.long),
            batch_origin=torch.tensor([0, 0, 1, 1]),
        )
        out = head(shared, {}, flat_inputs=fi)
        assert out.logits is not None
        assert out.logits.shape == (BN, C)

    def test_cls_pooling_uses_encoder_cls_when_no_source_words_survive(
        self,
        shared,
        flat_inputs,
    ):
        head = _make_head(pooling_type="cls")
        flat_inputs.words_embedding = torch.empty(B, 0, D)
        flat_inputs.mask = torch.empty(B, 0, dtype=torch.long)
        shared.token_embeds[:, 0] = torch.arange(
            B * D,
            dtype=shared.token_embeds.dtype,
        ).reshape(B, D)

        text_rep = head._pool_text(shared, flat_inputs)

        assert torch.equal(text_rep, shared.token_embeds[:, 0])

        output = head(shared, {}, flat_inputs=flat_inputs)
        assert output.logits.shape == (B, C)
        assert torch.isfinite(output.logits).all()

    def test_gradient_flows(self, shared, flat_inputs):
        head = _make_head()
        BN = flat_inputs.words_embedding.shape[0]
        labels = torch.zeros(BN, C)
        labels[:, 0] = 1.0
        flat_inputs.words_embedding.requires_grad_(True)
        out = head(shared, {}, flat_inputs=flat_inputs, cat_labels=labels)
        out.loss.backward()
        assert flat_inputs.words_embedding.grad is not None
        flat_inputs.words_embedding.requires_grad_(False)

    def test_anchor_refine_uses_words_embedding(self, shared, flat_inputs):
        head = _make_head(anchor_refine_layers=1)

        class SpyRefine(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.token_emb = None
                self.token_mask = None

            def forward(
                self,
                anchor_rep,
                token_emb,
                token_mask=None,
                query_mask=None,
            ):
                self.token_emb = token_emb
                self.token_mask = token_mask
                return anchor_rep

        spy = SpyRefine()
        head.anchor_refine = spy

        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.logits is not None
        assert spy.token_emb is flat_inputs.words_embedding
        assert spy.token_mask is flat_inputs.mask

    @pytest.mark.parametrize(
        "scorer_type",
        ["dot", "scaled-dot", "weighted-dot", "mlp", "hopfield"],
    )
    def test_forward_all_scorers(self, shared, flat_inputs, scorer_type):
        head = _make_head(scorer_type=scorer_type)
        out = head(shared, {}, flat_inputs=flat_inputs)
        BN = flat_inputs.words_embedding.shape[0]
        assert out.logits.shape == (BN, C)
