"""Tests for CountHead and CountModule."""

import pytest
import torch

from gliformer.tasks.count.model import CountHead, CountModule
from gliformer.tasks import TaskHeadOutput
from tests.heads.conftest import make_config, D, B, W, C
from dataclasses import asdict
from gliformer.config import CountHeadConfig


# ── CountModule ──────────────────────────────────────────────────────────

class TestCountModule:
    def test_regression_output_shape(self):
        m = CountModule(D, mode="regression")
        x = torch.randn(B, D)
        out = m(x)
        assert out.shape == (B, 1)

    def test_classification_output_shape(self):
        max_count = 10
        m = CountModule(D, mode="classification", max_count=max_count)
        x = torch.randn(B, D)
        out = m(x)
        assert out.shape == (B, max_count + 1)

    def test_regression_loss(self):
        m = CountModule(D, mode="regression")
        logits = torch.randn(B, 1)
        targets = torch.tensor([3.0, 5.0])
        loss = m.loss(logits, targets)
        assert loss.item() >= 0
        assert loss.isfinite()

    def test_classification_loss(self):
        m = CountModule(D, mode="classification", max_count=10)
        logits = torch.randn(B, 11)
        targets = torch.tensor([2, 5])
        loss = m.loss(logits, targets)
        assert loss.item() >= 0

    def test_classification_loss_clamped(self):
        m = CountModule(D, mode="classification", max_count=5)
        logits = torch.randn(1, 6)
        targets = torch.tensor([100])  # exceeds max_count, should be clamped
        loss = m.loss(logits, targets)
        assert loss.isfinite()


# ── CountHead construction ───────────────────────────────────────────────

class TestCountHeadConstruction:
    def test_from_config_disabled(self):
        config = make_config()
        config.count_config = None
        assert CountHead.from_config(config) is None

    def test_from_config_regression(self):
        config = make_config(count_config=asdict(CountHeadConfig(mode="regression")))
        head = CountHead.from_config(config)
        assert head is not None
        assert head.name == "count"

    def test_from_config_classification(self):
        config = make_config(count_config=asdict(CountHeadConfig(mode="classification", max_count=15)))
        head = CountHead.from_config(config)
        assert head is not None


# ── CountHead forward ────────────────────────────────────────────────────

class TestCountHeadForward:
    def _make_head(self, mode="regression", max_count=20):
        config = make_config(count_config=asdict(CountHeadConfig(mode=mode, max_count=max_count)))
        return CountHead.from_config(config)

    def test_inference_with_flat_inputs(self, flat_inputs, shared):
        head = self._make_head()
        out = head(shared, {}, flat_inputs=flat_inputs)
        assert out.loss is None
        assert out.logits is not None
        assert out.logits.shape[0] == flat_inputs.parent_embedding.shape[0]

    def test_training_regression(self, shared, flat_inputs):
        head = self._make_head(mode="regression")
        targets = torch.tensor([3.0, 5.0])
        out = head(shared, {}, flat_inputs=flat_inputs, count_targets=targets)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_training_classification(self, shared, flat_inputs):
        head = self._make_head(mode="classification", max_count=10)
        targets = torch.tensor([2, 5])
        out = head(shared, {}, flat_inputs=flat_inputs, count_targets=targets)
        assert out.loss is not None
        assert out.loss.item() >= 0

    def test_gradient_flows(self, shared, flat_inputs):
        head = self._make_head()
        flat_inputs.parent_embedding = flat_inputs.parent_embedding.detach().requires_grad_(True)
        targets = torch.tensor([1.0, 2.0])
        out = head(shared, {}, flat_inputs=flat_inputs, count_targets=targets)
        out.loss.backward()
        assert flat_inputs.parent_embedding.grad is not None
        flat_inputs.parent_embedding.requires_grad_(False)
