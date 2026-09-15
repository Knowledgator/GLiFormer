"""Check the app's GPU boundaries without downloading weights or allocating CUDA."""

import functools
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from gliformer import GLiFormer


@pytest.mark.parametrize(
    ("zero_gpu", "cuda_available", "expected_device"),
    [(True, False, "cuda"), (False, False, "cpu"), (False, True, "cuda")],
)
def test_app_gpu_lifecycle(monkeypatch, zero_gpu, cuda_available, expected_device):
    pytest.importorskip("gradio", minversion="6.0")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("SPACES_ZERO_GPU", "1" if zero_gpu else "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    registered = set()
    active = False
    spaces = ModuleType("spaces")

    def gpu(fn=None, **options):
        def decorate(handler):
            @functools.wraps(handler)
            def wrapped(*args, **kwargs):
                nonlocal active
                assert not active, "GPU requests must not be nested"
                active = True
                try:
                    return handler(*args, **kwargs)
                finally:
                    active = False

            registered.add(wrapped)
            return wrapped

        return decorate(fn) if fn is not None else decorate

    spaces.GPU = gpu
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    model = Mock()

    def infer(*args, **kwargs):
        assert active, "Inference must execute inside a GPU callback"
        return {"ner": [[]]}

    model.inference.side_effect = infer

    def load(*args, **kwargs):
        if zero_gpu:
            assert not active, "ZeroGPU weights must load before worker requests"
        return model

    loader = Mock(side_effect=load)
    monkeypatch.setattr(GLiFormer, "from_pretrained", loader)
    app = runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"))
    assert loader.call_count == int(zero_gpu)
    callbacks = {
        event.fn for event in app["demo"].fns.values()
        if event.fn and event.fn.__name__.startswith(("run_", "score_all_"))
    }
    assert len(callbacks) == 16
    assert callbacks <= registered

    predictions, _ = app["run_ner"](
        "Alice", '[{"parent": "general", "labels": ["person"]}]',
        0.5, True, False, "",
    )
    assert predictions == "[]"
    assert "NER" in app["score_all_ner"](0.5, True, False)
    assert model.inference.call_count == 1 + len(app["NER_EXAMPLES"])
    loader.assert_called_once()
    model.to.assert_called_once_with(expected_device)
    model.eval.assert_called_once_with()
