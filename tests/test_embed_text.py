from types import SimpleNamespace

import torch

from glinext.glinext import BaseGLiNExT
from glinext.layers.pooling import MeanPooling


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        assert all(isinstance(text, str) for text in texts)
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }


class DummyEmbeddingModel:
    def __init__(self):
        self.heads = {
            "embedding": SimpleNamespace(
                _project=lambda embeddings: embeddings,
                pooling=MeanPooling(),
            )
        }

    def encode_embedding_tokens(self, input_ids, attention_mask):
        return input_ids.float().unsqueeze(-1).repeat(1, 1, 2)


def test_embed_text_tokenizes_raw_strings_like_embedding_training():
    tokenizer = RecordingTokenizer()
    wrapper = SimpleNamespace(
        config=SimpleNamespace(hidden_size=2),
        data_processor=SimpleNamespace(transformer_tokenizer=tokenizer),
        model=DummyEmbeddingModel(),
        device=torch.device("cpu"),
        eval=lambda: None,
        _require_task_heads=lambda *tasks: None,
        _filter_valid_texts=BaseGLiNExT._filter_valid_texts,
    )
    texts = ["Don't split punctuation.", "Keep raw strings!"]

    output = BaseGLiNExT.embed_text(wrapper, texts, batch_size=2)

    assert output.shape == (2, 2)
    assert tokenizer.calls == [
        (
            texts,
            {
                "return_tensors": "pt",
                "truncation": True,
                "padding": "longest",
            },
        )
    ]
