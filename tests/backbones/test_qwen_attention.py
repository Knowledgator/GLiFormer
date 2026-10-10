"""Attention options of the Qwen3 / Qwen3.5 backbones.

``relative_pos_encoding`` may be left unset, and every combination of
causal/bidirectional masking has to give the same numbers on each backend
``_attn_implementation`` can name.
"""

import copy

import pytest
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from gliformer.backbones import qwen3_5
from gliformer.backbones.qwen3 import GLiFormerQwen3Model, _relative_position_type

FLASH = "flash_attention_2"
# The Gated DeltaNet layers of Qwen3.5 only have a CUDA kernel.
needs_cuda = pytest.mark.skipif(
    not (qwen3_5.QWEN3_5_AVAILABLE and torch.cuda.is_available()),
    reason="Qwen3.5 linear attention needs transformers support and CUDA",
)


def _qwen3(**options):
    config = Qwen3Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        pad_token_id=0,
    )
    for name, value in options.items():
        setattr(config, name, value)
    return GLiFormerQwen3Model(config).eval()


def _qwen3_5_config(**options):
    config = qwen3_5.Qwen3_5TextConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        pad_token_id=0,
    )
    for name, value in options.items():
        setattr(config, name, value)
    return config


def _qwen3_5(**options):
    return qwen3_5.GLiFormerQwen3_5TextModel(_qwen3_5_config(**options)).eval().cuda()


BUILDERS = [
    pytest.param(_qwen3, id="qwen3"),
    pytest.param(_qwen3_5, id="qwen3_5", marks=needs_cuda),
]


def _batch(model):
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, 100, (3, 12), generator=generator)
    attention_mask = torch.ones(3, 12, dtype=torch.long)
    attention_mask[1, 8:] = 0
    attention_mask[2, 5:] = 0
    return input_ids.to(model.device), attention_mask.to(model.device)


def _run(model, attn_implementation, input_ids, attention_mask):
    model.config._attn_implementation_internal = attn_implementation
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def _same(left, right, attention_mask=None):
    if attention_mask is not None:
        left, right = left[attention_mask.bool()], right[attention_mask.bool()]
    return torch.allclose(left, right, atol=1e-4)


@pytest.fixture
def flash(monkeypatch):
    """Stand-in for flash-attn that holds callers to its contract: a 2D padding mask and ``is_causal``."""

    def forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        assert attention_mask is None or attention_mask.dim() == 2
        length = query.shape[2]
        mask = torch.zeros(query.shape[0], 1, length, length, device=query.device)
        if attention_mask is not None:
            mask = mask.masked_fill(attention_mask[:, None, None, :].eq(0), float("-inf"))
        if module.is_causal if is_causal is None else is_causal:
            mask = mask + torch.full((length, length), float("-inf"), device=query.device).triu(1)
        return sdpa_attention_forward(module, query, key, value, mask, dropout=dropout, scaling=scaling, is_causal=False)

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, FLASH, forward)


class TestRelativePositionType:
    @pytest.mark.parametrize("value", [None, "none", "None"])
    def test_null_means_none(self, value):
        config = Qwen3Config()
        config.relative_pos_encoding = value
        assert _relative_position_type(config) == "none"

    def test_unset_means_none(self):
        assert _relative_position_type(Qwen3Config()) == "none"

    def test_unknown_is_rejected(self):
        config = Qwen3Config()
        config.relative_pos_encoding = "rope2d"
        with pytest.raises(ValueError, match="relative position encoding"):
            _relative_position_type(config)


@pytest.mark.parametrize("build", BUILDERS)
class TestAttentionBackends:
    @pytest.mark.parametrize("bidirectional", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_backends_agree_without_relative_bias(self, build, flash, bidirectional, masked):
        model = build(bidirectional_attention=bidirectional)
        input_ids, attention_mask = _batch(model)
        mask = attention_mask if masked else None
        eager = _run(model, "eager", input_ids, mask)
        assert _same(_run(model, "sdpa", input_ids, mask), eager, mask)
        assert _same(_run(model, FLASH, input_ids, mask), eager, mask)

    @pytest.mark.parametrize("attn_implementation", ["eager", "sdpa", FLASH])
    def test_bidirectional_differs_from_causal(self, build, flash, attn_implementation):
        model = build(bidirectional_attention=True)
        causal = build()
        causal.load_state_dict(model.state_dict())
        input_ids, _ = _batch(model)
        assert not _same(
            _run(model, attn_implementation, input_ids, None),
            _run(causal, attn_implementation, input_ids, None),
        )

    @pytest.mark.parametrize("relative_pos_encoding", ["t5", "deberta", "alibi"])
    @pytest.mark.parametrize("bidirectional", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_sdpa_applies_relative_bias(self, build, relative_pos_encoding, bidirectional, masked):
        model = build(relative_pos_encoding=relative_pos_encoding, bidirectional_attention=bidirectional)
        input_ids, attention_mask = _batch(model)
        mask = attention_mask if masked else None
        assert _same(_run(model, "sdpa", input_ids, mask), _run(model, "eager", input_ids, mask), mask)

    def test_flash_rejects_relative_bias(self, build, flash):
        model = build(relative_pos_encoding="deberta")
        with pytest.raises(ValueError, match="attention bias"):
            _run(model, FLASH, *_batch(model))


@needs_cuda
class TestQwen3_5Native:
    @pytest.mark.parametrize("attn_implementation", ["eager", "sdpa", FLASH])
    def test_default_config_matches_stock_model(self, flash, attn_implementation):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        model = _qwen3_5()
        stock = Qwen3_5TextModel(copy.deepcopy(model.config)).eval().cuda()
        stock.load_state_dict(model.state_dict())
        input_ids, attention_mask = _batch(model)
        assert _same(
            _run(model, attn_implementation, input_ids, attention_mask),
            _run(stock, attn_implementation, input_ids, attention_mask),
            attention_mask,
        )

    def test_wrapper_passes_options_to_the_text_model(self):
        config = qwen3_5.Qwen3_5Config(
            text_config=_qwen3_5_config().to_dict(),
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64},
        )
        config.bidirectional_attention = True
        config.relative_pos_encoding = None
        config._attn_implementation = "eager"
        model = qwen3_5.GLiFormerQwen3_5Model(config)
        attention = model.language_model.layers[1].self_attn
        assert model.language_model.bidirectional_attention
        assert attention.relative_pos_encoding == "none"
        assert attention.config._attn_implementation == "eager"
