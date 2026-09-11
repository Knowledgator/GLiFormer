"""Flash attention for the deberta_2d backbone.

`LayoutDebertaModel` adds a 2D layout bias inside the attention scores and is
fed packed block masks by `Transformer._forward_deberta`, neither of which the
stock FlashDeBERTa attention can express. It therefore drives the flashdeberta
Triton kernels itself: the disentangled kernel when the batch is plain padded
text, the bias kernel whenever a layout bias or a packed mask is in play. Both
must agree with the eager path they replace.
"""


import pytest
import torch

from gliformer.backbones import deberta_2d, flash_deberta
from gliformer.backbones.deberta_2d import LayoutDebertaConfig, LayoutDebertaModel
from gliformer.backbones.flash_deberta import (
    EAGER,
    FLASH_AUTO,
    FLASH_BIAS,
    FLASH_DISENTANGLED,
    additive_mask_bias,
    flash_kernels_available,
    is_flash_kernel,
    normalize_attn_kernel,
    padding_lengths,
)
from gliformer.encoders.base import _configure_layout_flash_attention, _requested_flash_kernel

B, L = 2, 48
PAD_FROM = 30

requires_flash = pytest.mark.skipif(
    not (torch.cuda.is_available() and flash_kernels_available()),
    reason="the flashdeberta Triton kernels need CUDA and the flashdeberta package",
)


def _config(attn_kernel=EAGER, layout=False, **overrides):
    base = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        max_2d_position_embeddings=128,
        relative_attention=True,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        norm_rel_ebd="layer_norm",
        layer_norm_eps=1e-7,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layout_embedding_type="both" if layout else "absolute",
        attn_kernel=attn_kernel,
    )
    base.update(overrides)
    return LayoutDebertaConfig(**base)


def _batch(device="cpu"):
    torch.manual_seed(1)
    input_ids = torch.randint(0, 256, (B, L), device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    attention_mask[1, PAD_FROM:] = 0
    return input_ids, attention_mask


def _boxes(device="cpu"):
    torch.manual_seed(2)
    bbox = torch.randint(0, 100, (B, L, 4), device=device)
    bbox[..., 2:] = bbox[..., :2] + 4
    return bbox


def _packed_block_mask(device="cpu"):
    """Two independent segments per row, the shape packing produces."""
    mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    mask[:, : L // 2, : L // 2] = True
    mask[:, L // 2 :, L // 2 :] = True
    return mask


def _pair(attn_kernel, layout=False, dtype=torch.float32, device="cpu"):
    """An eager and a flash model sharing one set of weights."""
    torch.manual_seed(0)
    eager = LayoutDebertaModel(_config(EAGER, layout)).to(device=device, dtype=dtype).eval()
    flash = LayoutDebertaModel(_config(attn_kernel, layout)).to(device=device, dtype=dtype).eval()
    flash.load_state_dict(eager.state_dict())
    return eager, flash


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """`warn_once` dedupes per process, which would hide warnings from later tests."""
    flash_deberta._WARNED.clear()


class TestKernelNames:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("flash_attention_2", FLASH_AUTO),
            ("flash", FLASH_AUTO),
            (True, FLASH_AUTO),
            ("flash_bias", FLASH_BIAS),
            ("flash-disentangled", FLASH_DISENTANGLED),
        ],
    )
    def test_flash_names_are_recognized(self, value, expected):
        assert is_flash_kernel(value) == expected
        assert normalize_attn_kernel(value) == expected

    @pytest.mark.parametrize("value", [None, False, "eager", "sdpa"])
    def test_non_flash_names_mean_eager(self, value):
        assert is_flash_kernel(value) is None
        assert normalize_attn_kernel(value) == EAGER

    def test_unknown_kernel_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown attn_kernel"):
            normalize_attn_kernel("triton_v9")

    def test_hub_kernel_names_are_not_mistaken_for_flash(self):
        """`_attn_implementation` carries names this module knows nothing about."""
        assert is_flash_kernel("kernels-community/vllm-flash-attn3") is None


class TestMaskAnalysis:
    """Only right padding has a per-example-length equivalent."""

    def test_right_padding_gives_lengths(self):
        _, attention_mask = _batch()
        assert torch.equal(padding_lengths(attention_mask), torch.tensor([L, PAD_FROM], dtype=torch.int32))

    def test_left_padding_is_rejected(self):
        mask = torch.ones(B, L, dtype=torch.long)
        mask[1, :5] = 0
        assert padding_lengths(mask) is None

    def test_square_padding_mask_gives_lengths(self):
        _, attention_mask = _batch()
        valid = attention_mask.bool()
        block = (valid.unsqueeze(-1) & valid.unsqueeze(-2)).unsqueeze(1)
        assert torch.equal(padding_lengths(block), torch.tensor([L, PAD_FROM], dtype=torch.int32))

    def test_packed_block_mask_is_rejected(self):
        assert padding_lengths(_packed_block_mask()) is None

    def test_per_head_mask_is_rejected(self):
        _, attention_mask = _batch()
        valid = attention_mask.bool()
        block = (valid.unsqueeze(-1) & valid.unsqueeze(-2)).unsqueeze(1).expand(B, 4, L, L)
        assert padding_lengths(block) is None

    def test_mask_bias_penalty_stays_finite(self):
        """A fully masked row must fall back to a uniform softmax, not to NaN."""
        mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
        mask[:, :, :, :4] = True
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            bias = additive_mask_bias(mask, dtype)
            assert bias.shape == (B, 1, L, L)
            assert bias.dtype == dtype
            assert torch.isfinite(bias).all()
            assert (bias[..., :4] == 0).all()
            assert (bias[..., 4:] < 0).all()
            # Summing with the other bias terms must not overflow to -inf.
            assert torch.isfinite(bias + bias.new_full((), -100.0)).all()


class TestConfiguration:
    def test_default_is_eager(self):
        assert _config().attn_kernel == EAGER

    def test_kernel_survives_a_config_round_trip(self):
        restored = LayoutDebertaConfig(**_config(FLASH_AUTO).to_dict())
        assert restored.attn_kernel == FLASH_AUTO

    @pytest.mark.parametrize(
        ("attn_implementation", "env", "expected"),
        [
            (None, None, None),
            ("eager", None, None),
            ("flash_attention_2", None, FLASH_AUTO),
            (None, "1", FLASH_AUTO),
            ("flash_bias", None, FLASH_BIAS),
        ],
    )
    def test_request_comes_from_config_or_env(self, monkeypatch, attn_implementation, env, expected):
        monkeypatch.delenv("USE_FLASHDEBERTA", raising=False)
        if env is not None:
            monkeypatch.setenv("USE_FLASHDEBERTA", env)
        config = _config()
        config._attn_implementation = attn_implementation
        assert _requested_flash_kernel(config) == expected

    def test_env_switch_also_covers_the_labels_encoder(self, monkeypatch):
        monkeypatch.setenv("USE_FLASHDEBERTA", "1")
        config = _config()
        config._attn_implementation = None
        assert _requested_flash_kernel(config, labels_encoder=True) == FLASH_AUTO

    def test_attn_implementation_stays_on_the_text_encoder(self, monkeypatch):
        monkeypatch.delenv("USE_FLASHDEBERTA", raising=False)
        config = _config()
        config._attn_implementation = "flash_attention_2"
        assert _requested_flash_kernel(config, labels_encoder=True) is None

    def test_hf_attn_implementation_is_neutralized(self):
        """transformers rejects flash_attention_2 on a model that does not declare it."""
        config = _config()
        config._attn_implementation = "flash_attention_2"
        _configure_layout_flash_attention(config, FLASH_AUTO)
        assert config._attn_implementation == EAGER
        assert config.attn_kernel == FLASH_AUTO
        # The model would refuse to build otherwise.
        LayoutDebertaModel(config)

    def test_missing_package_falls_back_to_eager(self, monkeypatch):
        monkeypatch.setattr("gliformer.encoders.base.flash_kernels_available", lambda: False)
        config = _config()
        with pytest.warns(UserWarning, match="pip install flashdeberta"):
            _configure_layout_flash_attention(config, FLASH_AUTO)
        assert config.attn_kernel == EAGER


class TestEagerFallback:
    """Anything the kernels cannot run must quietly produce the eager result."""

    def test_cpu_falls_back_and_warns(self):
        eager, flash = _pair(FLASH_AUTO)
        input_ids, attention_mask = _batch()
        with torch.no_grad():
            expected = eager(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            with pytest.warns(UserWarning, match="Falling back to eager attention"):
                actual = flash(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        assert torch.equal(actual, expected)

    def test_output_attentions_uses_the_eager_path(self):
        """The kernels never materialize probabilities, so they cannot return them."""
        eager, flash = _pair(FLASH_AUTO)
        input_ids, attention_mask = _batch()
        with torch.no_grad():
            actual = flash(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            expected = eager(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        assert actual.attentions[0] is not None
        assert torch.equal(actual.attentions[0], expected.attentions[0])

    def test_missing_kernels_fall_back(self, monkeypatch):
        monkeypatch.setattr("gliformer.backbones.deberta_2d.flash_kernels_available", lambda: False)
        eager, flash = _pair(FLASH_AUTO)
        input_ids, attention_mask = _batch()
        with torch.no_grad(), pytest.warns(UserWarning, match="pip install flashdeberta"):
            actual = flash(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            expected = eager(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        assert torch.equal(actual, expected)


class TestKernelSelection:
    """Which kernel a batch qualifies for, decided once per encoder pass."""

    def _context(self, model, attention_mask, **kwargs):
        encoder = model.encoder
        return encoder._resolve_flash_context(
            encoder.get_attention_mask(attention_mask),
            custom_relative_pos=kwargs.get("custom_relative_pos", False),
            query_states=kwargs.get("query_states"),
            output_attentions=kwargs.get("output_attentions", False),
        )

    def test_padded_text_qualifies_for_the_disentangled_kernel(self):
        _, flash = _pair(FLASH_AUTO)
        context = self._context(flash, _batch()[1])
        assert context.seq_lengths is not None

    def test_packed_masks_fall_to_the_bias_kernel(self):
        _, flash = _pair(FLASH_AUTO)
        assert self._context(flash, _packed_block_mask()).seq_lengths is None

    def test_forced_bias_kernel_skips_the_mask_probe(self):
        _, flash = _pair(FLASH_BIAS)
        assert self._context(flash, _batch()[1]).seq_lengths is None

    def test_custom_relative_positions_skip_the_disentangled_kernel(self):
        """That kernel rebuilds buckets from indices, ignoring supplied positions."""
        _, flash = _pair(FLASH_AUTO)
        assert self._context(flash, _batch()[1], custom_relative_pos=True).seq_lengths is None

    def test_eager_config_has_no_context(self):
        eager, _ = _pair(FLASH_AUTO)
        assert self._context(eager, _batch()[1]) is None


@requires_flash
class TestKernelParity:
    """Valid tokens must match the eager path within fp16 rounding.

    Padded rows are excluded: the disentangled kernel leaves them at zero while
    the eager path fills them with the meaningless output of a uniform softmax
    over masked scores. Nothing downstream reads them.
    """

    ATOL = 2e-2

    def _compare(self, attn_kernel, layout=False, **forward):
        eager, flash = _pair(attn_kernel, layout, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = eager(**forward).last_hidden_state.float()
            actual = flash(**forward).last_hidden_state.float()
        assert torch.isfinite(actual).all()
        return expected, actual

    def test_padded_text_matches_eager(self):
        input_ids, attention_mask = _batch("cuda")
        expected, actual = self._compare(FLASH_AUTO, input_ids=input_ids, attention_mask=attention_mask)
        valid = attention_mask.bool()
        assert torch.allclose(actual[valid], expected[valid], atol=self.ATOL, rtol=0)

    def test_bias_kernel_matches_eager_everywhere(self):
        """With a full L x L bias the kernel also reproduces the padded rows."""
        input_ids, attention_mask = _batch("cuda")
        expected, actual = self._compare(FLASH_BIAS, input_ids=input_ids, attention_mask=attention_mask)
        assert torch.allclose(actual, expected, atol=self.ATOL, rtol=0)

    def test_layout_bias_matches_eager(self):
        input_ids, attention_mask = _batch("cuda")
        expected, actual = self._compare(
            FLASH_AUTO,
            layout=True,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=_boxes("cuda"),
            layout_input_mask=torch.ones(B, dtype=torch.bool, device="cuda"),
        )
        assert torch.allclose(actual, expected, atol=self.ATOL, rtol=0)

    def test_packed_block_mask_matches_eager(self):
        """The whole point of the bias kernel: no attention across segments."""
        input_ids, _ = _batch("cuda")
        eager, flash = _pair(FLASH_AUTO, dtype=torch.float16, device="cuda")
        block = _packed_block_mask("cuda")
        embeddings = eager.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            mask=torch.ones(B, L, dtype=torch.long, device="cuda"),
        )
        with torch.no_grad():
            expected = eager.encoder(embeddings, block).last_hidden_state.float()
            actual = flash.encoder(embeddings, block).last_hidden_state.float()
        assert torch.allclose(actual, expected, atol=self.ATOL, rtol=0)

    def test_disentangled_kernel_actually_runs(self, monkeypatch):
        calls = []
        original = deberta_2d.flash_attention_disentangled

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(deberta_2d, "flash_attention_disentangled", spy)
        input_ids, attention_mask = _batch("cuda")
        self._compare(FLASH_AUTO, input_ids=input_ids, attention_mask=attention_mask)
        assert len(calls) == 2  # one per layer

    def test_layout_bias_routes_to_the_bias_kernel(self, monkeypatch):
        calls = []
        original = deberta_2d.flash_attention_bias

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(deberta_2d, "flash_attention_bias", spy)
        input_ids, attention_mask = _batch("cuda")
        self._compare(
            FLASH_DISENTANGLED,
            layout=True,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=_boxes("cuda"),
            layout_input_mask=torch.ones(B, dtype=torch.bool, device="cuda"),
        )
        assert len(calls) == 2


@requires_flash
class TestGradientParity:
    """Training is the reason to run these kernels, so the backward pass matters."""

    @pytest.mark.parametrize("attn_kernel", [FLASH_AUTO, FLASH_BIAS])
    @pytest.mark.parametrize("layout", [False, True], ids=["text", "layout"])
    def test_gradients_match_eager(self, attn_kernel, layout):
        input_ids, attention_mask = _batch("cuda")
        forward = dict(input_ids=input_ids, attention_mask=attention_mask)
        if layout:
            forward["bbox"] = _boxes("cuda")
            forward["layout_input_mask"] = torch.ones(B, dtype=torch.bool, device="cuda")
        torch.manual_seed(3)
        weights = torch.randn(B, L, 64, device="cuda")
        valid = attention_mask.float().unsqueeze(-1)

        def gradients(model):
            model.train()
            hidden = model(**forward).last_hidden_state.float()
            (hidden * weights * valid).sum().backward()
            return {n: p.grad.detach().float() for n, p in model.named_parameters() if p.grad is not None}

        eager, flash = _pair(attn_kernel, layout, dtype=torch.float16, device="cuda")
        expected, actual = gradients(eager), gradients(flash)

        compared = 0
        for name, reference in expected.items():
            norm = reference.norm().item()
            if norm < 1e-3:
                # Gradients this small are pure fp16 cancellation noise; the
                # eager path does not reproduce itself there either.
                continue
            compared += 1
            error = (reference - actual[name]).norm().item() / norm
            assert error < 0.05, f"{name}: relative gradient error {error:.3g}"
        assert compared > 10


@requires_flash
class TestPackedGuard:
    """Stock FlashDeBERTa cannot see a block mask; it must not silently ignore one."""

    def _transformer(self):
        from transformers import DebertaV2Config

        from gliformer.config import GLiFormerConfig
        from gliformer.encoders.base import Transformer

        encoder_config = DebertaV2Config(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
            relative_attention=True,
            position_buckets=16,
            pos_att_type=["p2c", "c2p"],
        )
        config = GLiFormerConfig(model_name="deberta", _attn_implementation="flash_attention_2")
        config.encoder_config = encoder_config
        return Transformer("deberta", config, from_pretrained=False)

    def test_packed_mask_is_refused(self):
        transformer = self._transformer()
        assert type(transformer.model).__name__ == "FlashDebertaV2Model"
        input_ids = torch.randint(0, 256, (B, L))
        with pytest.raises(ValueError, match="packed segments or left padding"):
            transformer(
                input_ids=input_ids,
                attention_mask=torch.ones(B, L, dtype=torch.long),
                pair_attention_mask=_packed_block_mask(),
            )

    def test_padding_mask_is_allowed(self):
        transformer = self._transformer().cuda().half()
        input_ids = torch.randint(0, 256, (B, L), device="cuda")
        _, attention_mask = _batch("cuda")
        valid = attention_mask.bool()
        with torch.no_grad():
            output = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pair_attention_mask=valid.unsqueeze(-1) & valid.unsqueeze(-2),
            )
        assert output.shape == (B, L, 64)


@requires_flash
class TestTransformerIntegration:
    """`Transformer._forward_deberta` is how GLiFormer actually reaches the backbone."""

    def _transformer(self, attn_implementation, layout):
        from gliformer.config import GLiFormerConfig
        from gliformer.encoders.base import Transformer

        kwargs = {"_attn_implementation": attn_implementation} if attn_implementation else {}
        config = GLiFormerConfig(
            model_name="layout",
            backbone_type="deberta_2d",
            encoder_config=_config(EAGER, layout),
            **kwargs,
        )
        torch.manual_seed(0)
        return Transformer("layout", config, from_pretrained=False).cuda().half().eval()

    @pytest.mark.parametrize("layout", [False, True], ids=["text", "layout"])
    def test_packed_batch_matches_eager(self, layout):
        eager = self._transformer(None, layout)
        flash = self._transformer("flash_attention_2", layout)
        flash.load_state_dict(eager.state_dict())
        assert eager.model.config.attn_kernel == EAGER
        assert flash.model.config.attn_kernel == FLASH_AUTO

        input_ids, attention_mask = _batch("cuda")
        valid = attention_mask.bool()
        forward = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_attention_mask=_packed_block_mask("cuda") & valid.unsqueeze(-1) & valid.unsqueeze(-2),
        )
        if layout:
            forward["bbox"] = _boxes("cuda")
            forward["layout_input_mask"] = torch.ones(B, dtype=torch.bool, device="cuda")

        with torch.no_grad():
            expected = eager(**forward).float()
            actual = flash(**forward).float()
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual[valid], expected[valid], atol=2e-2, rtol=0)
