"""GLiFormer-level settings that configure a custom backbone must reach its config class.

`max_page_embeddings` lives on `GLiFormerConfig`, but the thing it sizes -- the page
embedding table -- lives on `LayoutDebertaConfig`. `_coerce_backbone_config` builds the
latter from the *encoder* config, so without explicit forwarding the knob is inert.
"""

import pytest

from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Config

from gliformer.backbones import get_backbone
from gliformer.backbones.deberta_2d import LayoutDebertaConfig, LayoutDebertaEmbeddings
from gliformer.encoders.base import _coerce_backbone_config


class FakeGLiFormerConfig:
    """Only the attributes `_coerce_backbone_config` reads."""

    def __init__(self, max_page_embeddings=None):
        if max_page_embeddings is not None:
            self.max_page_embeddings = max_page_embeddings


@pytest.fixture
def backbone():
    return get_backbone("deberta_2d")


@pytest.fixture
def hub_config():
    """A stock HF config, as pulled off the hub: it predates the layout backbone."""
    return DebertaV2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=0,
    )


class TestForwarding:
    @pytest.mark.parametrize("pages", [0, 8, 64, 1024])
    def test_gliformer_value_reaches_the_backbone(self, hub_config, backbone, pages):
        coerced = _coerce_backbone_config(hub_config, backbone, FakeGLiFormerConfig(pages))
        assert coerced.max_page_embeddings == pages

    @pytest.mark.parametrize("pages,expected_rows", [(0, None), (8, 8), (64, 64)])
    def test_value_sizes_the_page_embedding_table(self, hub_config, backbone, pages, expected_rows):
        coerced = _coerce_backbone_config(hub_config, backbone, FakeGLiFormerConfig(pages))
        embeddings = LayoutDebertaEmbeddings(coerced)
        if expected_rows is None:
            assert embeddings.page_embeddings is None
        else:
            assert embeddings.page_embeddings.num_embeddings == expected_rows

    def test_absent_gliformer_config_keeps_the_class_default(self, hub_config, backbone):
        default = LayoutDebertaConfig().max_page_embeddings
        assert _coerce_backbone_config(hub_config, backbone).max_page_embeddings == default
        assert (
            _coerce_backbone_config(hub_config, backbone, FakeGLiFormerConfig()).max_page_embeddings
            == default
        )


class TestSavedConfigWins:
    """A config from an earlier run describes the shapes of its saved weights."""

    @pytest.fixture
    def saved(self, hub_config, backbone):
        return _coerce_backbone_config(hub_config, backbone, FakeGLiFormerConfig(8))

    def test_saved_instance_is_returned_unchanged(self, saved, backbone):
        coerced = _coerce_backbone_config(saved, backbone, FakeGLiFormerConfig(999))
        assert coerced is saved
        assert coerced.max_page_embeddings == 8

    def test_saved_dict_beats_a_mismatched_gliformer_value(self, saved, backbone):
        coerced = _coerce_backbone_config(saved.to_dict(), backbone, FakeGLiFormerConfig(999))
        assert coerced.max_page_embeddings == 8


class TestPassthrough:
    def test_no_backbone_returns_the_encoder_config_untouched(self, hub_config):
        assert _coerce_backbone_config(hub_config, None, FakeGLiFormerConfig(8)) is hub_config

    def test_uncoercible_encoder_config_still_raises(self, backbone):
        with pytest.raises(TypeError, match="Cannot coerce encoder_config"):
            _coerce_backbone_config(object(), backbone, FakeGLiFormerConfig(8))
