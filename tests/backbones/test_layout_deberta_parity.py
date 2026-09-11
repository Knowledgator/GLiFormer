"""LayoutDebertaModel must reduce to stock DeBERTa-v2 when no layout is supplied.

Most GLiFormer tasks are text-only, and `configs/multitask.yaml` batches those rows
alongside layout rows under `backbone_type: deberta_2d`. A text-only row must
therefore be bit-identical to the pretrained backbone it was initialized from,
otherwise every text task pays for the layout branches it never uses.
"""

import pytest
import torch

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Model,
)

from gliformer.backbones.deberta_2d import LayoutDebertaConfig, LayoutDebertaModel

B, L = 2, 12


def _configs(**overrides):
    base = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=16,
        max_relative_positions=-1,
        pos_att_type=["p2c", "c2p"],
        norm_rel_ebd="layer_norm",
        type_vocab_size=0,
        layer_norm_eps=1e-7,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    base.update(overrides)
    hf_config = DebertaV2Config(**base)
    layout_dict = hf_config.to_dict()
    layout_dict.pop("model_type", None)
    return hf_config, LayoutDebertaConfig(**layout_dict)


@pytest.fixture
def models():
    """Stock and layout backbones sharing one set of DeBERTa weights."""
    torch.manual_seed(0)
    hf_config, layout_config = _configs()
    # Defaults matter: this is what `_coerce_backbone_config` actually builds.
    assert layout_config.max_page_embeddings > 0
    assert layout_config.layout_embedding_type == "absolute"

    reference = DebertaV2Model(hf_config).eval()
    layout = LayoutDebertaModel(layout_config).eval()
    layout.load_state_dict(reference.state_dict(), strict=False)
    return reference, layout


@pytest.fixture
def batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 256, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    attention_mask[1, L - 3:] = 0
    return input_ids, attention_mask


def _last_hidden(model, input_ids, attention_mask, **kwargs):
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state


def _boxes():
    bbox = torch.zeros(B, L, 4, dtype=torch.long)
    bbox[1, :, 0] = 100
    bbox[1, :, 1] = 200
    bbox[1, :, 2] = 150
    bbox[1, :, 3] = 220
    return bbox


class TestTextOnlyParity:
    def test_no_layout_inputs_matches_stock_deberta(self, models, batch):
        reference, layout = models
        expected = _last_hidden(reference, *batch)
        assert torch.equal(_last_hidden(layout, *batch), expected)

    @pytest.mark.parametrize(
        "page_token_ids",
        [None, "zeros", "nonzero"],
        ids=["no_page_ids", "page_zero", "page_nonzero"],
    )
    def test_disabled_layout_mask_matches_stock_deberta(self, models, batch, page_token_ids):
        """layout_input_mask=False must silence every layout branch, page ids included."""
        reference, layout = models
        input_ids, attention_mask = batch
        pages = {
            None: None,
            "zeros": torch.zeros(B, L, dtype=torch.long),
            "nonzero": torch.full((B, L), 3, dtype=torch.long),
        }[page_token_ids]

        kwargs = dict(bbox=_boxes(), layout_input_mask=torch.zeros(B, dtype=torch.bool))
        if pages is not None:
            kwargs["page_token_ids"] = pages

        expected = _last_hidden(reference, input_ids, attention_mask)
        assert torch.equal(_last_hidden(layout, input_ids, attention_mask, **kwargs), expected)

    def test_mixed_batch_leaves_text_rows_untouched(self, models, batch):
        """The processor emits page_token_ids per batch, so row 0 must stay clean."""
        reference, layout = models
        input_ids, attention_mask = batch
        pages = torch.zeros(B, L, dtype=torch.long)
        pages[1] = 3

        expected = _last_hidden(reference, input_ids, attention_mask)
        actual = _last_hidden(
            layout,
            input_ids,
            attention_mask,
            bbox=_boxes(),
            page_token_ids=pages,
            layout_input_mask=torch.tensor([False, True]),
        )
        assert torch.equal(actual[0], expected[0])
        assert not torch.allclose(actual[1], expected[1])


class TestLayoutStaysActive:
    def test_enabled_layout_mask_changes_output(self, models, batch):
        reference, layout = models
        input_ids, attention_mask = batch
        expected = _last_hidden(reference, input_ids, attention_mask)
        actual = _last_hidden(
            layout,
            input_ids,
            attention_mask,
            bbox=_boxes(),
            layout_input_mask=torch.ones(B, dtype=torch.bool),
        )
        assert not torch.allclose(actual, expected)

    def test_page_ids_change_output_when_layout_enabled(self, models, batch):
        reference, layout = models
        input_ids, attention_mask = batch
        common = dict(bbox=_boxes(), layout_input_mask=torch.ones(B, dtype=torch.bool))
        page_zero = _last_hidden(
            layout, input_ids, attention_mask, page_token_ids=torch.zeros(B, L, dtype=torch.long), **common
        )
        page_three = _last_hidden(
            layout, input_ids, attention_mask, page_token_ids=torch.full((B, L), 3, dtype=torch.long), **common
        )
        assert not torch.allclose(page_zero, page_three)

    def test_page_token_ids_shape_is_validated(self, models, batch):
        _, layout = models
        input_ids, attention_mask = batch
        with pytest.raises(ValueError, match="page_token_ids must have shape"):
            layout(
                input_ids=input_ids,
                attention_mask=attention_mask,
                page_token_ids=torch.zeros(B, L + 1, dtype=torch.long),
            )
