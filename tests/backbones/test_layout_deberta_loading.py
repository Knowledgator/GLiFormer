"""`from_pretrained` must keep the DeBERTa checkpoint and rebuild its buffers.

`LayoutDebertaModel` adds parameters that no stock DeBERTa checkpoint contains,
so loading one always runs `_init_weights` over the model. That initializer
writes through `.data`, which Transformers' guarded initializers cannot see, and
`position_ids` is non-persistent, so it is in no checkpoint at all and comes
back from a meta-device load as uninitialized memory.
"""

import pytest
import torch

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Model,
)

from gliformer.backbones.deberta_2d import LayoutDebertaConfig, LayoutDebertaModel


def _base_kwargs(**overrides):
    kwargs = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=16,
        type_vocab_size=0,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def loaded(tmp_path):
    """A layout model loaded from a stamped stock DeBERTa checkpoint."""

    def _load(**overrides):
        kwargs = _base_kwargs(**overrides)
        torch.manual_seed(0)
        reference = DebertaV2Model(DebertaV2Config(**kwargs))
        with torch.no_grad():
            # Stamp recognizable values so any re-initialization is visible.
            for index, parameter in enumerate(reference.parameters()):
                parameter.fill_(0.5 + index * 0.001)
        saved = {name: value.clone() for name, value in reference.state_dict().items()}
        reference.save_pretrained(tmp_path)
        layout_config = LayoutDebertaConfig(
            **kwargs,
            layout_embedding_type="absolute",
            max_2d_position_embeddings=1024,
            max_page_embeddings=0,
        )
        model = LayoutDebertaModel.from_pretrained(tmp_path, config=layout_config)
        return model, saved

    return _load


class TestPretrainedWeightsSurvive:
    def test_checkpoint_tensors_are_not_reinitialized(self, loaded):
        model, saved = loaded()
        state = model.state_dict()

        clobbered = [
            name
            for name, reference in saved.items()
            if not torch.equal(state[name].cpu(), reference)
        ]

        assert saved, "the stock checkpoint must contribute tensors"
        assert clobbered == []

    def test_layout_parameters_are_initialized(self, loaded):
        model, saved = loaded()
        state = model.state_dict()

        new_names = [name for name in state if name not in saved]
        spatial = [name for name in new_names if "spatial_embeddings" in name]

        assert spatial, "the layout branch must add parameters"
        for name in new_names:
            assert torch.isfinite(state[name]).all(), name


class TestPositionIdsBuffer:
    @pytest.mark.parametrize("position_biased_input", [False, True])
    def test_buffer_is_rebuilt_after_loading(self, loaded, position_biased_input):
        model, _ = loaded(position_biased_input=position_biased_input)

        expected = torch.arange(model.config.max_position_embeddings).expand((1, -1))
        assert torch.equal(model.embeddings.position_ids.cpu(), expected)

    def test_absolute_positions_stay_usable(self, loaded):
        """Garbage position ids index out of the position embedding table."""
        model, _ = loaded(position_biased_input=True)
        model.eval()
        input_ids = torch.randint(0, 128, (2, 16))

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones(2, 16, dtype=torch.long),
            )

        assert torch.isfinite(output.last_hidden_state).all()
