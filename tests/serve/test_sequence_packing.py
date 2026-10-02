from types import SimpleNamespace

import torch
from torch import nn

from gliformer import InferencePackingConfig
from gliformer.encoders.omni import LayoutEncoder


class RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(config=SimpleNamespace(pad_token_id=0))
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, pair_attention_mask=None, **kwargs):
        self.calls.append({
            "shape": tuple(input_ids.shape),
            "pair_attention_mask": pair_attention_mask,
        })
        return input_ids.to(torch.float32).unsqueeze(-1)


def make_layout_encoder():
    encoder = object.__new__(LayoutEncoder)
    nn.Module.__init__(encoder)
    encoder.bert_layer = RecordingBackbone()
    return encoder.eval()


def test_text_only_layout_encoder_packs_and_restores_original_batch_shape():
    encoder = make_layout_encoder()
    input_ids = torch.tensor([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 0, 0, 0, 0],
        [6, 7, 8, 9, 0, 0],
    ])
    attention_mask = input_ids.ne(0).long()

    output = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_lengths=[3, 2, 4],
        packing_config=InferencePackingConfig(max_length=16),
    )

    assert encoder.bert_layer.calls[0]["shape"] == (1, 9)
    assert encoder.bert_layer.calls[0]["pair_attention_mask"].shape == (1, 9, 9)
    assert output.shape == (3, 6, 1)
    assert output.squeeze(-1).long().tolist() == input_ids.tolist()


def test_layout_inputs_fall_back_without_forwarding_packing_control():
    encoder = make_layout_encoder()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    bbox = torch.zeros(1, 3, 4, dtype=torch.long)

    output = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bbox=bbox,
        packing_config=InferencePackingConfig(max_length=16),
        token_lengths=[3],
    )

    assert encoder.bert_layer.calls[0]["shape"] == (1, 3)
    assert encoder.bert_layer.calls[0]["pair_attention_mask"] is None
    assert output.shape == (1, 3, 1)
