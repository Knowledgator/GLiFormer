from types import SimpleNamespace

import pytest
import torch
from torch import nn

import gliformer.encoders.media as media_module
from gliformer.encoders.media import MediaBackboneEncoder
from gliformer.encoders.vision import (
    VisionEncoder,
    VisionPathEmbeddings,
    vision_token_mask,
)
from gliformer.layers.position import LearnedGrid2DPositionEmbedding


def _config(**overrides):
    values = {
        "hidden_size": 8,
        "image_size": (16, 16),
        "vision_patch_size": (8, 8),
        "vision_in_channels": 3,
        "vision_encoder_type": "patch",
        "vision_model_name": None,
        "vision_position_embedding_type": "learned_grid2d",
        "vision_position_embedding_kwargs": {"init_std": 0.01},
        "vision_feature_stride": None,
        "vision_feature_spatial_shape": None,
        "vision_feature_prefix_tokens": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_local_vision_encoder_uses_position_hierarchy_on_rectangular_images():
    encoder = VisionEncoder(_config())

    output = encoder.forward_features(torch.randn(2, 3, 16, 24))

    assert isinstance(encoder.model, VisionPathEmbeddings)
    assert isinstance(
        encoder.model.position_embedding,
        LearnedGrid2DPositionEmbedding,
    )
    assert output.token_embeddings.shape == (2, 6, 8)
    assert output.spatial_shape.tolist() == [[2, 3], [2, 3]]
    assert output.prefix_tokens.tolist() == [0, 0]


class _FlatBackbone(nn.Module):
    def __init__(self, token_count, hidden_size=8):
        super().__init__()
        self.token_count = token_count
        self.config = SimpleNamespace(hidden_size=hidden_size, patch_size=16)
        self.kwargs = None

    def forward(self, pixel_values, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            last_hidden_state=torch.zeros(
                pixel_values.shape[0],
                self.token_count,
                self.config.hidden_size,
                device=pixel_values.device,
            )
        )


def test_external_vision_backbone_uses_shared_construction_and_projection(
    monkeypatch,
):
    external_config = SimpleNamespace(hidden_size=12)
    backbone = _FlatBackbone(token_count=5, hidden_size=12)

    class FakeAutoModel:
        @staticmethod
        def from_config(model_config, **kwargs):
            assert model_config is external_config
            assert kwargs == {"trust_remote_code": True}
            return backbone

    monkeypatch.setattr(media_module, "AutoModel", FakeAutoModel)
    encoder = VisionEncoder(
        _config(
            vision_encoder_type="auto",
            vision_encoder_config=external_config,
        )
    )

    token_embeddings = encoder(torch.randn(2, 3, 32, 32))

    assert isinstance(encoder, MediaBackboneEncoder)
    assert encoder.model is backbone
    assert token_embeddings.shape == (2, 5, 8)
    assert encoder.projection.in_features == 12
    assert encoder.projection.out_features == 8


def test_configured_final_stride_defines_flat_backbone_spatial_contract():
    encoder = VisionEncoder(
        _config(
            vision_feature_stride=(32, 32),
            vision_feature_prefix_tokens=1,
        )
    )
    encoder.model = _FlatBackbone(token_count=7)

    output = encoder.forward_features(
        torch.randn(2, 3, 64, 96),
        interpolate_pos_encoding=True,
    )

    assert output.spatial_shape.tolist() == [[2, 3], [2, 3]]
    assert output.prefix_tokens.tolist() == [1, 1]
    assert encoder.model.kwargs["interpolate_pos_encoding"] is True


def test_configured_spatial_contract_rejects_backbone_token_mismatch():
    encoder = VisionEncoder(
        _config(
            vision_feature_spatial_shape=(2, 3),
            vision_feature_prefix_tokens=1,
        )
    )
    encoder.model = _FlatBackbone(token_count=6)

    with pytest.raises(ValueError, match="describes 7 tokens"):
        encoder.forward_features(torch.randn(1, 3, 64, 96))


def test_dense_only_vision_mask_is_prefixed_for_cls_tokens():
    tokens = torch.randn(2, 7, 8)
    dense_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]]
    )

    mask = vision_token_mask(tokens, dense_mask, torch.ones(2, dtype=torch.long))

    assert mask.shape == (2, 7)
    assert mask[:, 0].tolist() == [1, 1]
    assert torch.equal(mask[:, 1:], dense_mask)


def test_incompatible_vision_mask_fails_closed():
    with pytest.raises(ValueError, match="cannot be aligned"):
        vision_token_mask(
            torch.randn(1, 7, 8),
            torch.ones(1, 4),
            torch.ones(1, dtype=torch.long),
        )


def test_spatial_vision_mask_is_resized_to_dense_grid_and_prefixed():
    tokens = torch.randn(1, 7, 8)
    pixel_mask = torch.zeros(1, 4, 6)
    pixel_mask[:, :2, :2] = 1

    mask = vision_token_mask(
        tokens,
        pixel_mask,
        prefix_tokens=torch.ones(1, dtype=torch.long),
        spatial_shape=torch.tensor([[2, 3]]),
    )

    assert mask.tolist() == [[1, 1, 0, 0, 0, 0, 0]]
