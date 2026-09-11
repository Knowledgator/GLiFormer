"""Tests for DeBERTa-style disentangled anchor refinement."""

import pytest
import torch
from torch import nn

from gliformer.layers.disentangled_attention import (
    DisentangledMultiheadAttention,
    DisentangledPositionSpec,
    DisentangledRelativePositions,
    cross_relative_position_ids,
    self_relative_position_ids,
)
from gliformer.layers.groups import AnchorCrossAttentionLayer

D = 32
H = 4
CPU = torch.device("cpu")


class TestDisentangledPositionSpec:
    @pytest.mark.parametrize(
        "config",
        [None, False, "none", "disabled", {"enabled": False}, {"type": "disabled"}],
    )
    def test_disabled_spellings(self, config):
        assert DisentangledPositionSpec.from_config(config) is None

    @pytest.mark.parametrize("config", [True, "deberta", "disentangled", {}])
    def test_enabled_spellings(self, config):
        spec = DisentangledPositionSpec.from_config(config)
        assert spec is not None
        assert spec.sharing == "per_layer"

    def test_nested_params(self):
        spec = DisentangledPositionSpec.from_config(
            {"type": "deberta", "params": {"sharing": "shared"}, "bucket_scaling": True}
        )
        assert spec.sharing == "shared"
        assert spec.bucket_scaling is True

    def test_spec_is_passed_through(self):
        spec = DisentangledPositionSpec()
        assert DisentangledPositionSpec.from_config(spec) is spec

    @pytest.mark.parametrize(
        "config",
        [
            {"max_self_relative_position": 4},
            {"sharing": "sometimes"},
            {"max_cross_relative_positions": 0},
            {"self_attention": False, "cross_attention": False},
            {"content_to_position": False, "position_to_content": False},
            {"type": "bogus"},
        ],
    )
    def test_rejects_invalid_options(self, config):
        with pytest.raises(ValueError):
            DisentangledPositionSpec.from_config(config)

    def test_rejects_non_boolean(self):
        with pytest.raises(TypeError):
            DisentangledPositionSpec.from_config({"bucket_scaling": 1})


class TestRelativePositionIds:
    def test_self_ids_are_slot_differences(self):
        positions = DisentangledRelativePositions(D, H, 8)
        ids = self_relative_position_ids(5, positions, device=CPU)
        index = torch.arange(5)
        assert ids.shape == (1, 1, 5, 5)
        assert torch.equal(ids[0, 0], index[:, None] - index[None, :] + 8)

    def test_self_table_covers_every_distance(self):
        # 2 * max + 1 vectors span [-max, max]; max >= N - 1 covers N anchors.
        positions = DisentangledRelativePositions(D, H, 7)
        ids = self_relative_position_ids(8, positions, device=CPU)
        assert positions.num_buckets == 15
        assert int(ids.min()) == 0
        assert int(ids.max()) == positions.num_buckets - 1

    def test_cross_ids_place_anchors_by_even_spacing(self):
        positions = DisentangledRelativePositions(D, H, 64)
        anchors, length = 4, 10
        token_mask = torch.ones(2, length, dtype=torch.bool)
        token_mask[1, 6:] = False
        ids = cross_relative_position_ids(
            anchors, length, length, token_mask, positions, device=CPU
        )
        assert ids.shape == (2, 1, anchors, length)
        for row, valid in enumerate((10, 6)):
            expected = torch.tensor(
                [
                    [(i * valid) // anchors - j for j in range(length)]
                    for i in range(anchors)
                ]
            )
            assert torch.equal(ids[row, 0], expected + 64)

    def test_cross_ids_skip_the_memory_sentinel(self):
        positions = DisentangledRelativePositions(D, H, 64)
        anchors, length = 3, 6
        token_mask = torch.cat(
            [torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, length, dtype=torch.bool)],
            dim=1,
        )
        ids = cross_relative_position_ids(
            anchors, length + 1, length, token_mask, positions, device=CPU
        )
        plain = cross_relative_position_ids(
            anchors, length, length, torch.ones(1, length, dtype=torch.bool),
            positions, device=CPU,
        )
        assert torch.equal(ids[..., 1:], plain)
        assert torch.equal(ids[..., 0], plain[..., 0])

    def test_cross_ids_without_a_mask(self):
        positions = DisentangledRelativePositions(D, H, 32)
        ids = cross_relative_position_ids(4, 9, 9, None, positions, device=CPU)
        assert ids.shape == (1, 1, 4, 9)

    def test_rejects_impossible_sentinel_width(self):
        positions = DisentangledRelativePositions(D, H, 32)
        with pytest.raises(ValueError):
            cross_relative_position_ids(4, 12, 9, None, positions, device=CPU)

    @pytest.mark.parametrize("bucket_scaling", [False, True])
    def test_bucketing_is_odd_symmetric(self, bucket_scaling):
        # The position-to-content term reuses `2 * max - index` for the mirrored
        # distance, which is only correct when bucketing is odd-symmetric.
        positions = DisentangledRelativePositions(
            D, H, 9, bucket_scaling=bucket_scaling
        )
        delta = torch.arange(-40, 41)
        forward = positions.bucket_indices(delta, max_distance=40)
        assert torch.equal(
            2 * positions.max_relative_positions - forward,
            positions.bucket_indices(-delta, max_distance=40),
        )
        assert int(forward.min()) >= 0
        assert int(forward.max()) < positions.num_buckets

    def test_distances_beyond_the_table_are_clipped(self):
        positions = DisentangledRelativePositions(D, H, 3)
        ids = positions.bucket_indices(torch.tensor([-99, 0, 99]))
        assert ids.tolist() == [0, 3, 6]


class TestDisentangledRelativePositions:
    def test_scale_factor_counts_active_terms(self):
        both = DisentangledRelativePositions(D, H, 4)
        c2p = DisentangledRelativePositions(D, H, 4, position_to_content=False)
        p2c = DisentangledRelativePositions(D, H, 4, content_to_position=False)
        assert (both.scale_factor, c2p.scale_factor, p2c.scale_factor) == (3, 2, 2)

    def test_scores_match_a_naive_reference(self):
        torch.manual_seed(0)
        positions = DisentangledRelativePositions(D, H, 6, bucket_scaling=True)
        batch, queries, keys, head_dim = 2, 3, 5, D // H
        query = torch.randn(batch, H, queries, head_dim)
        key = torch.randn(batch, H, keys, head_dim)
        ids = cross_relative_position_ids(
            queries, keys, keys, None, positions, device=CPU
        )
        scores = positions(query, key, ids)

        weights = positions.norm(positions.embeddings.weight)
        key_vectors = positions.key_proj(weights).view(-1, H, head_dim)
        query_vectors = positions.query_proj(weights).view(-1, H, head_dim)
        expected = torch.zeros(batch, H, queries, keys)
        for b in range(batch):
            for h in range(H):
                for i in range(queries):
                    for j in range(keys):
                        forward = int(ids[0, 0, i, j])
                        mirrored = 2 * positions.max_relative_positions - forward
                        expected[b, h, i, j] = (
                            query[b, h, i] @ key_vectors[forward, h]
                            + key[b, h, j] @ query_vectors[mirrored, h]
                        )
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_rejects_indivisible_hidden_size(self):
        with pytest.raises(ValueError):
            DisentangledRelativePositions(30, H, 4)

    def test_rejects_mismatched_ids(self):
        positions = DisentangledRelativePositions(D, H, 4)
        query = torch.randn(1, H, 3, D // H)
        key = torch.randn(1, H, 5, D // H)
        with pytest.raises(ValueError):
            positions(query, key, torch.zeros(1, 1, 3, 4, dtype=torch.long))


class TestDisentangledMultiheadAttention:
    def _reference(self, attention):
        reference = nn.MultiheadAttention(
            D, H, dropout=0.0, batch_first=True
        )
        with torch.no_grad():
            reference.in_proj_weight.copy_(
                torch.cat(
                    [
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ]
                )
            )
            reference.in_proj_bias.copy_(
                torch.cat(
                    [
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ]
                )
            )
            reference.out_proj.weight.copy_(attention.out_proj.weight)
            reference.out_proj.bias.copy_(attention.out_proj.bias)
        return reference

    @pytest.mark.parametrize("mask_shape", ["none", "flat", "batched"])
    def test_matches_multihead_attention_without_positions(self, mask_shape):
        torch.manual_seed(0)
        attention = DisentangledMultiheadAttention(D, H, dropout=0.0)
        reference = self._reference(attention)
        batch, queries, keys = 2, 3, 7
        query = torch.randn(batch, queries, D)
        key = torch.randn(batch, keys, D)
        value = torch.randn(batch, keys, D)
        if mask_shape == "none":
            attn_mask = None
        elif mask_shape == "flat":
            attn_mask = torch.randn(queries, keys)
        else:
            attn_mask = torch.randn(batch * H, queries, keys)
        mine = attention(query, key, value, attn_mask=attn_mask)[0]
        theirs = reference(query, key, value, attn_mask=attn_mask)[0]
        assert torch.allclose(mine, theirs, atol=1e-5)

    def test_key_padding_mask_is_honoured(self):
        torch.manual_seed(0)
        attention = DisentangledMultiheadAttention(D, H, dropout=0.0)
        query = torch.randn(1, 2, D)
        key = torch.randn(1, 5, D)
        padding = torch.zeros(1, 5, dtype=torch.bool)
        padding[0, 3:] = True
        masked = attention(query, key, key, key_padding_mask=padding)[0]
        key[:, 3:] = torch.randn(1, 2, D)
        again = attention(query, key, key, key_padding_mask=padding)[0]
        assert torch.allclose(masked, again, atol=1e-6)

    def test_positions_change_the_output(self):
        torch.manual_seed(0)
        attention = DisentangledMultiheadAttention(D, H, dropout=0.0)
        positions = DisentangledRelativePositions(D, H, 8)
        query = torch.randn(1, 4, D)
        key = torch.randn(1, 6, D)
        ids = cross_relative_position_ids(4, 6, 6, None, positions, device=CPU)
        plain = attention(query, key, key)[0]
        biased = attention(
            query,
            key,
            key,
            relative_positions=positions,
            relative_position_ids=ids,
        )[0]
        assert not torch.allclose(plain, biased, atol=1e-4)

    def test_rejects_indivisible_hidden_size(self):
        with pytest.raises(ValueError):
            DisentangledMultiheadAttention(30, H)


class TestAnchorCrossAttentionLayerDisentangled:
    def test_disabled_by_default(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=H, num_layers=2)
        assert layer.disentangled_position is None
        assert layer.self_relative_positions is None
        assert isinstance(layer.layers[0].self_attn, nn.MultiheadAttention)
        assert not any("relative" in key for key in layer.state_dict())

    def test_per_layer_tables_are_independent(self):
        layer = AnchorCrossAttentionLayer(
            D, num_heads=H, num_layers=2, disentangled_position=True
        )
        assert layer.self_relative_positions is None
        assert layer.cross_relative_positions is None
        first, second = layer.layers
        assert first.self_relative_positions is not second.self_relative_positions
        assert first.cross_relative_positions is not second.cross_relative_positions
        # Self and cross use separate tables sized for their own distance scale.
        assert first.self_relative_positions.num_buckets == 2 * 64 + 1
        assert first.cross_relative_positions.num_buckets == 2 * 128 + 1

    def test_shared_tables_are_reused_by_every_layer(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=H,
            num_layers=3,
            disentangled_position={"sharing": "shared"},
        )
        assert layer.self_relative_positions is not None
        assert all(block.self_relative_positions is None for block in layer.layers)
        keys = [key for key in layer.state_dict() if "relative" in key]
        assert keys and all(not key.startswith("layers.") for key in keys)

    @pytest.mark.parametrize("norm_style", ["pre_norm", "post_norm"])
    @pytest.mark.parametrize("sharing", ["per_layer", "shared"])
    def test_forward_with_masks(self, norm_style, sharing):
        torch.manual_seed(0)
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=H,
            num_layers=2,
            norm_style=norm_style,
            disentangled_position={
                "sharing": sharing,
                "max_self_relative_positions": 8,
                "max_cross_relative_positions": 16,
            },
        )
        anchors = torch.randn(3, 5, D, requires_grad=True)
        tokens = torch.randn(3, 11, D, requires_grad=True)
        token_mask = torch.ones(3, 11, dtype=torch.bool)
        token_mask[1, 7:] = False
        token_mask[2, :] = False  # exercises the all-masked sentinel path
        query_mask = torch.ones(3, 5, dtype=torch.bool)
        query_mask[1, 4:] = False
        out = layer(anchors, tokens, token_mask=token_mask, query_mask=query_mask)
        assert out.shape == (3, 5, D)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert anchors.grad.abs().sum() > 0
        assert tokens.grad.abs().sum() > 0

    def test_only_cross_attention_is_disentangled(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=H,
            num_layers=1,
            disentangled_position={"self_attention": False},
        )
        block = layer.layers[0]
        assert isinstance(block.self_attn, nn.MultiheadAttention)
        assert isinstance(block.cross_attn, DisentangledMultiheadAttention)
        assert block.self_relative_positions is None
        assert block.cross_relative_positions is not None
        assert layer(torch.randn(1, 3, D), torch.randn(1, 6, D)).shape == (1, 3, D)

    def test_anchor_order_and_text_order_matter(self):
        torch.manual_seed(0)
        layer = AnchorCrossAttentionLayer(
            D, num_heads=H, num_layers=1, disentangled_position=True
        ).eval()
        anchors = torch.randn(1, 6, D)
        tokens = torch.randn(1, 12, D)
        permutation = torch.tensor([3, 1, 0, 5, 4, 2])
        with torch.no_grad():
            baseline = layer(anchors, tokens)
            permuted = layer(anchors[:, permutation], tokens)[
                :, permutation.argsort()
            ]
            flipped = layer(anchors, tokens.flip(1))
        assert not torch.allclose(baseline, permuted, atol=1e-4)
        assert not torch.allclose(baseline, flipped, atol=1e-4)

    def test_from_config_accepts_the_disentangled_alias(self):
        layer = AnchorCrossAttentionLayer.from_config(
            {
                "type": "cross_attention",
                "params": {
                    "num_heads": H,
                    "num_layers": 1,
                    "disentangled": {"sharing": "shared", "bucket_scaling": True},
                },
            },
            D,
            dropout=0.1,
        )
        assert layer.disentangled_position.sharing == "shared"
        assert layer.disentangled_position.bucket_scaling is True

    def test_from_config_rejects_invalid_options(self):
        with pytest.raises(ValueError):
            AnchorCrossAttentionLayer.from_config(
                {"params": {"disentangled_position": {"sharing": "sometimes"}}},
                D,
            )
