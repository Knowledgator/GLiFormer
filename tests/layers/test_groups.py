"""Tests for group/anchor generation layers."""

from unittest.mock import Mock

import torch

from glinext.layers.groups import (
    AnchorCrossAttentionLayer,
    QueryGroupRNN,
    QueryGroupTransformer,
    RotaryGroupRNN,
)

D = 16


class TestAnchorCrossAttentionLayer:
    def test_output_shape(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=2)
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 10, D)
        out = layer(anchors, tokens)
        assert out.shape == (2, 3, D)

    def test_single_layer(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        out = layer(anchors, tokens)
        assert out.shape == (1, 2, D)

    def test_with_mask(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        mask = torch.tensor([[True, True, True, False, False]])
        out = layer(anchors, tokens, token_mask=mask)
        assert out.shape == (1, 2, D)

    def test_gradient_flows(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D, requires_grad=True)
        tokens = torch.randn(1, 5, D, requires_grad=True)
        out = layer(anchors, tokens)
        out.sum().backward()
        assert anchors.grad is not None
        assert tokens.grad is not None

    def test_memory_positions_are_keys_only_by_default(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        positions = torch.randn(1, 5, D)
        cross_attn = layer.layers[0].cross_attn
        cross_attn.forward = Mock(wraps=cross_attn.forward)

        layer(anchors, tokens, memory_pos_emb=positions)

        _, memory_key, memory_value = cross_attn.forward.call_args.args
        torch.testing.assert_close(memory_key, tokens + positions)
        torch.testing.assert_close(memory_value, tokens)

    def test_memory_positions_can_be_added_to_values(self):
        layer = AnchorCrossAttentionLayer(D, num_heads=4, num_layers=1)
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        positions = torch.randn(1, 5, D)
        cross_attn = layer.layers[0].cross_attn
        cross_attn.forward = Mock(wraps=cross_attn.forward)

        layer(
            anchors,
            tokens,
            memory_pos_emb=positions,
            memory_position_in_values=True,
        )

        _, memory_key, memory_value = cross_attn.forward.call_args.args
        torch.testing.assert_close(memory_key, tokens + positions)
        torch.testing.assert_close(memory_value, tokens + positions)

    def test_default_decoder_is_post_norm(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        norm_inputs = []
        layer.layers[0].norm0.register_forward_pre_hook(
            lambda _module, inputs: norm_inputs.append(inputs[0].detach().clone())
        )

        layer(anchors, tokens)

        assert not layer.layers[0].norm_first
        assert len(norm_inputs) == 1
        assert not torch.equal(norm_inputs[0], anchors)

    def test_default_post_norm_matches_legacy_equations(self):
        torch.manual_seed(5)
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        block = layer.layers[0]
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 5, D)
        query_positions = torch.randn(1, 3, D)
        memory_positions = torch.randn(1, 5, D)

        query = anchors + query_positions
        self_attended = block.self_attn(query, query, anchors)[0]
        expected = block.norm0(anchors + self_attended)
        cross_query = expected + query_positions
        memory_key = tokens + memory_positions
        cross_attended = block.cross_attn(cross_query, memory_key, tokens)[0]
        expected = block.norm1(expected + cross_attended)
        expected = block.norm2(expected + block.ffn(expected))

        actual = layer(
            anchors,
            tokens,
            query_pos_emb=query_positions,
            memory_pos_emb=memory_positions,
        )

        torch.testing.assert_close(actual, expected)
        assert not any("_scale" in name for name, _ in layer.named_parameters())

    def test_pre_norm_with_zero_layer_scale_is_identity(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            norm_first=True,
            layer_scale_init=0.0,
        ).eval()
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 5, D)

        out = layer(anchors, tokens)

        torch.testing.assert_close(out, anchors)
        for block in layer.layers:
            assert block.norm_first
            for scale in (
                block.self_attn_scale,
                block.cross_attn_scale,
                block.ffn_scale,
            ):
                assert scale.shape == (D,)
                assert scale.requires_grad
                assert not scale.detach().any()

    def test_pre_norm_normalizes_before_self_attention(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            norm_first=True,
        ).eval()
        anchors = torch.randn(1, 2, D)
        tokens = torch.randn(1, 5, D)
        norm_inputs = []
        layer.layers[0].norm0.register_forward_pre_hook(
            lambda _module, inputs: norm_inputs.append(inputs[0].detach().clone())
        )

        layer(anchors, tokens)

        assert len(norm_inputs) == 1
        torch.testing.assert_close(norm_inputs[0], anchors)

    def test_cross_attention_bias_expands_heads_and_absorbs_padding(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 5, D)
        token_mask = torch.tensor(
            [
                [True, True, False, True, False],
                [True, False, True, True, False],
            ]
        )
        bias = torch.arange(2 * 3 * 5, dtype=torch.float64).reshape(2, 3, 5)
        cross_attn = layer.layers[0].cross_attn
        cross_attn.forward = Mock(wraps=cross_attn.forward)

        layer(
            anchors,
            tokens,
            token_mask=token_mask,
            cross_attention_bias=bias,
        )

        call = cross_attn.forward.call_args
        attention_mask = call.kwargs["attn_mask"]
        assert call.kwargs["key_padding_mask"] is None
        assert attention_mask.shape == (2 * 4, 3, 5)
        assert attention_mask.dtype == anchors.dtype
        expected = bias.float()[:, None].expand(2, 4, 3, 5).clone()
        expected.masked_fill_(~token_mask[:, None, None], float("-inf"))
        torch.testing.assert_close(attention_mask, expected.reshape(8, 3, 5))

    def test_head_specific_cross_attention_bias_keeps_head_order(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 5, D)
        bias = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(
            2,
            4,
            3,
            5,
        )
        cross_attn = layer.layers[0].cross_attn
        cross_attn.forward = Mock(wraps=cross_attn.forward)

        layer(anchors, tokens, cross_attention_bias=bias)

        attention_mask = cross_attn.forward.call_args.kwargs["attn_mask"]
        torch.testing.assert_close(attention_mask, bias.reshape(8, 3, 5))

    def test_attention_bias_is_padded_for_safe_memory_sentinel(self):
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 5, D)
        token_mask = torch.tensor(
            [
                [True, True, False, False, False],
                [False, False, False, False, False],
            ]
        )
        bias = torch.zeros(2, 3, 5)
        cross_attn = layer.layers[0].cross_attn
        cross_attn.forward = Mock(wraps=cross_attn.forward)

        out = layer(
            anchors,
            tokens,
            token_mask=token_mask,
            cross_attention_bias=bias,
        )

        attention_mask = cross_attn.forward.call_args.kwargs["attn_mask"]
        attention_mask = attention_mask.reshape(2, 4, 3, 6)
        assert torch.isfinite(out).all()
        assert torch.isneginf(attention_mask[0, :, :, 0]).all()
        assert not attention_mask[1, :, :, 0].any()
        assert torch.isneginf(attention_mask[1, :, :, 1:]).all()

    def test_prepared_memory_and_individual_layers_match_forward(self):
        torch.manual_seed(9)
        layer = AnchorCrossAttentionLayer(
            D,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            norm_first=True,
            layer_scale_init=0.2,
        ).eval()
        anchors = torch.randn(2, 3, D)
        tokens = torch.randn(2, 5, D)
        token_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ]
        )
        query_positions = torch.randn(1, 3, D)
        memory_positions = torch.randn(1, 5, D)
        bias = torch.randn(2, 3, 5)

        expected = layer(
            anchors,
            tokens,
            token_mask=token_mask,
            query_pos_emb=query_positions,
            memory_pos_emb=memory_positions,
            memory_position_in_values=True,
            cross_attention_bias=bias,
        )
        memory = layer.prepare_memory(tokens, token_mask, memory_positions)
        actual = anchors
        assert layer.num_layers == 2
        for layer_index in range(layer.num_layers):
            actual = layer.forward_layer(
                layer_index,
                actual,
                memory,
                query_pos_emb=query_positions,
                memory_position_in_values=True,
                cross_attention_bias=bias,
            )

        torch.testing.assert_close(actual, expected)


class TestRotaryGroupRNN:
    def test_output_shape(self):
        layer = RotaryGroupRNN(D, max_count=10)
        pc_emb = torch.randn(3, D)  # 3 fields
        out = layer(pc_emb, count_val=4)
        assert out.shape == (4, 3, D)  # (count, M, D)

    def test_single_count(self):
        layer = RotaryGroupRNN(D, max_count=10)
        pc_emb = torch.randn(2, D)
        out = layer(pc_emb, count_val=1)
        assert out.shape == (1, 2, D)

    def test_exceeds_max_count(self):
        layer = RotaryGroupRNN(D, max_count=5)
        pc_emb = torch.randn(2, D)
        out = layer(pc_emb, count_val=8)
        # Should handle counts exceeding max by repeating
        assert out.shape == (8, 2, D)

    def test_single_field(self):
        layer = RotaryGroupRNN(D, max_count=5)
        pc_emb = torch.randn(1, D)
        out = layer(pc_emb, count_val=3)
        assert out.shape == (3, 1, D)


class TestQueryGroupRNN:
    def test_with_count(self):
        layer = QueryGroupRNN(D, max_count=10)
        context = torch.randn(2, D)  # batched (B, D)
        token_emb = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        out, mask = layer(context, token_emb, count_val=count)
        assert out.shape[0] == 2  # batch
        assert out.shape[2] == D
        assert mask.shape[0] == 2

    def test_without_count(self):
        layer = QueryGroupRNN(D, max_count=10)
        context = torch.randn(2, D)
        token_emb = torch.randn(2, 8, D)
        out, mask = layer(context, token_emb, count_val=None, threshold=0.5)
        assert out.shape == (2, 8, D)  # all tokens
        assert mask.shape == (2, 8)

    def test_output_types(self):
        layer = QueryGroupRNN(D, max_count=10)
        context = torch.randn(1, D)
        token_emb = torch.randn(1, 5, D)
        count = torch.tensor([3])
        out, mask = layer(context, token_emb, count_val=count)
        assert out.dtype == torch.float32
        assert mask.dtype == torch.bool


class TestQueryGroupTransformer:
    def test_with_count(self):
        layer = QueryGroupTransformer(D, num_heads=4, num_layers=1)
        context = torch.randn(2, D)
        token_emb = torch.randn(2, 10, D)
        count = torch.tensor([2, 3])
        out, mask = layer(context, token_emb, count_val=count)
        assert out.shape[0] == 2
        assert out.shape[2] == D
        assert mask.shape[0] == 2

    def test_without_count(self):
        layer = QueryGroupTransformer(D, num_heads=4, num_layers=1)
        context = torch.randn(2, D)
        token_emb = torch.randn(2, 8, D)
        out, mask = layer(context, token_emb, count_val=None, threshold=0.5)
        assert out.shape == (2, 8, D)
        assert mask.shape == (2, 8)

    def test_gradient_flows(self):
        layer = QueryGroupTransformer(D, num_heads=4, num_layers=1)
        context = torch.randn(1, D, requires_grad=True)
        token_emb = torch.randn(1, 5, D, requires_grad=True)
        count = torch.tensor([3])
        out, mask = layer(context, token_emb, count_val=count)
        out.sum().backward()
        assert context.grad is not None
        assert token_emb.grad is not None
