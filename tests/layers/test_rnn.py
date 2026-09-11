"""Tests for RNN encoder layers."""

import torch

from gliformer.layers.rnn import RnnSeq2SeqEncoder


class TestRnnSeq2SeqEncoder:
    def test_output_shape(self):
        enc = RnnSeq2SeqEncoder(input_size=16, hidden_size=32)
        x = torch.randn(2, 10, 16)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = enc(x, mask)
        assert out.shape == (2, 10, 32)

    def test_bidirectional(self):
        enc = RnnSeq2SeqEncoder(input_size=16, hidden_size=32, bidirectional=True)
        x = torch.randn(2, 10, 16)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = enc(x, mask)
        assert out.shape == (2, 10, 64)  # 2 * hidden_size

    def test_variable_lengths(self):
        enc = RnnSeq2SeqEncoder(input_size=16, hidden_size=32)
        x = torch.randn(2, 8, 16)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ], dtype=torch.bool)
        out = enc(x, mask)
        # Output should still be padded to max length
        assert out.shape == (2, 8, 32)

    def test_multi_layer(self):
        enc = RnnSeq2SeqEncoder(input_size=16, hidden_size=32, num_layers=3, dropout=0.1)
        x = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out = enc(x, mask)
        assert out.shape == (2, 5, 32)

    def test_single_token(self):
        enc = RnnSeq2SeqEncoder(input_size=16, hidden_size=32)
        x = torch.randn(1, 1, 16)
        mask = torch.ones(1, 1, dtype=torch.bool)
        out = enc(x, mask)
        assert out.shape == (1, 1, 32)
