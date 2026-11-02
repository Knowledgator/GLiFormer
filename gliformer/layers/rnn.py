"""RNN encoder layers."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(RnnSeq2SeqEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through RNN
        packed_output, hidden = self.rnn(packed_x, hidden)

        # Unpacking the output sequence. Keep the input width: without
        # total_length the output ends at the longest row's mask sum, which is
        # shorter than the mask when invalid positions sit inside it (layout
        # image tokens appended after padded words).
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])

        return output
