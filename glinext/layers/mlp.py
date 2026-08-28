"""MLP and projection utilities."""

import torch
from torch import nn
from transformers.activations import ACT2FN


def create_mlp(
    input_dim,
    intermediate_dims,
    output_dim,
    dropout=0.1,
    activation="gelu",
    add_layer_norm=False,
    include_dropout=False,
):
    """
    Creates a multi-layer perceptron (MLP) with specified dimensions and activation functions.
    """
    activation_mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU
    }
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        if add_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation_mapping[activation]())
        if dropout > 0 or include_dropout:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class FeaturesProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config.encoder_config.hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.dropout = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.hidden_size, config.encoder_config.hidden_size, bias=True)

    def forward(self, features):
        hidden_states = self.linear_1(features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
