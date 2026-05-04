import torch
from torch import nn

class SpatialEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        hidden_size = getattr(config, "hidden_size")
        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-7)
        dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.max_2d_position_embeddings = getattr(config, "max_2d_position_embeddings", 1024)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.x_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, config.shape_size)

        self.proj = nn.Linear(config.coordinate_size * 4 + config.shape_size * 2, hidden_size)

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        max_position = self.max_2d_position_embeddings - 1
        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, max_position))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, max_position))

        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def forward(self, bbox):

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        embeddings = self.proj(spatial_position_embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
