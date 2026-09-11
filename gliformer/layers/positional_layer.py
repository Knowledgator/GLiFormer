import torch
from torch import nn

class SpatialEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        hidden_size = getattr(config, "hidden_size")
        dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.max_2d_position_embeddings = getattr(config, "max_2d_position_embeddings", 1024)

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

        # Deliberately no LayerNorm on this branch. The caller adds the result
        # to the word embeddings and normalizes the sum with its own shared
        # LayerNorm, which is what LayoutLMv2/v3 do. Normalizing here instead
        # rescales the branch to unit per-dim variance: at initialization that
        # is ~8x the norm of the word embeddings, it leaves the encoder input
        # nearly orthogonal to the text-only input (cosine 0.08), and the model
        # then predicts nothing whenever boxes are supplied. The LayerNorm also
        # normalizes away the gradient along its own scale direction, so the
        # gain never shrinks: after 10k steps it had moved 1.0 -> 0.999951,
        # against the ~0.01 it would need to stop dominating.
        embeddings = self.proj(spatial_position_embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings
