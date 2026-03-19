import torch
import torch.nn as nn


CARD_FEATURE_DIM = 52
CARD_RANKS = 13
CARD_SUITS = 4


def _build_mlp(input_dim, hidden_layers, use_layer_norm):
    layers = []
    last_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim

    return nn.Sequential(*layers), last_dim


class FlexibleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, use_layer_norm=True):
        super().__init__()
        hidden_layers = hidden_layers or [64, 64]
        self.backbone, last_dim = _build_mlp(input_dim, hidden_layers, use_layer_norm)
        self.head = nn.Linear(last_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class CardMatrixEncoder(nn.Module):
    def __init__(self, channels=None, embedding_dim=128):
        super().__init__()
        channels = channels or [16, 32]

        conv_layers = []
        in_channels = 1

        for out_channels in channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers) # out: (B, 32, 13, 4)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * CARD_RANKS * CARD_SUITS, embedding_dim),
            nn.ReLU(),
        ) # out: (B, embedding_dim)

    def forward(self, x):
        batch_size = x.size(0)
        card_bits = x[:, :CARD_FEATURE_DIM]
        card_matrix = card_bits.view(batch_size, CARD_SUITS, CARD_RANKS).transpose(1, 2).unsqueeze(1)
        encoded = self.conv(card_matrix)
        return self.projection(encoded)


class CardAwareNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers=None,
        use_layer_norm=True,
        card_encoder_channels=None,
        card_embedding_dim=128,
    ):
        super().__init__()
        if input_dim < CARD_FEATURE_DIM:
            raise ValueError(f"Expected at least {CARD_FEATURE_DIM} input features, got {input_dim}")

        hidden_layers = hidden_layers or [256, 256]
        self.card_encoder = CardMatrixEncoder(
            channels=card_encoder_channels,
            embedding_dim=card_embedding_dim,
        )

        non_card_dim = input_dim - CARD_FEATURE_DIM
        self.backbone, last_dim = _build_mlp(
            card_embedding_dim + non_card_dim,
            hidden_layers,
            use_layer_norm,
        )
        self.head = nn.Linear(last_dim, output_dim)

    def forward(self, x):
        card_features = self.card_encoder(x)
        other_features = x[:, CARD_FEATURE_DIM:]
        combined = torch.cat([card_features, other_features], dim=-1) # concatenate with other features
        features = self.backbone(combined)
        return self.head(features)


class PolicyNet(CardAwareNet):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)


class ValueNet(CardAwareNet):
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, 1, **kwargs)
