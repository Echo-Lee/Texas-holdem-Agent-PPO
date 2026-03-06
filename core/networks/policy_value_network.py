import torch.nn as nn

class FlexibleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], use_layer_norm=True):
        super().__init__()
        layers = []
        last_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class PolicyNet(FlexibleNet):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

class ValueNet(FlexibleNet):
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, 1, **kwargs)