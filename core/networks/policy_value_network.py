import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Actor networks
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # logits for action probabilities
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    """
    Critic networks
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # state value estimation
        )

    def forward(self, x):
        return self.net(x)