# opponent_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class OpponentModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # logits for opponent action prediction
        )

    def forward(self, x):
        return self.net(x)

class OpponentPredictor:
    """Train a simple supervised model to predict opponent's next action."""
    def __init__(self, obs_dim, act_dim, device=None, lr=1e-3):
        self.model = OpponentModel(obs_dim, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def train_step(self, state, action):
        """
        state: numpy array of shape (n, obs_dim)
        action: numpy array of shape (n,) with integer action indices
        """
        
        self.model.train()

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if len(state.shape) == 1: # add batch dimension if missing
            state = state.unsqueeze(0)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        if len(action.shape) == 0:
            action = action.unsqueeze(0)
        
        logits = self.model(state)
        loss = self.loss_fn(logits, action)

        probs = torch.softmax(logits, dim=-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pred = torch.argmax(probs, dim=1)
        correct = (pred == action).float().mean().item()

        return loss.item(), correct

    def predict(self, state):
        """Return one-hot vector of predicted opponent action probabilities."""
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            is_single = (state.dim() == 1)

            if is_single: # add batch dimension if missing
                state = state.unsqueeze(0) # (obs_dim,) -> (1, obs_dim)

            logits = self.model(state)
            probs = torch.softmax(logits, dim=-1) # (B, act_dim)

            if is_single:
                return probs.squeeze(0).cpu().numpy()
            return probs.cpu().numpy()