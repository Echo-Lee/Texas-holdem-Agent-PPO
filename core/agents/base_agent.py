import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

"""
This file defines a abstract RL agent class, which can be extended for specific algorithms (e.g., PPO, DQN) and architectures. 
It also includes the get_action and update_policy_value methods.

get_action: takes in an observation and action mask, returns the selected action and its log probability. 
update: to be implemented by subclasses, will handle the policy and value network updates based on collected trajectories and rewards.
"""

class BaseRLAgent:
    def __init__(self, policy_net, value_net, lr=1e-3, gamma=0.99, device="cpu"):
        self.device = device
        self.gamma = gamma

        self.policy = policy_net.to(self.device)
        self.value = value_net.to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr * 0.2)

    def get_action(self, obs, mask, deterministic=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: numpy array of shape (n, obs_dim)
            mask: numpy array of shape (n, action_dim) with 1 for valid actions and 0 for invalid actions
            deterministic: if True, select the action with highest probability; otherwise, sample from the distribution
        Returns:
            action: torch tensor (n,), the selected action indices  
            log_probs: torch tensor (n,), the log probability of the selected action
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if obs.dim() == 1: # add batch dimension if missing
            obs = obs.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        # Get action probabilities from the policy network
        logits = self.policy(obs)
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample() # (n,)
        
        return action.detach(), dist.log_prob(action).detach()

    def save(self, checkpoint_path):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def load(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        print(f"Model loaded from {checkpoint_path}")

    def update(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")