import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from .base_agent import BaseRLAgent

class PPOAgent(BaseRLAgent):
    def __init__(self, policy_net, value_net, lr=1e-3, gamma=0.99, device="cpu",
                 clip_param=0.2, policy_epochs=5, value_epochs=5, mini_batch_size=64, 
                 gae_lambda=0.95, entropy_coef=0.01): 
        super().__init__(policy_net, value_net, lr, gamma, device)

        self.clip_epsilon = clip_param
        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

    def update(self, trajectories):
        """
        Update policy and value networks using PPO loss.
        trajectories: dict of tensors for states, actions, log_probs, rewards, dones, masks
        """
        states = trajectories['states'] # (B * T, obs_dim)
        actions = trajectories['actions'] # (B * T,)
        old_log_probs = trajectories['log_probs'] # (B * T,)
        rewards = trajectories['rewards'] # (B * T,)
        dones = trajectories['dones'] # (B * T,)
        masks = trajectories['masks']   # (B * T, action_dim)


        batch_size = states.size(0)
        indices = np.arange(batch_size)

        # total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0

        # Value Training
        with torch.no_grad():
            old_values = self.value(states).squeeze(-1) # (B * T,)
            
        returns = torch.zeros_like(rewards).to(self.device)
        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = old_values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            returns[t] = gae + old_values[t]

        # Value network update
        for _ in range(self.value_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_returns = returns[mb_idx]

                new_values = self.value(mb_states).squeeze(-1)
                value_loss = F.mse_loss(new_values, mb_returns)

                # total_value_loss += value_loss.item()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
                self.value_optimizer.step()
        
        # Compute updated advantages using GAE and new value estimates
        with torch.no_grad():
            updated_values = self.value(states).squeeze(-1)
            
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lambda = 0

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = updated_values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - updated_values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy network update
        for _ in range(self.policy_epochs):
            np.random.shuffle(indices)

            # Mini-batch update
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                # get mini-batch data
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_masks = masks[mb_idx]

                # Compute new log probabilities and entropy
                logits = self.policy(mb_states)
                masked_logits = logits.masked_fill(~mb_masks, -1e9)
                probs = torch.softmax(masked_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO surrogate loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.policy_optimizer.zero_grad()
                
                policy_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                
                self.policy_optimizer.step()
        
        # return total_policy_loss, total_value_loss, entropy.item()