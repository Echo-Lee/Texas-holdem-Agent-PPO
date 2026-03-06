"""
This file defines the Runner class, 
which is responsible for managing the interaction between the agent and the environment. 
"""
import torch
import numpy as np

from core.agents.ppo_agent import PPOAgent

class Runner:
    def __init__(self, env, agent, opponent_model=None, opponent_agent=None, use_model_logic=True):
        self.env = env
        self.agent = agent
        self.opp_model = opponent_model
        self.opp_agent = opponent_agent
        self.device = agent.device
        self.self_play_mode = isinstance(opponent_agent, PPOAgent)
        self.use_model_logic = use_model_logic
    
    def _get_processed_obs(self, raw_obs, blind=False) -> np.ndarray:
        """If use opponent model, concatenate the predicted opponent action distribution to the raw observation."""
        # blind: whether to blind the RL agent from the opponent model's prediction, used for ablation comparison
        if blind or self.opp_model is None:
            padding = np.zeros(4, dtype=np.float32)
            return np.concatenate([raw_obs, padding])

        opp_pred = self.opp_model.predict(raw_obs)
        return np.concatenate([raw_obs, opp_pred])
    
    def run_one_episode(self) -> tuple[dict[str, list], float]:
        """
        Run one episode of interaction with the environment using the agent's policy.
        Return trajectory of player_0.
        states: list of numpy arrays, each of shape (obs_dim,) or (obs_dim + opp_pred_dim,)
        actions: list of torch tensors, each of shape (1,) containing the action index
        log_probs: list of torch tensors, each of shape (1,) containing the log probability of the action
        rewards: list of floats, the reward received after taking the action
        dones: list of floats (0.0 or 1.0), indicating whether the episode ended after the action
        masks: list of numpy arrays, each of shape (action_dim,) with 1 for valid actions and 0 for invalid actions
        """
        self.env.reset()

        # Trajectory storage for player_0
        trajectory = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'dones': [], 'masks': [],
            'opp_states': [], 'opp_actions': []
        }

        player_0_last_raw_obs = None
        p0_final_reward = 0.0

        for name in self.env.agent_iter():
            # obs is a numpy array
            obs, reward, termination, truncation, _ = self.env.last()
            done = termination or truncation
            raw_obs = obs["observation"]

            if name == "player_0":
                player_0_last_raw_obs = raw_obs.copy()
                # Append reward and done for the previous step (except for the first step)
                if len(trajectory['states']) > 0:
                    trajectory['rewards'].append(reward)
                    trajectory['dones'].append(float(done))
                
                if done:
                    self.env.step(None)
                    p0_final_reward = reward
                    continue
                
                p0_is_blind = not self.use_model_logic
                state = self._get_processed_obs(raw_obs, blind=p0_is_blind)
                mask = obs["action_mask"] # already numpy array

                # torch.Tensor: action, log_prob, value
                action, log_prob = self.agent.get_action(state, mask)

                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['log_probs'].append(log_prob)
                trajectory['masks'].append(mask)

                # must input an int item to env.step(), not a tensor
                self.env.step(action.item())
            
            else:  # opponent moves
                if done:
                    self.env.step(None)
                    continue
                
                if self.opp_agent is not None:
                    if self.self_play_mode:
                        # Self-play: feed in 76 dim obs with zero padding (blind to opponent model)
                        opp_state = self._get_processed_obs(raw_obs, blind=True)
                        action_tensor, _ = self.opp_agent.get_action(opp_state, obs["action_mask"])
                    else:
                        # Rule-based opponent: feed in the raw 72 dim obs without opponent model prediction
                        action_tensor, _ = self.opp_agent.get_action(obs["observation"], obs["action_mask"])
                    action = action_tensor.item()
                else:
                    # random opponent
                    valid_actions = np.where(obs["action_mask"] == 1)[0]
                    action = np.random.choice(valid_actions)

                if player_0_last_raw_obs is not None:
                        trajectory['opp_states'].append(player_0_last_raw_obs)
                        trajectory['opp_actions'].append(action)
                # Step the environment with opponent's action
                self.env.step(int(action))
        return trajectory, p0_final_reward

    def collect_batch(self, num_episodes):
        """Collect a batch of trajectories by running multiple episodes.
         Return: dict of tensors for states, actions, log_probs, values, rewards, dones, masks; average reward; win rate."""
        
        batch_trajectories = {k: [] for k in ['states', 'actions', 'log_probs', 'rewards', 'dones', 'masks', 'opp_states', 'opp_actions']}
        total_reward = 0
        win_count = 0

        for _ in range(num_episodes):
            traj, final_reward = self.run_one_episode()

            # The reward of the episode is determined by the last reward in the trajectory (win/loss).
            total_reward += final_reward
            if final_reward > 0: win_count += 1

            for k in batch_trajectories.keys():
                # Flatten the trajectory lists and extend to batch storage, a big list for each key
                batch_trajectories[k].extend(traj[k])
        
        # Train opponent model with collected opponent trajectories
        opp_loss, opp_acc = 0.0, 0.0
        if self.opp_model is not None and self.use_model_logic and len(batch_trajectories['opp_states']) > 0:
            opp_states_batch = np.stack(batch_trajectories['opp_states'])
            opp_act_batch = np.array(batch_trajectories['opp_actions'])
            
            opp_loss, opp_acc = self.opp_model.train_step(opp_states_batch, opp_act_batch)
        
        ppo_keys = ['states', 'actions', 'log_probs', 'rewards', 'dones', 'masks']
        ppo_batch = {k: batch_trajectories[k] for k in ppo_keys}
        batch_tensors = self._to_tensor(ppo_batch)

        return batch_tensors, total_reward / num_episodes, win_count / num_episodes, opp_loss, opp_acc
    
    def _to_tensor(self, batch) -> dict[str, torch.Tensor]:
        tensors = {}
        
        # States (Numpy -> Tensor)
        tensors['states'] = torch.as_tensor(np.stack(batch['states']), 
                                           dtype=torch.float32, device=self.device)
        
        # Actions, Log_probs (List of Tensors -> One Big Tensor)
        tensors['actions'] = torch.cat(batch['actions']).to(self.device)
        tensors['log_probs'] = torch.cat(batch['log_probs']).to(self.device)
        
        
        # Rewards, Dones, Masks (List of Floats -> Tensor)
        tensors['rewards'] = torch.as_tensor(batch['rewards'], 
                                            dtype=torch.float32, device=self.device)
        tensors['dones'] = torch.as_tensor(batch['dones'], 
                                          dtype=torch.float32, device=self.device)
        tensors['masks'] = torch.as_tensor(np.stack(batch['masks']), 
                                           dtype=torch.bool, device=self.device)
        
        # (batch_size, ...)
        return tensors