# main.py
import torch
import yaml
import wandb
from pettingzoo.classic import leduc_holdem_v4
from core.agents import PPOAgent, RuleBasedAgent
from .utils.utils import plot_results, moving_average
from core.environments import Runner
from core.networks import PolicyNet, ValueNet
from core.networks import OpponentPredictor
from statistics import mean

def load_config(config_path="running_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config(config_path="running_config.yaml")

    wandb.init(
        project=cfg['wandb']['project'],
        name=f"ppo_{cfg['opponent']['style']}_use_oppo_model_lr{cfg['system']['use_opponent_model']}",
        config=cfg,
    )

    if cfg['system']['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg['system']['device']
    print(f"Using device: {device}")

    env = leduc_holdem_v4.env()
    env.reset()

    first_obs, _, _, _, _ = env.last()
    # Leduc official specs:
    # Observation shape = (36,)
    # Actions = {0: Call, 1: Raise, 2: Fold, 3: Check}
    raw_obs_dim = len(first_obs["observation"])
    action_dim = 4
    use_opponent_model = cfg['system']['use_opponent_model']

    print(f"Detected observation dim = {raw_obs_dim}")

    agent_obs_dim = raw_obs_dim + (action_dim if use_opponent_model else 0)

    # Instantiate agents and models
    if use_opponent_model:
        opp_predictor = OpponentPredictor(
            obs_dim=raw_obs_dim, 
            act_dim=action_dim,
            device=device,
            lr=cfg['opponent']['predictor_lr']
        )
    else:
        opp_predictor = None

    policy_net = PolicyNet(input_dim=agent_obs_dim, output_dim=action_dim)
    value_net = ValueNet(input_dim=agent_obs_dim)
    opponent_agent = RuleBasedAgent(device=device, style=cfg['opponent']['style'])

    agent = PPOAgent(
        policy_net=policy_net,
        value_net=value_net,
        device=device,
        lr=cfg['train']['lr'],
        gamma=cfg['train']['gamma'],
        clip_param=cfg['ppo']['clip_param'],
        policy_epochs=cfg['ppo']['policy_epochs'],
        value_epochs=cfg['ppo']['value_epochs'],
        mini_batch_size=cfg['train']['mini_batch_size'],
        gae_lambda=cfg['ppo']['gae_lambda'],
        entropy_coef=cfg['ppo']['entropy_coef']
    )

    runner = Runner(env, agent, opp_predictor, opponent_agent=opponent_agent)

    print(f"Starting training against {cfg['opponent']['style']} opponent...")

    reward_history, win_rate_history, opp_loss_history, opp_acc_history = [], [], [], []
    # policy_loss_history, value_loss_history, entropy_history = [], [], []

    for iteration in range(cfg['train']['num_iterations']):
        # Collect trajectories by running episodes in the environment
        batch_tensors, avg_reward, win_rate, opp_loss, opp_acc = runner.collect_batch(
            num_episodes=cfg['train']['episodes_per_batch']
            )

        # PPO update
        agent.update(batch_tensors)
        # policy_loss, value_loss, entropy = agent.update(batch_tensors)

        # Record metrics for plotting
        reward_history.append(avg_reward)
        win_rate_history.append(win_rate)
        opp_loss_history.append(opp_loss)
        opp_acc_history.append(opp_acc)
        # policy_loss_history.append(policy_loss)
        # value_loss_history.append(value_loss)
        # entropy_history.append(entropy)

        wandb.log({
            "iteration": iteration,
            "performance/avg_reward": avg_reward,
            "performance/win_rate": win_rate,
            "opponent/model_loss": opp_loss,
            "opponent/model_acc": opp_acc,
        })

        # Logging
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Avg Reward = {avg_reward:.3f}, Win Rate = {win_rate:.3f}, Opp Loss = {opp_loss:.4f}, Opp Acc = {opp_acc:.3f}")

    wandb.finish()
    print("Training completed.")
    env.close()

    # Plot results
    # plot_results(reward_history, win_rate_history, opp_loss_history, opp_acc_history, policy_loss_history, value_loss_history, entropy_history)

if __name__ == "__main__":
    main()