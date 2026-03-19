"""
Agent Battle Script
Allows two agents to compete against each other, regardless of architecture.
"""
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from pettingzoo.classic import texas_holdem_v4

from core.agents import PPOAgent
from core.networks import PolicyNet, ValueNet, FlexibleNet


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_model_type_from_checkpoint(checkpoint_path):
    """Detect model type from checkpoint filename."""
    checkpoint_name = Path(checkpoint_path).name
    if "_mlp" in checkpoint_name.lower():
        return "mlp"
    elif "_cnn" in checkpoint_name.lower():
        return "cnn"
    else:
        # Default to cnn if not specified
        print(f"Warning: Cannot detect model type from {checkpoint_name}, defaulting to CNN")
        return "cnn"


def create_agent_from_checkpoint(checkpoint_path, config_path, device):
    """Create and load an agent from checkpoint, auto-detecting architecture."""
    model_type = detect_model_type_from_checkpoint(checkpoint_path)

    # Auto-select config file based on model type if using default config
    if config_path == "running_config.yaml":
        if model_type == "mlp":
            # Try to use MLP-specific config if it exists
            mlp_config = "running_config_mlp.yaml"
            if Path(mlp_config).exists():
                config_path = mlp_config
                print(f"Auto-selected {mlp_config} for MLP model")

    cfg = load_config(config_path)

    # Environment dimensions
    env = texas_holdem_v4.env(num_players=2)
    env.reset()
    first_obs, _, _, _, _ = env.last()
    raw_obs_dim = len(first_obs["observation"])
    action_dim = 4
    agent_obs_dim = raw_obs_dim + action_dim
    env.close()

    # Build network kwargs based on model type
    agent_cfg = cfg.get("model", {}).get("agent", {})

    if model_type == "mlp":
        network_kwargs = {
            "hidden_layers": agent_cfg.get("mlp_hidden_layers", [256, 256, 128]), # MLP-specific setting
            "use_layer_norm": agent_cfg.get("use_layer_norm", True),
        }
        policy_net = FlexibleNet(
            input_dim=agent_obs_dim,
            output_dim=action_dim,
            **network_kwargs,
        )
        value_net = FlexibleNet(
            input_dim=agent_obs_dim,
            output_dim=1,
            **network_kwargs,
        )
    else:  # cnn
        network_kwargs = {
            "hidden_layers": agent_cfg.get("hidden_layers", [256, 256, 128]),
            "use_layer_norm": agent_cfg.get("use_layer_norm", True),
            "card_encoder_channels": agent_cfg.get("card_encoder_channels", [16, 32]),
            "card_embedding_dim": agent_cfg.get("card_embedding_dim", 128),
        }
        policy_net = PolicyNet(
            input_dim=agent_obs_dim,
            output_dim=action_dim,
            **network_kwargs,
        )
        value_net = ValueNet(
            input_dim=agent_obs_dim,
            **network_kwargs,
        )

    # Create PPO agent
    agent = PPOAgent(
        policy_net=policy_net,
        value_net=value_net,
        device=device,
        lr=0.0001,  # Not used for evaluation
        gamma=0.99,
        clip_param=0.2,
        policy_epochs=1,
        value_epochs=1,
        mini_batch_size=64,
        gae_lambda=0.95,
        entropy_coef=0.01,
    )

    # Load checkpoint
    agent.load(checkpoint_path)
    agent.policy.eval()
    agent.value.eval()

    print(f"Loaded {model_type.upper()} agent from {checkpoint_path}")
    return agent, model_type


def get_processed_obs(raw_obs):
    """Process observation (add zero padding for opponent model features)."""
    padding = np.zeros(4, dtype=np.float32)
    return np.concatenate([raw_obs, padding])


def run_battle(agent1, agent2, num_games=1000, device="cpu"):
    """
    Run battle between two agents.

    Returns:
        dict with statistics
    """
    env = texas_holdem_v4.env(num_players=2)

    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    agent1_rewards = []
    agent2_rewards = []

    for game_idx in range(num_games):
        env.reset()

        game_rewards = {"player_0": [], "player_1": []}

        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, _ = env.last()
            done = termination or truncation

            # Store rewards
            game_rewards[agent_name].append(reward)

            if done:
                env.step(None)
                continue

            raw_obs = obs["observation"]
            state = get_processed_obs(raw_obs)
            mask = obs["action_mask"]

            # Select agent
            if agent_name == "player_0":
                agent = agent1
            else:
                agent = agent2

            # Get action
            with torch.no_grad():
                action, _ = agent.get_action(state, mask)

            env.step(action.item())

        # Determine winner based on final rewards
        p0_final_reward = game_rewards["player_0"][-1] if len(game_rewards["player_0"]) > 0 else 0
        p1_final_reward = game_rewards["player_1"][-1] if len(game_rewards["player_1"]) > 0 else 0

        agent1_rewards.append(p0_final_reward)
        agent2_rewards.append(p1_final_reward)

        if p0_final_reward > 0:
            agent1_wins += 1
        elif p1_final_reward > 0:
            agent2_wins += 1
        else:
            ties += 1

        # Progress update
        if (game_idx + 1) % 100 == 0:
            print(f"Completed {game_idx + 1}/{num_games} games...")

    env.close()

    # Calculate statistics
    total_games = num_games
    agent1_win_rate = agent1_wins / total_games
    agent2_win_rate = agent2_wins / total_games
    tie_rate = ties / total_games

    agent1_avg_reward = np.mean(agent1_rewards)
    agent2_avg_reward = np.mean(agent2_rewards)

    return {
        "total_games": total_games,
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "ties": ties,
        "agent1_win_rate": agent1_win_rate,
        "agent2_win_rate": agent2_win_rate,
        "tie_rate": tie_rate,
        "agent1_avg_reward": agent1_avg_reward,
        "agent2_avg_reward": agent2_avg_reward,
        "agent1_rewards": agent1_rewards,
        "agent2_rewards": agent2_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent Battle: Compare two poker agents")
    parser.add_argument("--agent1", required=True, help="Path to agent 1 checkpoint")
    parser.add_argument("--agent2", required=True, help="Path to agent 2 checkpoint")
    parser.add_argument("--config", default="running_config.yaml", help="Config file (for network architecture)")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("Agent Battle Setup")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of games: {args.games}")
    print(f"Agent 1: {args.agent1}")
    print(f"Agent 2: {args.agent2}")
    print("=" * 60)
    print()

    # Load agents
    print("Loading agents...")
    agent1, type1 = create_agent_from_checkpoint(args.agent1, args.config, device)
    agent2, type2 = create_agent_from_checkpoint(args.agent2, args.config, device)
    print()

    # Run battle
    print(f"Starting battle: {type1.upper()} (Agent 1) vs {type2.upper()} (Agent 2)")
    print("=" * 60)
    results = run_battle(agent1, agent2, num_games=args.games, device=device)

    # Print results
    print()
    print("=" * 60)
    print("Battle Results")
    print("=" * 60)
    print(f"Total Games: {results['total_games']}")
    print()
    print(f"Agent 1 ({type1.upper()}):")
    print(f"  Wins: {results['agent1_wins']} ({results['agent1_win_rate']*100:.1f}%)")
    print(f"  Avg Reward: {results['agent1_avg_reward']:.3f}")
    print()
    print(f"Agent 2 ({type2.upper()}):")
    print(f"  Wins: {results['agent2_wins']} ({results['agent2_win_rate']*100:.1f}%)")
    print(f"  Avg Reward: {results['agent2_avg_reward']:.3f}")
    print()
    print(f"Ties: {results['ties']} ({results['tie_rate']*100:.1f}%)")
    print("=" * 60)

    # Save results
    output_file = Path("battle_results.txt")
    with output_file.open("w") as f:
        f.write(f"Agent Battle Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Agent 1: {args.agent1} ({type1.upper()})\n")
        f.write(f"Agent 2: {args.agent2} ({type2.upper()})\n")
        f.write(f"Total Games: {results['total_games']}\n\n")
        f.write(f"Agent 1 Wins: {results['agent1_wins']} ({results['agent1_win_rate']*100:.1f}%)\n")
        f.write(f"Agent 1 Avg Reward: {results['agent1_avg_reward']:.3f}\n\n")
        f.write(f"Agent 2 Wins: {results['agent2_wins']} ({results['agent2_win_rate']*100:.1f}%)\n")
        f.write(f"Agent 2 Avg Reward: {results['agent2_avg_reward']:.3f}\n\n")
        f.write(f"Ties: {results['ties']} ({results['tie_rate']*100:.1f}%)\n")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
