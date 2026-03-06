# main.py
import pathlib

import torch
import yaml
import wandb
from pettingzoo.classic import texas_holdem_v4
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

    use_opp = cfg['system']['use_opponent_model']
    sp_enabled = cfg['selfplay']['enabled']

    model_tag = "w-OppM" if use_opp else "no-OppM"
    mode_tag = f"SP-U{cfg['selfplay']['update_every']}" if sp_enabled else f"Static-{cfg['opponent']['style']}"
    
    run_name = f"{mode_tag}_{model_tag}_lr{cfg['train']['lr']}"
    group_name = "Self-Play-Exploration" if sp_enabled else "Baseline-Tests"

    agent_cfg = cfg.get('model', {}).get('agent', {'hidden_layers': [256, 256], 'use_layer_norm': True})

    wandb.init(
        project=cfg['wandb']['project'],
        name=run_name,
        group=group_name,
        tags=["Texas-Holdem", mode_tag, model_tag],
        config=cfg,
    )

    if cfg['system']['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg['system']['device']
    print(f"Using device: {device}")

    env = texas_holdem_v4.env()
    env.reset()

    first_obs, _, _, _, _ = env.last()
    # Texas holdem official specs:
    # Observation shape = (72,)
    # Actions = {0: Call, 1: Raise, 2: Fold, 3: Check}
    raw_obs_dim = len(first_obs["observation"])
    action_dim = 4

    print(f"Detected observation dim = {raw_obs_dim}")

    agent_obs_dim = raw_obs_dim + action_dim

    # Instantiate agents and models
    policy_net = PolicyNet(
        input_dim=agent_obs_dim, 
        output_dim=action_dim, 
        hidden_layers=agent_cfg['hidden_layers'],
        use_layer_norm=agent_cfg['use_layer_norm']
    )
    
    value_net = ValueNet(
        input_dim=agent_obs_dim,
        hidden_layers=agent_cfg['hidden_layers'],
        use_layer_norm=agent_cfg['use_layer_norm']
    )
    
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

    # set up opponent model based on config
    if use_opp:
        opp_cfg = cfg['model']['opponent_predictor']
        
        opp_predictor = OpponentPredictor(
            obs_dim=raw_obs_dim, 
            act_dim=action_dim,
            device=device,
            lr=cfg['opponent']['predictor_lr'],
            hidden_layers=opp_cfg['hidden_layers'],
            use_layer_norm=opp_cfg['use_layer_norm']
        )
    else:
        opp_predictor = None


    # set up opponent agent based on config
    if sp_enabled:
        mirror_policy = PolicyNet(
            input_dim=agent_obs_dim, 
            output_dim=action_dim, 
            hidden_layers=agent_cfg['hidden_layers'],
            use_layer_norm=agent_cfg['use_layer_norm']
        )
        mirror_value = ValueNet(
            input_dim=agent_obs_dim,
            hidden_layers=agent_cfg['hidden_layers'],
            use_layer_norm=agent_cfg['use_layer_norm']
        )
        opponent_agent = PPOAgent(
            policy_net=mirror_policy,
            value_net=mirror_value,
            device=device
        )
        opponent_agent.policy.load_state_dict(agent.policy.state_dict())
        opponent_agent.value.load_state_dict(agent.value.state_dict())
    else:
        opponent_agent = RuleBasedAgent(style=cfg['opponent']['style'], device=device)

    # use_model_logic: whether the RLagent should use the opponent model's prediction as part of its observation input for action selection
    # If not, concatenate a zero vector instead to keep the input dimension consistent.
    runner = Runner(env, agent, opp_predictor, opponent_agent=opponent_agent, use_model_logic=use_opp)

    if sp_enabled:
        print(f"Starting self-play training against mirror opponent...")
    elif use_opp:
        print(f"Starting training against {cfg['opponent']['style']} opponent with opponent model...")
    else:
        print(f"Starting training against {cfg['opponent']['style']} opponent without opponent model...")

    # reward_history, win_rate_history, opp_loss_history, opp_acc_history = [], [], [], []
    # policy_loss_history, value_loss_history, entropy_history = [], [], []

    for iteration in range(cfg['train']['num_iterations']):
        # Self-play update: periodically update opponent agent to mirror current policy
        if sp_enabled and iteration > 0 and iteration % cfg['selfplay']['update_every'] == 0:
            print(f"Iteration {iteration}: [Self-Play] updating opponent agent to mirror current policy...")
            opponent_agent.policy.load_state_dict(agent.policy.state_dict())
            opponent_agent.value.load_state_dict(agent.value.state_dict())

        # Collect trajectories by running episodes in the environment
        batch_tensors, avg_reward, win_rate, opp_loss, opp_acc = runner.collect_batch(
            num_episodes=cfg['train']['episodes_per_batch']
            )

        # PPO update
        agent.update(batch_tensors)
        # policy_loss, value_loss, entropy = agent.update(batch_tensors)

        # Record metrics for plotting
        # reward_history.append(avg_reward)
        # win_rate_history.append(win_rate)
        # opp_loss_history.append(opp_loss)
        # opp_acc_history.append(opp_acc)
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

    save_dir = cfg['save'].get('save_dir', 'models/')
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_filename = f"{run_name}_final.pth"
    full_save_path = pathlib.Path(save_dir) / model_filename
    torch.save(agent.policy.state_dict(), full_save_path)
    
    print(f"Training completed. Model saved to: {full_save_path}")
    env.close()

    # Plot results
    # plot_results(reward_history, win_rate_history, opp_loss_history, opp_acc_history, policy_loss_history, value_loss_history, entropy_history)

if __name__ == "__main__":
    main()