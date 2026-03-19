import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from pettingzoo.classic import texas_holdem_v4

try:
    import wandb
except ImportError:
    wandb = None

from core.agents import PPOAgent, RuleBasedAgent
from core.environments import Runner
from core.networks import OpponentPredictor, PolicyNet, ValueNet, FlexibleNet
from core.utils.replay_buffer import ReplayBatchBuffer, clone_batch_to_cpu, merge_batches
from core.utils.utils import plot_stage_metrics, write_metrics_csv, write_summary_json


def load_config(config_path="running_config.yaml"):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config_snapshot(config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg):
    requested = cfg["system"].get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def get_stage_train_params(cfg, stage_cfg):
    params = dict(cfg["train"])
    stage_key_map = {
        "iterations": "num_iterations",
        "episodes_per_batch": "episodes_per_batch",
        "mini_batch_size": "mini_batch_size",
        "lr": "lr",
        "gamma": "gamma",
        "seed": "seed",
    }
    for stage_key, param_key in stage_key_map.items():
        if stage_key in stage_cfg:
            params[param_key] = stage_cfg[stage_key]
    return params


def detect_env_dims():
    env = texas_holdem_v4.env(num_players=2)
    env.reset()
    first_obs, _, _, _, _ = env.last()
    raw_obs_dim = len(first_obs["observation"])
    env.close()
    return raw_obs_dim, 4


def build_agent_network_kwargs(cfg):
    agent_cfg = cfg.get("model", {}).get("agent", {})
    model_type = cfg.get("system", {}).get("model_type", "cnn")

    if model_type == "mlp":
        return {
            "hidden_layers": agent_cfg.get("mlp_hidden_layers", [512, 400, 256]),
            "use_layer_norm": agent_cfg.get("use_layer_norm", True),
        }
    else:  # cnn
        return {
            "hidden_layers": agent_cfg.get("hidden_layers", [256, 256, 128]),
            "use_layer_norm": agent_cfg.get("use_layer_norm", True),
            "card_encoder_channels": agent_cfg.get("card_encoder_channels", [16, 32]),
            "card_embedding_dim": agent_cfg.get("card_embedding_dim", 128),
        }


def create_ppo_agent(cfg, input_dim, action_dim, device, train_params=None):
    train_params = train_params or cfg["train"]
    network_kwargs = build_agent_network_kwargs(cfg)
    model_type = cfg.get("system", {}).get("model_type", "cnn")

    if model_type == "mlp":
        # Use pure MLP networks
        policy_net = FlexibleNet(
            input_dim=input_dim,
            output_dim=action_dim,
            **network_kwargs,
        )
        value_net = FlexibleNet(
            input_dim=input_dim,
            output_dim=1,
            **network_kwargs,
        )
    else:  # cnn
        # Use card-aware CNN networks
        policy_net = PolicyNet(
            input_dim=input_dim,
            output_dim=action_dim,
            **network_kwargs,
        )
        value_net = ValueNet(
            input_dim=input_dim,
            **network_kwargs,
        )
    return PPOAgent(
        policy_net=policy_net,
        value_net=value_net,
        device=device,
        lr=train_params["lr"],
        gamma=train_params["gamma"],
        clip_param=cfg["ppo"]["clip_param"],
        policy_epochs=cfg["ppo"]["policy_epochs"],
        value_epochs=cfg["ppo"]["value_epochs"],
        mini_batch_size=train_params["mini_batch_size"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        entropy_coef=cfg["ppo"]["entropy_coef"],
    )


def create_opponent_predictor(cfg, raw_obs_dim, action_dim, device):
    if not cfg["system"].get("use_opponent_model", True):
        return None

    predictor_cfg = cfg.get("model", {}).get("opponent_predictor", {})
    return OpponentPredictor(
        obs_dim=raw_obs_dim,
        act_dim=action_dim,
        device=device,
        lr=cfg["opponent"]["predictor_lr"],
        hidden_layers=predictor_cfg.get("hidden_layers", [128, 128]),
        use_layer_norm=predictor_cfg.get("use_layer_norm", False),
    )


def append_seed_batch(seed_batches, batch_tensors, max_items):
    if max_items <= 0:
        return
    seed_batches.append(clone_batch_to_cpu(batch_tensors))
    while len(seed_batches) > max_items:
        seed_batches.pop(0)


def maybe_init_wandb(cfg, stage_name, run_name):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        print("wandb is not installed. Continuing without experiment tracking.")
        return None

    return wandb.init(
        project=wandb_cfg.get("project", "Texas-Holdem-PPO-selfplay"),
        name=run_name,
        group=stage_name,
        config=cfg,
        reinit=True,
        mode=wandb_cfg.get("mode", "offline"),
        tags=["Texas-Holdem", stage_name],
    )


def train_stage(
    cfg,
    stage_name,
    agent,
    runner,
    train_params,
    output_dir,
    replay_buffer=None,
    replay_settings=None,
    seed_batch_limit=0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = maybe_init_wandb(cfg, stage_name, stage_name)
    metrics = []
    rewards = []
    replay_settings = replay_settings or {}
    stage1_replay_per_update = replay_settings.get("stage1_batches_per_update", 0)
    stage2_replay_per_update = replay_settings.get("stage2_batches_per_update", 0)
    replay_every = replay_settings.get("replay_every", 1)
    replay_start_iteration = replay_settings.get("replay_start_iteration", 0)
    seed_replay_batches = []

    for iteration in range(train_params["num_iterations"]):
        batch_tensors, avg_reward, win_rate, opp_loss, opp_acc = runner.collect_batch(
            num_episodes=train_params["episodes_per_batch"]
        )
        update_batches = [batch_tensors]
        use_replay = (
            replay_buffer is not None
            and iteration >= replay_start_iteration
            and (iteration - replay_start_iteration) % replay_every == 0
        )
        if use_replay:
            replay_batches = replay_buffer.sample(
                num_stage1_batches=stage1_replay_per_update,
                num_stage2_batches=stage2_replay_per_update,
                device=agent.device,
            )
            update_batches.extend(replay_batches)

        update_batch = merge_batches(update_batches, device=agent.device) if len(update_batches) > 1 else batch_tensors
        agent.update(update_batch)

        rewards.append(avg_reward)
        row = {
            "iteration": iteration,
            "avg_reward": float(avg_reward),
            "win_rate": float(win_rate),
            "opp_loss": float(opp_loss),
            "opp_acc": float(opp_acc),
        }
        metrics.append(row)

        if run is not None:
            wandb.log(
                {
                    "iteration": iteration,
                    "performance/avg_reward": avg_reward,
                    "performance/win_rate": win_rate,
                    "opponent/model_loss": opp_loss,
                    "opponent/model_acc": opp_acc,
                }
            )

        if iteration % 5 == 0 or iteration == train_params["num_iterations"] - 1:
            print(
                f"[{stage_name}] iteration {iteration:03d} | "
                f"avg_reward={avg_reward:.3f} | win_rate={win_rate:.3f} | "
                f"opp_loss={opp_loss:.4f} | opp_acc={opp_acc:.3f}"
            )

        append_seed_batch(seed_replay_batches, batch_tensors, seed_batch_limit)
        if replay_buffer is not None:
            replay_buffer.add_stage2_batch(batch_tensors)

    if run is not None:
        wandb.finish()

    checkpoint_path = output_dir / "agent_checkpoint.pt"
    agent.save(checkpoint_path)

    write_metrics_csv(metrics, output_dir / "metrics.csv")
    summary = {
        "stage_name": stage_name,
        "iterations": train_params["num_iterations"],
        "episodes_per_batch": train_params["episodes_per_batch"],
        "final_avg_reward": metrics[-1]["avg_reward"] if metrics else None,
        "best_avg_reward": max(rewards) if rewards else None,
        "final_win_rate": metrics[-1]["win_rate"] if metrics else None,
        "checkpoint_path": str(checkpoint_path),
    }
    write_summary_json(summary, output_dir / "summary.json")

    return {
        "stage_name": stage_name,
        "metrics": metrics,
        "rewards": rewards,
        "checkpoint_path": checkpoint_path,
        "summary": summary,
        "seed_replay_batches": seed_replay_batches,
    }


def run_stage1(cfg, raw_obs_dim, action_dim, agent_obs_dim, device, results_dir):
    stage1_cfg = cfg.get("stages", {}).get("stage1", {})
    num_agents = stage1_cfg.get("num_agents", 2)
    train_params = get_stage_train_params(cfg, stage1_cfg)
    base_seed = train_params.get("seed", 7)
    opponent_style = stage1_cfg.get("opponent_style", cfg["opponent"]["style"])
    stage1_replay_batches = stage1_cfg.get("replay_seed_batches", 0)

    outputs = []
    for agent_index in range(num_agents):
        stage_name = f"stage1_agent_{agent_index + 1}"
        seed = base_seed + agent_index
        print(f"Starting {stage_name} with seed={seed} against {opponent_style} opponent.")
        set_seed(seed)

        env = texas_holdem_v4.env(num_players=2)
        stage_train_params = dict(train_params)
        stage_train_params["seed"] = seed
        learner = create_ppo_agent(cfg, agent_obs_dim, action_dim, device, train_params=stage_train_params)
        opponent_model = create_opponent_predictor(cfg, raw_obs_dim, action_dim, device)
        opponent_agent = RuleBasedAgent(style=opponent_style, device=device)
        runner = Runner(
            env,
            learner,
            opponent_model=opponent_model,
            opponent_agent=opponent_agent,
            use_model_logic=cfg["system"].get("use_opponent_model", True),
        )

        output = train_stage(
            cfg=cfg,
            stage_name=stage_name,
            agent=learner,
            runner=runner,
            train_params=stage_train_params,
            output_dir=results_dir / stage_name,
            seed_batch_limit=stage1_replay_batches,
        )

        # Save stage1 checkpoints to models/ directory
        save_dir = Path(cfg.get("save", {}).get("save_dir", "models"))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_type = cfg.get("system", {}).get("model_type", "cnn")
        stage1_model_name = f"stage1_agent_{agent_index + 1}_{model_type}_policy.pt"
        stage1_model_path = save_dir / stage1_model_name
        learner.save(stage1_model_path)
        shutil.copy2(stage1_model_path, results_dir / stage_name / stage1_model_name)
        output["final_model_path"] = stage1_model_path

        outputs.append(output)
        env.close()

    return outputs


def run_finetune_stage(
    cfg,
    stage_name,
    raw_obs_dim,
    action_dim,
    agent_obs_dim,
    device,
    results_dir,
    stage_outputs,
):
    stage_cfg = cfg.get("stages", {}).get(stage_name, {})
    train_params = get_stage_train_params(cfg, stage_cfg)
    base_seed = train_params.get("seed", 7)
    seed = base_seed + stage_cfg.get("seed_offset", 1000)

    learning_source = stage_cfg["learning_source"]
    opponent_source = stage_cfg["opponent_source"]
    if learning_source not in stage_outputs:
        raise ValueError(f"{stage_name}.learning_source={learning_source} is not available.")
    if opponent_source not in stage_outputs:
        raise ValueError(f"{stage_name}.opponent_source={opponent_source} is not available.")

    print(
        f"Starting {stage_name} with learner={learning_source} and "
        f"opponent={opponent_source}, seed={seed}."
    )
    set_seed(seed)

    env = texas_holdem_v4.env(num_players=2)
    stage_train_params = dict(train_params)
    stage_train_params["seed"] = seed
    learner = create_ppo_agent(cfg, agent_obs_dim, action_dim, device, train_params=stage_train_params)
    learner.load(stage_outputs[learning_source]["checkpoint_path"])

    fixed_opponent = create_ppo_agent(cfg, agent_obs_dim, action_dim, device, train_params=stage_train_params)
    fixed_opponent.load(stage_outputs[opponent_source]["checkpoint_path"])

    opponent_model = create_opponent_predictor(cfg, raw_obs_dim, action_dim, device)
    runner = Runner(
        env,
        learner,
        opponent_model=opponent_model,
        opponent_agent=fixed_opponent,
        use_model_logic=cfg["system"].get("use_opponent_model", True),
    )

    replay_cfg = stage_cfg.get("replay", {})
    replay_buffer = ReplayBatchBuffer(
        max_stage1_batches=replay_cfg.get("max_prev_stage_batches", replay_cfg.get("max_stage1_batches", 0)),
        max_stage2_batches=replay_cfg.get("max_current_stage_batches", replay_cfg.get("max_stage2_batches", 0)),
    )
    replay_buffer.stage1_batches = list(stage_outputs[learning_source].get("seed_replay_batches", []))

    output = train_stage(
        cfg=cfg,
        stage_name=stage_name,
        agent=learner,
        runner=runner,
        train_params=stage_train_params,
        output_dir=results_dir / stage_name,
        replay_buffer=replay_buffer,
        replay_settings=replay_cfg,
        seed_batch_limit=stage_cfg.get("replay_seed_batches", replay_cfg.get("max_current_stage_batches", replay_cfg.get("max_stage2_batches", 0))),
    )

    save_dir = Path(cfg.get("save", {}).get("save_dir", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)
    model_type = cfg.get("system", {}).get("model_type", "cnn")
    base_model_name = cfg.get("save", {}).get(f"{stage_name}_model_name", f"{stage_name}_final_policy.pt")
    # Insert model_type before .pt extension
    final_model_name = base_model_name.replace(".pt", f"_{model_type}.pt")
    final_model_path = save_dir / final_model_name
    learner.save(final_model_path)
    shutil.copy2(final_model_path, results_dir / stage_name / final_model_name)
    env.close()

    output["final_model_path"] = final_model_path
    return output


def run_training_pipeline(cfg, results_dir=None, stages_to_run=None):
    results_dir = Path(results_dir or cfg.get("results", {}).get("dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, results_dir / "run_config.yaml")

    device = resolve_device(cfg)
    raw_obs_dim, action_dim = detect_env_dims()
    agent_obs_dim = raw_obs_dim + action_dim

    print(f"Using device: {device}")
    print(f"Detected raw observation dim: {raw_obs_dim}")

    selected_stages = stages_to_run or ["stage1", "stage2", "stage3"]

    stage1_outputs = []
    stage_outputs = {} # used for replay buffer

    if "stage1" in selected_stages:
        # If we run stage1
        stage1_outputs = run_stage1(cfg, raw_obs_dim, action_dim, agent_obs_dim, device, results_dir)
        for output in stage1_outputs:
            stage_outputs[output["stage_name"]] = output

    if "stage2" in selected_stages:
        stage2_output = run_finetune_stage(
            cfg,
            "stage2",
            raw_obs_dim,
            action_dim,
            agent_obs_dim,
            device,
            results_dir,
            stage_outputs,
        )
        stage_outputs["stage2"] = stage2_output

    if "stage3" in selected_stages:
        stage3_output = run_finetune_stage(
            cfg,
            "stage3",
            raw_obs_dim,
            action_dim,
            agent_obs_dim,
            device,
            results_dir,
            stage_outputs,
        )
        stage_outputs["stage3"] = stage3_output

    if stage1_outputs:
        plot_stage_metrics(
            {output["stage_name"]: output["metrics"] for output in stage1_outputs},
            results_dir / "stage1_reward_winrate.png",
            title="Stage 1",
        )
    if "stage2" in stage_outputs:
        plot_stage_metrics(
            {"stage2": stage_outputs["stage2"]["metrics"]},
            results_dir / "stage2_reward_winrate.png",
            title="Stage 2",
        )
    if "stage3" in stage_outputs:
        plot_stage_metrics(
            {"stage3": stage_outputs["stage3"]["metrics"]},
            results_dir / "stage3_reward_winrate.png",
            title="Stage 3",
        )

    run_summary = {
        "device": device,
        "raw_obs_dim": raw_obs_dim,
        "agent_obs_dim": agent_obs_dim,
        "stage1": [output["summary"] for output in stage1_outputs],
    }
    if "stage2" in stage_outputs:
        run_summary["stage2"] = stage_outputs["stage2"]["summary"]
        run_summary["stage2_final_model_path"] = str(stage_outputs["stage2"]["final_model_path"])
    if "stage3" in stage_outputs:
        run_summary["stage3"] = stage_outputs["stage3"]["summary"]
        run_summary["stage3_final_model_path"] = str(stage_outputs["stage3"]["final_model_path"])
    write_summary_json(run_summary, results_dir / "run_summary.json")

    print(f"Training completed. Results saved to: {results_dir}")
    if "stage3" in stage_outputs:
        print(f"Final stage3 checkpoint saved to: {stage_outputs['stage3']['final_model_path']}")
    elif "stage2" in stage_outputs:
        print(f"Final stage2 checkpoint saved to: {stage_outputs['stage2']['final_model_path']}")

    return {
        "results_dir": results_dir,
        "device": device,
        "raw_obs_dim": raw_obs_dim,
        "action_dim": action_dim,
        "agent_obs_dim": agent_obs_dim,
        "stage1_outputs": stage1_outputs,
        "stage_outputs": stage_outputs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage Texas Hold'em PPO training")
    parser.add_argument("--config", default="running_config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()
