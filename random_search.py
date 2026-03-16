import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from core.main import (
    detect_env_dims,
    load_config,
    resolve_device,
    run_finetune_stage,
    run_stage1,
    run_training_pipeline,
)


STAGE_ORDER = ["stage1", "stage2", "stage3"]


def iter_with_progress(iterable, total, description):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=description, dynamic_ncols=True)


def progress_write(message):
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential random search for 3-stage poker training")
    parser.add_argument("--config", default="running_config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "stage1", "stage2", "stage3"],
        help="Highest stage to tune. Default tunes stage1-stage3 sequentially.",
    )
    return parser.parse_args()


def loguniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def get_search_stages(stage_arg):
    if stage_arg == "all":
        return list(STAGE_ORDER)
    return STAGE_ORDER[: STAGE_ORDER.index(stage_arg) + 1]


def apply_search_overrides(cfg, stage_name):
    overrides = cfg.get("random_search", {}).get("search_train_overrides", {}).get(stage_name, {})
    cfg["stages"][stage_name].update(overrides)


def sample_stage_params(stage_name, rng):
    if stage_name == "stage1":
        return {
            "lr": round(loguniform(rng, 1e-4, 7e-4), 8),
            "mini_batch_size": rng.choice([32, 48, 64, 96]),
            "gamma": round(rng.uniform(0.985, 0.995), 5),
            "entropy_coef": round(rng.uniform(0.002, 0.02), 5),
        }

    if stage_name == "stage2":
        return {
            "lr": round(loguniform(rng, 1e-5, 8e-5), 8),
            "mini_batch_size": rng.choice([64, 96, 128]),
            "replay_every": rng.choice([3, 4, 5, 6]),
            "replay_start_iteration": rng.choice([4, 8, 12]),
            "max_prev_stage_batches": rng.choice([4, 6, 8]),
            "max_current_stage_batches": rng.choice([2, 4, 6]),
        }

    return {
        "lr": round(loguniform(rng, 5e-6, 5e-5), 8),
        "mini_batch_size": rng.choice([64, 96, 128]),
        "replay_every": rng.choice([3, 4, 5, 6]),
        "replay_start_iteration": rng.choice([4, 8, 12]),
        "max_prev_stage_batches": rng.choice([2, 4, 6]),
        "max_current_stage_batches": rng.choice([2, 4, 6]),
    }


def apply_stage_params(cfg, stage_name, params):
    stage_cfg = cfg["stages"][stage_name]

    if "lr" in params:
        stage_cfg["lr"] = params["lr"]
    if "mini_batch_size" in params:
        stage_cfg["mini_batch_size"] = params["mini_batch_size"]
    if "gamma" in params:
        cfg["train"]["gamma"] = params["gamma"]
    if "entropy_coef" in params:
        cfg["ppo"]["entropy_coef"] = params["entropy_coef"]

    if stage_name in {"stage2", "stage3"}:
        replay_cfg = stage_cfg.setdefault("replay", {})
        for key in ["replay_every", "replay_start_iteration", "max_prev_stage_batches", "max_current_stage_batches"]:
            if key in params:
                replay_cfg[key] = params[key]


def mean_tail(metrics, key, tail=3):
    tail_metrics = metrics[-tail:] if len(metrics) >= tail else metrics
    return sum(row[key] for row in tail_metrics) / max(1, len(tail_metrics))


def compute_stage_score(metrics, score_weights):
    reward_score = mean_tail(metrics, "avg_reward")
    win_rate_score = mean_tail(metrics, "win_rate")
    return score_weights["reward"] * reward_score + score_weights["win_rate"] * win_rate_score


def compute_stage1_score(outputs, score_weights):
    scores = [compute_stage_score(output["metrics"], score_weights) for output in outputs]
    return sum(scores) / len(scores)


def plot_search_scores(trial_records, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trial_numbers = [record["trial"] for record in trial_records]
    scores = [record["score"] for record in trial_records]
    best_so_far = []
    running_best = float("-inf")
    for score in scores:
        running_best = max(running_best, score)
        best_so_far.append(running_best)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(trial_numbers, scores, marker="o", linewidth=1.5, label="trial score")
    axis.plot(trial_numbers, best_so_far, linewidth=2.0, label="best so far")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def evaluate_stage1_trial(cfg, device, raw_obs_dim, action_dim, agent_obs_dim, trial_dir, score_weights):
    outputs = run_stage1(cfg, raw_obs_dim, action_dim, agent_obs_dim, device, trial_dir)
    score = compute_stage1_score(outputs, score_weights)
    return score, outputs


def evaluate_finetune_trial(stage_name, cfg, device, raw_obs_dim, action_dim, agent_obs_dim, trial_dir, stage_outputs, score_weights):
    output = run_finetune_stage(
        cfg=cfg,
        stage_name=stage_name,
        raw_obs_dim=raw_obs_dim,
        action_dim=action_dim,
        agent_obs_dim=agent_obs_dim,
        device=device,
        results_dir=trial_dir,
        stage_outputs=stage_outputs,
    )
    score = compute_stage_score(output["metrics"], score_weights)
    return score, output


def stage_output_map(stage1_outputs=None, stage2_output=None, stage3_output=None):
    output_map = {}
    for output in stage1_outputs or []:
        output_map[output["stage_name"]] = output
    if stage2_output is not None:
        output_map["stage2"] = stage2_output
    if stage3_output is not None:
        output_map["stage3"] = stage3_output
    return output_map


def main():
    args = parse_args()
    rng = random.Random(7)
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    raw_obs_dim, action_dim = detect_env_dims()
    agent_obs_dim = raw_obs_dim + action_dim

    score_weights = cfg.get("random_search", {}).get("score_weights", {"reward": 1.0, "win_rate": 4.0})
    trials_per_stage = cfg.get("random_search", {}).get("trials_per_stage", 30)
    stages_to_tune = get_search_stages(args.stage)

    search_root = Path(cfg.get("results", {}).get("dir", "results")) / "random_search"
    search_root.mkdir(parents=True, exist_ok=True)

    tuned_params = {}
    tuned_outputs = {}
    stage1_outputs_full = None
    stage2_output_full = None

    for stage_name in stages_to_tune:
        progress_write(f"[random-search] starting {stage_name} search with {trials_per_stage} trials")
        stage_dir = search_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        trial_records = []
        best_score = float("-inf")
        best_params = None

        trial_iterator = iter_with_progress(
            range(1, trials_per_stage + 1),
            total=trials_per_stage,
            description=f"{stage_name} trials",
        )
        for trial in trial_iterator:
            trial_cfg = copy.deepcopy(cfg)
            for previous_stage in tuned_params:
                apply_stage_params(trial_cfg, previous_stage, tuned_params[previous_stage])

            sampled_params = sample_stage_params(stage_name, rng)
            apply_stage_params(trial_cfg, stage_name, sampled_params)
            apply_search_overrides(trial_cfg, stage_name)

            trial_dir = stage_dir / f"trial_{trial:02d}"
            if stage_name == "stage1":
                score, _ = evaluate_stage1_trial(
                    trial_cfg,
                    device,
                    raw_obs_dim,
                    action_dim,
                    agent_obs_dim,
                    trial_dir,
                    score_weights,
                )
            else:
                prior_outputs = stage_output_map(stage1_outputs_full, stage2_output_full)
                score, _ = evaluate_finetune_trial(
                    stage_name,
                    trial_cfg,
                    device,
                    raw_obs_dim,
                    action_dim,
                    agent_obs_dim,
                    trial_dir,
                    prior_outputs,
                    score_weights,
                )

            record = {
                "trial": trial,
                "score": score,
                "params": sampled_params,
            }
            trial_records.append(record)
            progress_write(
                f"[random-search:{stage_name}] trial {trial:02d}/{trials_per_stage} | "
                f"score={score:.4f} | params={sampled_params}"
            )

            if score > best_score:
                best_score = score
                best_params = sampled_params
                progress_write(
                    f"[random-search:{stage_name}] new best at trial {trial:02d} | "
                    f"score={best_score:.4f} | params={best_params}"
                )

        tuned_params[stage_name] = best_params
        write_json(
            {"stage": stage_name, "best_score": best_score, "best_params": best_params, "trials": trial_records},
            stage_dir / "search_results.json",
        )
        plot_search_scores(trial_records, stage_dir / "search_scores.png", title=f"{stage_name} Random Search")
        progress_write(f"[random-search:{stage_name}] best params locked in: {best_params}")

        full_cfg = copy.deepcopy(cfg)
        for tuned_stage, params in tuned_params.items():
            apply_stage_params(full_cfg, tuned_stage, params)

        if stage_name == "stage1":
            full_stage1_dir = stage_dir / "best_stage1_full"
            progress_write(f"[random-search:{stage_name}] running full stage1 with best params")
            stage1_outputs_full = run_stage1(full_cfg, raw_obs_dim, action_dim, agent_obs_dim, device, full_stage1_dir)
        elif stage_name == "stage2":
            full_stage2_dir = stage_dir / "best_stage2_full"
            progress_write(f"[random-search:{stage_name}] running full stage2 with best params")
            stage2_output_full = run_finetune_stage(
                cfg=full_cfg,
                stage_name="stage2",
                raw_obs_dim=raw_obs_dim,
                action_dim=action_dim,
                agent_obs_dim=agent_obs_dim,
                device=device,
                results_dir=full_stage2_dir,
                stage_outputs=stage_output_map(stage1_outputs_full),
            )
        elif stage_name == "stage3":
            full_stage3_dir = stage_dir / "best_stage3_full"
            progress_write(f"[random-search:{stage_name}] running full stage3 with best params")
            run_finetune_stage(
                cfg=full_cfg,
                stage_name="stage3",
                raw_obs_dim=raw_obs_dim,
                action_dim=action_dim,
                agent_obs_dim=agent_obs_dim,
                device=device,
                results_dir=full_stage3_dir,
                stage_outputs=stage_output_map(stage1_outputs_full, stage2_output_full),
            )

    final_cfg = copy.deepcopy(cfg)
    for stage_name, params in tuned_params.items():
        apply_stage_params(final_cfg, stage_name, params)

    write_json({"tuned_params": tuned_params}, search_root / "best_params.json")

    final_results_dir = search_root / "final"
    progress_write("[random-search] running final tuned multi-stage training")
    run_training_pipeline(final_cfg, results_dir=final_results_dir, stages_to_run=stages_to_tune)

    progress_write(f"Random search completed. Best params saved to: {search_root / 'best_params.json'}")
    progress_write(f"Final tuned run saved to: {final_results_dir}")


if __name__ == "__main__":
    main()
