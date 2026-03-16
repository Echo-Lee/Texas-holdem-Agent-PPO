import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_metrics_csv(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["iteration", "avg_reward", "win_rate", "opp_loss", "opp_acc"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def write_summary_json(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_stage_metrics(stage_metrics, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (reward_ax, win_rate_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for label, metrics in stage_metrics.items():
        if not metrics:
            continue

        iterations = [0] + [row["iteration"] + 1 for row in metrics]
        rewards = [0.0] + [row["avg_reward"] for row in metrics]
        win_rates = [0.0] + [row["win_rate"] for row in metrics]

        reward_ax.plot(iterations, rewards, label=label)
        win_rate_ax.plot(iterations, win_rates, label=label)

    reward_ax.set_ylabel("Reward")
    reward_ax.set_title(f"{title} Reward")
    reward_ax.grid(alpha=0.2)
    reward_ax.legend()

    win_rate_ax.set_xlabel("Training Iteration")
    win_rate_ax.set_ylabel("Win Rate")
    win_rate_ax.set_title(f"{title} Win Rate")
    win_rate_ax.set_ylim(0.0, 1.0)
    win_rate_ax.grid(alpha=0.2)
    win_rate_ax.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
