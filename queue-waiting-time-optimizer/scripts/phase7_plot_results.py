from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation results for baseline vs DQN.")
    parser.add_argument("--summary-csv", type=str, default="outputs/metrics/evaluation_summary.csv")
    parser.add_argument("--trace-csv", type=str, default="outputs/metrics/evaluation_traces.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/plots")
    return parser.parse_args()


def plot_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "mean_episode_reward",
        "mean_queue_length",
        "mean_waiting_time",
        "mean_open_counters",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot = summary_df.pivot(index="scenario", columns="agent", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    out_path = out_dir / "summary_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_traces(trace_df: pd.DataFrame, out_dir: Path) -> None:
    scenarios = ["easy", "medium", "hard"]

    for scenario in scenarios:
        scenario_df = trace_df[trace_df["scenario"] == scenario]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for agent in ["baseline", "dqn"]:
            part = scenario_df[scenario_df["agent"] == agent]
            axes[0].plot(part["t"], part["queue_length"], label=agent)
            axes[1].plot(part["t"], part["average_waiting_time"], label=agent)
            axes[2].plot(part["t"], part["open_counters"], label=agent)
            axes[3].plot(part["t"], part["cumulative_reward"], label=agent)

        axes[0].set_title(f"{scenario.title()} Queue Length")
        axes[1].set_title(f"{scenario.title()} Avg Waiting Time")
        axes[2].set_title(f"{scenario.title()} Open Counters")
        axes[3].set_title(f"{scenario.title()} Cumulative Reward")

        for ax in axes:
            ax.set_xlabel("Timestep")
            ax.legend()

        plt.tight_layout()
        out_path = out_dir / f"trace_{scenario}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    trace_df = pd.read_csv(args.trace_csv)

    plot_summary(summary_df, out_dir)
    plot_traces(trace_df, out_dir)

    print("=== Phase 7 Plotting Complete ===")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
