from __future__ import annotations

import argparse
from pathlib import Path

import csv
import numpy as np

from qwt_optimizer.agents.rule_based import RuleBasedAgent
from qwt_optimizer.envs.queue_gym_env import QueueGymEnv
from qwt_optimizer.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs DQN across scenarios.")
    parser.add_argument("--model-path", type=str, default="outputs/models/dqn_queue_optimizer")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/metrics")
    return parser.parse_args()


def _resolve_model_path(model_path_arg: str) -> Path:
    model_path = Path(model_path_arg)
    if model_path.suffix == ".zip":
        return model_path
    if model_path.with_suffix(".zip").exists():
        return model_path.with_suffix(".zip")
    return model_path


def run_policy_episode(
    env: QueueGymEnv,
    agent_name: str,
    model,
    baseline: RuleBasedAgent,
    max_steps: int,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    obs, info = env.reset()

    rewards: list[float] = []
    queue_vals: list[float] = []
    wait_vals: list[float] = []
    counter_vals: list[float] = []

    trace_rows: list[dict[str, float | int | str]] = []
    cumulative_reward = 0.0

    for t in range(max_steps):
        if agent_name == "baseline":
            action = int(baseline.act(info["state_dict"]))
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, step_info = env.step(action)

        state = step_info["state_dict"]
        cumulative_reward += float(reward)
        rewards.append(float(reward))
        queue_vals.append(float(state["queue_length"]))
        wait_vals.append(float(state["average_waiting_time"]))
        counter_vals.append(float(state["open_counters"]))

        trace_rows.append(
            {
                "t": t + 1,
                "agent": agent_name,
                "queue_length": float(state["queue_length"]),
                "average_waiting_time": float(state["average_waiting_time"]),
                "open_counters": float(state["open_counters"]),
                "reward": float(reward),
                "cumulative_reward": float(cumulative_reward),
            }
        )

        info = step_info
        if terminated or truncated:
            break

    metrics = {
        "episode_reward": float(np.sum(rewards)),
        "mean_queue_length": float(np.mean(queue_vals) if queue_vals else 0.0),
        "mean_waiting_time": float(np.mean(wait_vals) if wait_vals else 0.0),
        "mean_open_counters": float(np.mean(counter_vals) if counter_vals else 0.0),
        "steps": float(len(rewards)),
    }
    return metrics, trace_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    try:
        from stable_baselines3 import DQN
    except Exception as exc:
        raise RuntimeError("Stable-Baselines3 is required for evaluation.") from exc

    model_path = _resolve_model_path(args.model_path)
    model = DQN.load(str(model_path))
    baseline = RuleBasedAgent()

    summary_file = out_dir / "evaluation_summary.csv"
    trace_file = out_dir / "evaluation_traces.csv"

    scenarios = ["easy", "medium", "hard"]
    agents = ["baseline", "dqn"]

    summary_rows: list[dict[str, str | float | int]] = []
    trace_rows_all: list[dict[str, str | float | int]] = []

    for scenario in scenarios:
        for agent_name in agents:
            episode_metrics: list[dict[str, float]] = []

            for episode_idx in range(args.episodes):
                env_seed = args.seed + episode_idx
                env = QueueGymEnv(scenario_name=scenario, seed=env_seed)
                env.reset(seed=env_seed)

                metrics, traces = run_policy_episode(
                    env=env,
                    agent_name=agent_name,
                    model=model,
                    baseline=baseline,
                    max_steps=args.max_steps,
                )
                env.close()

                episode_metrics.append(metrics)

                if episode_idx == 0:
                    for row in traces:
                        row["scenario"] = scenario
                        trace_rows_all.append(row)

            summary_rows.append(
                {
                    "scenario": scenario,
                    "agent": agent_name,
                    "episodes": args.episodes,
                    "mean_episode_reward": float(
                        np.mean([m["episode_reward"] for m in episode_metrics])
                    ),
                    "mean_queue_length": float(
                        np.mean([m["mean_queue_length"] for m in episode_metrics])
                    ),
                    "mean_waiting_time": float(
                        np.mean([m["mean_waiting_time"] for m in episode_metrics])
                    ),
                    "mean_open_counters": float(
                        np.mean([m["mean_open_counters"] for m in episode_metrics])
                    ),
                    "mean_steps": float(np.mean([m["steps"] for m in episode_metrics])),
                }
            )

    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "agent",
                "episodes",
                "mean_episode_reward",
                "mean_queue_length",
                "mean_waiting_time",
                "mean_open_counters",
                "mean_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with trace_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "t",
                "agent",
                "queue_length",
                "average_waiting_time",
                "open_counters",
                "reward",
                "cumulative_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(trace_rows_all)

    print("=== Phase 6 Evaluation Complete ===")
    print(f"Summary CSV: {summary_file}")
    print(f"Trace CSV: {trace_file}")


if __name__ == "__main__":
    main()
