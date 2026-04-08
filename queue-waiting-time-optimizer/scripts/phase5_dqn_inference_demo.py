from __future__ import annotations

import argparse
from pathlib import Path

from qwt_optimizer.envs.queue_gym_env import QueueGymEnv
from qwt_optimizer.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a trained DQN model.")
    parser.add_argument("--scenario", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="outputs/models/dqn_queue_optimizer.zip")
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    try:
        from stable_baselines3 import DQN
    except Exception as exc:
        raise RuntimeError(
            "Stable-Baselines3 is not available. Install compatible versions of torch and stable-baselines3 first."
        ) from exc

    env = QueueGymEnv(scenario_name=args.scenario, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    model_path = Path(args.model_path)
    if model_path.suffix != ".zip" and model_path.with_suffix(".zip").exists():
        model_path = model_path.with_suffix(".zip")

    model = DQN.load(str(model_path))

    total_reward = 0.0
    print("=== Phase 5 DQN Inference Demo ===")
    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward

        state = info["state_dict"]
        print(
            f"t={t + 1:02d} action={int(action)} queue={state['queue_length']:.1f} "
            f"wait={state['average_waiting_time']:.2f} counters={state['open_counters']:.0f} reward={reward:.3f}"
        )

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
