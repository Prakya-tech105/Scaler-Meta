from __future__ import annotations

import argparse
from pathlib import Path

from qwt_optimizer.envs.queue_gym_env import QueueGymEnv
from qwt_optimizer.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on queue optimization env.")
    parser.add_argument("--scenario", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--model-out", type=str, default="outputs/models/dqn_queue_optimizer")
    parser.add_argument("--log-dir", type=str, default="outputs/metrics")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        raise RuntimeError(
            "Stable-Baselines3 is not available. Install compatible versions of torch and stable-baselines3 first."
        ) from exc

    log_dir = Path(args.log_dir)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = QueueGymEnv(scenario_name=args.scenario, seed=args.seed)
    env = Monitor(env, filename=str(log_dir / f"train_{args.scenario}_seed{args.seed}.csv"))

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
    model.save(str(model_out))
    env.close()

    print("=== Phase 5 DQN Training Complete ===")
    print(f"Saved model to: {model_out}.zip")
    print(f"Training log CSV: {log_dir / f'train_{args.scenario}_seed{args.seed}.csv'}")


if __name__ == "__main__":
    main()
