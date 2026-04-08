from __future__ import annotations

import numpy as np

from qwt_optimizer.envs.queue_gym_env import QueueGymEnv


def main() -> None:
    env = QueueGymEnv(scenario_name="medium", seed=42)

    obs, info = env.reset(seed=42)
    total_reward = 0.0

    print("=== Phase 4 Gymnasium Demo ===")
    print(f"Initial obs: {obs}")
    print(f"Initial state dict: {info['state_dict']}")

    for t in range(5):
        action = int(np.random.randint(0, 3))
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        print(
            f"t={t + 1:02d} action={action} obs={obs.tolist()} "
            f"reward={reward:.3f} done={terminated or truncated}"
        )

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
