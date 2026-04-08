from __future__ import annotations

from qwt_optimizer.envs.queue_simulator import QueueSimulator


def main() -> None:
    env_a = QueueSimulator(scenario_name="medium", seed=123)
    env_b = QueueSimulator(scenario_name="medium", seed=123)

    state_a = env_a.reset(seed=123)
    state_b = env_b.reset(seed=123)

    step_a = env_a.step(1)
    step_b = env_b.step(1)

    print("=== Phase 8 Seed Check ===")
    print(f"Reset state A: {state_a}")
    print(f"Reset state B: {state_b}")
    print(f"First step A: {step_a[0]}")
    print(f"First step B: {step_b[0]}")
    print(f"States match after reset: {state_a == state_b}")
    print(f"States match after first step: {step_a[0] == step_b[0]}")
    print(f"Rewards match: {step_a[1] == step_b[1]}")


if __name__ == "__main__":
    main()
