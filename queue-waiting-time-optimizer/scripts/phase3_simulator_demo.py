from __future__ import annotations

from qwt_optimizer.agents.rule_based import RuleBasedAgent
from qwt_optimizer.envs.queue_simulator import QueueSimulator


def main() -> None:
    simulator = QueueSimulator(scenario_name="medium", seed=42)
    baseline = RuleBasedAgent()

    state = simulator.reset()
    total_reward = 0.0

    print("=== Phase 3 Simulator Demo ===")
    print(f"Initial state: {state}")

    for t in range(10):
        action = baseline.act(state)
        state, reward, done, info = simulator.step(action)
        total_reward += reward

        print(
            f"t={t + 1:02d} action={int(action)} queue={state['queue_length']:.1f} "
            f"wait={state['average_waiting_time']:.2f} counters={state['open_counters']:.0f} "
            f"arrivals={info['arrivals']:.0f} served={info['served']:.1f} reward={reward:.3f}"
        )

        if done:
            break

    print(f"Total reward (10 steps max): {total_reward:.3f}")


if __name__ == "__main__":
    main()
