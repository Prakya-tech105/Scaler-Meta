from __future__ import annotations

from qwt_optimizer.core import (
    Action,
    DEFAULT_SEED,
    RewardConfig,
    SCENARIOS,
    build_state,
    calculate_reward,
)


def main() -> None:
    reward_cfg = RewardConfig()
    scenario = SCENARIOS["medium"]

    # Example state to validate keys and reward calculation behavior.
    state = build_state(
        queue_length=12,
        average_waiting_time=4.5,
        open_counters=scenario.initial_open_counters,
        incoming_rate=scenario.base_arrival_rate,
    )

    previous_action = Action.DO_NOTHING
    action = Action.OPEN_COUNTER
    reward = calculate_reward(state, action, previous_action, reward_cfg)

    print("=== Phase 1 Smoke Test ===")
    print(f"Seed: {DEFAULT_SEED}")
    print(f"Scenario: {scenario.name}")
    print(f"Action mapping: close={Action.CLOSE_COUNTER}, hold={Action.DO_NOTHING}, open={Action.OPEN_COUNTER}")
    print(f"State: {state}")
    print(f"Reward: {reward:.4f}")


if __name__ == "__main__":
    main()
