from __future__ import annotations

from qwt_optimizer.agents.rule_based import RuleBasedAgent, RuleBasedThresholds
from qwt_optimizer.core import build_state


def main() -> None:
    agent = RuleBasedAgent(
        RuleBasedThresholds(open_if_queue_at_least=15.0, close_if_queue_at_most=5.0)
    )

    sample_states = [
        build_state(
            queue_length=2,
            average_waiting_time=0.8,
            open_counters=2,
            incoming_rate=4.0,
        ),
        build_state(
            queue_length=9,
            average_waiting_time=2.2,
            open_counters=2,
            incoming_rate=6.0,
        ),
        build_state(
            queue_length=22,
            average_waiting_time=6.7,
            open_counters=3,
            incoming_rate=9.5,
        ),
    ]

    print("=== Phase 2 Baseline Demo ===")
    for i, state in enumerate(sample_states, start=1):
        action = agent.act(state)
        print(
            f"Sample {i}: queue={state['queue_length']:.1f}, "
            f"wait={state['average_waiting_time']:.1f}, action={int(action)}"
        )


if __name__ == "__main__":
    main()
