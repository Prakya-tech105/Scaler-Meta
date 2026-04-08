from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from qwt_optimizer.core import (
    Action,
    RewardConfig,
    SCENARIOS,
    ScenarioConfig,
    build_state,
    calculate_reward,
)


@dataclass
class QueueSimulator:
    """Core queue dynamics without Gymnasium wrappers.

    This simulator is intentionally simple so behavior is easy to understand:
    - arrivals follow a Poisson process
    - total service capacity is proportional to open counters
    - queue evolves as previous + arrivals - served
    """

    scenario_name: str = "medium"
    seed: int = 42
    reward_config: RewardConfig = RewardConfig()

    def __post_init__(self) -> None:
        if self.scenario_name not in SCENARIOS:
            raise KeyError(f"Unknown scenario: {self.scenario_name}")

        self.scenario: ScenarioConfig = SCENARIOS[self.scenario_name]
        self.rng = np.random.default_rng(self.seed)

        self.step_count: int = 0
        self.queue_length: float = 0.0
        self.average_waiting_time: float = 0.0
        self.open_counters: int = self.scenario.initial_open_counters
        self.incoming_rate: float = self.scenario.base_arrival_rate
        self.previous_action: Action = Action.DO_NOTHING

    def reset(self, seed: int | None = None) -> Dict[str, float]:
        """Reset simulator state and return the initial state."""

        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.step_count = 0
        self.queue_length = 0.0
        self.average_waiting_time = 0.0
        self.open_counters = self.scenario.initial_open_counters
        self.incoming_rate = self._sample_next_incoming_rate()
        self.previous_action = Action.DO_NOTHING

        return self.state()

    def state(self) -> Dict[str, float]:
        """Current observable state used by policies."""

        return build_state(
            queue_length=self.queue_length,
            average_waiting_time=self.average_waiting_time,
            open_counters=self.open_counters,
            incoming_rate=self.incoming_rate,
        )

    def step(self, action: int | Action) -> tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """Advance one timestep.

        Returns (state, reward, done, info) similar to classic RL APIs.
        """

        action_enum = Action(int(action))

        if action_enum == Action.OPEN_COUNTER:
            self.open_counters += 1
        elif action_enum == Action.CLOSE_COUNTER:
            self.open_counters -= 1

        self.open_counters = int(
            np.clip(
                self.open_counters,
                self.scenario.min_open_counters,
                self.scenario.max_open_counters,
            )
        )

        arrivals = float(self.rng.poisson(lam=max(self.incoming_rate, 0.0)))
        capacity = float(self.open_counters * self.scenario.service_rate_per_counter)
        served = float(min(self.queue_length + arrivals, capacity))

        self.queue_length = max(0.0, self.queue_length + arrivals - served)

        instant_wait = self.queue_length / max(capacity, 1e-6)
        self.average_waiting_time = 0.8 * self.average_waiting_time + 0.2 * instant_wait

        reward = calculate_reward(
            state=self.state(),
            action=action_enum,
            previous_action=self.previous_action,
            reward_cfg=self.reward_config,
        )

        self.previous_action = action_enum
        self.step_count += 1
        done = self.step_count >= self.scenario.episode_length

        current_rate = self.incoming_rate
        self.incoming_rate = self._sample_next_incoming_rate()

        info: Dict[str, Any] = {
            "arrivals": arrivals,
            "served": served,
            "capacity": capacity,
            "step": self.step_count,
            "applied_incoming_rate": current_rate,
        }

        return self.state(), reward, done, info

    def _sample_next_incoming_rate(self) -> float:
        noise = self.rng.normal(loc=0.0, scale=self.scenario.arrival_rate_noise)
        rate = self.scenario.base_arrival_rate * (1.0 + noise)
        return float(max(0.1, rate))
