from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Dict

DEFAULT_SEED = 42


class Action(IntEnum):
    """Discrete action mapping required by the project."""

    CLOSE_COUNTER = 0
    DO_NOTHING = 1
    OPEN_COUNTER = 2


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights for balancing service quality and operating cost."""

    waiting_time_weight: float = 1.0
    open_counter_weight: float = 0.35
    queue_length_weight: float = 0.2
    action_smoothness_weight: float = 0.05


@dataclass(frozen=True)
class ScenarioConfig:
    """Traffic and infrastructure parameters per scenario."""

    name: str
    episode_length: int
    min_open_counters: int
    max_open_counters: int
    initial_open_counters: int
    service_rate_per_counter: float
    base_arrival_rate: float
    arrival_rate_noise: float


SCENARIOS: Dict[str, ScenarioConfig] = {
    "easy": ScenarioConfig(
        name="easy",
        episode_length=250,
        min_open_counters=1,
        max_open_counters=6,
        initial_open_counters=2,
        service_rate_per_counter=3.0,
        base_arrival_rate=4.0,
        arrival_rate_noise=0.15,
    ),
    "medium": ScenarioConfig(
        name="medium",
        episode_length=300,
        min_open_counters=1,
        max_open_counters=8,
        initial_open_counters=2,
        service_rate_per_counter=2.5,
        base_arrival_rate=6.0,
        arrival_rate_noise=0.2,
    ),
    "hard": ScenarioConfig(
        name="hard",
        episode_length=350,
        min_open_counters=1,
        max_open_counters=10,
        initial_open_counters=3,
        service_rate_per_counter=2.2,
        base_arrival_rate=8.5,
        arrival_rate_noise=0.25,
    ),
}


def build_state(
    queue_length: float,
    average_waiting_time: float,
    open_counters: float,
    incoming_rate: float,
) -> Dict[str, float]:
    """Return a consistent state dictionary used by simulator and wrappers."""

    return {
        "queue_length": float(queue_length),
        "average_waiting_time": float(average_waiting_time),
        "open_counters": float(open_counters),
        "incoming_rate": float(incoming_rate),
    }


def calculate_reward(
    state: Dict[str, float],
    action: Action,
    previous_action: Action,
    reward_cfg: RewardConfig,
) -> float:
    """Negative-cost reward: lower waiting and fewer counters improves score."""

    waiting_penalty = reward_cfg.waiting_time_weight * state["average_waiting_time"]
    counter_penalty = reward_cfg.open_counter_weight * state["open_counters"]
    queue_penalty = reward_cfg.queue_length_weight * state["queue_length"]
    action_change_penalty = (
        reward_cfg.action_smoothness_weight * abs(int(action) - int(previous_action))
    )

    total_penalty = (
        waiting_penalty + counter_penalty + queue_penalty + action_change_penalty
    )
    return -float(total_penalty)


def scenario_to_dict(name: str) -> Dict[str, float | int | str]:
    """Helper for logging and debugging configs."""

    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {name}")
    return asdict(SCENARIOS[name])
