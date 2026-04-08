from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qwt_optimizer.core import RewardConfig
from qwt_optimizer.envs.queue_simulator import QueueSimulator


class QueueGymEnv(gym.Env[np.ndarray, int]):
    """Gymnasium wrapper around the queue simulator for RL training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_name: str = "medium",
        seed: int = 42,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()

        self.simulator = QueueSimulator(
            scenario_name=scenario_name,
            seed=seed,
            reward_config=reward_config or RewardConfig(),
        )

        self.action_space = spaces.Discrete(3)

        max_counters = float(self.simulator.scenario.max_open_counters)
        max_rate = float(self.simulator.scenario.base_arrival_rate * 3.0)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e4, 1e4, max_counters, max_rate], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        state_dict = self.simulator.reset(seed=seed)
        obs = self._state_to_obs(state_dict)
        info = {"state_dict": state_dict}
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state_dict, reward, done, info = self.simulator.step(action)
        obs = self._state_to_obs(state_dict)

        terminated = bool(done)
        truncated = False

        info_out: dict[str, Any] = dict(info)
        info_out["state_dict"] = state_dict

        return obs, float(reward), terminated, truncated, info_out

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    @staticmethod
    def _state_to_obs(state_dict: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                state_dict["queue_length"],
                state_dict["average_waiting_time"],
                state_dict["open_counters"],
                state_dict["incoming_rate"],
            ],
            dtype=np.float32,
        )
