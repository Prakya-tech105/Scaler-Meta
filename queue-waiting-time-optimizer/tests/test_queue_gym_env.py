import numpy as np

from qwt_optimizer.envs.queue_gym_env import QueueGymEnv


def test_reset_returns_valid_observation() -> None:
    env = QueueGymEnv(scenario_name="easy", seed=123)
    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert env.observation_space.contains(obs)
    assert "state_dict" in info


def test_step_returns_gymnasium_signature() -> None:
    env = QueueGymEnv(scenario_name="easy", seed=123)
    env.reset(seed=123)

    obs, reward, terminated, truncated, info = env.step(1)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "state_dict" in info
