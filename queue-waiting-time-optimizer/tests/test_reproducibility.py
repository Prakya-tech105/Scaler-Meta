from qwt_optimizer.envs.queue_simulator import QueueSimulator


def test_same_seed_produces_same_initial_state_and_first_step() -> None:
    env_a = QueueSimulator(scenario_name="medium", seed=999)
    env_b = QueueSimulator(scenario_name="medium", seed=999)

    state_a = env_a.reset(seed=999)
    state_b = env_b.reset(seed=999)
    assert state_a == state_b

    step_a = env_a.step(1)
    step_b = env_b.step(1)

    assert step_a[0] == step_b[0]
    assert step_a[1] == step_b[1]
