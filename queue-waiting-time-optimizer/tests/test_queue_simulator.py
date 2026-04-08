from qwt_optimizer.core import Action
from qwt_optimizer.envs.queue_simulator import QueueSimulator


def test_reset_returns_required_state_fields() -> None:
    env = QueueSimulator(scenario_name="easy", seed=7)
    state = env.reset()

    assert set(state.keys()) == {
        "queue_length",
        "average_waiting_time",
        "open_counters",
        "incoming_rate",
    }


def test_open_close_respects_counter_bounds() -> None:
    env = QueueSimulator(scenario_name="easy", seed=11)
    env.reset()

    for _ in range(20):
        env.step(Action.CLOSE_COUNTER)
    assert env.open_counters >= env.scenario.min_open_counters

    for _ in range(20):
        env.step(Action.OPEN_COUNTER)
    assert env.open_counters <= env.scenario.max_open_counters
