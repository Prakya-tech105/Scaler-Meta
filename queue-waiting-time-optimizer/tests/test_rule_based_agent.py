from qwt_optimizer.agents.rule_based import RuleBasedAgent
from qwt_optimizer.core import Action, build_state


def test_rule_based_agent_close_when_queue_small() -> None:
    agent = RuleBasedAgent()
    state = build_state(3, 1.0, 2, 4.0)
    assert agent.act(state) == Action.CLOSE_COUNTER


def test_rule_based_agent_hold_when_queue_medium() -> None:
    agent = RuleBasedAgent()
    state = build_state(10, 2.0, 2, 6.0)
    assert agent.act(state) == Action.DO_NOTHING


def test_rule_based_agent_open_when_queue_large() -> None:
    agent = RuleBasedAgent()
    state = build_state(20, 6.0, 3, 9.0)
    assert agent.act(state) == Action.OPEN_COUNTER
