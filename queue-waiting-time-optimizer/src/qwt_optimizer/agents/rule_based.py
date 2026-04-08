from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from qwt_optimizer.core import Action


@dataclass(frozen=True)
class RuleBasedThresholds:
    """Thresholds used by the baseline policy."""

    open_if_queue_at_least: float = 15.0
    close_if_queue_at_most: float = 5.0


class RuleBasedAgent:
    """Simple heuristic policy for queue counter control.

    Rules:
    - If queue is large, open a counter.
    - If queue is small, close a counter.
    - Otherwise do nothing.
    """

    def __init__(self, thresholds: RuleBasedThresholds | None = None) -> None:
        self.thresholds = thresholds or RuleBasedThresholds()

    def act(self, state: Mapping[str, float]) -> Action:
        queue_length = float(state["queue_length"])

        if queue_length >= self.thresholds.open_if_queue_at_least:
            return Action.OPEN_COUNTER

        if queue_length <= self.thresholds.close_if_queue_at_most:
            return Action.CLOSE_COUNTER

        return Action.DO_NOTHING
