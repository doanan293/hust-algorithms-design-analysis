"""Acceptance keys of search methods; a larger key is a better plan."""

from typing import Callable

from models.evaluate import EvaluationResult
from models.scenario import Scenario

Key = tuple
Objective = Callable[[Scenario, EvaluationResult], Key]
FRACTION_DIGITS = 9


def lexicographic_key(scenario: Scenario, result: EvaluationResult) -> Key:
    """(timely_count, connected_count, -round(shortfall, 9)) summed over realizations."""
    return result.key


def leximin_key(scenario: Scenario, result: EvaluationResult) -> Key:
    """Mean delivered fraction per alert sorted ascending, then timely_count and connected_count."""
    count = len(result.realizations)
    fractions = sorted(
        round(
            sum(min(item.delivered_bits[alert.id] / alert.size_bits, 1.0) for item in result.realizations) / count,
            FRACTION_DIGITS,
        )
        for alert in scenario.alerts
    )
    return (*fractions, result.key[0], result.key[1])


OBJECTIVES: dict[str, Objective] = {"lex": lexicographic_key, "leximin": leximin_key}
