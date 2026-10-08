"""Congestion and residual capacity of the incumbent plan, read from the ledgers of its design realizations."""

from collections import defaultdict
from dataclasses import dataclass

from models.channel import LinkId
from models.evaluate import EvaluationResult
from models.ledger import Ledger
from models.scenario import Alert, Scenario

CONGESTION_TOLERANCE = 1e-6
RESIDUAL_FLOOR = 1e-3


def congested_slots(ledger: Ledger) -> dict[LinkId, set[int]]:
    """Slots in which a link carried its whole recorded capacity (within 1e-6 relative)."""
    carried: dict[tuple[int, LinkId], float] = defaultdict(float)
    for slot, link, _, bits in ledger.transfers:
        carried[(slot, link)] += bits
    congested: dict[LinkId, set[int]] = defaultdict(set)
    for slot, link, capacity in ledger.capacity_bits:
        if carried.get((slot, link), 0.0) >= capacity * (1.0 - CONGESTION_TOLERANCE):
            congested[link].add(slot)
    return congested


def alert_window(scenario: Scenario, alert: Alert) -> range:
    """Slots in which the alert can still be delivered, clamped to the horizon."""
    return range(alert.release_slot, min(alert.deadline_slot, scenario.time.num_slots))


@dataclass(frozen=True)
class LedgerSummary:
    """Per-realization congested slots and carried bits per (link, slot) summed over the realizations."""

    realization_count: int
    congested: tuple[dict[LinkId, set[int]], ...]
    carried_bits: dict[tuple[LinkId, int], float]

    @classmethod
    def from_result(cls, result: EvaluationResult) -> "LedgerSummary":
        congested = []
        carried: dict[tuple[LinkId, int], float] = defaultdict(float)
        for item in result.realizations:
            if item.ledger is None:
                raise ValueError("a ledger summary needs ledgers; score the plan with keep_ledger=True")
            congested.append(congested_slots(item.ledger))
            for slot, link, _, bits in item.ledger.transfers:
                carried[(link, slot)] += bits
        return cls(len(result.realizations), tuple(congested), dict(carried))

    def congestion_count(self, link: LinkId, window: range) -> int:
        """Congested slots of the link inside the window, summed over the realizations."""
        return sum(len(slots.get(link, set()).intersection(window)) for slots in self.congested)

    def congested_realizations(self, link: LinkId, slot: int) -> int:
        return sum(slot in slots.get(link, ()) for slots in self.congested)

    def carried_bps(self, link: LinkId, window: range, slot_s: float) -> float:
        """Mean bits per second the link carried over the window and the realizations."""
        if len(window) == 0 or self.realization_count == 0:
            return 0.0
        total = sum(self.carried_bits.get((link, slot), 0.0) for slot in window)
        return total / (self.realization_count * len(window) * slot_s)


def residual_bps(scenario: Scenario, summary: LedgerSummary, link: LinkId, window: range) -> float:
    """Backhaul capacity left on the link during the window, floored at a thousandth of the capacity."""
    capacity = scenario.backhaul_capacity_bps(link[1], link[2])
    return max(capacity - summary.carried_bps(link, window, scenario.time.slot_s), capacity * RESIDUAL_FLOOR)
