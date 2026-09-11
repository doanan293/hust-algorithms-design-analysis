from collections import defaultdict
import math
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkChannel, LinkId
from .ledger import Ledger
from .paths import CandidatePath
from .plan import Plan, StaticSchedule
from .scenario import Scenario

ABSOLUTE_TOLERANCE_BITS = 1e-6
RELATIVE_TOLERANCE = 1e-9


class SimulatorInvariantError(RuntimeError):
    def __init__(self, violations: list[str]) -> None:
        super().__init__("simulator invariant violated: " + "; ".join(violations[:5]))
        self.violations = tuple(violations)


def check_ledger(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    plan: Plan,
    channels: Mapping[LinkId, LinkChannel],
    ledger: Ledger,
    delivered_bits: Mapping[str, float] | None = None,
) -> list[str]:
    """Replay the ledger without simulator code and list every broken invariant."""
    paths = {
        alert.id: None if plan.path_choice[alert.id] is None else candidates[alert.id][plan.path_choice[alert.id]]
        for alert in scenario.alerts
    }
    violations: list[str] = []
    bandwidth = _check_bandwidth(scenario, plan, ledger, violations)
    capacity = _check_capacity(scenario, channels, ledger, bandwidth, violations)
    flows = _check_transfers(ledger, paths, capacity, violations)
    drops: dict[tuple[str, int, int], float] = defaultdict(float)
    for slot, alert_id, holder, bits in ledger.drops:
        path = paths.get(alert_id)
        if path is None or holder not in path.holders:
            violations.append(f"slot {slot}: alert {alert_id} drops bits at {holder} outside its chosen path")
            continue
        drops[(alert_id, slot, path.holders.index(holder))] += bits
    replayed = _replay_buffers(scenario, paths, flows, drops, violations)
    if delivered_bits is not None:
        for alert in scenario.alerts:
            if not math.isclose(delivered_bits[alert.id], replayed[alert.id], abs_tol=ABSOLUTE_TOLERANCE_BITS):
                violations.append(
                    f"alert {alert.id}: reported {delivered_bits[alert.id]:.3f} delivered bits, "
                    f"ledger gives {replayed[alert.id]:.3f}"
                )
    return violations


def _check_bandwidth(
    scenario: Scenario, plan: Plan, ledger: Ledger, violations: list[str]
) -> dict[tuple[int, LinkId], float]:
    bandwidth: dict[tuple[int, LinkId], float] = {}
    totals: dict[tuple[int, str], float] = defaultdict(float)
    active_downlinks: dict[int, int] = defaultdict(int)
    for slot, link, hz in ledger.bandwidth:
        bandwidth[(slot, link)] = hz
        if link[0] not in (ACCESS, DOWNLINK):
            violations.append(f"slot {slot}: bandwidth assigned to non-radio link {link}")
            continue
        if hz < 0.0:
            violations.append(f"slot {slot}: negative bandwidth on {link}")
        totals[(slot, link[0])] += hz
        if link[0] == DOWNLINK and hz > 0.0:
            active_downlinks[slot] += 1
        if isinstance(plan.bandwidth, StaticSchedule) and not math.isclose(
            hz, plan.bandwidth.at(link, slot), rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-9
        ):
            violations.append(f"slot {slot}: bandwidth on {link} differs from the static schedule")
    limit = scenario.spectrum.b_tot_hz * (1.0 + RELATIVE_TOLERANCE)
    for (slot, kind), total in sorted(totals.items()):
        if total > limit:
            violations.append(f"slot {slot}: {kind} bandwidth {total:.1f} Hz exceeds B_tot")
    for slot, count in sorted(active_downlinks.items()):
        if count > scenario.uav.max_active_downlinks:
            violations.append(f"slot {slot}: {count} active downlinks exceed the cap")
    return bandwidth


def _check_capacity(
    scenario: Scenario,
    channels: Mapping[LinkId, LinkChannel],
    ledger: Ledger,
    bandwidth: Mapping[tuple[int, LinkId], float],
    violations: list[str],
) -> dict[tuple[int, LinkId], float]:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    capacity: dict[tuple[int, LinkId], float] = {}
    for slot, link, bits in ledger.capacity_bits:
        capacity[(slot, link)] = bits
        if link[0] == BACKHAUL:
            expected = slot_s * scenario.backhaul_capacity_bps(link[1], link[2])
        else:
            duration = access_s if link[0] == ACCESS else slot_s - access_s
            expected = duration * channels[link].rate_bps(slot, bandwidth.get((slot, link), 0.0))
        if not math.isclose(bits, expected, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE_BITS):
            violations.append(f"slot {slot}: recorded capacity {bits:.3f} of {link} differs from {expected:.3f}")
    return capacity


def _check_transfers(
    ledger: Ledger,
    paths: Mapping[str, CandidatePath | None],
    capacity: Mapping[tuple[int, LinkId], float],
    violations: list[str],
) -> dict[tuple[str, int, int], float]:
    load: dict[tuple[int, LinkId], float] = defaultdict(float)
    flows: dict[tuple[str, int, int], float] = defaultdict(float)
    for slot, link, alert_id, bits in ledger.transfers:
        path = paths.get(alert_id)
        if path is None or link not in path.links:
            violations.append(f"slot {slot}: alert {alert_id} uses {link} outside its chosen path")
            continue
        if bits <= 0.0:
            violations.append(f"slot {slot}: alert {alert_id} records non-positive transfer on {link}")
        load[(slot, link)] += bits
        flows[(alert_id, slot, path.links.index(link))] += bits
    for (slot, link), bits in sorted(load.items()):
        limit = capacity.get((slot, link))
        if limit is None:
            violations.append(f"slot {slot}: {link} carries bits without a recorded capacity")
        elif bits > limit * (1.0 + RELATIVE_TOLERANCE) + ABSOLUTE_TOLERANCE_BITS:
            violations.append(f"slot {slot}: {link} carries {bits:.3f} bits over capacity {limit:.3f}")
    return flows


def _replay_buffers(
    scenario: Scenario,
    paths: Mapping[str, CandidatePath | None],
    flows: Mapping[tuple[str, int, int], float],
    drops: Mapping[tuple[str, int, int], float],
    violations: list[str],
) -> dict[str, float]:
    delivered: dict[str, float] = {}
    for alert in scenario.alerts:
        delivered[alert.id] = 0.0
        path = paths[alert.id]
        if path is None:
            continue
        hops = len(path.links)
        buffer = [0.0] * hops
        for slot in range(scenario.time.num_slots):
            if slot == alert.release_slot:
                buffer[0] += float(alert.size_bits)
            for hop in range(hops):
                dropped = drops.get((alert.id, slot, hop), 0.0)
                if dropped > 0.0 and slot < alert.deadline_slot:
                    violations.append(f"slot {slot}: alert {alert.id} dropped before its deadline")
                if dropped > buffer[hop] + ABSOLUTE_TOLERANCE_BITS:
                    violations.append(
                        f"slot {slot}: alert {alert.id} drops {dropped:.3f} bits at hop {hop} "
                        f"holding only {buffer[hop]:.3f}"
                    )
                buffer[hop] -= dropped
                if slot >= alert.deadline_slot and buffer[hop] > ABSOLUTE_TOLERANCE_BITS:
                    violations.append(f"slot {slot}: alert {alert.id} keeps expired bits at hop {hop}")
            incoming = [0.0] * (hops + 1)
            for hop in range(hops):
                sent = flows.get((alert.id, slot, hop), 0.0)
                if sent > 0.0 and slot < alert.release_slot:
                    violations.append(f"slot {slot}: alert {alert.id} transmits before release")
                if sent > buffer[hop] + ABSOLUTE_TOLERANCE_BITS:
                    violations.append(
                        f"slot {slot}: alert {alert.id} sends {sent:.3f} bits at hop {hop} "
                        f"holding only {buffer[hop]:.3f}"
                    )
                buffer[hop] -= sent
                incoming[hop + 1] += sent
            for hop in range(1, hops):
                buffer[hop] += incoming[hop]
            delivered[alert.id] += incoming[hops]
        if delivered[alert.id] > alert.size_bits + ABSOLUTE_TOLERANCE_BITS:
            violations.append(f"alert {alert.id}: delivered {delivered[alert.id]:.3f} bits exceed its size")
    return delivered
