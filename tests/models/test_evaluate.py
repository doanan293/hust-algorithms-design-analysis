import math

import numpy as np
import pytest

from models.evaluate import EXPECTED, INFEASIBLE_KEY, REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory
from models.scenario import scenario_from_dict
from scenario_builders import make_alert


def test_mode_arguments_are_checked(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    with pytest.raises(ValueError):
        evaluate(scenario, plan, EXPECTED, (0,))
    with pytest.raises(ValueError):
        evaluate(scenario, plan, REALIZED)


def test_invalid_plan_is_infeasible_and_counted(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    counter = EvaluationCounter()
    plan = Plan(stationary_trajectory(scenario), {}, EqualSplitBacklogged())
    result = evaluate(scenario, plan, REALIZED, (0, 1, 2), counter=counter)
    assert result.feasible is False
    assert result.key == INFEASIBLE_KEY
    assert result.realizations == ()
    assert counter.calls == 1


def test_realized_results_are_reproducible(raw_scenario):
    raw_scenario["alerts"].append(make_alert("alert-0001", "s01", 0, 20, 20000))
    scenario = scenario_from_dict(raw_scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 0}, EqualSplitBacklogged())
    counter = EvaluationCounter()
    first = evaluate(scenario, plan, REALIZED, (0, 1), counter=counter, keep_ledger=True)
    second = evaluate(scenario, plan, REALIZED, (0, 1), counter=counter)
    assert counter.calls == 2
    assert first.feasible and first.key == second.key
    assert [r.delivered_bits for r in first.realizations] == [r.delivered_bits for r in second.realizations]
    assert first.realizations[0].ledger is not None and second.realizations[0].ledger is None
    assert first.connectivity_ratio_without_uav == 0.5
    assert first.scenario_sha256 == scenario.sha256 and first.paths_version == 1


def _out_and_back(scenario, target, hover_until):
    start = np.asarray(scenario.uav.start_xy_m)
    goal = np.asarray(target)
    steps = math.ceil(np.linalg.norm(goal - start) / (scenario.uav.v_max_mps * scenario.time.slot_s))
    outward = [start + (goal - start) * j / steps for j in range(steps + 1)]
    hover = [goal] * (hover_until - steps)
    back = [goal + (start - goal) * j / steps for j in range(1, steps + 1)]
    return np.asarray(outward + hover + back)


def test_flying_to_far_source_beats_hovering_in_expected_mode(raw_scenario):
    raw_scenario["time"]["num_slots"] = 80
    raw_scenario["alerts"] = [make_alert("alert-0001", "s01", 0, 80, 200000)]
    scenario = scenario_from_dict(raw_scenario)
    trajectory = _out_and_back(scenario, (100.0, 100.0), 54)
    assert trajectory.shape == (81, 2)
    flying = evaluate(scenario, Plan(trajectory, {"alert-0001": 0}, EqualSplitBacklogged()), EXPECTED)
    hovering = evaluate(
        scenario, Plan(stationary_trajectory(scenario), {"alert-0001": 0}, EqualSplitBacklogged()), EXPECTED
    )
    assert flying.feasible and hovering.feasible
    assert flying.timely_ratio == 1.0
    assert hovering.timely_ratio == 0.0
    assert flying.key > hovering.key


def test_random_valid_plans_never_break_invariants(raw_scenario):
    raw_scenario["alerts"] = [
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s00", 3, 20, 700000),
        make_alert("alert-0002", "s01", 1, 18, 50000),
        make_alert("alert-0003", "s01", 6, 20, 90000),
    ]
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    links = radio_links(candidates)
    rng = np.random.default_rng(20260911)
    for _ in range(15):
        steps = rng.uniform(-35.0, 35.0, size=(10, 2))
        outward = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        offsets = np.vstack([outward, outward[-2::-1]])
        trajectory = np.asarray(scenario.uav.start_xy_m) + offsets
        choice = {
            alert.id: None if rng.random() < 0.2 else int(rng.integers(0, len(candidates[alert.id])))
            for alert in scenario.alerts
        }
        if rng.random() < 0.5:
            policy = EqualSplitBacklogged()
        else:
            values = {link: rng.random(20) for link in links}
            for kind in ("access", "down"):
                group = [link for link in links if link[0] == kind]
                total = sum(values[link] for link in group)
                for link in group:
                    values[link] = values[link] / total * scenario.spectrum.b_tot_hz
            downlinks = [link for link in links if link[0] == "down"]
            for slot in range(20):
                keep = set(rng.choice(len(downlinks), size=min(2, len(downlinks)), replace=False))
                for index, link in enumerate(downlinks):
                    if index not in keep:
                        values[link][slot] = 0.0
            policy = StaticSchedule(values)
        result = evaluate(scenario, Plan(trajectory, choice, policy), REALIZED, (0, 1), candidates=candidates)
        assert result.feasible, result.violations
        for realization in result.realizations:
            for alert in scenario.alerts:
                assert realization.delivered_bits[alert.id] <= alert.size_bits + 1e-6
