import copy

from models.scenario import scenario_from_dict
from scenario_builders import BASE_SCENARIO, make_alert


def base_scenario(alerts, **changes):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    for key, value in changes.items():
        raw[key] = value
    return scenario_from_dict(raw)


def three_zone_raw(num_slots=50):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["damage_zones"] = [
        {"x_m": 1230.0, "y_m": 1000.0, "radius_m": 400.0},
        {"x_m": 1000.0, "y_m": 1400.0, "radius_m": 300.0},
        {"x_m": 100.0, "y_m": 100.0, "radius_m": 200.0},
    ]
    raw["sources"] = [
        {"id": "s00", "x_m": 1230.0, "y_m": 1000.0, "zone": 0},
        {"id": "s01", "x_m": 100.0, "y_m": 100.0, "zone": 2},
        {"id": "s02", "x_m": 1000.0, "y_m": 1500.0, "zone": 1},
        {"id": "s03", "x_m": 1100.0, "y_m": 1300.0, "zone": 1},
    ]
    raw["alerts"] = [
        make_alert("alert-0000", "s00", 0, 10, 100000),
        make_alert("alert-0001", "s02", 0, 20, 100000),
        make_alert("alert-0002", "s03", 5, 45, 100000),
    ]
    raw["time"]["num_slots"] = num_slots
    return raw


def hopeless_raw():
    """One ground candidate per alert; a hopeless 12 Mbit alert holds the 1 Mbit/s backhaul until its deadline."""
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["paths"] = {"k": 1, "max_ground": 1, "version": 1}
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 10, 12000000), make_alert("alert-0001", "s00", 0, 11, 1500000)]
    return raw


def bandwidth_raw():
    """Two access links share B_tot; alert-0000 is timely only with more than an equal share."""
    raw = copy.deepcopy(BASE_SCENARIO)
    for edge in raw["network"]["edges"]:
        edge["capacity_bps"] = 1e8
    raw["sources"].append({"id": "s02", "x_m": 1000.0, "y_m": 1430.0, "zone": 0})
    raw["time"]["num_slots"] = 40
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 5, 4500000), make_alert("alert-0001", "s02", 0, 40, 4000000)]
    return raw
