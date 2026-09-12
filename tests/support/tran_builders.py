import copy

from scenario_builders import BASE_SCENARIO, make_alert


def tran_raw(num_slots=40, uav_size=200000):
    """Ground-served source s00 and UAV-only source s02; alert-0001 and alert-0002 travel through the UAV to n1."""
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["sources"].append({"id": "s02", "x_m": 1500.0, "y_m": 1400.0, "zone": 0})
    raw["damage_zones"].append({"x_m": 1500.0, "y_m": 1400.0, "radius_m": 400.0})
    raw["time"]["num_slots"] = num_slots
    raw["alerts"] = [
        make_alert("alert-0000", "s00", 0, 20, 200000),
        make_alert("alert-0001", "s02", 0, 40, uav_size),
        make_alert("alert-0002", "s02", 5, 40, uav_size // 2),
    ]
    return raw
