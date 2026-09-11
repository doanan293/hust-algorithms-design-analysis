BASE_SCENARIO = {
    "schema_version": 2,
    "scenario_id": "Tiny-r0",
    "split": "eval",
    "network": {
        "network_id": "Tiny",
        "nodes": [
            {"id": "n0", "x_m": 1000.0, "y_m": 1000.0},
            {"id": "n1", "x_m": 1200.0, "y_m": 1000.0},
            {"id": "n2", "x_m": 1000.0, "y_m": 1400.0},
            {"id": "n3", "x_m": 1700.0, "y_m": 1000.0},
        ],
        "edges": [
            {"source": "n0", "target": "n1", "failed": False, "capacity_bps": 1000000.0},
            {"source": "n0", "target": "n2", "failed": False, "capacity_bps": 1000000.0},
            {"source": "n1", "target": "n3", "failed": False, "capacity_bps": 1000000.0},
        ],
        "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
    },
    "center": "n0",
    "relays": ["n1", "n2", "n3"],
    "damage_zones": [{"x_m": 1230.0, "y_m": 1000.0, "radius_m": 400.0}],
    "sources": [
        {"id": "s00", "x_m": 1230.0, "y_m": 1000.0, "zone": 0},
        {"id": "s01", "x_m": 100.0, "y_m": 100.0, "zone": 0},
    ],
    "alerts": [
        {
            "id": "alert-0000",
            "source": "s00",
            "release_slot": 0,
            "deadline_slot": 10,
            "size_bits": 1000000,
            "size_ratio": 1.0,
            "provenance": "derived-from-sndlib-demand",
        }
    ],
    "uav": {
        "altitude_m": 100.0,
        "v_max_mps": 50.0,
        "start_xy_m": [1000.0, 1000.0],
        "end_xy_m": [1000.0, 1000.0],
        "total_power_w": 0.1,
        "max_active_downlinks": 2,
    },
    "time": {"slot_s": 1.0, "num_slots": 20, "access_fraction": 0.5},
    "spectrum": {"b_tot_hz": 1000000.0, "noise_psd_dbm_per_hz": -140.0},
    "channel": {
        "uav": {
            "plos_a": 9.61,
            "plos_b": 0.16,
            "nlos_attenuation": 0.2,
            "alpha_los": 2.2,
            "alpha_nlos": 3.3,
            "beta0_db": -50.0,
        },
        "ground": {"alpha": 2.8, "beta0_db": -50.0},
        "usable_rate_bps": 10000.0,
    },
    "source_power_w": 0.1,
    "paths": {"k": 3, "max_ground": 2, "version": 1},
    "provenance": {"scenario_seed": 7, "generator_version": 2},
}


def make_alert(alert_id, source, release, deadline, size):
    return {
        "id": alert_id,
        "source": source,
        "release_slot": release,
        "deadline_slot": deadline,
        "size_bits": size,
        "size_ratio": 1.0,
        "provenance": "derived-from-sndlib-demand",
    }


def make_alert(alert_id, source, release, deadline, size):
    return {
        "id": alert_id,
        "source": source,
        "release_slot": release,
        "deadline_slot": deadline,
        "size_bits": size,
        "size_ratio": 1.0,
        "provenance": "derived-from-sndlib-demand",
    }
