import csv
import importlib.util
import json
from pathlib import Path
import sys

from runner.provenance import source_hash

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
SPEC = importlib.util.spec_from_file_location("report_tables_extensions", EXPERIMENTS / "report_tables_extensions.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

SUMMARY_COLUMNS = ["set_id", "method", "scenario_count", "timely_ratio_mean", "timely_ratio_ci_low", "timely_ratio_ci_high", "conn_ratio_mean", "gap_mean", "planning_runtime_s_mean"]
ROW_COLUMNS = ["set_id", "scenario_id", "method", "realization_id", "alert_count", "planning_runtime_s", "evaluation_runtime_s"]


def _write(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _experiment(folder: Path, values: dict, source: str, extra_manifest=None):
    _write(folder / "summary.csv", SUMMARY_COLUMNS, [
        {"set_id": set_id, "method": method, "scenario_count": "30", "timely_ratio_mean": timely, "timely_ratio_ci_low": timely - 0.05,
         "timely_ratio_ci_high": timely + 0.05, "conn_ratio_mean": 1.0, "gap_mean": 0.25, "planning_runtime_s_mean": 120.0}
        for (set_id, method), timely in values.items()
    ])
    _write(folder / "method_realizations.csv", ROW_COLUMNS, [
        {"set_id": set_id, "scenario_id": "Agis-r0", "method": method, "realization_id": "0", "alert_count": "40", "planning_runtime_s": "3600", "evaluation_runtime_s": "0"}
        for (set_id, method) in values
    ])
    _write(folder / "bounds.csv", ["status", "runtime_s"], [{"status": "optimal", "runtime_s": "1800"}])
    manifest = {"status": "complete", "source_hash": source, "task_count": 10, "environment": {"cvxpy": "1.9.2", "clarabel": "0.11.1"}}
    (folder / "manifest.json").write_text(json.dumps({**manifest, **(extra_manifest or {})}), encoding="utf-8")


def _results(root: Path, source: str):
    phase2, phase1 = root / "phase2", root / "phase1"
    sets = ("v0", "v0-bh50")
    _experiment(phase2 / "main", {(s, m): v for s in sets for m, v in (("B1", 0.5), ("B2", 0.55), ("P", 0.55))}, source)
    for name in ("budget", "sensitivity", "design"):
        _experiment(phase2 / name, {("v0", "P"): 0.5}, source)
    _experiment(phase1, {("v0", "B1"): 0.5}, source)
    _write(phase1 / "milp_crosscheck.csv", ["milp_runtime_s"], [{"milp_runtime_s": "3600"}])
    _experiment(phase2 / "literature", {(s, m): v for s in sets for m, v in (("B1", 0.5), ("TRAN", 0.53))}, source)
    _experiment(phase2 / "case_study", {("rescuenet", m): v for m, v in (("B0", 0.3), ("B1", 0.4), ("B2", 0.45), ("P", 0.45), ("TRAN", 0.44))}, source)
    _write(phase2 / "statistics_literature.csv", ["set_id", "comparison", "mean_difference", "ci_low", "ci_high", "rank_biserial", "p_value", "p_holm", "topology_count"], [
        {"set_id": s, "comparison": c, "mean_difference": 0.02, "ci_low": -0.01, "ci_high": 0.05, "rank_biserial": 0.4, "p_value": 0.36, "p_holm": 0.72, "topology_count": 10}
        for s in sets for c in ("P - TRAN", "B2 - TRAN")
    ])
    pilot = phase2 / "tran_pilot"
    pilot.mkdir(parents=True)
    (pilot / "choice.json").write_text(json.dumps({"theta": 1 / 3, "mu": 10.0, "block_slots": 1, "median_runtime_step1_s": 134.12, "scenarios": ["A-r0", "B-r0"],
                                                   "mean_timely_ratio": 0.5436, "better_than_initial_count": 2}), encoding="utf-8")
    run_columns = ["step", "scenario_id", "theta", "mu", "block_slots", "planning_runtime_s", "timely_ratio"]
    _write(pilot / "runs.csv", run_columns, [
        {"step": "runtime", "scenario_id": "A-r0", "theta": 0.5, "mu": 1.0, "block_slots": 1, "planning_runtime_s": 1800, "timely_ratio": 0.5},
        {"step": "grid", "scenario_id": "A-r0", "theta": 0.5, "mu": 1.0, "block_slots": 1, "planning_runtime_s": 1800, "timely_ratio": 0.5},
        {"step": "grid", "scenario_id": "A-r0", "theta": 0.3333333333333333, "mu": 10.0, "block_slots": 1, "planning_runtime_s": 1800, "timely_ratio": 0.6},
    ])
    _write(pilot / "channel_fit.csv", ["scenario_id", "beta", "alpha", "r_squared", "max_log_error"], [
        {"scenario_id": "A-r0", "beta": 0.002, "alpha": 3.35, "r_squared": 0.9937, "max_log_error": 0.5},
        {"scenario_id": "B-r0", "beta": 0.003, "alpha": 3.42, "r_squared": 0.9947, "max_log_error": 0.4},
    ])
    trace = phase2 / "case_study" / "trace"
    _write(trace / "targets.csv", ["source_id", "target_id", "x_m", "y_m", "damage_class", "zone"], [
        {"source_id": "s00", "target_id": "1-r01", "x_m": 1000, "y_m": 500, "damage_class": 4, "zone": 0}])
    _write(trace / "nodes.csv", ["node_id", "x_m", "y_m", "role"], [
        {"node_id": "n0", "x_m": 0, "y_m": 0, "role": "center"}, {"node_id": "n1", "x_m": 2000, "y_m": 0, "role": "relay"}])
    _write(trace / "edges.csv", ["source", "target", "source_x_m", "source_y_m", "target_x_m", "target_y_m", "failed"], [
        {"source": "n0", "target": "n1", "source_x_m": 0, "source_y_m": 0, "target_x_m": 2000, "target_y_m": 0, "failed": 1}])
    _write(trace / "trajectories.csv", ["method", "slot", "x_m", "y_m"], [
        {"method": method, "slot": slot, "x_m": 100 * slot, "y_m": 0} for method in ("B1", "P", "TRAN") for slot in range(3)])
    _write(trace / "progress.csv", ["method", "slot", "mean_timely_alerts"], [
        {"method": method, "slot": slot, "mean_timely_alerts": 10 * slot} for method in ("B1", "P", "TRAN") for slot in range(2)])
    (trace / "trace.json").write_text(json.dumps({"scenario_id": "Agis-r0", "methods": {m: {"timely_ratio_trace": 0.4} for m in ("B1", "P", "TRAN")}}), encoding="utf-8")
    manifests = root / "manifests"
    (manifests).mkdir()
    (manifests / "rescuenet_targets.json").write_text(json.dumps({
        "images": [{"pair_id": "1", "gsd_m_per_px": 0.0196, "focal_source": "calibrated", "targets": 2}, {"pair_id": "2", "gsd_m_per_px": 0.0246, "focal_source": "fallback", "targets": 0}],
        "region_count": 3, "target_count": 2, "dropped_regions": [], "merged_pairs": [{"distance_m": 2.7}, {"distance_m": 8.2}], "normalization": {"scale": 0.693323}}), encoding="utf-8")
    for set_id, fraction in (("rescuenet", 0.376), ("v0", 0.509)):
        _write(manifests / f"scenarios_{set_id}.csv", ["scenario_id", "split", "ground_connected_source_fraction"], [
            {"scenario_id": "Agis-r0", "split": "eval", "ground_connected_source_fraction": fraction},
            {"scenario_id": "Dev-r0", "split": "dev", "ground_connected_source_fraction": 0.0}])
    return phase2, phase1, manifests


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_extension_report_files_follow_the_results(tmp_path: Path):
    phase2, phase1, manifests = _results(tmp_path, source_hash())
    output = tmp_path / "data"
    arguments = ["--results", str(phase2), "--phase1", str(phase1), "--manifests", str(manifests), "--output", str(output)]
    assert report.main(arguments) == 0
    assert _lines(output / "ext_literature.csv")[:2] == ["set;backhaul;method;timely;ci;gap;planning",
                                                        "v0;\\qty{1000}{\\kilo\\bit\\per\\second};B1;\\num{0.500};[\\num{0.450}, \\num{0.550}];\\num{0.250};\\num{120.0}"]
    assert _lines(output / "ext_literature.csv")[4].split(";")[2:4] == ["TRAN", "\\num{0.530}"]
    assert _lines(output / "ext_literature_statistics.csv")[1] == "v0;P $-$ TRAN;\\num{0.020};[\\num{-0.010}, \\num{0.050}];\\num{0.40};\\num{0.3600};\\num{0.7200}"
    assert _lines(output / "ext_case_study.csv")[5] == "TRAN;\\num{0.440};[\\num{0.390}, \\num{0.490}];\\num{1.000};\\num{0.250};\\num{120.0}"
    assert _lines(output / "ext_quality_points.csv") == ["method,planning,timely", "B1,120.000,0.500000", "TRAN,120.000,0.530000"]
    assert _lines(output / "ext_map_targets.csv") == ["x,y,class", "1.0000,0.5000,4"]
    assert _lines(output / "ext_map_edges_alive.csv") == ["x,y", "nan,nan"]
    assert _lines(output / "ext_map_edges_failed.csv") == ["x,y", "0.0000,0.0000", "2.0000,0.0000", "nan,nan"]
    assert _lines(output / "ext_map_trajectory_tran.csv") == ["x,y", "0.0000,0.0000", "0.1000,0.0000", "0.2000,0.0000"]
    assert _lines(output / "ext_progress.csv") == ["slot,B1,P,TRAN", "0,0.000000,0.000000,0.000000", "1,0.250000,0.250000,0.250000"]
    compute = [line.split(";") for line in _lines(output / "ext_compute.csv")]
    assert [row[0] for row in compute[1:]][5] == "Pilot TRAN trên tập phát triển" and compute[6][1:] == ["2", "\\num{1.0}"]
    assert compute[1][1:] == ["10", "\\num{2.5}"] and len(compute) == 9
    macros = (output / "ext_results.tex").read_text(encoding="utf-8")
    for expected in (
        "\\newcommand{\\ResultExtTimelyVzeroTRAN}{\\num{0.530}}", "\\newcommand{\\ResultExtHolmVzeroBhfivezeroPTran}{\\num{0.7200}}",
        "\\newcommand{\\ResultExtPilotTheta}{\\num{0.333}}", "\\newcommand{\\ResultExtPilotMu}{\\num{10}}", "\\newcommand{\\ResultExtPilotGridSpread}{\\num{0.100}}",
        "\\newcommand{\\ResultExtCaseTimelyTRAN}{\\num{0.440}}", "\\newcommand{\\ResultExtMergeDistanceMax}{\\num{8.2}}",
        "\\newcommand{\\ResultExtGsdMax}{\\num{2.46}}", "\\newcommand{\\ResultExtFallbackPairs}{2}", "\\newcommand{\\ResultExtGroundFractionRescuenet}{\\num{0.376}}",
        "\\newcommand{\\ResultExtTraceScenario}{\\texttt{Agis-r0}}", "\\newcommand{\\ResultExtComputeHours}{\\num{24.5}}", "\\newcommand{\\ResultExtCvxpyVersion}{1.9.2}",
    ):
        assert expected in macros, expected


def test_extension_report_tables_refuse_results_from_other_source_code(tmp_path: Path, capsys):
    phase2, phase1, manifests = _results(tmp_path, source_hash())
    manifest = json.loads((phase2 / "case_study" / "manifest.json").read_text(encoding="utf-8"))
    (phase2 / "case_study" / "manifest.json").write_text(json.dumps({**manifest, "source_hash": "0" * 64}), encoding="utf-8")
    output = tmp_path / "data"
    assert report.main(["--results", str(phase2), "--phase1", str(phase1), "--manifests", str(manifests), "--output", str(output)]) == 1
    assert "case_study: the results were produced by different source code" in capsys.readouterr().err and not output.exists()
