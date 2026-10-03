"""Write report CSVs and result macros for docs/report from a completed Phase 1 run."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

import yaml

from runner.provenance import result_provenance_problem

DIGIT_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
MILP_STATUS = {"optimal": "optimal", "time_limit": "time limit"}


def num(value: float, digits: int = 3) -> str:
    return f"\\num{{{float(value):.{digits}f}}}"


def macro_name(*parts: str) -> str:
    words = []
    for part in parts:
        for piece in part.replace("_", "-").split("-"):
            spelled = "".join(DIGIT_WORDS.get(char, char) for char in piece)
            words.append(spelled[:1].upper() + spelled[1:])
    return "Result" + "".join(words)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_semicolon(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    lines = [";".join(header)] + [";".join(row) for row in rows]
    for line in lines[1:]:
        if line.count(";") != len(header) - 1:
            raise ValueError(f"{path.name}: a cell contains a semicolon: {line}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def kbit(bits_per_second: float) -> str:
    return f"\\qty{{{bits_per_second / 1000:g}}}{{\\kilo\\bit\\per\\second}}"


def params_rows(scenario: Mapping, stressed: Mapping, experiment: Mapping, data_profile: Mapping) -> list[list[str]]:
    topology, workload, failures = scenario["topology"], scenario["workload"], scenario["failures"]
    time, uav, spectrum, channel = scenario["time"], scenario["uav"], scenario["spectrum"], scenario["channel"]
    span = experiment["realization_ids"]
    return [
        ["Simulation area side", "$L_{\\mathrm{box}}$", f"\\qty{{{data_profile['box_size_m']:g}}}{{\\metre}}"],
        ["Evaluation / development topologies", "---", f"{topology['strata'] - topology['dev_count']} / {topology['dev_count']}"],
        ["Replicates per topology (evaluation / development)", "---", f"{topology['eval_replicates']} / {topology['dev_replicates']}"],
        ["Channel realizations", "$M$", str(span["stop"] - span["start"])],
        ["Slot length / number of slots", "$\\Delta$, $N$", f"\\qty{{{time['slot_s']:g}}}{{\\second}} / {time['num_slots']}"],
        ["Access-phase fraction", "$\\tau$", num(time["access_fraction"], 1)],
        ["UAV altitude / maximum speed", "$H$, $V_{\\max}$", f"\\qty{{{uav['altitude_m']:g}}}{{\\metre}} / \\qty{{{uav['v_max_mps']:g}}}{{\\metre\\per\\second}}"],
        ["UAV power / concurrent downlinks", "$P_U$", f"\\qty{{{uav['total_power_w']:g}}}{{\\watt}} / {uav['max_active_downlinks']}"],
        ["Source power", "$p_s$", f"\\qty{{{scenario['source_power_w']:g}}}{{\\watt}}"],
        ["Bandwidth per phase", "$B_{\\mathrm{tot}}$", f"\\qty{{{spectrum['b_tot_hz'] / 1e6:g}}}{{\\mega\\hertz}}"],
        ["Noise power spectral density", "$N_0$", f"\\qty{{{spectrum['noise_psd_dbm_per_hz']:g}}}{{\\dBm\\per\\hertz}}"],
        ["PrLoS parameters", "$(a, b)$", f"({num(channel['uav']['plos_a'], 2)}, {num(channel['uav']['plos_b'], 2)})"],
        ["NLoS attenuation / LoS, NLoS exponents", "$\\mu$, $\\alpha_L$, $\\alpha_N$", f"{num(channel['uav']['nlos_attenuation'], 1)} / {num(channel['uav']['alpha_los'], 1)}, {num(channel['uav']['alpha_nlos'], 1)}"],
        ["Reference gain", "$\\beta_0$", f"\\qty{{{channel['uav']['beta0_db']:g}}}{{\\dB}}"],
        ["Ground path-loss exponent", "$\\alpha_G$", num(channel["ground"]["alpha"], 1)],
        ["Usable-rate threshold", "$R_{\\min}$", kbit(channel["usable_rate_bps"])],
        ["Candidate paths (max.\\ ground paths)", "$K$", f"{scenario['paths']['k']} ({scenario['paths']['max_ground']})"],
        ["Damage zones: count / separation / radius", "---", f"{workload['zone_count']} / \\qty{{{workload['zone_min_separation_m']:g}}}{{\\metre}} / \\qty{{{workload['zone_radius_m']:g}}}{{\\metre}}"],
        ["Sources / position std.\\ dev.", "$|S|$, $\\sigma$", f"{workload['source_count']} / \\qty{{{workload['source_sigma_m']:g}}}{{\\metre}}"],
        ["Alerts / deadline window", "$|A|$, $d_a-r_a$", f"{workload['alert_count']} / {workload['deadline_min_slots']}--{workload['deadline_max_slots']} slots"],
        ["Reference size (clip range, instance)", "$L_{\\mathrm{ref}}$", f"\\qty{{{workload['size_ref_bits'] / 1000:g}}}{{\\kilo\\bit}} ([{num(workload['size_ratio_min'], 2)}, {num(workload['size_ratio_max'], 0)}], \\texttt{{{workload['sndlib_instance']}}})"],
        ["Edge failure prob.\\ inside / outside zones", "$p_{\\mathrm{in}}$, $p_{\\mathrm{out}}$", f"{num(failures['p_in'], 2)} / {num(failures['p_out'], 2)}"],
        ["Backhaul capacity (v0 / v0-bh50)", "$C_e$", f"{kbit(scenario['backhaul_capacity_bps'])} / {kbit(stressed['backhaul_capacity_bps'])}"],
        ["LP tangents / bootstrap resamples", "---", f"{experiment['bounds']['tangents']} / {experiment['bootstrap']['resamples']}"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report-tables")
    parser.add_argument("--results", type=Path, default=Path("results/phase1"))
    parser.add_argument("--experiment", type=Path, default=Path("configs/experiments/phase1.yaml"))
    parser.add_argument("--scenario-config", type=Path, default=Path("configs/scenarios/v0.yaml"))
    parser.add_argument("--stressed-config", type=Path, default=Path("configs/scenarios/v0-bh50.yaml"))
    parser.add_argument("--data-profile", type=Path, default=Path("configs/data/paper.yaml"))
    parser.add_argument("--output", type=Path, default=Path("docs/report/data"))
    args = parser.parse_args(argv)

    manifest_path = args.results / "manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}; run experiments/run_phase1.py first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result_provenance_problem(manifest):
        print("results are incomplete or were produced by different source code; rerun the experiment", file=sys.stderr)
        return 1

    load = lambda path: yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario, stressed, experiment = load(args.scenario_config), load(args.stressed_config), load(args.experiment)
    capacities = {scenario["set_id"]: scenario["backhaul_capacity_bps"], stressed["set_id"]: stressed["backhaul_capacity_bps"]}
    args.output.mkdir(parents=True, exist_ok=True)
    _write_semicolon(args.output / "params.csv", ["parameter", "symbol", "value"], params_rows(scenario, stressed, experiment, load(args.data_profile)))

    summary = _read_rows(args.results / "summary.csv")
    main_rows, macros = [], []
    for row in summary:
        set_id, method = row["set_id"], row["method"]
        main_rows.append(
            [
                set_id, kbit(capacities[set_id]), method, num(row["timely_ratio_mean"]),
                f"[{num(row['timely_ratio_ci_low'])}, {num(row['timely_ratio_ci_high'])}]",
                num(row["conn_ratio_mean"]), num(row["bound_ratio_mean"]), num(row["gap_mean"]),
                num(row["planning_runtime_s_mean"]), num(row["evaluation_runtime_s_per_realization_mean"]), num(row["bound_runtime_s_mean"]),
            ]
        )
        for metric, column in (
            ("Timely", "timely_ratio_mean"), ("TimelyLow", "timely_ratio_ci_low"), ("TimelyHigh", "timely_ratio_ci_high"),
            ("Conn", "conn_ratio_mean"), ("Bound", "bound_ratio_mean"), ("Gap", "gap_mean"),
            ("Planning", "planning_runtime_s_mean"), ("Evaluation", "evaluation_runtime_s_per_realization_mean"),
            ("BoundTime", "bound_runtime_s_mean"),
        ):
            macros.append((macro_name(metric, set_id, method), num(row[column])))
        macros.append((macro_name("GapUndefined", set_id, method), row["gap_undefined_count"]))
    _write_semicolon(
        args.output / "phase1_main.csv",
        ["set", "backhaul", "method", "timely", "ci", "conn", "bound", "gap", "planning", "evaluation", "boundtime"],
        main_rows,
    )

    milp_rows = []
    crosscheck = _read_rows(args.results / "milp_crosscheck.csv")
    for row in crosscheck:
        milp_rows.append(
            [
                row["scenario_id"], row["alert_count"], num(row["method_value"], 0), num(row["milp_plan_value"], 0), num(row["milp_incumbent"], 0),
                num(row["milp_dual_bound"], 2), num(row["lp_ub"], 2), MILP_STATUS.get(row["milp_status"], row["milp_status"]),
                num(row["milp_runtime_s"], 1), num(row["tangent_max_overestimate"], 4),
            ]
        )
    _write_semicolon(
        args.output / "phase1_milp.csv",
        ["scenario", "alerts", "method", "plan", "incumbent", "dual", "lp", "status", "runtime", "tangent"],
        milp_rows,
    )

    bounds = _read_rows(args.results / "bounds.csv")
    with (args.output / "lp_runtime.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variables", "runtime"])
        writer.writerows([row["variables"], f"{float(row['runtime_s']):.6f}"] for row in bounds)

    runtimes = [float(row["runtime_s"]) for row in bounds]
    variables = [int(row["variables"]) for row in bounds]
    method_rows = _read_rows(args.results / "method_realizations.csv")
    planning = {(row["set_id"], row["scenario_id"], row["method"]): float(row["planning_runtime_s"]) for row in method_rows}
    compute_s = (
        sum(runtimes)
        + sum(float(row["milp_runtime_s"]) for row in crosscheck)
        + sum(float(row["evaluation_runtime_s"]) for row in method_rows)
        + sum(planning.values())
    )
    macros += [
        ("ResultLpCount", str(len(bounds))),
        ("ResultLpRuntimeMedian", num(statistics.median(runtimes), 2)),
        ("ResultLpRuntimeMax", num(max(runtimes), 1)),
        ("ResultLpVariablesMedian", str(int(statistics.median(variables)))),
        ("ResultLpVariablesMax", str(max(variables))),
        ("ResultMilpCount", str(len(crosscheck))),
        ("ResultMilpOptimalCount", str(sum(1 for row in crosscheck if row["milp_status"] == "optimal"))),
        ("ResultTangentOverestimateMax", num(max(float(row["tangent_max_overestimate"]) for row in crosscheck), 4)),
        ("ResultComputeMinutes", num(compute_s / 60.0, 0)),
        ("ResultMilpMethodAtOptimumCount", str(sum(1 for row in crosscheck if float(row["method_value"]) >= float(row["milp_incumbent"]) - 1e-9 and row["milp_status"] == "optimal"))),
        ("ResultMilpStaticTotal", str(int(sum(float(row["milp_plan_static_value"]) for row in crosscheck)))),
        ("ResultMilpEqualSplitTotal", str(int(sum(float(row["milp_plan_equal_split_value"]) for row in crosscheck)))),
        ("ResultMilpPlanAtOptimumCount", str(sum(1 for row in crosscheck if float(row["milp_plan_value"]) >= float(row["milp_incumbent"]) - 1e-9 and row["milp_status"] == "optimal"))),
        ("ResultWorkers", str(manifest["workers"])),
        ("ResultCpuCount", str(manifest["environment"]["cpu_count"])),
        ("ResultScenarioCount", summary[0]["scenario_count"] if summary else "0"),
        ("ResultPythonVersion", manifest["environment"]["python"]),
        ("ResultScipyVersion", manifest["environment"]["scipy"]),
        ("ResultHighsVersion", manifest["environment"]["highs"]),
    ]
    scenario_means: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in method_rows:
        scenario_means.setdefault((row["set_id"], row["method"]), {}).setdefault(row["scenario_id"], []).append(float(row["timely_ratio"]))
    for set_id in sorted({row["set_id"] for row in method_rows}):
        first, second = scenario_means.get((set_id, "B0")), scenario_means.get((set_id, "B1"))
        if first and second:
            means = {scenario: (statistics.fmean(first[scenario]), statistics.fmean(second[scenario])) for scenario in first}
            macros.append((macro_name("BoneWins", set_id), str(sum(1 for b0, b1 in means.values() if b1 > b0))))
            macros.append((macro_name("BoneTies", set_id), str(sum(1 for b0, b1 in means.values() if b1 == b0))))
    lines = ["% Generated by experiments/report_tables.py from results/phase1; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    (args.output / "phase1_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"params": len(params_rows(scenario, stressed, experiment, load(args.data_profile))), "summary": len(main_rows), "milp": len(milp_rows), "lp": len(bounds)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
