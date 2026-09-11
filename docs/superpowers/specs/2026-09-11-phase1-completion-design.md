# Phase 1 Completion Design (Sub-Project B)

**Date:** 2026-09-11
**Status:** Approved for implementation planning
**Roadmap:** `docs/superpowers/specs/2026-09-11-implementation-roadmap.md`
**Builds on:** `docs/superpowers/specs/2026-09-11-model-evaluator-design.md` (sub-project A, merged)
**Requirements:** `docs/issue.md` Sections 3.4, 9.1, 9.4, 10, 11, 13; course Phase 1 requirement in `docs/references/huong-nghien-cuu-de-so-5.md`

## 1. Purpose

Deliver course Phase 1: baselines B0 and B1 on the published-benchmark scenarios, a valid LP upper bound with gaps, an exact-MILP cross-check on reduced instances, a reproducible multi-seed runner, a complexity and hardness analysis, and a Phase 1 state of the single research report in `docs/report`.

## 2. Scope

### 2.1 Included

- Methods B0 (no UAV) and B1 (fixed zone-tour trajectory, estimated-delay path choice) behind a common `Method` interface.
- Per-realization LP upper bound for a fixed trajectory and candidate set, with a ground-only variant for B0.
- MILP cross-check on five reduced instances.
- Backhaul-stressed scenario family `v0-bh50`.
- Experiment runner with parallel execution, resumable shards, provenance, and aggregation.
- Complexity analysis, NP-hardness proposition with proof, and verification of the scheduling results cited by the issue.
- Report updates for Phase 1, generated tables, and a runtime figure.

### 2.2 Excluded

B2, B3, P, ablations, sensitivity sweeps, paired hypothesis tests, the literature baseline, SAA, and the RescueNet case study. The roadmap assigns them to sub-projects C and D.

## 3. Decisions and Considered Approaches

### 3.1 Bound definition

- **Expected-rate LP — rejected.** One LP on expected rates is cheap but does not bound the timely ratio measured on channel realizations; a method can exceed it and the gap becomes meaningless.
- **Per-realization LP averaged over realizations — selected.** For each realization the LP knows the realized channel. No plan with the same trajectory and candidate set can exceed it on that realization, so the averaged gap is always valid.

### 3.2 Solver and formulation

- **`scipy` (HiGHS) with sparse matrices and tangent outer approximation — selected.** One dependency covers LP, MILP, and the statistics that sub-project C needs.
- **`cvxpy` with a conic solver — rejected.** Exact logarithm, but much slower at this size, heavier dependencies, and no dependable open-source mixed-integer conic solver for the cross-check.
- **PuLP/CBC or OR-Tools — rejected.** CBC is slower on LP; OR-Tools is heavy; `scipy` would still be needed for statistics.

A throwaway spike on three evaluation scenarios with the B1 zone-tour trajectory and realization 0 measured 25,730–74,062 variables, 0.1–0.2 s to build, and 0.5–2.4 s per LP. The bounds (13.65, 21.88, 10.80 timely alerts) exceeded the evaluated zone-tour results (3, 4, 1). Full-size MILPs reached the optimum (12) only on the smallest instance within 120 s, so the MILP cross-check uses reduced instances.

### 3.3 Environment

The project environment is managed by uv in `.venv` (commit `8876feb`). `scipy>=1.14,<2` is a core dependency. `pandas>=2.2,<4` is added as optional extra `report`, used only to run the LaTeX skill's `csv_to_latex.py`; it is invoked with `uv run --extra report` so the script never falls back to its own `pip install`.

## 4. Architecture

```text
configs/scenarios/v0.yaml, v0-bh50.yaml ──► data.cli scenarios ──► scenario sets + manifests
configs/experiments/phase1.yaml ──► experiments/run_phase1.py ──► runner
    runner: method tasks (plan + evaluate) │ bound tasks (LP) │ MILP cross-check
          └─► results/phase1/shards/ (ignored) ──► merge ──► CSV + summary + manifest.json
results/phase1/ ──► experiments/report_tables.py ──► docs/report/tables/*.tex, docs/report/data/*.csv
docs/report ──► skill compile_latex.sh ──► main.pdf
```

| File | Responsibility |
|---|---|
| `src/baselines/__init__.py` | `Method` protocol and method registry |
| `src/baselines/trajectories.py` | `stationary` and `zone_tour_trajectory` (moved from `experiments/calibrate_v0.py`) |
| `src/baselines/b0.py` | B0 |
| `src/baselines/b1.py` | B1 |
| `src/bounds/formulation.py` | Sparse LP/MILP model from scenario, candidates, trajectory, channels |
| `src/bounds/solve.py` | HiGHS LP/MILP via `scipy`, `BoundResult` |
| `src/bounds/crosscheck.py` | Reduced instances, MILP solution to plan conversion |
| `src/runner/config.py` | Experiment config loading and validation |
| `src/runner/tasks.py` | Task definitions, shard keys, execution |
| `src/runner/aggregate.py` | Merging, per-scenario means, cluster bootstrap, gaps, validity checks |
| `src/runner/provenance.py` | `manifest.json` contents and source hash |
| `experiments/run_phase1.py` | Thin entry point |
| `experiments/report_tables.py` | Display CSVs, parameter table, figure data |
| `experiments/calibrate_v0.py` | Imports trajectories from `src/baselines` |
| `configs/scenarios/v0-bh50.yaml` | Backhaul-stressed family |
| `configs/experiments/phase1.yaml` | Phase 1 experiment |

`src/models` imports none of these packages. `src/baselines` and `src/bounds` import `models` only. `src/runner` imports `models`, `baselines`, and `bounds`.

## 5. Baselines

### 5.1 Common rules

Both baselines use the shared evaluator, candidate set, EDF dispatcher, and `EqualSplitBacklogged` bandwidth (issue Section 9.1). Planning uses expected rates only and never reads evaluation realizations (issue Section 8.8).

```python
class Method(Protocol):
    name: str
    uses_uav: bool
    def plan(self, scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], counter: EvaluationCounter) -> Plan: ...
```

`uses_uav` selects the connectivity metric and the bound variant in the runner.

### 5.2 B0 — no UAV

- Trajectory: stationary at the start point.
- Path: the first ground candidate (lowest ground cost); an alert without a ground candidate is unserved.
- Connectivity reported: `connectivity_ratio_without_uav`, because the hovering UAV would otherwise count as a usable link.
- Bound: ground-only LP (Section 6.3).
- Planning cost: $O(|A|K)$.

### 5.3 B1 — fixed trajectory

- Trajectory: `zone_tour_trajectory` (Section 10.3 of the A spec), shared with the calibration script.
- Path: for each alert and each candidate, simulate that alert alone on the B1 trajectory with expected-rate channels and `EqualSplitBacklogged` (the alert gets the full band). Choose the candidate with the earliest delivery slot, ties by lower index. If no candidate delivers before the deadline, the alert is unserved.
- Connectivity reported: `connectivity_ratio`.
- Bound: trajectory LP on the B1 trajectory (Section 6.3).
- Planning cost: $O(|A| K N H)$ slot operations for $H$ maximum hops; planning simulations are not evaluator calls and do not increment the counter.

## 6. Upper Bound

### 6.1 Formulation

Inputs: scenario, candidate set, trajectory $q$, realization $\omega$ (channel states from `channel_uniforms`, identical to the evaluator). Variables:

- $x_{a,p}\in[0,1]$ per alert and candidate; $z_a\in[0,1]$ per alert.
- $f_{a,p,h,n}\ge 0$ and buffer $B_{a,p,h,n}\ge 0$ for hop $h$ of candidate $p$ and slot $n\in[r_a+h,\,d_a-1]$.
- $b_{e,n}\ge 0$ for every radio link $e$ in the candidate set and slot $n$ on which some flow variable uses $e$.

Constraints:

1. $\sum_p x_{a,p}\le 1$.
2. $B_{a,p,0,r_a}=L_a x_{a,p}$; for $h\ge1$, $B_{a,p,h,r_a+h}=f_{a,p,h-1,r_a+h-1}$.
3. $B_{a,p,h,n+1}=B_{a,p,h,n}-f_{a,p,h,n}+f_{a,p,h-1,n}$ for $n+1\le d_a-1$ (the last term absent for $h=0$); $f_{a,p,h,n}\le B_{a,p,h,n}$.
4. $L_a z_a\le\sum_p\sum_n f_{a,p,H_p-1,n}$.
5. Radio link $e$ in slot $n$ with realized SNR per Hz $s$ (LoS or NLoS value by the realized state): $\sum f\le t_e\Delta\,(R(b_k,s)+R'(b_k,s)(b_{e,n}-b_k))$ for $b_k=B_{\mathrm{tot}}2^{-k}$, $k=0,\dots,9$; if $s=0$, $\sum f\le 0$.
6. Backhaul edge in slot $n$: $\sum f\le\Delta C_e$.
7. Per slot: $\sum_{e\in\mathrm{access}} b_{e,n}\le B_{\mathrm{tot}}$ and $\sum_{e\in\mathrm{down}} b_{e,n}\le B_{\mathrm{tot}}$.

Objective: maximize $\sum_a z_a$. The bound ratio of a realization is the optimum divided by $|A|$; the scenario bound is the mean over realizations.

### 6.2 Validity

**Proposition.** For fixed $q$, candidate set, and $\omega$, the LP optimum is at least the number of timely alerts of every plan with trajectory $q$ evaluated on $\omega$.

**Proof sketch for the report.** Any plan's ledger defines a feasible LP point: $x$ and $z$ from the chosen paths and timely flags, $f$ and $B$ from transfers and buffers, $b$ from allocated bandwidth. Constraints 2–4 and 6–7 hold because the evaluator enforces them; constraint 5 holds because a concave function lies below each tangent. The LP additionally relaxes integrality, EDF order, the two-downlink cap, and non-anticipation, which only enlarge the feasible set.

The report states that the bound holds for the given trajectory and candidate set only.

### 6.3 Variants and cost

- **Trajectory bound** for B1: full candidate set, B1 trajectory.
- **Ground-only bound** for B0: candidates restricted to ground paths; no UAV links; trajectory irrelevant.

Phase 1 solves 2 sets × 30 scenarios × 30 realizations × 2 variants = 3,600 LPs. The HiGHS status must be optimal; the problem is always feasible (all variables zero) and bounded ($z\le1$).

### 6.4 MILP cross-check

- Instances: from set `v0`, sort evaluation scenarios by `(flow_variable_count, scenario_id)`, where `flow_variable_count` is $\sum_a\sum_p H_p(d_a-r_a)$ over the full scenario, and take positions $\mathrm{round}(i\cdot 29/4)$ for $i=0,\dots,4$; keep the ten alerts with the smallest IDs; B1 trajectory; realization 0.
- **LP-UB:** Section 6.1 LP. **MILP-UB:** the same model with $x,z$ binary, time limit 300 s; record the incumbent, the dual bound, and the status.
- **MILP plan value:** convert the incumbent to a plan (chosen paths; `StaticSchedule` from $b$; where a slot has more than two positive downlinks, keep the two largest and zero the rest) and evaluate it on realization 0. The plan uses realized channel knowledge and is used only to measure bound tightness.
- Required chain within $10^{-6}$ alerts: MILP plan value ≤ MILP dual bound ≤ LP-UB, and MILP incumbent ≤ MILP dual bound.
- Diagnostic: the maximum relative overestimate of the ten-tangent envelope over $b\in[B_{\mathrm{tot}}/512,\,B_{\mathrm{tot}}]$ for the SNR values present in the cross-check instances.

## 7. Backhaul-Stressed Family

- `configs/scenarios/v0-bh50.yaml` equals `configs/scenarios/v0.yaml` except `set_id: v0-bh50` and `backhaul_capacity_bps: 50000.0`. A test enforces this.
- Seeds are unchanged, so topologies, zones, sources, alerts, and failure patterns match `v0`; candidate rankings can change because their cost includes backhaul time.
- Outputs: `data/processed/scenarios/v0-bh50/` (ignored), `data/manifests/scenarios_v0-bh50.csv` and `results/calibration/v0-bh50.json` (committed). The calibration check must pass with the A spec bounds.
- Rationale: in exploratory runs with five realizations on the ground-only calibration plan, backhaul link-slots were saturated in 5 of 55,885 used slots at 1,000 kbit/s, 11.4% at 100 kbit/s, and 31.7% at 50 kbit/s. At 250 kbit/s the timely ratio stayed at 0.415 (0.413 on `v0` with 30 realizations); at 50 kbit/s it fell to 0.370.
- Runner results key on `(set_id, scenario_id)` because scenario IDs repeat across sets.

## 8. Runner

### 8.1 Configuration

```yaml
experiment_id: phase1
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
  - set_id: v0-bh50
    manifest: data/manifests/scenarios_v0-bh50.csv
    scenario_dir: data/processed/scenarios/v0-bh50
split: eval
realization_ids: {start: 0, stop: 30}
methods: [B0, B1]
bounds: {tangents: 10}
milp_crosscheck: {set_id: v0, scenario_count: 5, alert_count: 10, realization_id: 0, time_limit_s: 300}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase1
```

### 8.2 Tasks, parallelism, and resume

- **Method task** `(set_id, scenario_id, method)`: plan once, evaluate on all realization IDs, emit one row per realization.
- **Bound task** `(set_id, scenario_id, variant, realization_id)`: build and solve one LP.
- **Cross-check task** per reduced instance.
- Tasks run in `concurrent.futures.ProcessPoolExecutor(max_workers=workers)` and depend only on their inputs.
- Each finished task writes `results/phase1/shards/<task_key>.json` with a `task_hash` over the task fields, the scenario `sha256`, the config hash, and a source hash of all files under `src/models`, `src/baselines`, `src/bounds`, and `src/runner`. A rerun skips shards whose hash matches and recomputes the rest. `results/phase1/shards/` is added to `.gitignore`.
- After all tasks, shards are merged into CSV files sorted by their key columns.

### 8.3 Outputs (committed)

| File | Columns |
|---|---|
| `method_realizations.csv` | `set_id, scenario_id, topology_id, method, realization_id, alert_count, timely_count, timely_ratio, source_count, connected_count, conn_ratio, shortfall, planning_runtime_s, evaluation_runtime_s, evaluate_calls` |
| `bounds.csv` | `set_id, scenario_id, topology_id, variant, realization_id, value, ratio, status, runtime_s, variables, rows, nonzeros` |
| `milp_crosscheck.csv` | `scenario_id, alert_count, lp_ub, milp_incumbent, milp_dual_bound, milp_status, milp_runtime_s, milp_plan_value, tangent_max_overestimate` |
| `summary.csv` | `set_id, method, scenario_count, timely_ratio_mean, timely_ratio_ci_low, timely_ratio_ci_high, conn_ratio_mean, bound_ratio_mean, gap_mean, gap_undefined_count, shortfall_mean, planning_runtime_s_mean, evaluation_runtime_s_per_realization_mean, bound_runtime_s_mean` |
| `manifest.json` | experiment ID, config hash, git commit and dirty flag, source hash, scenario manifest SHA-256 per set, Python/numpy/scipy versions, HiGHS version string reported by scipy, platform, CPU count, workers, start and end UTC time, command line |

### 8.4 Aggregation

- Scenario mean: average over realizations of `timely_ratio`, `conn_ratio`, `shortfall`, and bound `ratio`.
- Set mean: average of scenario means over the 30 evaluation scenarios.
- Confidence interval: cluster bootstrap over topologies (resample the 10 topologies with replacement, recompute the mean over all their scenarios), percentile interval, `numpy.random.default_rng(seed)`.
- Gap per scenario: $(\overline{UB}-\overline{P})/\overline{UB}$ with the bound variant of the method; scenarios with $\overline{UB}=0$ are counted in `gap_undefined_count` and excluded from `gap_mean`.
- Validity checks (Section 11) run before summaries are written.

## 9. Complexity, Hardness, and Literature

### 9.1 Complexity

The report gives: candidate generation $O(|V|+|E|+|A||V|\log|V|)$; evaluation $O(MN(|\mathcal L|+|A|\log|A|))$, with the checker of the same order and connectivity $O(MN(|V|+|E|+|S||V|))$; B0 planning $O(|A|K)$; B1 planning $O(|A|KNH)$; LP size $\sum_a\sum_p H_p(d_a-r_a)$ flow and buffer variables each, plus at most $|\mathcal L_{\mathrm{radio}}|N$ bandwidth variables and $10|\mathcal L_{\mathrm{radio}}|N$ tangent rows. No worst-case polynomial claim is made for HiGHS; the report plots measured LP runtime against variable count for all 3,600 LPs.

### 9.2 Hardness proposition

**Proposition.** With a fixed trajectory, bit-divisible transmission across slots, two ground relays, and $K=2$ candidates per alert, maximizing the number of timely alerts is NP-hard.

**Proof (reduction from PARTITION).** Given positive integers $s_1,\dots,s_m$ with even sum $2S$, build one source within range of relays $v_1,v_2$, each with a single backhaul edge to the center of capacity $S$ bits per slot, and access SNR high enough that access capacity in one half-slot at bandwidth $B_{\mathrm{tot}}/2$ exceeds $2S$ bits. Alert $i$ has size $s_i$, release 0, and deadline 2 (upload in slot 0, backhaul in slot 1). Its two candidates are the paths through $v_1$ and $v_2$. All $m$ alerts are timely if and only if the sizes split into two sets of sum $S$. The report states the scope: weak NP-hardness only; no membership claim, because bandwidth is continuous; no claim for $K=1$. The hardness comes from single-path selection, consistent with issue Section 9.4.

### 9.3 Literature verification

Verify with web sources before citing: strong NP-hardness of $1|r_j|\sum U_j$ (expected: Lenstra, Rinnooy Kan, and Brucker, 1977) and polynomial solvability of $1|r_j,\mathrm{pmtn}|\sum U_j$ (expected: Lawler, 1990). A reference enters `docs/report/references.bib` only with verified authors, title, venue, volume, pages, year, and DOI or stable URL, and only if the source states the result as cited. An unverifiable result is removed from the report text. These results provide context; the proposition does not rely on them.

## 10. Report Updates

- `preamble.tex`: add `\usepackage{pgfplots}` with `\pgfplotsset{compat=1.18}`, and set `\sisetup{output-decimal-marker={,}}` to match Vietnamese text.
- `tables/params.tex`: generated by `experiments/report_tables.py` from `configs/scenarios/v0.yaml` and `configs/experiments/phase1.yaml`. **Deviation from the skill:** it is written as LaTeX directly because the skill's `csv_to_latex.py` escapes `$` and would break symbols and `siunitx` units.
- `tables/phase1_main.tex` and `tables/phase1_milp.tex`: display CSVs in `results/phase1/tables/` (numbers pre-formatted with decimal commas) converted by the skill's `csv_to_latex.py` through `uv run --extra report`.
- `data/lp_runtime.csv`: variables and runtime per LP, read by a `pgfplots` figure in `experiments.tex`.
- `method.tex`: full system model and constraints; B0 and B1; bound proposition and proof; hardness proposition and proof; complexity subsection. The BCD-repair algorithm stays as the planned Phase 2 method and is labelled as not yet evaluated.
- `experiments.tex`: setup (data sources, scenario generation, both families, seeds, realizations, hardware); Phase 1 results table; MILP cross-check table; LP runtime figure. The illustrative table with B2/B3/P rows and the convergence placeholder figure are removed.
- `abstract.tex`, `introduction.tex`: contributions restated to Phase 1 facts; Phase 2 items marked as planned.
- Build with `.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview`; inspect the preview pages.

## 11. Error Handling

- A non-optimal LP status or a failed task is recorded in its shard; the runner writes merged partial results, lists failed tasks, and exits with code 1.
- Validity violations stop the run with exit code 1 before summaries: a scenario-realization where the method's timely count exceeds its bound value by more than $10^{-6}$, or a broken MILP chain.
- Missing scenario sets or manifests raise an error naming the configured path.
- `report_tables.py` refuses to run when `results/phase1/manifest.json` is missing or its source hash differs from the current source hash.

## 12. Testing

All tests run offline with `uv run --extra test pytest`.

- **Baselines:** B0 selects only ground candidates and leaves alerts without one unserved; B1 picks the earliest estimated delivery with the lower-index tie-break and leaves hopeless alerts unserved; `experiments/calibrate_v0.py` uses the `zone_tour_trajectory` object from `src/baselines/trajectories.py`, and that function produces a feasible tour that visits every zone that fits on the test scenarios from sub-project A.
- **Bounds:** the tangent envelope is at least $R(b)$ on a grid; a single alert on a single link gives $z=\min(1,\text{capacity}/L_a)$; the two-hop deadline boundary matches the evaluator (deliverable exactly by slot $d_a-1$ gives bound 1, one slot less gives 0); for random valid plans on small scenarios the bound is at least the evaluated timely count; MILP optimum ≤ LP optimum; the ground-only bound uses no UAV link.
- **Runner:** one worker and two workers produce identical CSV rows apart from runtime columns; a rerun skips completed tasks; a shard with a mismatching hash is recomputed; aggregation reproduces hand-computed means, gaps, undefined-gap counts, and a bootstrap interval with a fixed seed on a small table; `manifest.json` has every field in Section 8.3.
- **Configs and tables:** `v0-bh50.yaml` differs from `v0.yaml` only in the two fields; the generated parameter table contains every configured value.

## 13. Acceptance Criteria

1. The full test suite passes in `.venv`.
2. After moving `zone_tour_trajectory`, rerunning `experiments/calibrate_v0.py` on `v0` reproduces every per-scenario ratio in the committed `results/calibration/v0.json`. `v0-bh50` is generated reproducibly and passes calibration; its manifest and calibration result are committed.
3. `uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml` completes with all LPs optimal and no validity violation; `results/phase1/` (except shards) is committed.
4. Running the same command again skips every task.
5. `docs/report` compiles with the skill's script and contains Phase 1 numbers, both propositions with proofs, the complexity analysis, verified references, and no illustrative numbers or unsupported claims.
6. `README.md` documents generating `v0-bh50`, running Phase 1, building report tables, and compiling the report.

## 14. Decision Log

- Per-realization LP bound instead of an expected-rate bound (Section 3.1).
- `scipy`/HiGHS with ten tangents (Section 3.2).
- MILP cross-check on reduced instances after full-size MILPs hit time limits in the spike (Section 3.2).
- `v0-bh50` at 50 kbit/s (Section 7).
- B0 reports connectivity without the UAV and is compared with the ground-only bound (Section 5.2).
- Parameter table generated directly instead of through `csv_to_latex.py` (Section 10).
- Decimal comma in the report (Section 10).
- `pandas` as optional extra `report` for the skill script (Section 3.3).
