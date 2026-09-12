# Phase 2 Core Design (Sub-Project C)

**Date:** 2026-09-11
**Status:** Approved for implementation planning
**Roadmap:** `docs/superpowers/specs/2026-09-11-implementation-roadmap.md`
**Builds on:** release `v0.1.0` — sub-project A (`2026-09-11-model-evaluator-design.md`) and sub-project B (`2026-09-11-phase1-completion-design.md`)
**Requirements:** `docs/issue.md` Sections 5, 8.7–8.8, 9.1–9.4, 10, 12

## 1. Purpose

Deliver the core of course Phase 2: the Huang-style decomposition B2, the objective-control variant B3, the proposed method P with deadline-aware repair, experiments that answer RQ1–RQ2 with paired statistics and RQ3–RQ5 with ablations and sensitivity, a literature review that selects the strong literature baseline, and the Phase 2 state of the report.

## 2. Scope and Delivery

### 2.1 Included

- Evaluator extensions: design-realization namespace and weighted bandwidth policy (Section 5).
- Design scoring with an evaluation budget (Section 6).
- Methods B2, B3, P and their ablation variants as configurations of one search framework (Sections 7–9).
- Runner extensions for parameterized methods, per-method trajectory bounds, and the union bound (Section 10).
- Main comparison, ablations, quality–runtime curves, and sensitivity (Section 11).
- Paired statistical analysis (Section 12).
- Literature review and strong-baseline selection (Section 13).
- Report updates (Section 14).

### 2.2 Excluded

Implementation and evaluation of the literature baseline, SAA beyond the design-realization scoring used here, the RescueNet case study, the reproducibility package, and the final report polish belong to sub-project D.

### 2.3 Two plans

- **Plan C.1 — Methods:** Sections 5–10 and the development-set pilot (Section 17.1).
- **Plan C.2 — Experiments, statistics, literature, report:** Sections 11–14, written after the C.1 pilot numbers exist. Experiment sizes, budgets, and figure choices may be adjusted in C.2 from pilot evidence; every adjustment is recorded in Section 18.
- **Plan C.3 — Report (added by plan C.2, see Section 18):** Section 14 and acceptance criterion 17.2.6, written after the C.2 results exist.

## 3. Evidence and Considered Approaches

Three throwaway spikes on six `v0` evaluation scenarios (`Marwan-r0`, `Easynet-r0`, `Garr200404-r0`, `Agis-r0`, `BtNorthAmerica-r0`, `Cesnet200603-r2`) with ten evaluation realizations shaped the design. One `evaluate` call on a v0 scenario took 6–14 ms in expected mode and 6–15 ms per realization, checker included.

### 3.1 Spike 1 — bandwidth from the LP

An expected-rate version of the Section 6.1 LP of sub-project B was solved with the B1 trajectory; its bandwidth was used as a static schedule or as weights, and its path choice was rounded.

| Variant | Mean timely ratio |
|---|---|
| B1 (B1 paths, equal split) | 0.410 |
| B1 paths + static LP bandwidth | 0.128 |
| B1 paths + LP bandwidth as weights | 0.404 |
| LP paths + equal split | 0.412 |
| LP paths + LP weights | 0.386 |

LP guidance does not transfer to the EDF dispatcher: static schedules collapse and LP weights or paths do not beat B1.

### 3.2 Spikes 2 and 3 — evaluator-guided search

Starting from B1, one path sweep (every alert tries "unserved" and every candidate), a trajectory search over zone orders and hover splits, and a second path sweep were scored either on expected rates (spike 2) or on five separate realizations (spike 3).

| Scenario | B1 | Expected-rate search | Sampled search (5 realizations) |
|---|---|---|---|
| Marwan-r0 | 0.372 | 0.305 | 0.360 |
| Easynet-r0 | 0.125 | 0.145 | 0.158 |
| Garr200404-r0 | 0.645 | 0.672 | 0.675 |
| Agis-r0 | 0.450 | 0.502 | 0.460 |
| BtNorthAmerica-r0 | 0.393 | 0.390 | 0.505 |
| Cesnet200603-r2 | 0.472 | 0.545 | 0.535 |
| **Mean** | **0.410** | **0.427** | **0.449** |

Searches used 495–635 evaluate calls; sampled search took 36–57 s per scenario. Expected-rate scoring can overfit (Marwan-r0 fell from 0.372 to 0.305); sampled scoring was better on average and did not collapse.

### 3.3 Approaches

- **A. Evaluator-guided BCD — selected.** Keeps the three-block alternating structure and stopping rule of Huang; each block is a local search scored by the evaluator on design realizations. Supported by spikes 2–3; consistent with evaluator semantics; cheap.
- **B. Convex BCD as in Huang (LP/SCA blocks) — rejected.** Spike 1 shows LP-derived bandwidth and paths do not beat B1; SCA for the PrLoS trajectory has no guaranteed valid surrogate (issue Section 9.2).
- **C. Hybrid (LP paths, evaluator trajectory) — rejected.** LP path choice matched B1 (0.412 versus 0.410) while adding complexity.

**Deviation from issue Section 9.2:** the bandwidth and path blocks are not convex problems; the report states the reason with the spike numbers.

## 4. Architecture

```text
models (extended: design namespace, WeightedBacklogged)
   ▲
optimization: scoring ─ tours ─ objectives ─ blocks ─ repair ─ bcd (B2, B3, P variants)
   ▲                                   ▲
baselines (B0, B1: initial plan)     runner (method specs, bounds in method tasks, union bound)
   ▲                                   ▲
analysis.statistics ◄── experiments/run_phase2.py, experiments/report_tables_phase2.py
```

| File | Responsibility |
|---|---|
| `src/models/channel.py` | `channel_uniforms(..., namespace="eval")` |
| `src/models/plan.py` | `WeightedBacklogged`; validation |
| `src/models/simulator.py` | Weighted allocation |
| `src/models/evaluate.py` | `channel_namespace` argument and result field |
| `src/optimization/scoring.py` | `DesignScorer`, `BudgetExhausted` |
| `src/optimization/tours.py` | Parameterized tour family |
| `src/optimization/objectives.py` | Lexicographic and leximin keys |
| `src/optimization/blocks.py` | Path, bandwidth, trajectory blocks |
| `src/optimization/repair.py` | Risk detection, bandwidth shift, path change |
| `src/optimization/bcd.py` | `SearchMethod` (B2, B3, P, ablation variants) |
| `src/runner/methods.py` | Builds B0, B1, and search variants from method specs |
| `src/runner/config.py`, `tasks.py`, `experiment.py`, `aggregate.py` | Method specs, per-method bounds, union bound |
| `src/analysis/statistics.py` | Paired topology-level tests, Holm, effect sizes |
| `experiments/run_phase2.py` | Entry point for Phase 2 experiment configs |
| `experiments/report_tables_phase2.py` | Report CSVs and macros for Phase 2 |
| `configs/experiments/phase2_pilot.yaml`, `phase2_main.yaml`, `phase2_budget.yaml`, `phase2_sensitivity.yaml` | Experiment configurations |
| `configs/scenarios/sens-*.yaml` | Sensitivity scenario families |
| `docs/literature/review.md` | Literature review protocol and records |

Import rules: `models` imports none of the new packages; `baselines` is unchanged and does not import `optimization`; `optimization` imports `models` and `baselines`; `analysis` imports only `numpy` and `scipy`; `runner` imports `models`, `baselines`, `bounds`, `optimization`, and builds every method from its spec in `runner/methods.py`. No new third-party dependency.

## 5. Evaluator Extensions

These extensions are backward compatible. Every existing call returns identical results, and re-running the Phase 1 configuration must reproduce the committed `results/phase1` rows outside runtime columns.

### 5.1 Design realizations

- `channel_uniforms(scenario, realization_id, namespace="eval")` draws from `named_rng(scenario_seed, f"channel-{namespace}:{realization_id}")` with `namespace ∈ {"eval", "design"}`; any other value raises `ValueError`.
- `evaluate(scenario, plan, mode, realization_ids=(), counter=None, candidates=None, keep_ledger=False, channel_namespace="eval")` passes the namespace; `EvaluationResult` gains `channel_namespace`.
- Planners never call `evaluate`; they receive a `DesignScorer` fixed to the `"design"` namespace (Section 6). Final evaluation by the runner uses `"eval"`.

### 5.2 Weighted bandwidth policy

- `WeightedBacklogged(weights: Mapping[LinkId, np.ndarray])` holds a non-negative weight per radio link per slot; missing links weigh 0.
- In each half-slot the evaluator takes the backlogged access links, and the backlogged downlinks limited to `max_active_downlinks` by earliest head-of-queue deadline as for equal split, and assigns $b_e = B_{\mathrm{tot}} w_e / \sum w$ over that group; a zero weight sum gives an equal split of the group.
- `validate_plan` requires finite, non-negative arrays of length $N$ keyed by radio links of the candidate set.
- The policy is non-anticipative: weights are fixed offline and service depends only on current backlog. Equal split is the special case of all weights equal to one.
- The ledger checker needs no change.

## 6. Design Scoring and Budget

- `DesignScorer(scenario, candidates, design_ids, budget, objective)` exposes `score(plan) -> key`, `score_with_ledger(plan) -> (key, EvaluationResult)`, `calls`, and `remaining`.
- With `design_ids` non-empty it evaluates in `REALIZED` mode on namespace `"design"`; with `design_ids = ()` it evaluates in `EXPECTED` mode (the expected-rate design ablation). It never exposes the `"eval"` namespace.
- Every call counts as one evaluation regardless of the number of design realizations. When `calls == budget`, `score` raises `BudgetExhausted`; search methods catch it and return the best plan found.
- Defaults: `design_ids = (0, 1, 2, 3, 4)`, `budget = 2000`. Both are confirmed or changed by the C.1 pilot.

## 7. Method B2

### 7.1 State and initialization

The search state is `(trajectory, path_choice, weights)`. It starts from the B1 plan with all radio-link weights equal to one. Keys use the lexicographic objective of sub-project A summed over design realizations: `(timely_count, connected_count, -round(shortfall, 9))`. A change is accepted only if the key increases strictly; ties keep the incumbent.

### 7.2 Iteration

Up to `max_iterations = 10` iterations, each applying the blocks in order; the search stops after an iteration that accepts nothing, or when the budget is exhausted.

1. **Path block.** Alerts in `(deadline_slot, id)` order. For each alert, options in the order `None, 0, 1, …, K-1` except the current one; each option is scored from the current incumbent and accepted if strictly better.
2. **Bandwidth block.** Radio links used by chosen paths, ordered by the total size of the alerts routed over them (descending, ties by link). For each link, try multiplying its whole weight array by 2, then by 1/2; accept a strictly better result.
3. **Trajectory block.** Enumerate the tour family of Section 7.3 in a fixed order, score each tour with the current paths and weights, and accept a strictly better one.

### 7.3 Tour family

A tour starts and ends at the center, flies straight segments at speed $V_{\max}$ with $\lceil d/(V_{\max}\Delta)\rceil$ steps per segment, hovers at each visited stop, and fills the remaining slots at the center; it is feasible by construction. Parameters:

- **Visited zones and order:** every non-empty subset of the damage zones in every order, enumerated by subset size descending, then lexicographic order of zone indices; combinations whose travel exceeds $N$ slots are skipped.
- **Hover split:** `uniform`; `load` (weights = number of alerts whose source belongs to the zone); `urgency` (weights = $\sum 1/(d_a-r_a)$ over the zone's alerts). Spare slots are split by $\lfloor \mathrm{spare}\cdot w_i/\sum w\rfloor$; a zero weight sum falls back to uniform.
- **Hover point:** the zone center, or the centroid of the zone's source points (the zone center when the zone has no source).

With three zones the family has at most $15 \times 3 \times 2 = 90$ tours.

### 7.4 Complexity

One iteration uses at most $(K+1)|A| + 2|\mathcal L_{\mathrm{used}}| + 90$ evaluations; each costs $O(M_d N(|\mathcal L| + |A|\log|A|))$. The budget bounds the total.

## 8. Method B3

B3 is B2 with a different acceptance key. For each alert, $\bar f_a$ is the delivered fraction $D_a/L_a$ averaged over design realizations; this normalizes Huang's max–min average rate, since an alert's average delivery rate $D_a/((d_a-r_a)\Delta)$ divided by its required rate $L_a/((d_a-r_a)\Delta)$ equals $D_a/L_a$. Because a single unreachable alert makes $\min_a \bar f_a = 0$ for every plan, B3 uses the leximin key: the tuple of $\bar f_a$ values sorted ascending and rounded to $10^{-9}$, compared lexicographically, followed by `timely_count` and `connected_count`. Unserved alerts contribute $\bar f_a = 0$. B3 is evaluated with the same timely ratio as every other method; B2 − B3 measures the cost of the objective.

## 9. Method P

P is B2 plus a repair round after each iteration's trajectory block. An iteration that accepts nothing in its blocks and its repair round stops the search.

### 9.1 Risk detection

One `score_with_ledger` call (one evaluation) on the incumbent. For each served alert and design realization:

- **Slack:** `deadline_slot - 1 - delivery_slot` if timely; otherwise $-1$.
- **Bottleneck hop:** replaying the ledger, the hop whose buffer held bits of the alert in the most slots; ties go to the earliest hop.
- **Congested link:** a link is congested in a slot if the bits it carried reached its recorded capacity within $10^{-6}$ relative.

The risk score of an alert is the fraction of design realizations in which it is late or has slack at most $\theta = 2$ slots. The at-risk set holds the served alerts with positive risk, ordered by `(-risk, deadline_slot, id)`, truncated to $R = 10$.

### 9.2 Operation 1 — bandwidth shift

If the alert's most frequent bottleneck hop across design realizations is a radio link, multiply that link's weights by 2 in slots $[r_a, d_a)$ only. Because weights are relative within a half-slot, bandwidth moves from other links in the same half-slot and slot. Skipped when the bottleneck is backhaul.

### 9.3 Operation 2 — path change

Try the alert's other candidates ordered by `(number of links congested in the alert's window in any design realization, cost_s, index)` and accept the first strictly better plan.

### 9.4 Acceptance, budget, variants

- Each operation is accepted only if the global key increases strictly, so rescuing one alert never costs more timely alerts than it gains.
- Repair evaluations consume the same budget as the blocks; P and B2 receive identical budgets.
- A repair round costs at most $1 + R\,K$ evaluations.
- `SearchMethod` parameters and the variants they define:

| Parameter | B2 | B3 | P | `P_op1` | `P_op2` | `P_fixed_paths` | `P_fixed_trajectory` | `P_equal_bandwidth` | `P_no_backlog` | `P_expected_design` |
|---|---|---|---|---|---|---|---|---|---|---|
| objective | lex | leximin | lex | lex | lex | lex | lex | lex | lex | lex |
| path block | on | on | on | on | on | off | on | on | on | on |
| bandwidth block | on | on | on | on | on | on | on | off | on | on |
| trajectory block | on | on | on | on | on | on | off | on | on | on |
| operation 1 | off | off | on | on | off | on | on | off | on | on |
| operation 2 | off | off | on | off | on | off | on | on | on | on |
| backlog information | — | — | on | on | on | on | on | on | off | on |
| design ids | 0–4 | 0–4 | 0–4 | 0–4 | 0–4 | 0–4 | 0–4 | 0–4 | 0–4 | expected |

Without backlog information, risk uses slack only, operation 1 targets the alert's first radio hop, and operation 2 orders candidates by `(cost_s, index)`.

## 10. Runner Extensions

- **Method specs.** `methods` entries may be a name (Phase 1 form) or `{id, base, params, bounds}`; `base` is `B0`, `B1`, or `search`; `params` set Section 9.4 parameters, `budget`, and `max_iterations`; `bounds: true` computes trajectory bounds.
- **Bounds inside method tasks.** A search method's trajectory exists only after planning, so a method task with `bounds: true` solves the trajectory LP for its own trajectory on every evaluation realization after evaluating and emits bound rows with `trajectory_method = id`. B0 and B1 bounds keep their Phase 1 bound tasks. Plans stay in the ignored shard files.
- **Union bound.** Per scenario and realization, the maximum over all `trajectory` bound rows, including B1's bound tasks and the bounds computed inside search-method tasks (issue Section 9.4: $\max\{\mathrm{UB}(q_{B1}),\mathrm{UB}(q_{B2}),\mathrm{UB}(q_P)\}$, here also $q_{B3}$); reported as `union_bound_ratio_mean` and `union_gap_mean`.
- **Compatibility.** The Phase 1 configuration, run with the extended runner, produces the same task keys and the same non-runtime rows as `results/phase1`.

## 11. Experiments (Plan C.2)

All runs use the evaluation split, 30 evaluation realizations, and the runner's validity checks. Wall-time estimates with 12 workers, from spike timings (about 100 s of planning per search method per scenario at budget 2000, 2–7 s per trajectory LP): main comparison and ablations about 2 h, quality–runtime about 0.5 h, sensitivity about 2 h. The C.1 pilot refines these estimates.

### 11.1 Main comparison — `configs/experiments/phase2_main.yaml`

Sets `v0` and `v0-bh50`; methods B0, B1, B2, B3, P with bounds for B2, B3, P; ablation variants `P_op1`, `P_op2`, `P_fixed_paths`, `P_fixed_trajectory`, `P_equal_bandwidth`, `P_no_backlog`, `P_expected_design` without bounds. Output `results/phase2/main/`.

### 11.2 Quality–runtime — `configs/experiments/phase2_budget.yaml`

Set `v0`; B2 and P with budgets 250, 500, 1000, 2000, 4000. Output `results/phase2/budget/` and a figure of mean timely ratio against mean evaluate calls and runtime.

### 11.3 Sensitivity — `configs/experiments/phase2_sensitivity.yaml`

One-at-a-time families generated from `v0` with `set_id: sens-<name>`; each config differs from `v0.yaml` only in the listed fields (a test enforces this). Methods B1, B2, P; no bounds.

| Family | Changed fields |
|---|---|
| `sens-lref125k`, `sens-lref500k`, `sens-lref1m` | `workload.size_ref_bits` |
| `sens-bh100` | `backhaul_capacity_bps: 100000.0` (with `v0` and `v0-bh50` as the other points) |
| `sens-physical` | `channel.uav.beta0_db: -40.0`, `channel.ground.beta0_db: -40.0`, `spectrum.noise_psd_dbm_per_hz: -164.0` |
| `sens-deadline-short`, `sens-deadline-long` | deadline window [10, 30] and [40, 120] |
| `sens-btot05`, `sens-btot2` | `spectrum.b_tot_hz` 500000.0 and 2000000.0 |
| `sens-alerts20`, `sens-alerts60` | `workload.alert_count` |
| `sens-k2`, `sens-k5` | `paths.k: 2` with `paths.max_ground: 1`; `paths.k: 5` with `paths.max_ground: 2` (so every alert keeps at least one UAV candidate) |

The design-realization count for P (expected, 1, 5, 10) runs on `v0` as method variants in the same config. Output `results/phase2/sensitivity/`. Sensitivity families are not calibration-gated; degenerate families are reported as such. Excluded from C (stated in the report): failure probabilities, power, mission time, UAV speed and altitude, PrLoS parameters, $\Delta$, and dataset conversion rules.

## 12. Statistics — `src/analysis/statistics.py`

- **Unit:** per-scenario means over the 30 evaluation realizations, averaged per topology, giving 10 paired values per comparison.
- **Primary comparisons, fixed before running:** P − B2, B2 − B1, B2 − B3, P − B1 on `v0` and on `v0-bh50` (eight comparisons).
- **Test:** exact two-sided sign-flip permutation test on the 10 topology differences (all $2^{10}$ sign assignments; p = fraction with |mean| at least the observed |mean|).
- **Multiplicity:** Holm correction over the eight comparisons.
- **Effect sizes:** mean difference with a 95% bootstrap interval over topologies (10,000 resamples, seed 20260911) and matched-pairs rank-biserial correlation $r = (W^+ - W^-)/(W^+ + W^-)$ with zero differences excluded.
- **Output:** `results/phase2/statistics.csv` with columns `set_id, comparison, mean_difference, ci_low, ci_high, rank_biserial, p_value, p_holm, topology_count`.
- Ablations and sensitivity are reported descriptively with bootstrap intervals and no further tests.

## 13. Literature Review and Strong Baseline — `docs/literature/review.md`

- **Protocol:** fixed queries recorded in the file, run on web search, arXiv, and Crossref, covering deadline-aware UAV relaying, relay selection with backhaul, time-expanded network flow, stochastic timely delivery, joint trajectory and bandwidth design, and decomposition with repair.
- **Inclusion criteria:** UAV relaying or data collection with time constraints, or relay selection with backhaul.
- **Target size:** 12–20 included papers; excluded candidates are listed with reasons.
- **Verification:** every included paper's metadata is checked against Crossref or arXiv, and statements come only from text that was read (abstract, arXiv version, or open copy).
- **Comparison table** with columns key, year, venue, objective, UAVs, relay selection, deadline, connectivity, channel uncertainty, bandwidth/power, method, guarantee; stored as `docs/report/data/related_work.csv` and rendered with `csvsimple`.
- **Selection criteria,** in order: objective close to maximizing timely alerts; offline design with known demand; joint trajectory and radio resources; enough detail or code to reproduce; adaptable without changing the method's nature.
- **Current candidates, verified during brainstorming:** Tran, Nguyen, Chatzinotas, Vu, Ottersten, "UAV Relay-Assisted Emergency Communications in IoT Networks: Resource Allocation and Trajectory Optimization", *IEEE TWC* 21(3):1621–1637, 2022, DOI 10.1109/TWC.2021.3105821 (UAV relay with storage, latency-constrained devices, maximizes served devices, bandwidth/power/trajectory, inner approximation; no public code found); and Samir, Sharafeddine, Assi, Nguyen, Ghrayeb, "UAV Trajectory Planning for Data Collection from Time-Constrained IoT Devices", *IEEE TWC* 19(1):34–46, 2020, DOI 10.1109/TWC.2019.2940447 (maximizes served devices with deadlines; SCA; unofficial MATLAB/CVX implementation under MIT license). Neither has ground relay selection or finite backhaul.
- **Default decision:** Tran et al. (half-duplex variant), adapted to UAV paths that drop at the center, unless the review finds a closer match. The final choice and the adaptation are recorded in Section 18 by plan C.2; implementation belongs to sub-project D.

## 14. Report Updates

The Phase 1 content stays; new numbers come from `docs/report/data/phase2_results.tex` macros written by `experiments/report_tables_phase2.py`.

- **`method.tex`:**
  - weighted policy and design realizations in the model;
  - the algorithm of B2/P (blocks and repair) replacing the planned Phase 2 algorithm;
  - the leximin definition of B3;
  - complexity per iteration and the budget bound;
  - the reason blocks are not convex, with spike numbers.
- **`related-work.tex`:** the review table and the baseline-selection rationale.
- **`experiments.tex`:**
  - main comparison table (methods × sets with intervals, gap to own and union bounds, runtime, evaluate calls);
  - statistics table;
  - ablation table;
  - quality–runtime figure;
  - sensitivity figure grid (`pgfplots` from CSV);
  - answers to RQ1–RQ5.
- **Abstract, introduction, discussion, conclusion:** updated to Phase 2 facts, including a plain statement if P does not beat B2.
- **Build and checks:** built with the skill's `compile_latex.sh`; no skill-analysis warnings, no `??` in the PDF text, no placeholders, and preview pages inspected.

## 15. Error Handling

- `DesignScorer` raises `BudgetExhausted`; search methods return their incumbent, which is always feasible because every accepted plan passed evaluation.
- An infeasible candidate plan (`feasible=False`) scores below every feasible key and is never accepted.
- `channel_uniforms` rejects unknown namespaces; `validate_plan` rejects malformed weights.
- Runner validity checks from sub-project B apply to every bound row, including bounds computed inside method tasks.
- Sensitivity generation failures (for example, zones that cannot be placed) stop with the family name.

## 16. Testing

### 16.1 Plan C.1

- **Evaluator:** `"eval"` reproduces previous uniforms and results; `"design"` differs; unknown namespace rejected; a 3:1 weight instance yields 0.75/0.25 of $B_{\mathrm{tot}}$; all-ones weights give a ledger identical to equal split; malformed weights rejected; random weighted plans pass the checker.
- **Scoring:** uses `"design"` (results differ from `"eval"` for the same ids); counts calls; raises at the budget; expected mode when `design_ids = ()`.
- **Tours:** every tour passes `validate_plan`; infeasible subsets are skipped; hover splits and hover points follow Section 7.3.
- **Objectives:** lexicographic and leximin keys match hand values and tie rules.
- **Blocks:**
  - the path block drops a hopeless alert when that lets another alert finish;
  - the trajectory block picks a tour reaching a far source when only that delivers;
  - the bandwidth block raises the weight of a congested link when that makes an alert timely.
- **Methods:**
  - keys never decrease across iterations;
  - calls never exceed the budget;
  - identical plans on repeated runs;
  - feasible outputs;
  - B3 differs from B2 only through its key;
  - P with both operations off returns exactly the B2 plan;
  - operation 1 changes weights only inside $[r_a, d_a)$;
  - operation 2 follows its ordering;
  - the no-backlog variant ignores bottleneck data.
- **Runner:**
  - method specs parse (name and dictionary forms);
  - bound rows from method tasks carry `trajectory_method`;
  - the union bound is the per-realization maximum;
  - the Phase 1 configuration yields the Phase 1 task keys.

### 16.2 Plan C.2

- **Statistics:** 10 equal positive differences give p = 2/1024; symmetric differences give p = 1; Holm and rank-biserial match hand examples; the bootstrap is deterministic with its seed.
- **Sensitivity configs:** each differs from `v0.yaml` only in its listed fields.
- **Phase 2 report data:** CSVs and macros match fixture inputs.

## 17. Acceptance Criteria

### 17.1 Plan C.1

1. The full test suite passes.
2. Running `configs/experiments/phase1.yaml` with the extended code reproduces every non-runtime row of the committed `results/phase1` files.
3. `configs/experiments/phase2_pilot.yaml` runs B1, B2, B3, P (bounds on for B2, B3, P) on the six development scenarios of `v0`, with 30 evaluation realizations, and completes without validity violations. Its results and a short pilot note (runtime per method, timely ratios, confirmation or change of `budget` and `design_ids`) are committed under `results/phase2/pilot/`. Evaluation-split scenarios are not used for tuning.

### 17.2 Plan C.2

1. The full test suite passes.
2. The main comparison and ablations on `v0` and `v0-bh50` complete with all LPs optimal and no validity violation, a rerun skips every task, and `results/phase2/main/` is committed.
3. Budget and sensitivity runs complete; sensitivity families are generated reproducibly with committed manifests; results are committed.
4. `results/phase2/statistics.csv` reports all eight primary comparisons with Holm-adjusted p-values, whatever their significance.
5. `docs/literature/review.md`, `docs/report/data/related_work.csv`, verified bibliography entries, and the recorded baseline decision exist.
6. The report compiles cleanly with the skill script, answers RQ1–RQ5, contains no placeholders, and `README.md` documents the Phase 2 commands.

## 18. Decision Log

- Evaluator-guided BCD instead of convex blocks (Section 3, spike 1).
- Design scoring on five sampled design realizations instead of expected rates (spikes 2–3); expected-rate design kept as an ablation.
- Bandwidth expressed as weights for a backlog-proportional policy, never as static schedules (spike 1; sub-project B cross-check).
- Trajectory search over a feasible-by-construction tour family instead of free waypoint moves.
- B3 uses leximin over delivered fractions to avoid a flat max–min objective.
- Trajectory bounds for search methods are computed inside their method tasks.
- Two plans (C.1 methods, C.2 experiments) under one spec, chosen by the user.

Adjustments made by plan C.2 from pilot evidence, the literature-baseline decision, and any change to budgets or families are appended here with date and reason.

### Plan C.2 adjustments (2026-09-11)

- **Report in plan C.3.** A plan cannot state results that do not exist yet, so the report work of Section 14, including `experiments/report_tables_phase2.py`, moves to plan C.3, written after the C.2 results exist. Plan C.2 covers Sections 11–13 and acceptance criteria 17.2.1–17.2.5; plan C.3 covers criterion 17.2.6. The method and related-work sections move with the rest of the report, so the report is edited in one plan.
- **Faster design scoring.** The C.1 pilot planned for 367–476 s per search task. Profiling put 41% of evaluation time in the ledger checker and 26% in connectivity, which depends only on the trajectory and the channel draw. `evaluate` gains `check=True` and `connectivity_cache=None`; `DesignScorer` scores without the checker and with one connectivity cache. On two development scenarios B2 and P return identical plans and keys and run 2.9 times faster. The runner's final evaluation still runs the checker.
- **Replicate filter.** A scenario set in an experiment configuration may list `replicates`; only those replicate numbers of the split are run.
- **Sensitivity and design-realization count on one replicate per topology.** Both run on replicate 0 of the ten evaluation topologies (ten scenarios, 30 realizations each) instead of all 30 scenarios, which keeps the C.2 compute near 3.5 hours with 12 workers; these results stay descriptive (Section 12). `phase2_sensitivity.yaml` also runs `v0` and `v0-bh50` as reference points. The design-realization variants (expected rates, 1, 10) run in `phase2_design.yaml`, because a configuration applies every method to every set; the five-realization point is P on `v0` in `phase2_sensitivity.yaml`.
- **Budget experiment.** `phase2_budget.yaml` includes budget 2000, so the quality–runtime curve comes from one run; its method ids are `B2_b<budget>` and `P_b<budget>`. In the smoke run, B2 and P with budget 4000 stopped after 2372 and 2136 calls because an iteration accepted nothing.
- **Degenerate family.** In `sens-physical` every source is ground-connected (mean ground-connected source fraction 1.0 against 0.506 in the other families), so the family is reported as degenerate for the role of the UAV.
- **Literature baseline.** The review in `docs/literature/review.md` (16 included papers) confirms Tran et al. (half-duplex variant) as the strong literature baseline; Samir et al. is the fallback because an implementation is available. Adaptation and implementation stay in sub-project D.
- **LP bounds re-solved without presolve (2026-09-12).** In the first main run HiGHS returned model status Unknown for 2 of 9000 trajectory bounds, the identical B2 and P trajectories of `v0-bh50/Garr200404-r2` at realization 18, whose constraint coefficients span 1.1e-13 to 1e6. `solve_lp` now re-solves an LP without presolve when the first solve is not optimal; without presolve both HiGHS simplex variants return the optimum 26.3928 for that instance. First solves that are optimal are unchanged, so Phase 1 results are unaffected; the main experiment was run again because the source hash changed.
