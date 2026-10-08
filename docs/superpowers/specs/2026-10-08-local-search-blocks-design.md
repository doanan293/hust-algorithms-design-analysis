# Local Search Blocks Design (Sub-Project E)

**Date:** 2026-10-08
**Status:** Draft for user review
**Builds on:** `2026-09-11-phase2-core-design.md` (search framework, Sections 5–9), `2026-09-12-extensions-final-delivery-design.md` (experiments, report, slides)
**Trigger:** reviewer feedback that the path and trajectory blocks of B2/B3/P "search" by switching among pre-generated options instead of modifying the incumbent

## 1. Purpose

Make the path and trajectory blocks of the evaluator-guided BCD (`src/optimization/`) genuine local search: each block derives its candidates from the incumbent plan and the state of the other blocks, and each move changes one part of the incumbent. The bandwidth block, the repair operations, the dispatcher ("transmit as soon as possible", EDF), the three decision variables, and the evaluator stay as they are. All Phase 2 experiments that involve a search method are rerun, and the report and slides are updated.

## 2. Diagnosis (what is wrong today)

- `trajectory_block` (`blocks.py:85-92`) replaces the whole trajectory by one of up to 90 tours enumerated once before the search (`tours.py:81-90`, `bcd.py:105`). The incumbent has no influence on the options.
- `path_block` (`blocks.py:45-55`) switches one alert at a time among the `None` option and the K = 3 paths of `candidate_paths` (`paths.py:75-112`). The move is a genuine 1-flip, but no path is ever generated during the search. On the 1440 alerts of `v0`, 503 alerts have exactly one UAV path, so for them the UAV route (downlink node and backhaul) never changes; for every alert the backhaul route is the fixed fewest-hop route of the entry node.
- `bandwidth_block` already perturbs the incumbent (weights of one link × 2 or × 0.5) and is kept unchanged.

## 3. Scope

### 3.1 Included

1. Plan representation: a plan stores its paths, not indices into a candidate set (Section 4).
2. Path neighbourhood generated from the incumbent path and the incumbent ledger (Section 5).
3. Trajectory neighbourhood: segment shifts of the incumbent trajectory, guided by the incumbent ledger (Section 6).
4. Search parameters and the ablation variant `P_catalog_trajectory` (Section 7).
5. Rerun of every Phase 2 experiment with search methods, statistics, report data (Section 8).
6. Report and slide updates (Section 9).
7. Tests and acceptance criteria (Sections 10–11).

### 3.2 Excluded (recorded as future work in the report)

- Extending the LP bound (P2) and the MILP (P3) beyond the K-path candidate set.
- A transmission schedule for the UAV (holding bits until a better slot) or time-dependent downlink nodes.
- Changes to the bandwidth block, TRAN, B0, B1.

## 4. Plan representation

### 4.1 Change

`Plan.path_choice: Mapping[str, int | None]` becomes `Plan.paths: Mapping[str, CandidatePath | None]`. `CandidatePath` is unchanged (`kind`, `entry`, `links`, `cost_s`); a path is a value, so a plan is self-contained.

- `Plan.from_choice(trajectory, candidates, choice, bandwidth)` builds a plan from indices for the index-based producers: B0, B1, `plan_from_iterate` (TRAN), `plan_from_solution` (bound cross-check), `calibrate_v0.py`.
- `chosen_paths` returns `plan.paths`. `simulate`, `check_ledger`, `evaluate`, `validate_plan` read paths from the plan and drop their `candidates` parameter. `candidate_paths` is still computed by the runner for B0/B1/TRAN initialisation and for the bounds.
- Radio links for channel construction and bandwidth weights come from the paths in the plan: `radio_links_of(plan)` ∪ `all_uav_links(scenario)` (the union already exists in `evaluate.py:108`).
- `WeightedBacklogged.weight` already treats a missing link as weight 0; new links get weight 1 when they enter the plan (Section 5.5).

### 4.2 Validation (`validate_plan`)

For every alert with a path:

- `links[0] == (ACCESS, alert.source, x)`. If `x == UAV_ID`, `kind == VIA_UAV` and `links[1] == (DOWNLINK, UAV_ID, entry)`; otherwise `kind == GROUND`, `x == entry` and the source is within `ground_range_m` of `entry`.
- Every later link is `(BACKHAUL, u, v)` with `scenario.backhaul_capacity_bps(u, v) > 0` (alive edge), the chain is contiguous (`v` of one link is `u` of the next), the last `v` is `scenario.center`, and no node appears twice. Zero backhaul links are allowed when `entry == center`.

A plan built by `from_choice` from `candidate_paths` passes these checks (they describe what `candidate_paths` already produces).

### 4.3 Reproduction check

Before any new experiment runs, `reproduce.py --compare` on `results/phase1` and on the B0/B1/TRAN rows of `results/phase2/*` must show no differences (non-runtime columns). This is the gate for the representation change.

## 5. Path block

### 5.1 Neighbourhood of one alert

Module `src/optimization/path_moves.py`. Input: scenario, alert, incumbent path `p` (or `None`), `BackhaulRoutes` (fewest-hop routes, `paths.py:41-66`), congestion of the incumbent (Section 5.3), parameter `entry_neighbors = M`. Output: an ordered list of distinct paths, each valid under Section 4.2 and different from `p`.

1. **Kind flip at the same entry.** `GROUND ↔ VIA_UAV` with `entry` unchanged; the ground direction only if the source is within ground range of the entry. Backhaul part unchanged.
2. **Entry switch.** The `M` entry nodes nearest (Euclidean) to the current entry, excluding it, that are reachable (`routes.hop_count`). Kind unchanged; a ground path to an entry out of range is skipped. Backhaul = fewest-hop route of the new entry (same as `candidate_paths`).
3. **Backhaul detour.** Let `(u, v)` be the backhaul link of `p` with the most congested slots inside `[r_a, d_a)` (ties: the earlier link on the path); skip the move when no backhaul link of `p` is congested in that window. Compute the fewest-hop route from `u` to the center in the alive graph without edge `{u, v}` and without the nodes of `p` before `u` (ties broken by Euclidean length then node id, as in `backhaul_routes`). If a route exists and the result differs from `p`, the neighbour is prefix of `p` up to `u` + detour.
4. **Unserve.** `None`.

If `p is None`, the neighbourhood is the K candidate paths of the alert (restart from the catalogue), then nothing else.

`cost_s` of a generated path uses the same formula as `candidate_paths` (`access time + backhaul delay`), so Operation 2 can still order paths by cost.

### 5.2 Block procedure

```
path_block(scenario, routes, scorer, state, params):
    _, result = scorer.score_with_ledger(state.plan())      # 1 evaluation; gives congestion
    congestion = backhaul_congestion(result)                 # Section 5.3
    accepted = False
    for alert in alerts sorted by (deadline_slot, id):
        for q in path_neighbours(alert, state.paths[alert.id], congestion, ...):
            if state.try_change(scorer, paths={**state.paths, alert.id: q}, weights=with_unit_weights(q)):
                accepted = True            # first improvement, continue with the new incumbent
    return accepted
```

Acceptance stays "key increases strictly" (`SearchState.try_change`). Each `try_change` is one evaluation under the budget.

### 5.3 Congestion from the ledger

`congested_slots(ledger)` (`repair.py:53-62`) already gives, per link, the slots in which the link carried its whole capacity. For the path block: `congestion[link] = Σ_realizations |congested slots of link ∩ [r_a, d_a)|` computed per alert when ranking the backhaul links of its path. This makes the generated paths depend on the current trajectory and bandwidth (through the ledger).

### 5.4 Operation 2 of the repair round

`path_change_order` is replaced: for an at-risk alert, candidates are `path_neighbours(...)` (Section 5.1) ordered by `(number of links in risk.congested_links, cost_s)`; first improvement wins, as today. `backlog=False` keeps the order `(cost_s)` only.

### 5.5 Weights of new links

When a neighbour introduces a radio link that has no weight vector, `SearchState` adds `np.ones(num_slots)` for it before scoring. Weights of links that leave the plan are kept (harmless, weight is only read for links in the plan).

### 5.6 Budget per pass

1 + Σ_alerts |neighbourhood| ≈ 1 + 40 × (1 + 3 + ≤1 + 1) ≈ 240 evaluations with M = 3.

## 6. Trajectory block

### 6.1 Segment shift

Module `src/optimization/trajectory_moves.py`.

`shift_segment(trajectory, i, j, direction, fraction, step_m) -> np.ndarray | None` with `1 ≤ i ≤ j ≤ N−1`: adds `λ·d` to `q[i..j]`, where `d` is the unit direction and `λ = fraction · λ*`. `λ*` is the largest shift keeping both boundary steps feasible, `‖q[i−1] − (q[i] + λd)‖ ≤ step_m` and `‖(q[j] + λd) − q[j+1]‖ ≤ step_m`, obtained from the two quadratic inequalities (minimum of the two positive roots). Returns `None` when `λ* < 1e-6 m`. Interior steps are unchanged, `q[0]` and `q[N]` are never moved, so every output satisfies `_trajectory_violations`.

When `λ* ≈ 0` (a boundary lies on a cruise segment flown at `V_max`), the segment is widened by one point on each side, up to `max_widening = 10` times, until `λ* > 0`; otherwise the segment is dropped.

### 6.2 Segment selection from the ledger

```
trajectory_block(scenario, scorer, state, params):
    _, result = scorer.score_with_ledger(state.plan())              # 1 evaluation
    segments = ranked_uav_segments(scenario, state.plan(), result)  # below
    accepted = False
    for (i, j, direction) in segments[:W]:
        for fraction in (1.0, 0.5):
            t = shift_segment(state.trajectory, i, j, direction, fraction, step_m)
            if t is not None and state.try_change(scorer, trajectory=t):
                accepted = True
                break
    return accepted
```

`ranked_uav_segments`:

1. For each radio link of the plan with the UAV as receiver (uplink `(ACCESS, source, UAV)`) or transmitter (downlink `(DOWNLINK, UAV, node)`), and each slot `n`, `score(link, n) = (number of realizations in which the link is congested at n) × Σ_{alerts a on link, r_a ≤ n < d_a} size_a / (d_a − r_a)`. The second factor favours large and urgent alerts (the reviewer's "size A > size B or deadline A < deadline B" case).
2. Consecutive slots with `score > 0` on the same link form one segment; slot range `[n1, n2]` maps to points `i = max(n1, 1)`, `j = min(n2 + 1, N−1)`.
3. Segments are ranked by total score, descending; ties by link id and `n1`. The first `trajectory_segments = W` segments are tried.
4. Direction: unit vector from the centroid of `q[i..j]` to the ground point of the link (source position for an uplink, node position for a downlink). Zero vector → segment skipped.

No congested UAV link → no moves → the block returns `False`.

### 6.3 Catalogue initialisation and the catalogue mode

- `catalog_init = True` (default): once, before the BCD loop, the old catalogue pass runs (`tour_family`, try every tour, keep the best). Cost ≤ |family| ≈ 90 evaluations. It gives the local moves a good starting tour; it is initialisation, not search.
- `trajectory_mode = "local"` (default) uses Section 6.2 in every iteration. `trajectory_mode = "catalog"` uses the old catalogue pass in every iteration (today's behaviour); it exists only for the ablation `P_catalog_trajectory`.
- `trajectory_block = False` disables both the catalogue initialisation and the block, so `P_fixed_trajectory` keeps today's meaning (B1 trajectory throughout).

### 6.4 Budget per pass

1 + 2W = 11 evaluations with W = 5.

## 7. Parameters, methods, configuration

`SearchParams` gains:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `trajectory_mode` | `"local"` \| `"catalog"` | `"local"` | Section 6.3 |
| `catalog_init` | bool | `True` | Section 6.3 |
| `entry_neighbors` | int ≥ 1 | 3 | M of Section 5.1 |
| `trajectory_segments` | int ≥ 1 | 5 | W of Section 6.2 |
| `max_widening` | int ≥ 0 | 10 | Section 6.1 |

`from_mapping` validates them like the existing fields. `budget`, `max_iterations`, the block switches, `operation1/2`, `backlog`, `design_ids`, `objective` are unchanged.

The BCD loop order stays path → bandwidth → trajectory → repair; it stops when an iteration accepts nothing, `max_iterations` is reached, or `BudgetExhausted` is raised (accepted work survives, as today).

`configs/experiments/phase2_main.yaml` adds
`{id: P_catalog_trajectory, base: search, params: {operation1: true, operation2: true, trajectory_mode: catalog}}`;
`experiments/report_tables_phase2.py` `ABLATIONS` adds `("P_catalog_trajectory", "Catalogue trajectory block")`.

A pilot on the development split (`phase2_pilot.yaml`) with M ∈ {2, 3, 5} and W ∈ {3, 5, 10} picks the defaults; the chosen values and the pilot numbers go into Section 12 of this document.

## 8. Experiments

Order, from `docs/reproducibility.md` Section 7, restricted to what changes:

1. `reproduce.py --compare` gate (Section 4.3).
2. `phase2_pilot` (parameter choice, Section 7).
3. `phase2_main`, `phase2_budget`, `phase2_sensitivity`, `phase2_design`, `phase2_case_study`, `case_study_trace`.
4. `phase2_statistics.py`, `phase2_literature_statistics.py` (TRAN rows are reused; comparisons against P and B2 are recomputed).
5. `report_tables.py`, `report_tables_phase2.py`, `report_tables_extensions.py`.
6. `reproduce.py --level tables`, `--level scenarios`; `--level experiments --subset Agis`.

Expectations: planning runtime of a search task is about 145 s today; `phase2_main` has about 660 search tasks (11 search variants × 60 scenarios) on 12 workers, so 2–3 hours; the whole chain 6–10 machine hours. Runs use `systemd-run --user --scope -p MemoryMax=14G` as in the README. Old shards of search methods are removed before rerunning (shards are keyed by method id; a stale shard would be silently reused).

The K-sensitivity panel (`sens-k2`, `sens-k5`) is kept; its meaning becomes "sensitivity to the initial candidate set and to the bound", stated in the report.

Bounds (P2)/(P3) stay on the K-path set. The report calls them bounds of the K-restricted problem; a method value above such a bound is legitimate and is reported as a negative gap, not hidden.

Result manifests record the new git commit; `docs/reproducibility.md` Sections 4–7 are updated for the new variant and the rerun.

## 9. Report and slides

Report (`docs/report/sections/`):

- Proposed Method: the two blocks are described as neighbourhoods of the incumbent (Sections 5–6 of this document), with updated pseudocode; the catalogue is presented as initialisation.
- Experimental Evaluation: ablation table gains the `P_catalog_trajectory` row; text for the K panel and for the K-restricted bounds; all numbers regenerated.
- Discussion / Conclusions: future work gains the excluded items of Section 3.2.
- Build with `compile_latex.sh main.tex --preview`; `main.pdf` is tracked and committed.

Slides (`docs/presentation/uav-alert-delivery.pptx`, edited in place with the pptx skill): method slides (block descriptions, pseudocode figure if any) and result slides (main table, ablation, sensitivity figures) follow the new numbers; layout unchanged.

## 10. Tests

- `tests/models/test_plan.py`: `from_choice` equals a plan built from the same paths; validation rejects a broken chain, a repeated node, a dead edge, a ground entry out of range, a wrong first link.
- `tests/optimization/test_path_moves.py`: every neighbour passes `validate_plan`; kind flip respects ground range; entry switch returns ≤ M distinct entries; detour avoids the congested link and the prefix nodes, and is absent without congestion; `None` incumbent yields the K candidates.
- `tests/optimization/test_trajectory_moves.py`: `shift_segment` output satisfies `_trajectory_violations` for random segments and directions; returns `None` on a V_max cruise boundary without widening; widening finds a feasible segment; `fraction = 1.0` touches the limit within tolerance.
- `tests/optimization/test_blocks.py`, `test_search_methods.py`, `test_repair.py`: updated for `paths`, the ledger call, `trajectory_mode`, `catalog_init`; a test that `trajectory_mode="catalog"` reproduces the old block's choice on a small scenario.
- `tests/runner`: config parsing of the new parameters.

## 11. Acceptance criteria

1. `uv run --extra test pytest -q` passes.
2. Section 4.3 gate: B0, B1, TRAN, Phase 1 bounds reproduce exactly.
3. Every plan returned by a search method passes `validate_plan` on every evaluation scenario (runner already raises on an infeasible plan).
4. Every evaluation during search goes through `DesignScorer` (budget respected; `evaluate_calls ≤ budget + 30` per task, the 30 being the final evaluation realizations).
5. `phase2_main` summary contains the 13 methods including `P_catalog_trajectory`; statistics and report tables regenerate without manual edits.
6. `main.pdf` and the slides reflect the regenerated numbers; `reproduce.py --level tables` passes.

## 12. Pilot results and decisions

To be filled during implementation: chosen M and W, pilot table (development split, B2/P with local vs catalogue trajectory mode), and any deviation from this document with its reason.
