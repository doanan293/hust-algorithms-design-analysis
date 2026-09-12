# Extensions and Final Delivery Design (Sub-Project D)

**Date:** 2026-09-12
**Status:** Approved for implementation planning
**Roadmap:** `docs/superpowers/specs/2026-09-11-implementation-roadmap.md` Section 3.4
**Builds on:** sub-project C (`2026-09-11-phase2-core-design.md`, plans C.1–C.3, merged on `develop`)
**Requirements:** `docs/issue.md` Sections 1.1, 6.3–6.4, 7, 10, 13–14

## 1. Purpose

Finish the course project: implement and evaluate the strong literature baseline chosen in `docs/literature/review.md` (Tran et al. 2022, half-duplex variant), run a case study whose targets come from RescueNet damage masks, restate the design-realization study as sample average approximation (SAA), build a reproducibility package that regenerates and checks the main tables, and complete the report.

## 2. Scope and Delivery

### 2.1 Included

- Tran HD adaptation as a planning method `TRAN`, its development-set pilot, a comparison experiment on `v0` and `v0-bh50`, paired statistics against P and B2, and its point on the quality–runtime figure (Sections 5–7).
- RescueNet target extraction with georeferencing from image metadata, a replaceable source-point generator, the `rescuenet` scenario set, the case-study experiment, and the map and delivery-progress trace (Sections 8–10).
- SAA framing of the C design-realization results in the report (Section 11).
- `experiments/reproduce.py` with three check levels and `docs/reproducibility.md` (Section 12).
- Final report updates (Section 13).

### 2.2 Excluded

- Tran full-duplex variant, throughput maximization (Algorithm 4), transmit-power optimization.
- A new SAA or chance-constrained method.
- Samir et al. 2020, unless the fallback rule of Section 5.8 fires.
- Docker or other container packaging.
- RescueNet train/test archives, damage-level priority weights, no-fly zones.
- Redistribution of RescueNet images or masks, or figures derived from them.

### 2.3 Three plans

- **Plan D.1 — literature baseline:** Sections 5–7.
- **Plan D.2 — RescueNet case study:** Sections 8–10. It runs after D.1 because the case study uses the pilot-selected `TRAN` parameters.
- **Plan D.3 — reproducibility and final report:** Sections 11–13, written after the D.1 and D.2 results exist.

## 3. Evidence and Considered Approaches

### 3.1 Tran et al. 2022 (arXiv 2008.00218, full text read)

- **Model:** K devices relay data through a UAV to one gateway (GW). Device k uploads in $\mathcal{T}_{1k}=\{n_{\mathrm{start},k},\dots,n_{\mathrm{end},k}\}$ and the UAV forwards in $\mathcal{T}_{2k}=\{n_{\mathrm{end},k}+1,\dots,N\}$ (eq. 15). Device k is served when $\delta_t\min(R_{1k},R_{2k})\ge\lambda_k S_k$ with binary $\lambda_k$ (eqs. 16, 50c).
- **HD problem (50):** maximize $\|\lambda\|_1$ over trajectory, bandwidth fractions $a_{ik}[n]$, power, and $\lambda$, subject to per-slot bandwidth sums $\le 1$, speed, endpoints, power budgets, and UAV storage $C$.
- **Algorithm 3:** relax $\lambda\in[0,1]$ with penalty $\mu\sum_k\lambda_k(\lambda_k-1)$ linearized at the current point (eq. 30); slack $z$ with $H^2+\|q-w\|^2\le z^{2/\alpha}$; concave lower bounds of the rate from Lemma 3 and a difference-of-convex Taylor bound of $a\Phi$ (eqs. 51–57); solve the convex program (58) repeatedly until convergence. The paper names no solver, gives no numeric $\mu$ or tolerance, and states per-iteration complexity $O\big((N(7+8K)+4K)^{0.5}(3N(1+4K)+K)^3\big)$.
- **Benchmarks:** BHD1 (straight-line trajectory) and BHD2 (fixed equal resources).

Each iteration has second-order-cone and power constraints, so HiGHS (LP only) cannot solve it; `cvxpy` with its bundled Clarabel solver is added.

### 3.2 RescueNet metadata and masks (measured on the 30 selected pairs)

- The paper (Rahnemoonfar et al., Scientific Data 2023) states DJI Mavic Pro imagery from Mexico Beach, 11–14 October 2018, 3000×4000 px; it gives no altitude or ground sampling distance (GSD).
- Every validation image has EXIF GPS and DJI XMP fields `RelativeAltitude` (58.4–75.1 m, median 60.9 m), `GimbalPitchDegree` (−90.0 to −89.9) and `GimbalYawDegree`. 428 of 449 images (camera FC220) carry `CalibratedFocalLength` (3032.3 or 3051.6 px); 21 images (camera FC2103, 4.5 mm lens) do not, one of them among the selected pairs.
- GSD of the selected images: 1.96–2.46 cm/px, so one image covers about 80 × 60 m.
- Damage classes 3–5 form 68 eight-connected regions of 19.9–453 m² in 27 of the 30 images.
- Image centres span 2.85 × 1.49 km; nearest-neighbour distances are 19–347 m, so some images overlap.
- With the yaw rotation of Section 8.2, the same building seen in two overlapping images lands 2.7–8.2 m apart (five pairs); with the opposite rotation sign the closest pairs are ≥ 16 m apart. Merging within 10 m leaves 63 targets; 20 m starts merging distinct buildings.

### 3.3 Reproducibility state

`uv.lock` pins packages; `data.cli verify` checks raw-data SHA-256; scenario generation is deterministic and every result manifest records scenario-manifest SHA-256, config hash, source hash, git commit, and environment; report generators take `--output`. Result files contain runtime columns that differ between runs, and the run scripts always write to the configured output directory.

### 3.4 Approaches

| Part | Chosen | Rejected |
|---|---|---|
| Literature baseline | Tran HD inner approximation at slot resolution, with a pre-declared 5-slot aggregation fallback | Aggregation from the start (coarser than the paper); Samir et al. directly (no second hop) |
| Case-study geometry | Georeferenced targets from image metadata, normalized to the 2 km box | Assumed pixel–metre scale per image (ignores available metadata); true scale in a 3 km box (breaks comparability with `v0` physics) |
| Reproducibility | One checking script with three levels plus a mapping document | Docker (extra tooling, `uv.lock` already pins packages); README commands only (no automatic check) |

## 4. Architecture

```text
src/literature/                    new package (added to runner source hash)
  tran_model.py                    adaptation: gateways, windows, effective channel, instance
  tran_ia.py                       convex program (58) in cvxpy, IA loop
  tran_plan.py                     solution -> Plan with StaticSchedule
  tran.py                          TranHD method, TranParams
src/data/rescuenet_targets.py      metadata, regions, georeferencing, merging, normalization
src/data/scenarios.py              source-placement interface (synthetic | rescuenet)
src/data/cli.py                    command rescuenet-targets
src/runner/config.py, methods.py,  base TRAN; planned-trajectory bounds inside method tasks
  experiment.py, provenance.py
experiments/tran_pilot.py          development-set pilot
experiments/phase2_literature_statistics.py
experiments/case_study_trace.py
experiments/report_tables_extensions.py
experiments/reproduce.py
experiments/run_phase1.py, run_phase2.py   option --output-dir
configs/data/rescuenet_targets.yaml
configs/scenarios/rescuenet.yaml
configs/experiments/phase2_literature.yaml, phase2_case_study.yaml
docs/reproducibility.md
```

`src/models` does not change in sub-project D. The only new dependency is `cvxpy>=1.6,<2` in `[project].dependencies`, installed into `.venv` with `uv`; its version and Clarabel's are added to `provenance.environment()`.

## 5. Method TRAN — Tran HD Adaptation

### 5.1 Instance construction (`tran_model.py`)

- **Devices:** every alert $k$ is one device with source position $w_k$, release $r_k$, deadline $d_k$ (usable slots $[r_k,d_k)$, as in the simulator), and size $S_k$.
- **Path and gateway:** the alert uses its candidate with the smallest `cost_s` (ties: lower index). For a via-UAV candidate the gateway $g_k$ is its entry node; a ground candidate makes $k$ a ground device.
- **Backhaul slots:** $b_k=\lceil \text{backhaul delay}/\text{slot\_s}\rceil$ with the delay of `models.paths` for the chosen route.
- **Windows:** with $L_k=d_k-b_k-r_k$ radio slots and split fraction $\theta$, a UAV device collects in $\mathcal{T}_{1k}=\{r_k,\dots,r_k+\lceil\theta L_k\rceil-1\}$ and forwards in $\mathcal{T}_{2k}=\{r_k+\lceil\theta L_k\rceil,\dots,d_k-b_k-1\}$. A ground device transmits in $\{r_k,\dots,d_k-b_k-1\}$. A device whose window (either window for UAV devices) is empty gets $\lambda_k=0$ fixed.
- **Durations:** uplink throughput uses $\delta_1=\text{access\_fraction}\cdot\text{slot\_s}$, downlink uses $\delta_2=(1-\text{access\_fraction})\cdot\text{slot\_s}$.
- **Power:** fixed at `source_power_w` for uplinks and `downlink_power_w` for downlinks; no power variables.
- **Channel:** Tran's power law $\omega_0 d^{-\alpha}$ is replaced by an effective law $\beta_{\mathrm{eff}} d^{-\alpha_{\mathrm{eff}}}$ fitted by least squares on $\ln$ of the expected PrLoS gain $\beta_0\big(P_{\mathrm{LoS}}(\rho)d^{-\alpha_{\mathrm{LoS}}}+(1-P_{\mathrm{LoS}}(\rho))\eta\, d^{-\alpha_{\mathrm{NLoS}}}\big)$, $d=\sqrt{h^2+\rho^2}$, on horizontal distances $\rho\in[0,\sqrt2 L_{\mathrm{box}}]$ in 10 m steps. The fit's $R^2$ and maximum absolute log error are reported by the pilot.
- **Noise:** as in the paper, over the whole band: $\sigma^2=N_0 B_{\mathrm{tot}}$, so $\Phi=B_{\mathrm{tot}}\log_2(1+c/z)$ with $c=p\,\beta_{\mathrm{eff}}/\sigma^2$.
- **Ground devices:** constant rate $\rho_k=B_{\mathrm{tot}}\log_2(1+\mathrm{snr}_k/B_{\mathrm{tot}})$ with `ground_snr_per_hz` at the source–entry distance; they have no downlink stage and share the uplink bandwidth sum.
- **Storage:** constraint (50e) is dropped; the UAV buffer is unbounded in this model.

### 5.2 Convex program per iteration (`tran_ia.py`)

Variables: trajectory points $q[0..N]$; bandwidth fractions $a_{1k}[n]$ on $\mathcal{T}_{1k}$ and $a_{2k}[n]$ on $\mathcal{T}_{2k}$ (non-negative); $\lambda_k\in[0,1]$; for each ground point $w$ (sources and gateways) and slot $n$, slacks $z_w[n]$ and $\Phi^{\mathrm{lb}}_w[n]$; per device and window slot, rate slacks $\hat r_{ik}[n]$.

1. Objective: maximize $\sum_k\lambda_k+\mu\hat P^{(j)}(\lambda)$ with $\hat P^{(j)}(\lambda)=\sum_k\big(\lambda^{(j)}_k(2\lambda_k-1)-(\lambda^{(j)}_k)^2\big)$ (eq. 30).
2. Trajectory: $q[0]=q[N]=$ UAV start/end; $\|q[n+1]-q[n]\|\le v_{\max}\,\text{slot\_s}$; rates use slot midpoints $m[n]=(q[n]+q[n+1])/2$, as in the evaluator.
3. Distance slack: $h^2+\|m[n]-w\|^2\le z_w[n]^{2/\alpha_{\mathrm{eff}}}$ (the paper's $z$ constraints of (58)).
4. Rate lower bound: with fixed power, Lemma 3 reduces to the tangent of the convex function $\Phi(z)$: $\Phi^{\mathrm{lb}}_w[n]\le \Phi(z^{(j)})+\Phi'(z^{(j)})(z-z^{(j)})$.
5. Product bound (eq. 57): $\hat r_{ik}[n]\le\tfrac14\big[2(a^{(j)}+\Phi^{(j)})(a+\Phi^{\mathrm{lb}})-(a^{(j)}+\Phi^{(j)})^2\big]-\tfrac14(a-\Phi^{\mathrm{lb}})^2$, where $a=a_{ik}[n]$ and $\Phi$ belongs to $w_k$ for $i=1$ and $g_k$ for $i=2$.
6. Service (58c): $\delta_1\sum_{n\in\mathcal{T}_{1k}}\hat r_{1k}[n]\ge\lambda_kS_k$ and $\delta_2\sum_{n\in\mathcal{T}_{2k}}\hat r_{2k}[n]\ge\lambda_kS_k$; ground devices: $\delta_1\sum_n a_{1k}[n]\rho_k\ge\lambda_kS_k$. The paper's window-length constraints $\lambda_kS_k/\hat R_{ik}\le|\mathcal{T}_{ik}|\delta_t$ follow from these because $\delta_i\le\text{slot\_s}$, so they are not added; the storage constraint is dropped (Section 5.1).
7. Bandwidth: $\sum_k a_{1k}[n]\le1$ and $\sum_k a_{2k}[n]\le1$ per slot.

Sharing $z$ and $\Phi^{\mathrm{lb}}$ per ground point instead of per device is an exact reformulation (the paper's per-device slacks take equal values at the optimum) that shrinks the program.

### 5.3 Initialization and stopping

- Initial point: the B1 zone-tour trajectory, $a$ split equally among devices active in each slot, $\lambda=0$, and $z$, $\Phi$ evaluated at that trajectory. With $\lambda=0$ this point satisfies (58), so the feasibility search (44) is not needed; the report states this difference.
- Stop when the objective changes by less than `tolerance` (relative, default $10^{-3}$) or after `max_iterations` (default 30).
- If Clarabel returns a status other than optimal, stop and keep the last accepted iterate.

### 5.4 Plan conversion (`tran_plan.py`)

- Trajectory: $q$ as solved.
- `path_choice`: the candidate index chosen in Section 5.1.
- `StaticSchedule`: access link (source→UAV) gets $B_{\mathrm{tot}}\sum a_{1k}[n]$ over its UAV devices, ground access link (source→entry) gets $B_{\mathrm{tot}}\sum a_{1k}[n]$ over its ground devices, downlink (UAV→entry) gets $B_{\mathrm{tot}}\sum a_{2k}[n]$.
- In each slot only the `max_active_downlinks` downlinks with the largest bandwidth keep it (ties: link order); the others get zero.
- `validate_plan` must return no violation; otherwise the method raises.

### 5.5 Parameters (`TranParams`)

`theta` ∈ (0, 1), `mu` > 0, `max_iterations` (30), `tolerance` (1e-3), `block_slots` (1). Parameters are validated at construction.

### 5.6 Time aggregation fallback

With `block_slots = G > 1`, slots are grouped into blocks of $G$: trajectory points at block boundaries with speed limit $G\,v_{\max}\,\text{slot\_s}$, one bandwidth value per block, windows rounded inward to whole blocks. The plan interpolates the trajectory linearly inside each block (feasible) and repeats block bandwidth per slot. The fallback is used with $G=5$ when the pilot's median planning runtime per development scenario at $G=1$ exceeds 600 s.

### 5.7 Complexity

The report states the paper's per-iteration bound and the reduced counts of the shared-slack program: $2(N+1)$ trajectory variables, $K$ values of $\lambda$, $2N|W|$ point slacks for $|W|$ ground points, and $2\sum_k(|\mathcal{T}_{1k}|+|\mathcal{T}_{2k}|)$ bandwidth and rate variables, together with measured iterations and runtime.

### 5.8 Fallback to Samir et al.

If, after the Section 5.6 fallback, `TRAN` still fails to return a valid plan better than its initial point on at least half of the development scenarios, execution stops and the evidence goes to the user before any switch to Samir et al.

## 6. Runner Changes

- `config.py`: base `TRAN` with an id following the search-method id rule, `params` keys of Section 5.5, and optional `bounds`.
- `methods.py`: `build_method` returns `TranHD(spec.id, TranParams(**params))`; `bound_owner` treats `TRAN` like a search method.
- `experiment.py`: bound tasks are built only for B0/B1; methods whose trajectory exists only after planning (`search`, `TRAN`) compute bounds inside their method tasks.
- `provenance.py`: `literature` joins `SOURCE_PACKAGES`; environment adds `cvxpy` and `clarabel` versions.
- `run_phase1.py`, `run_phase2.py`: `--output-dir` overrides the configured output directory.

## 7. Literature Experiments (Plan D.1)

### 7.1 Pilot — `experiments/tran_pilot.py`

- Six development scenarios of `v0`, 30 evaluation realizations each.
- Step 1: $\theta=1/2$, $\mu=1$, $G=1$; apply the Section 5.6 rule.
- Step 2: grid $\theta\in\{1/3,1/2,2/3\}$ × $\mu\in\{1,10\}$ at the chosen $G$. Selection: highest mean timely ratio over the six scenarios; ties go to $\theta=1/2$, then the smaller $\mu$.
- Output `results/phase2/tran_pilot/`: `runs.csv` (scenario, θ, μ, G, iterations, final status, relaxed objective, penalty value, planning runtime, timely ratio), `channel_fit.csv` ($\beta_{\mathrm{eff}}$, $\alpha_{\mathrm{eff}}$, $R^2$, max log error per scenario), and `pilot_note.md` (choice and reasons). Evaluation-split scenarios are not used.

### 7.2 Comparison — `configs/experiments/phase2_literature.yaml`

B1 and `TRAN` (pilot parameters, `bounds: true`) on the 30 evaluation scenarios of `v0` and of `v0-bh50`, realizations 0–29. Results in `results/phase2/literature/`.

### 7.3 Statistics — `experiments/phase2_literature_statistics.py`

- **Anchor check:** B1 rows of `results/phase2/literature` must equal B1 rows of `results/phase2/main` on `timely_count`, `connected_count` and `shortfall` for every (set, scenario, realization). Any difference stops the script: it would mean scenarios or evaluator changed, and pairing with the C results would be invalid.
- **Comparisons:** P − TRAN and B2 − TRAN on `v0` and `v0-bh50` (four), with P and B2 from `results/phase2/main`, using `src/analysis/statistics.py` unchanged: topology-level mean differences, exact sign-flip test, Holm over these four, bootstrap interval, matched-pairs rank-biserial.
- Output `results/phase2/statistics_literature.csv` with the columns of `results/phase2/statistics.csv`.

### 7.4 Quality–runtime

The report's budget figure adds one `TRAN` point per scenario set: mean planning runtime against mean timely ratio.

## 8. RescueNet Targets (Plan D.2)

### 8.1 Configuration — `configs/data/rescuenet_targets.yaml`

Selection manifest `data/manifests/rescuenet_selection.csv`, damage classes `[3, 4, 5]`, `min_area_m2: 10`, `merge_radius_m: 10`, `box_size_m: 2000`, `utm_epsg: 32616`, `max_pitch_deviation_deg: 2`, camera fallback table `FC2103: {focal_mm: 4.5, sensor_width_mm: 6.17}`.

### 8.2 Extraction and georeferencing (`rescuenet_targets.py`)

1. **Metadata:** GPS latitude/longitude from EXIF; `RelativeAltitude`, `GimbalPitchDegree`, `GimbalYawDegree`, `CalibratedFocalLength`, `CalibratedOpticalCenterX/Y` from XMP. Focal length in pixels is `CalibratedFocalLength`, or `focal_mm × width_px / sensor_width_mm` from the fallback table. The optical centre defaults to the image centre.
2. **Regions:** eight-connected components of pixels whose class is in the damage list; components below `min_area_m2` are dropped. The representative point is the region pixel nearest the region centroid, so it lies inside the building. Each region records area (m²) and its largest damage class.
3. **Projection:** $\mathrm{GSD}=\text{RelativeAltitude}/f_{\mathrm{px}}$ (flat terrain, altitude above ground). With pixel column $u$, row $v$, centre $(c_x,c_y)$: $dx=(u-c_x)\,\mathrm{GSD}$, $dy=(c_y-v)\,\mathrm{GSD}$, and yaw $\psi$ clockwise from north: $E=E_0+dx\cos\psi+dy\sin\psi$, $N=N_0-dx\sin\psi+dy\cos\psi$, with $(E_0,N_0)$ the image centre in UTM zone 16N via `pyproj`.
4. **Merging:** single-linkage groups of targets within `merge_radius_m` (union–find, independent of order); the group's representative is its largest-area member; `merged_from` lists all members.
5. **Normalization:** $s=L_{\mathrm{box}}/\max(\Delta E,\Delta N)$ over merged targets; $x=(E-E_{\min})s+(L_{\mathrm{box}}-\Delta E\,s)/2$, $y=(N-N_{\min})s+(L_{\mathrm{box}}-\Delta N\,s)/2$.

### 8.3 Outputs — `uv run python -m data.cli rescuenet-targets --config configs/data/rescuenet_targets.yaml`

- `data/manifests/rescuenet_targets.csv`: `target_id`, `pair_id`, `region_label`, `pixel_u`, `pixel_v`, `area_m2`, `damage_class`, `lat`, `lon`, `utm_e`, `utm_n`, `x_m`, `y_m`, `merged_from`.
- `data/manifests/rescuenet_targets.json`: configuration hash, selection-manifest SHA-256, per-image metadata (GSD, yaw, pitch, focal source), normalization ($s$, offsets, extents), merged-pair distances as the georeferencing-error estimate, dropped regions with reasons.

## 9. Source Placement Interface and Case-Study Scenarios

- `scenarios.py` gains `place_sources(config, seed, box_m) -> (zones, sources)`, selected by `workload.source_generator` (`synthetic` default).
- **`synthetic`** runs the existing `sample_damage_zones` and `sample_sources` with the same named RNG streams, so every existing scenario set regenerates with unchanged manifest SHA-256.
- **`rescuenet`** reads `workload.targets_manifest` and requires its SHA-256 to match `workload.targets_manifest_sha256`:
  - sources `s00`, `s01`, … follow the target-manifest row order (the mapping to images and regions stays in the target manifest);
  - zones: deterministic k-means with `zone_count` clusters, farthest-point initialization from the target with the smallest $(x,y)$, Lloyd iterations until assignments stop changing (at most 100); zone centres are cluster means sorted by $(x,y)$; radius `zone_radius_m`; each source's `zone` is its cluster;
  - the `damage-zones` and `sources` RNG streams are not drawn; alert, size and failure streams are unchanged; `provenance` records the target-manifest SHA-256.
- `configs/scenarios/rescuenet.yaml`: identical to `v0.yaml` except `set_id: rescuenet`, `source_generator: rescuenet`, the target-manifest fields, and without the synthetic-only fields `source_count`, `source_sigma_m`, `zone_min_separation_m` and `zone_max_attempts`, which `load_scenario_set_config` then requires only for `synthetic`. The same selection seed and strata give the ten `v0` evaluation topologies (three replicates) and the `v0` development topologies.
- Validation errors: unknown generator, missing or mismatched target manifest, fewer targets than zones, synthetic fields given to `rescuenet`.

## 10. Case-Study Experiment and Trace

### 10.1 Experiment — `configs/experiments/phase2_case_study.yaml`

B0, B1, B2 (`bounds: true`), P (`bounds: true`), and `TRAN` (pilot parameters, `bounds: true`) on the 30 evaluation scenarios of `rescuenet`, realizations 0–29. Results in `results/phase2/case_study/`. Results are descriptive: one geographic layout, overlapping images, no hypothesis tests.

### 10.2 Trace — `experiments/case_study_trace.py`

- Representative scenario: the one whose P mean timely ratio is closest to the median over the 30 scenarios (ties: smallest `scenario_id`).
- Replans B1, P and `TRAN` on it and checks each timely ratio against `results/phase2/case_study/method_realizations.csv`; a mismatch stops the script.
- Writes `results/phase2/case_study/trace/`: `targets.csv` (x, y, damage class), `nodes.csv` (id, x, y, role), `edges.csv` (endpoints, failed flag), `trajectories.csv` (method, slot, x, y), `progress.csv` (method, slot, mean cumulative timely alerts over 30 realizations, from the simulator's delivery slots), and `trace.json` (scenario id, rule, checked ratios).

## 11. SAA Framing (Plan D.3)

The P design score averages the lexicographic objective over five sampled design realizations, which is an SAA of the expected objective with sample size 5; the expected-rate variant is the deterministic equivalent. The report restates the `phase2_design` results (expected, 1, 5, 10 realizations) as the SAA sample-size study, keeping design and evaluation seed namespaces disjoint. No new method or experiment is added.

## 12. Reproducibility Package (Plan D.3)

### 12.1 `experiments/reproduce.py --level {tables,scenarios,experiments} [--subset TOPOLOGY]`

- **`tables`:** runs `phase2_statistics.py`, `phase2_literature_statistics.py`, `report_tables.py`, `report_tables_phase2.py`, `report_tables_extensions.py` with outputs in a temporary directory, and compares bytes with `results/phase2/statistics*.csv` and the generated files in `docs/report/data/` (the `docs/report/tables/*.tex` files are hand-written and read those CSVs).
- **`scenarios`:** runs `data.cli verify --profile configs/data/paper.yaml`, `data.cli rescuenet-targets`, and `data.cli scenarios` for `v0`, `v0-bh50`, `sens-*` and `rescuenet`; compares every scenario-manifest SHA-256 with the values recorded in `results/**/manifest.json` and the target-manifest SHA-256 with `configs/scenarios/rescuenet.yaml`.
- **`experiments`:** writes temporary experiment configurations (output under a temporary directory; with `--subset`, scenario manifests filtered to that topology) for `phase1` and every `phase2_*` experiment, runs them and `tran_pilot.py` (skipped with `--subset`, which names an evaluation topology), and compares row by row on the non-runtime columns:
  - `method_realizations.csv`: `timely_count`, `connected_count`, `shortfall` (absolute tolerance 1e-9), `evaluate_calls`;
  - `bounds.csv`: bound values with relative tolerance 1e-6 and statuses;
  - `milp_crosscheck.csv`: objective where both runs report optimal, relative tolerance 1e-6;
  - `tran_pilot/runs.csv`: iterations, final status, timely ratio.
- Each run writes `results/reproducibility/<level>[-<subset>].json` (checks with pass/fail and detail, git commit, environment) and exits non-zero when any check fails. Differences on another machine are reported, not filtered.

### 12.2 `docs/reproducibility.md`

- Setup (`uv sync --extra test`), data acquisition and verification, licenses (RescueNet CC BY-NC-ND: images and masks are not redistributed; only manifests and conversion code).
- Table and figure map: report item → generator command → input files → source experiment.
- Recorded compute per experiment, summed from its result files, and the recorded environment.
- Acceptance map: each item of `docs/issue.md` Section 14 → evidence (file, table, report section).
- `README.md` gains a "Tái lập" section that points to this document.

## 13. Report (Plan D.3)

- `experiments/report_tables_extensions.py` writes `docs/report/data/ext_*.csv` and `ext_results.tex` (macros `\ResultLit…`, `\ResultCase…`); the hand-written tables `tables/ext_literature.tex` and `tables/ext_case_study.tex` read those CSVs with csvsimple; `main.tex` inputs `data/ext_results`.
- **Method:** subsections "Baseline từ tài liệu" (adaptation table of Section 5.1, differences from the paper, complexity of Section 5.7) and "Dựng scenario từ RescueNet" (Sections 8–9); an SAA paragraph (Section 11).
- **Experiments:** literature comparison with table and statistics; case study with table, target–relay–trajectory map and delivery-progress plot (pgfplots from the trace CSVs); quality–runtime figure with `TRAN` points; RQ answers updated.
- **Abstract, Introduction, Related Work, Discussion, Conclusion:** updated to the final results, including limitations (single geographic layout, non-independent images, adaptation choices that may disadvantage Tran, synthetic combination of sources).
- **Appendix:** reproducibility summary and the acceptance map.
- **Bibliography:** `rahnemoonfar2023rescuenet` is checked against the publisher record and its "chưa được đối chiếu" note removed.
- Compiled with the latex-document-skill script in a copy outside the workspace (the editor auto-builds `docs/report`); no undefined references, no overfull boxes, no placeholders.

## 14. Error Handling

- `TRAN`: invalid parameters raise at construction; non-optimal solver status keeps the last accepted iterate; an invalid plan raises and fails the task; the runner's existing failure reporting applies.
- Targets: missing GPS, altitude or yaw, pitch deviation above the limit, unknown mask class, missing focal length for a camera outside the fallback table, or an image–mask size mismatch stop the command with the pair id.
- Scenarios: Section 9 validation errors; target-manifest SHA mismatch.
- Statistics: anchor mismatch stops with the first differing rows.
- Trace: timely-ratio mismatch stops.
- Reproduce: every failed check is listed; the exit code is non-zero.

## 15. Testing

### 15.1 Plan D.1

- **Channel fit:** with $P_{\mathrm{LoS}}\equiv1$ the fit recovers $\beta_0$ and $\alpha_{\mathrm{LoS}}$.
- **Instance:** windows lie inside $[r_k,d_k-b_k)$ and follow $\theta$; empty windows fix $\lambda_k=0$; gateway and ground-device choice follow `cost_s`.
- **Bounds:** at the initial point the tangent and product bounds hold numerically and are tight.
- **IA on a small scenario** (few alerts, short horizon): relaxed objective does not decrease across iterations; $\lambda$ stays in $[0,1]$; repeated runs give identical plans (how close $\lambda$ ends to binary is reported by the pilot).
- **Plan conversion:** at most `max_active_downlinks` downlinks per slot; access and downlink totals ≤ $B_{\mathrm{tot}}$; `validate_plan` is empty; aggregation `G=5` gives a feasible trajectory.
- **Runner:** `TRAN` specs parse and reject bad params; bounds come from the method task; source hash covers `literature`; `--output-dir` leaves the configured directory untouched.
- **Statistics:** an anchor mismatch raises; four comparisons with Holm on fixture data.

### 15.2 Plan D.2

- **Projection:** yaw 0° maps image-up to north, yaw 90° to east; GSD formula; UTM round trip.
- **Regions:** a synthetic mask with two regions and one speck yields two targets with points inside their regions.
- **Merging:** result independent of input order; representative is the largest member.
- **Metadata:** XMP parsing from a sample string; fallback focal length; error cases of Section 14.
- **Real data:** tests on `data/raw` are skipped when the data is absent.
- **Generator:** `synthetic` regenerates `v0` with unchanged manifest SHA-256; `rescuenet` scenarios load through the scenario loader, are deterministic, and use the `v0` evaluation topologies; validation errors.
- **Trace:** replanned timely ratios equal the runner's on a fixture scenario.

### 15.3 Plan D.3

- **Reproduce comparisons:** runtime columns ignored; a changed `timely_count` or out-of-tolerance bound fails; the `tables` level passes on committed results.
- **Report data:** extension CSVs and macros match fixture inputs.

## 16. Acceptance Criteria

### 16.1 Plan D.1

1. The full test suite passes.
2. The pilot results and note are committed with the chosen θ, μ and G.
3. `phase2_literature.yaml` completes with all LPs optimal and no validity violation; a rerun skips every task; results are committed.
4. `results/phase2/statistics_literature.csv` reports all four comparisons after the anchor check passes.

### 16.2 Plan D.2

1. The full test suite passes.
2. `rescuenet_targets.csv`, `rescuenet_targets.json` and `scenarios_rescuenet.csv` are committed, and `scenarios_v0.csv` regenerates with unchanged SHA-256.
3. `phase2_case_study.yaml` completes with all LPs optimal and no validity violation; results and trace are committed.

### 16.3 Plan D.3

1. The full test suite passes.
2. `reproduce.py --level tables` and `--level scenarios` pass, and `--level experiments --subset <one evaluation topology>` passes; their JSON reports are committed.
3. `docs/reproducibility.md` and the README section exist.
4. The report compiles cleanly with the skill script, contains the literature comparison, case-study table and figures, SAA framing and appendix, has no placeholders, and answers every research question with the final results.

## 17. Decision Log

- Three plans under one spec (D.1 baseline, D.2 case study, D.3 reproducibility and report), baseline first because it carries the most risk.
- Tran HD adapted with fixed power, per-alert gateways, split windows ending at the end-to-end deadline, an effective power-law channel, full-band noise as in the paper, no storage limit, and a top-k downlink projection; output as a static schedule.
- `cvxpy` with Clarabel added as the only new dependency.
- Pairing with C results through a B1 anchor check instead of rerunning P and B2.
- RescueNet targets georeferenced from EXIF/XMP metadata instead of an assumed pixel–metre scale; yaw convention confirmed by overlapping images; 10 m merge radius from measured duplicates.
- SAA restated from existing results; no new SAA method.
- Reproducibility checked by a script with three levels; no container.

Adjustments made by plans D.1–D.3 from pilot evidence are appended here with date and reason.
