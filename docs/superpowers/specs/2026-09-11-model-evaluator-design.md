# Model and Evaluator Design (Sub-Project A)

**Date:** 2026-09-11
**Status:** Approved for implementation planning
**Roadmap:** `docs/superpowers/specs/2026-09-11-implementation-roadmap.md`
**Requirements:** `docs/issue.md` Sections 3.4, 7, 8, and 11

## 1. Purpose

Build a reproducible v0 scenario set and a trustworthy evaluator. Any method in later sub-projects produces a `Plan`; the evaluator scores every plan with the same rules, independently of how the plan was produced.

## 2. Scope

### 2.1 Included

- Scenario schema v2 and a deterministic generator that replaces schema v1 in `src/data/scenarios.py`.
- UAV PrLoS channel, ground access channel, backhaul capacity, and channel-realization sampling.
- Candidate paths with at most $K=3$ paths per alert.
- `Plan` with trajectory, per-alert path choice, and bandwidth policy, plus input validation.
- Slot simulator with EDF dispatcher, buffers, and slot causality.
- Metrics: timely ratio, time-expanded connectivity, shortfall, lexicographic key.
- An independent ledger checker.
- Offline tests, a scenario-generation command, and a calibration check.

### 2.2 Excluded

B0, B1, B2, B3, P, the LP bound, the experiment runner, no-fly zones, RescueNet source points, SAA design, and performance vectorization. The roadmap assigns these to sub-projects B–D.

## 3. Considered Evaluator Approaches

### 3.1 Slot simulation with EDF dispatcher — selected

The evaluator steps through slots, computes realized link capacity, and serves each queue in earliest-deadline-first order using only buffers present at the start of the slot. It matches the shared dispatcher of issue Section 8.8, uses no future channel information, needs no solver, and is independent of the optimization code. EDF can be suboptimal for a given plan, so the gap to the LP bound includes dispatcher loss; reports state this.

### 3.2 Optimal flow per realization — rejected

Solving a time-expanded max-flow per realization gives each plan its best flow but uses future channel states (an optimistic bound per issue Section 8.8), requires a solver, is much slower, and would share machinery with the sub-project B bound.

### 3.3 Simulation plus exact flow cross-check — deferred

The exact cross-check belongs to sub-project B, which introduces the solver for the LP bound. Sub-project A verifies the simulator with hand-computed instances.

## 4. Architecture

```text
data/processed/networks/*.json  +  SNDlib demand values
        │  scenario generator (seeded)
        ▼
data/processed/scenarios/v0/*.json  +  data/manifests/scenarios_v0.csv
        │  load + validate
        ▼
Scenario ──► candidate paths ──► [method outside A builds Plan]
                                          │
                                          ▼
                  evaluate(scenario, plan, mode, realization seeds) ──► EvaluationResult
```

| File | Responsibility |
|---|---|
| `src/data/scenarios.py` | Schema v2 generator; replaces schema v1 |
| `src/data/cli.py` | New `scenarios` stage; `preprocess` stops writing the v1 placeholder scenario |
| `configs/scenarios/v0.yaml` | Every v0 parameter and seed in Section 5 and Section 6 |
| `src/models/__init__.py` | Package marker |
| `src/models/scenario.py` | Typed `Scenario` dataclasses, JSON loading, validation |
| `src/models/channel.py` | PrLoS, gains, rates, expected rate, ground range, realization sampling |
| `src/models/paths.py` | Candidate path generation |
| `src/models/plan.py` | `Plan`, bandwidth policies, input validation |
| `src/models/simulator.py` | Slot simulation producing a bit ledger |
| `src/models/metrics.py` | Timely ratio, connectivity, shortfall, lexicographic key |
| `src/models/checker.py` | Independent ledger checker |
| `src/models/evaluate.py` | `evaluate(...)`, the only entry point used by methods |
| `experiments/calibrate_v0.py` | Calibration check in Section 10.3 |
| `tests/models/` | Tests in Section 10 |

`src/models/` imports nothing from `src/baselines/`, `src/optimization/`, or `experiments/`. No new third-party dependency is added; `numpy`, `networkx`, and `PyYAML` are already declared.

## 5. Scenario Schema v2

### 5.1 Generation rules

A scenario is one topology plus one workload-and-failure seed. Channel randomness is not part of a scenario.

| Element | Rule | v0 default |
|---|---|---|
| Topology selection | Sort eligible topologies by `(node_count, topology_id)`, cut into 13 contiguous strata of near-equal size, pick one per stratum with `named_rng(selection_seed, "topology-selection")`, then assign 3 of the 13 to the development split with `named_rng(selection_seed, "dev-split")` | 10 evaluation, 3 development topologies |
| Replicates | Scenario seed = first 8 bytes of `sha256(f"{scenario_base_seed}:{topology_id}:{replicate}")` as an unsigned big-endian integer; `selection_seed` and `scenario_base_seed` are set in `configs/scenarios/v0.yaml` | 3 per evaluation topology, 2 per development topology |
| Rescue center $c$ | Highest unweighted closeness centrality on the intact graph; ties go to the lexicographically smallest node ID | — |
| Relays | Every node except $c$ | $V\setminus\{c\}$ |
| Damage zones | Centers uniform in the box, rejection sampling for minimum pairwise separation, at most 1,000 attempts before raising an error | 3 zones, separation ≥ 600 m, radius $R_{\mathrm{dmg}}=400$ m |
| Source points $S$ | Assigned to zones round-robin; position drawn from an isotropic Gaussian around the zone center and clipped to the box; not graph nodes | 15 sources, $\sigma=150$ m |
| Alerts | Source uniform over $S$; release slot $r_a$ uniform integer in $[0, N-D_{\max}-1]$; deadline $d_a=r_a+D$ with $D$ uniform integer in $[D_{\min}, D_{\max}]$ | 40 alerts, $D\in[20,60]$ |
| Size | $L_a=\mathrm{round}(L_{\mathrm{ref}}\cdot\mathrm{clip}(v/\mathrm{median}(v),0.25,4))$, $v$ sampled with replacement from one fixed SNDlib instance; labelled derived | instance `abilene`, $L_{\mathrm{ref}}=2.5\times10^5$ bits |
| Failures | Each edge fails independently with $p_{\mathrm{in}}$ if its midpoint lies within $R_{\mathrm{dmg}}$ of any zone center, otherwise with $p_{\mathrm{out}}$ | $p_{\mathrm{in}}=0.6$, $p_{\mathrm{out}}=0.05$ |
| Backhaul | Surviving edge capacity $C_{\mathrm{bh}}$, shared by all flows on the edge; failed edge capacity 0 | $C_{\mathrm{bh}}=10^6$ bit/s |
| UAV | $q_s=q_t=$ position of $c$; fixed altitude | $H=100$ m, $V_{\max}=50$ m/s |
| Time | $V_{\max}N\Delta\ge 2\sqrt2 L_{\mathrm{box}}$ | $\Delta=1$ s, $N=150$ |

Zones, sources, alerts, sizes, and failures each draw from their own `named_rng(scenario_seed, concern)` stream, so changing one rule does not change the others.

### 5.2 JSON layout

```json
{
  "schema_version": 2,
  "scenario_id": "Aarnet-r0",
  "split": "eval",
  "network": {"network_id": "...", "nodes": [{"id": "n0", "x_m": 0.0, "y_m": 0.0}],
              "edges": [{"source": "n0", "target": "n1", "failed": false, "capacity_bps": 1000000.0}],
              "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0}},
  "center": "n5",
  "relays": ["n0", "n1"],
  "damage_zones": [{"x_m": 0.0, "y_m": 0.0, "radius_m": 400.0}],
  "sources": [{"id": "s00", "x_m": 0.0, "y_m": 0.0, "zone": 0}],
  "alerts": [{"id": "alert-0000", "source": "s00", "release_slot": 0, "deadline_slot": 20,
              "size_bits": 1000000, "size_ratio": 1.0, "provenance": "derived-from-sndlib-demand"}],
  "uav": {"altitude_m": 100.0, "v_max_mps": 50.0, "start_xy_m": [0.0, 0.0], "end_xy_m": [0.0, 0.0],
          "total_power_w": 0.1, "max_active_downlinks": 2},
  "time": {"slot_s": 1.0, "num_slots": 150, "access_fraction": 0.5},
  "spectrum": {"b_tot_hz": 1000000.0, "noise_psd_dbm_per_hz": -140.0},
  "channel": {"uav": {"plos_a": 9.61, "plos_b": 0.16, "nlos_attenuation": 0.2, "alpha_los": 2.2,
                      "alpha_nlos": 3.3, "beta0_db": -50.0},
              "ground": {"alpha": 2.8, "beta0_db": -50.0},
              "usable_rate_bps": 10000.0},
  "source_power_w": 0.1,
  "paths": {"k": 3, "max_ground": 2, "version": 1},
  "provenance": {"network_sha256": "...", "sndlib_instance": "abilene", "raw_sha256": ["..."],
                 "config_hash": "...", "scenario_seed": 0, "generator_version": 2}
}
```

`src/models/scenario.py` rejects a file with a missing field, a wrong schema version, an alert whose `deadline_slot` exceeds `num_slots`, an unknown source or node reference, or a center that is not a node, and names the offending field in the error.

### 5.3 Command and manifest

`python -m data.cli scenarios --config configs/scenarios/v0.yaml` reads normalized networks and SNDlib archives under the configured `data_root`, writes `data/processed/scenarios/v0/<scenario_id>.json` (Git-ignored) and `data/manifests/scenarios_v0.csv` (committed) with columns `scenario_id, split, topology_id, replicate, scenario_seed, sha256, node_count, alert_count, failed_edge_count, ground_connected_source_fraction`. Rerunning the command yields identical `sha256` values.

## 6. Channel Model

### 6.1 UAV links

Uplink (source → UAV) and downlink (UAV → drop node) use the Huang PrLoS model with the constants of Huang Table 4, inherited unchanged: $a=9.61$, $b=0.16$, $\mu=0.2$, $\alpha_L=2.2$, $\alpha_N=3.3$, $\beta_0=-50$ dB, $N_0=-140$ dBm/Hz. For slot $n\in\{0,\dots,N-1\}$ the UAV position is the segment midpoint $\bar q[n]=(q[n]+q[n+1])/2$:

$$d_k[n]=\sqrt{\|\bar q[n]-w_k\|^2+H^2},\quad \theta_k[n]=\tfrac{180}{\pi}\operatorname{atan2}(H,\|\bar q[n]-w_k\|),$$
$$P^L_k[n]=\frac{1}{1+a\exp[-b(\theta_k[n]-a)]},\quad g^L=\beta_0d^{-\alpha_L},\quad g^N=\mu\beta_0d^{-\alpha_N},$$
$$R(b_e,g)=b_e\log_2\!\Big(1+\frac{p_e g}{N_0 b_e}\Big),\qquad R(0,g)=0.$$

Gains and powers are linear; $N_0$ is converted to W/Hz. The expected rate is $P^L R(b,g^L)+(1-P^L)R(b,g^N)$, never the rate of the mean gain.

With these constants and 1 MHz, the expected UAV rate is about 2.3 Mbit/s directly above a ground point, 456 kbit/s at 200 m horizontal offset, and 18 kbit/s at 500 m. The physical-constant variant ($\beta_0=-40$ dB, $N_0=-164$ dBm/Hz) is a sensitivity case in sub-project C, not a v0 default.

### 6.2 Realizations

For realization $\omega$, one uniform number $u_{k,n,\omega}$ exists for every ground point $k\in S\cup V$ and slot $n$, drawn from `named_rng(scenario_seed, f"channel-eval:{omega}")`. The link to $k$ is LoS in slot $n$ when $u_{k,n,\omega}<P^L_k[n]$. The numbers do not depend on the plan, so different plans face common random numbers while their LoS probabilities still follow their own trajectories. Slots are independent; temporal correlation is a stated limitation. The namespace `channel-design:{omega}` is reserved for SAA design scenarios and never used for evaluation.

### 6.3 Ground access and backhaul

Source → relay links use deterministic log-distance path loss $g_G=\beta_0 d^{-\alpha_G}$ with $\alpha_G=2.8$. The link exists only when $R(B_{\mathrm{tot}},g_G)\ge 10$ kbit/s, which yields a range of about 350 m under v0 constants. Backhaul is a separate wired resource with capacity $\Delta C_{\mathrm{bh}}$ bits per slot and consumes no spectrum or radio power.

### 6.4 Spectrum and power

Each slot has an access half-slot of length $\tau\Delta$ and a downlink half-slot of length $(1-\tau)\Delta$ with $\tau=0.5$. Radio links active in one half-slot use orthogonal bands with total bandwidth at most $B_{\mathrm{tot}}=1$ MHz. Sources transmit at 0.1 W. The UAV has 0.1 W in total; each downlink uses 0.05 W and at most two downlinks are active in a slot, which keeps the per-device budget fixed as scenarios grow. Relaxing the two-downlink cap only enlarges the feasible set, so a bound that ignores it stays valid.

## 7. Candidate Paths

Each alert $a$ from source point $s_a$ receives at most three candidates, or may be left unserved by the plan.

- **Ground path:** $s_a$ → relay $v$ within ground range → backhaul route $v\leadsto c$.
- **UAV path:** $s_a$ → UAV → drop node $u\in V$ (including $c$) → backhaul route $u\leadsto c$.

Routes $u\leadsto c$ come from one breadth-first search from $c$ over surviving edges: fewest hops, ties by total Euclidean length, then by lexicographically smallest next node ID. Nodes without a surviving route are excluded. No path revisits the UAV after reaching the ground network.

Candidates are ranked within their own type with trajectory-free cost estimates:

$$\mathrm{cost}_G=\frac{L_a}{R(B_{\mathrm{tot}},g_G(d(s_a,v)))}+h(v)\Big(\frac{L_a}{C_{\mathrm{bh}}}+\Delta\Big),\qquad \mathrm{cost}_U=\frac{d(s_a,u)}{V_{\max}}+h(u)\Big(\frac{L_a}{C_{\mathrm{bh}}}+\Delta\Big),$$

where $h(\cdot)$ is the backhaul hop count. The candidate list holds the lowest-cost ground paths (at most two), then the lowest-cost UAV paths until three candidates exist or UAV options run out. Ties break by node ID. The list is ordered ground-then-UAV by cost, and a plan refers to candidates by index. A UAV path to $c$ always exists, so every alert has at least one candidate. Generation costs $O(|V|+|E|)$ for the search plus $O(|A||V|\log|V|)$ for ranking. Every result records `paths.version`; bounds and optima are valid only relative to this candidate set.

## 8. Evaluator

### 8.1 API

```python
evaluate(scenario, plan, mode, realization_ids=(), counter=None) -> EvaluationResult
```

- `mode="expected"` uses expected UAV rates and runs one deterministic pass; `realization_ids` must be empty.
- `mode="realized"` runs one pass per realization ID in `realization_ids`, which must be non-empty.
- `counter`, when given, increments by one per call regardless of the number of realizations.

`Plan` contains waypoints $q[0..N]$, one candidate index or `None` per alert, and a bandwidth policy. The radio link set of a scenario is every access link and UAV downlink that appears in at least one candidate path; a static schedule has one entry per radio link and slot.

- `StaticSchedule`: $b_e[n]$ for every radio link and slot, fixed offline; bandwidth assigned to an idle link is wasted.
- `EqualSplitBacklogged`: in each half-slot, $B_{\mathrm{tot}}$ is split equally among radio links whose queue is non-empty at the start of the slot; in the downlink half-slot at most two downlinks are served, chosen by earliest head-of-queue deadline, then node ID.

### 8.2 Input validation

Before simulation the evaluator checks endpoints $q[0]=q_s$ and $q[N]=q_t$ within $10^{-6}$ m, per-slot movement at most $V_{\max}\Delta(1+10^{-9})$, schedule shape, non-negative bandwidth, per-half-slot sums at most $B_{\mathrm{tot}}(1+10^{-9})$, at most two downlinks with positive bandwidth per slot, and valid candidate indices. Any failure returns `feasible=False` with the list of violations and no simulation.

### 8.3 Slot semantics

Slot $n$ covers $[n\Delta,(n+1)\Delta)$.

1. At the start of slot $n$, bits of every alert with $n\ge d_a$ are dropped from every queue and recorded as dropped.
2. Bits of alert $a$ exist at its source from the start of slot $r_a$.
3. Every transfer in slot $n$ uses only the buffer present at the start of slot $n$; received bits are added at the end of slot $n$ and can move on from slot $n+1$.
4. Access half-slot: each source → relay and source → UAV link carries at most $\tau\Delta R$ bits.
5. Downlink half-slot: each UAV → drop-node link carries at most $(1-\tau)\Delta R$ bits.
6. Backhaul: each surviving edge carries at most $\Delta C_{\mathrm{bh}}$ bits in total across alerts.
7. Each link serves its queue in EDF order by `(deadline_slot, alert_id)`; bits follow only the chosen candidate path.
8. UAV links use the realized state of Section 6.2 (or the expected rate in `expected` mode); ground links are deterministic.
9. Alert $a$ is timely in realization $\omega$ ($z_{a,\omega}=1$) when at least $L_a$ of its bits reach $c$ by the end of slot $d_a-1$.

Because each alert has exactly one next hop at each node, no buffer is spent twice within a slot and the processing order of links does not change the result.

### 8.4 Metrics

- **Timely ratio:** $\hat P_{\mathrm{timely}}=\frac{1}{M|A|}\sum_\omega\sum_a z_{a,\omega}$; per-realization vectors are kept for paired tests.
- **Shortfall:** $\sum_\omega\sum_a (L_a-\mathrm{delivered}_{a,\omega}(d_a))/L_a$; an unserved alert contributes 1 per realization.
- **Connectivity:** on a time-expanded graph with vertices $(x,n)$ for $x\in S\cup V\cup\{\mathrm{UAV}\}$ and $n\in\{0,\dots,N\}$, a wait edge links $(x,n)\to(x,n+1)$ and a transmission edge links $(x,n)\to(y,n+1)$ when link $x\to y$ is usable in slot $n$. Usable means: a ground access link within range; a UAV uplink or downlink whose rate with the full $B_{\mathrm{tot}}$ reaches 10 kbit/s in that slot's realized state (expected rate in `expected` mode); a surviving backhaul edge. Transmission edges follow the candidate-path structure only: source → relay, source → UAV, UAV → node, and backhaul between surviving neighbours; sources neither receive nor relay. $\mathrm{Conn}$ is the fraction of sources with a path from $(s,0)$ to some $(c,n)$, averaged over realizations. The result also reports $\mathrm{Conn}$ with UAV links removed and each source's disconnected time, defined as $\Delta$ times the number of slots $n<N$ for which no path leads from $(s,n)$ to any $(c,n')$.
- **Lexicographic key:** `(timely_count, connected_source_count, -round(shortfall, 9))`, with both counts summed over realizations.

**Deviation from issue Section 8.7:** the third criterion is shortfall instead of total lateness. The dispatcher drops expired alerts (issue Section 8.8), so a late alert has no completion time; shortfall measures the same preference and is defined for every alert.

### 8.5 Independent checker

The simulator records a ledger of bits per `(realization, slot, link, alert)`, allocated bandwidth per `(realization, slot, radio link)`, realized capacity per `(realization, slot, link)`, and drops. `src/models/checker.py` shares no code with `simulator.py` and verifies from the ledger alone: per-alert flow conservation at every node, non-negative buffers, no transmission before release, delivered bits at most $L_a$, link capacity in every half-slot and on every backhaul edge, start-of-slot causality, flow only on the chosen path, and the bandwidth and downlink limits. A checker violation after simulation raises `SimulatorInvariantError`, because it means the simulator is wrong rather than the plan.

### 8.6 Result and cost

`EvaluationResult` contains `feasible`, `violations`, `mode`, realization IDs, per-realization `z` vectors, delivered bits and delivery slots, shortfall, $\mathrm{Conn}$, $\mathrm{Conn}$ without UAV, disconnected times, the lexicographic key, `paths.version`, the scenario `sha256`, and wall-clock runtime. The ledger is returned only when requested.

One evaluation costs $O(M\,N(|\mathcal L|+|A|\log|A|))$ for $|\mathcal L|$ radio and backhaul links. Vectorization waits until sub-project C measures a need.

## 9. Error Handling

- Generator: raises with the topology ID and rule name when zone rejection sampling fails, the SNDlib instance is missing, or a topology is ineligible.
- Loader: raises with the JSON field path on schema violations (Section 5.2).
- Evaluator: returns `feasible=False` for invalid plans; raises `SimulatorInvariantError` for checker violations; a plan that delivers nothing is a valid zero result, not an error.

## 10. Testing

All tests run offline under `pytest` and follow test-driven development.

### 10.1 Unit tests

- **Channel:** $P^L$ against hand-computed values; $R(0,g)=0$; expected rate equals the state-weighted rate, not the rate of the mean gain; fixed $u$ gives LoS exactly when $u<P^L$; ground range matches the 10 kbit/s threshold.
- **Scenario:** same seed gives identical hashes; changing the alert rule leaves failures unchanged; development and evaluation topologies are disjoint; the center rule and tie-break; sources inside the box; release and deadline bounds; size clipping; loader rejections of Section 5.2.
- **Paths:** a UAV path to $c$ always exists; failed edges and disconnected relays are excluded; at most two ground paths and at least one UAV path; deterministic tie-breaks.

### 10.2 Simulator and checker tests

Hand-computable instances cover zero bandwidth, zero power, release at slot 0, the deadline boundary (arrival in slot $d_a-1$ counts, in slot $d_a$ does not), empty buffers, a backhaul bottleneck where two alerts share a 1 Mbit/s edge in EDF order, a disconnected ground graph where only the UAV path delivers, store-carry-forward (received in slot $n$, forwarded from $n+1$), expired-alert dropping, delivered bits never exceeding $L_a$, the two-downlink cap, and seed reproducibility. Connectivity tests include a source that reaches $c$ only after a UAV visit in slot 5 followed by backhaul.

Checker tests corrupt ledgers (over-capacity link, negative buffer, off-path flow, transmission before release, same-slot forwarding) and assert that each corruption is detected. A randomized test builds random valid plans on small scenarios and asserts zero checker violations.

### 10.3 Calibration check

`experiments/calibrate_v0.py` evaluates two simple plans on the 30 evaluation scenarios with realization IDs 0–29 and writes `results/calibration/v0.json` (committed):

- **Ground-only:** each alert takes its lowest-cost ground candidate or stays unserved; the UAV stays at $c$; `EqualSplitBacklogged`.
- **Zone tour:** the UAV visits zone centers in nearest-neighbour order from $c$ at speed $V_{\max}$, splits the remaining time equally as hover time at the visited zones, and returns to $c$; the zone added last is dropped until the tour length fits $V_{\max}N\Delta$; each alert takes its lowest-cost UAV candidate; `EqualSplitBacklogged`.

These plans exist only for calibration; they are not the B0 or B1 baselines. The check passes when both plans produce no violations and each plan's mean realized $\hat P_{\mathrm{timely}}$ over evaluation scenarios lies in $[0.05, 0.95]$. If it fails, adjust $L_{\mathrm{ref}}$ first, then the deadline window, and record the change in Section 12. Mean runtime per evaluation and whether the zone tour beats ground-only are reported but are not pass conditions.

## 11. Acceptance Criteria

- The offline test suite passes.
- `python -m data.cli scenarios --config configs/scenarios/v0.yaml` produces 30 evaluation and 6 development scenarios and `data/manifests/scenarios_v0.csv`; a rerun reproduces every `sha256`.
- The calibration check in Section 10.3 passes and its summary is committed.
- `README.md` documents the scenario command, the evaluator entry point, and the calibration command.
- Schema v1 generation, its placeholder scenario file, and its tests are removed or replaced.

## 12. Deviations and Change Log

Deviations from `docs/issue.md` adopted in this design:

- Shortfall replaces total lateness as the third lexicographic criterion (Section 8.4).
- Relays are all non-center nodes, and the center is chosen by closeness centrality rather than at random (Section 5.1).
- Source points are independent locations, not graph nodes (Section 5.1).
- Ground access channels are deterministic in v0; only UAV links are random (Section 6.3).
- The two radio stages of a slot are called access and downlink half-slots to avoid confusion with course phases (Section 6.4).

Calibration changes to $L_{\mathrm{ref}}$ or the deadline window are appended here with date and reason.

- **2026-09-11 — $L_{\mathrm{ref}}$ from $10^6$ to $2.5\times10^5$ bits.** A prototype of this design generated the v0 set from the pinned data and ran the Section 10.3 check. With $L_{\mathrm{ref}}=10^6$ bits the zone-tour plan reached a mean timely ratio of 0.042 (ground-only 0.261, two realizations), below the 0.05 bound. A bit-flow breakdown showed the UAV downlink as the bottleneck: of 325 Mbit uplinked to the UAV over eight scenarios, 56 Mbit were forwarded and 269 Mbit expired on board, because the tour hovers at zone centres and the 0.05 W downlink reaches drop nodes hundreds of metres away. With 30 realizations, $5\times10^5$ bits passed narrowly (ground-only 0.335, zone tour 0.069, two scenarios at zero), and $2.5\times10^5$ bits passed with margin (ground-only 0.413, zone tour 0.104, no scenario at zero). One evaluation of 30 realizations, checker included, took 0.30–0.57 s. The downlink bottleneck is a property of the naive calibration plan and is relevant to the trajectory–relay coupling studied in sub-projects B and C.
