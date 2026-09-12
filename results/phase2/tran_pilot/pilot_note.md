# TRAN pilot on the development split

Command `experiments/tran_pilot.py` on the 6 development scenarios of `v0` (Bbnplanet-r0, Bbnplanet-r1, Canerie-r0, Canerie-r1, Fccn-r0, Fccn-r1), 30 evaluation realizations per plan. Spec: `docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md` Sections 5.6, 5.8 and 7.1.

## Step 1: theta = 1/2, mu = 1, slot resolution

| Scenario | Status | Iterations | Planning time (s) | Initial timely ratio | TRAN timely ratio |
|---|---|---|---|---|---|
| Bbnplanet-r0 | converged | 27 | 141.2 | 0.200 | 0.388 |
| Bbnplanet-r1 | converged | 3 | 16.2 | 0.489 | 0.633 |
| Canerie-r0 | converged | 24 | 127.0 | 0.372 | 0.647 |
| Canerie-r1 | converged | 13 | 71.8 | 0.578 | 0.768 |
| Fccn-r0 | converged | 27 | 154.9 | 0.027 | 0.260 |
| Fccn-r1 | max_iterations | 30 | 159.5 | 0.138 | 0.493 |

Median planning time 134.1 s against the limit of 600 s, so the grid runs with block size 1.

## Grid

| theta | mu | Mean timely ratio | Mean planning time (s) | Mean iterations |
|---|---|---|---|---|
| 0.333 | 1 | 0.541 | 305.3 | 19.3 |
| 0.333 | 10 | 0.544 | 326.6 | 20.3 |
| 0.500 | 1 | 0.532 | 111.8 | 20.7 |
| 0.500 | 10 | 0.531 | 352.7 | 21.7 |
| 0.667 | 1 | 0.511 | 268.7 | 19.2 |
| 0.667 | 10 | 0.503 | 241.8 | 19.8 |

## Effective channel fit

| Scenario | beta | alpha | R² | Max log error |
|---|---|---|---|---|
| Bbnplanet-r0 | 2.526e-03 | 3.393 | 0.9944 | 0.482 |
| Bbnplanet-r1 | 2.789e-03 | 3.409 | 0.9946 | 0.468 |
| Canerie-r0 | 2.917e-03 | 3.416 | 0.9947 | 0.461 |
| Canerie-r1 | 2.821e-03 | 3.411 | 0.9946 | 0.466 |
| Fccn-r0 | 1.889e-03 | 3.346 | 0.9937 | 0.526 |
| Fccn-r1 | 1.798e-03 | 3.338 | 0.9936 | 0.533 |

## Decision

- **Parameters for the comparison:** theta = 0.333, mu = 10, block size 1 (highest mean timely ratio 0.544; the grid means span 0.041).
- **Samir rule:** the chosen TRAN beats its initial point on 6 of 6 scenarios, so the rule does not fire.
- **Solver statuses over all runs:** converged 33, max_iterations 9.
- Evaluation-split scenarios were not used; `configs/experiments/phase2_literature.yaml` was written from `choice.json`.
