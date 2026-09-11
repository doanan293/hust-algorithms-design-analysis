# Phase 2 pilot on the development split

Configuration `configs/experiments/phase2_pilot.yaml`: set `v0`, split `dev` (6 scenarios), 30 evaluation realizations, methods B1, B2, B3, P. Search methods use the spec C defaults (design realizations 0–4, budget 2000, at most 10 iterations) and solve their trajectory bounds inside their method tasks. Run: 204 tasks, 12 workers, status `complete`, 2026-09-11T14:01:22Z to 2026-09-11T14:19:03Z (UTC).

## Results by method

| Method | Timely ratio | Gap to own bound | Gap to union bound | Evaluate calls | Planning time (s) | LP time per bound (s) |
|---|---|---|---|---|---|---|
| B1 | 0.461 | 0.290 | 0.383 | 1 | 0.2 | 2.35 |
| B2 | 0.545 | 0.225 | 0.264 | 2001 | 454.3 | 2.11 |
| B3 | 0.396 | 0.466 | 0.479 | 2001 | 518.6 | 2.07 |
| P | 0.542 | 0.230 | 0.268 | 2001 | 456.4 | 2.17 |

## Timely ratio by scenario

| Scenario | B1 | B2 | B3 | P |
|---|---|---|---|---|
| Bbnplanet-r0 | 0.318 | 0.340 | 0.250 | 0.336 |
| Bbnplanet-r1 | 0.675 | 0.802 | 0.500 | 0.800 |
| Canerie-r0 | 0.484 | 0.694 | 0.618 | 0.684 |
| Canerie-r1 | 0.847 | 0.830 | 0.683 | 0.830 |
| Fccn-r0 | 0.093 | 0.147 | 0.080 | 0.147 |
| Fccn-r1 | 0.349 | 0.453 | 0.243 | 0.453 |

## Budget and runtime of search methods

- B2: budget exhausted on 6 of 6 scenarios; planning time 321–621 s per scenario.
- B3: budget exhausted on 6 of 6 scenarios; planning time 387–643 s per scenario.
- P: budget exhausted on 6 of 6 scenarios; planning time 317–632 s per scenario.

## Decision

- **Budget stays at 2000 and design realizations stay at 0–4 for plan C.2.** Every search task used its whole budget, so the pilot compares methods at equal evaluation cost, not at convergence. The C.2 quality–runtime experiment (budgets 250 to 4000) measures what a larger budget buys, and the design-realization sensitivity (expected, 1, 5, 10) measures the effect of the design sample; raising either default now would multiply the runtime of every C.2 experiment.
- **Observed on the development split, not tuned on:** B2 is above B1 on 5 scenarios, equal on 0, below on 1; P is above B2 on 0, equal on 3, below on 3; B3 is above B1 on 1, equal on 0, below on 5. No method parameter was changed after the pilot.
- **Runtime for plan C.2:** a search task planned for 476 s on average with 12 workers (range 317–643 s) and a trajectory LP took 2.1 s. At that rate the main comparison with ablations (60 scenario instances, 10 search methods, bounds for 3) needs about 6.9 h and the budget experiment about 2.6 h, instead of the 2 h and 0.5 h estimated in spec C Section 11; plan C.2 sizes its experiments from these numbers and records any change in spec Section 18.
