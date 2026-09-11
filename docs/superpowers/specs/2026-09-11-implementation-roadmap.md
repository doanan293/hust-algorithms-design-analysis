# Implementation Roadmap: Joint UAV Trajectory–Relay–Bandwidth Allocation

**Date:** 2026-09-11
**Status:** Approved for spec writing
**Source requirements:** `docs/issue.md`, `docs/references/huong-nghien-cuu-de-so-5.md`

## 1. Purpose

`docs/issue.md` describes a full ten-week research project. It is too large for a single specification, plan, or implementation session. This roadmap splits it into four sub-projects. Each sub-project receives its own design spec, implementation plan, and implementation cycle, written when the previous sub-project has produced the facts it depends on (solver behaviour, realistic scale, runtime, calibration).

This document fixes the order, the scope boundary, the interfaces between sub-projects, and the completion condition of each. It does not fix the internal design of sub-projects B, C, and D.

## 2. Baseline State

- Done: the research data pipeline (`docs/superpowers/specs/2026-09-10-research-data-preparation-design.md`). Topology Zoo, SNDlib, and the RescueNet validation subset are acquired, verified, inventoried, and normalized. 130 Topology Zoo topologies with 15–40 nodes are eligible and normalized into a 2 km square. 26 SNDlib instances parse.
- Placeholder: `src/data/scenarios.py` (schema v1) chooses the center and relays at random and has no UAV, channel, spectrum, backhaul, or candidate-path parameters. Sub-project A replaces it.
- Skeleton only: the LaTeX report in `docs/report/`. Its result tables contain illustrative numbers.

## 3. Sub-Projects

| ID | Name | Course phase | Framework weeks | Depends on |
|---|---|---|---|---|
| A | Model and evaluator | Phase 1 | 1 | data pipeline |
| B | Phase 1 completion | Phase 1 | 2 | A |
| C | Phase 2 core | Phase 2 | 3–6 | B |
| D | Extensions and final delivery | Phase 2 | 7–10 | C |

The ten-week mapping is the course's sample framework, not a confirmed remaining schedule. Report writing is not a separate sub-project: every sub-project ends by updating the report sections that its results support.

### 3.1 A — Model and evaluator

**Scope:** scenario schema v2 and generator, UAV PrLoS and ground channel models, candidate paths, slot simulator with EDF dispatcher, metrics (timely ratio, time-expanded connectivity, shortfall), independent constraint checker, tests, calibration check.

**Out of scope:** any optimization method, the LP bound, the experiment runner, no-fly zones, RescueNet targets, SAA.

**Done when:** the offline test suite passes; the v0 scenario set (30 evaluation + 6 development scenarios) is generated reproducibly with a manifest; the calibration check in the A spec passes.

**Spec:** `docs/superpowers/specs/2026-09-11-model-evaluator-design.md`.

### 3.2 B — Phase 1 completion

**Scope:** B0 (no UAV), B1 (fixed trajectory), LP upper bound $\mathrm{UB}(q)$ with outer approximation on a fixed trajectory and candidate set, exact MILP cross-check on small instances, multi-seed experiment runner with a result schema, solver selection, complexity analysis, verification of the two scheduling results cited in issue Section 9.4 against their original sources, and the Phase 1 report.

**Done when:** a table of objective, gap to UB, and runtime covers the 30 evaluation scenarios with 30 evaluation realizations each; the Phase 1 report compiles with real numbers; code and configuration reproduce the table.

### 3.3 C — Phase 2 core

**Scope:** B2 (Huang-style three-block BCD), B3 (Huang max–min objective on the B2 structure), P (B2 plus two-operation deadline-aware repair), shared evaluation budget, mandatory ablations (issue Section 10.3), controlled sensitivity (issue Section 10.4, including the physical-constant channel variant), paired statistical tests with effect sizes and multiple-comparison correction, literature review that selects the strong literature baseline, and the Method and Experiments report sections.

**Done when:** RQ1 and RQ2 are answered with paired tests on the evaluation set; every method row reports the gap to UB; the literature baseline is chosen with a written justification.

### 3.4 D — Extensions and final delivery

**Scope:** RescueNet case study (damage-mask targets replacing the synthetic source generator), implementation of the chosen literature baseline, SAA design if time allows, quality–runtime curves, reproducibility package, and the complete final report (Abstract, Introduction, Related Work, Proposed Method, Experiment Results, Discussion, Conclusion).

**Done when:** the case study figure and table exist; the reproducibility package regenerates the main tables from pinned data and configuration; the report is complete.

## 4. Interfaces Between Sub-Projects

| From | To | Contract |
|---|---|---|
| data pipeline | A | `data/processed/networks/*.json`, SNDlib parser, `named_rng`, source manifests |
| A | B, C, D | `Scenario` loader, candidate paths, `Plan`, `evaluate(...)`, `EvaluationResult`, lexicographic key, evaluation counter |
| B | C, D | experiment runner, result schema, `UB(q)` bound, solver choice |
| C | D | method implementations B2, B3, P; experiment configurations; statistical analysis code |
| A | D | replaceable source-point generator interface used by the RescueNet case study |

A change to a contract is made in the sub-project that owns it and recorded in that sub-project's spec.

## 5. Cross-Cutting Rules

- The evaluator in `src/models/` never imports optimization code. Every method is scored by the same `evaluate` call.
- All methods compared in one table share hardware budget (spectrum, power), candidate set, dispatcher, scenarios, and realization seeds.
- Design information and evaluation realizations use disjoint seed namespaces; the development set and evaluation set use disjoint topologies.
- Implementation follows test-driven development; each plan task ends with its own commit.
- Dependencies are added only when a sub-project needs them and the spec names them.
- Results that do not improve on baselines are reported and analysed, not hidden.
