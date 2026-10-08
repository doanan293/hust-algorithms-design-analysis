# Local Search Blocks — Plan E.2 (Experiments, Report, Slides) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun every experiment under the local-search code of Plan E.1, regenerate statistics and report data, rewrite the report's method and results text for the new blocks, rebuild `main.pdf`, update the slides, and leave the reproducibility checks passing.

**Architecture:** Shard hashes include the source hash of `src/`, so every committed result is stale after Plan E.1; all experiments rerun in the order of `docs/reproducibility.md` Section 7 inside one tmux session with the 14 GB `systemd-run` limit. Report numbers come from generated macros and CSVs (`experiments/report_tables*.py`); prose, figures drawn in TikZ, the algorithm box and the complexity paragraph in `docs/report/sections/method.tex` are rewritten by hand. Slides are edited in place with the pptx skill.

**Tech Stack:** Python ≥ 3.11 (uv), pytest, LaTeX via `.claude/skills/latex-document-skill/scripts/compile_latex.sh`, python-pptx via the pptx skill, tmux.

**Spec:** `docs/superpowers/specs/2026-10-08-local-search-blocks-design.md` (Sections 8–9, 11; Section 12 holds the pilot decisions).

## Global Constraints

- Long runs only inside detached tmux sessions with a log ending in `exit=<code>`, under `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0` (user rule; WSL has 24 GB).
- Search defaults fixed by the pilot: `entry_neighbors = 3`, `trajectory_segments = 5`, `catalog_probability = 0.1`, `catalog_init = False`, `seed = 0`.
- Bounds (P2)/(P3) stay on the K-path set; the report calls them bounds of the K-restricted problem and reports a method value above such a bound as a negative gap.
- Report numbers only through the generated files in `docs/report/data/`; no hand-typed result numbers in prose except where the existing text already quotes design-time pilot numbers (Section 4.1 of the report), which are updated by hand from `results/phase2/pilot_local/` and spec Section 12.
- No `Co-Authored-By` or other attribution trailers in commits.
- English in the report and slides (existing language); Vietnamese in README/reproducibility docs (existing language).

## Review Focus

1. **Negative gaps.** A search method may now exceed the K-restricted LP bound on some realization; report tables and statistics must print the negative gap rather than clip or crash. — Task 3 (report tables run) and Task 4 (text).
2. **New ablation row.** `P_catalog_trajectory` must appear in the ablation table, statistics and slides with its label, and no script may assume 12 methods. — Tasks 2 and 3.
3. **Stale shards.** A shard from the old code must never be reused; the runner's task hash covers this, and the manifests must record the new commit with `git_dirty: false`. — Task 2.
4. **Claims that the new numbers contradict.** Every qualitative sentence in abstract, introduction, results, discussion and conclusion is re-checked against the regenerated macros (e.g., which ablation matters, SBS-R vs SBS). — Task 4.
5. **PDF build hygiene.** No undefined references or overfull boxes introduced by the rewritten method section; the figure of the pipeline and the algorithm box match the code. — Task 5.

---

### Task 1: Fix the Plan E.1 review findings

**Files:** as named by the review; tests next to them.

- [ ] For each Critical/Important finding: write the reproducing test, watch it fail, fix, run `uv run --extra test pytest -q`, commit (`fix: ...`).
- [ ] Ledger every Minor as deferred and every declined finding as a ruling.

### Task 2: Ablation label and full rerun

**Files:**
- Modify: `experiments/report_tables_phase2.py` (`ABLATIONS`: add `("P_catalog_trajectory", "Catalogue trajectory block")` after `P_expected_design`)
- Create: `scripts`-free run: a shell command block in tmux (below)
- Output: every `results/*/manifest.json`, `summary.csv`, `method_realizations.csv`, `bounds.csv`, `crosscheck.csv`, `results/phase2/tran_pilot/*`, `results/phase2/case_study/trace/*`

- [ ] **Step 1:** add the label, run `uv run --extra test pytest tests/runner -q` → the phase-2 report test fails on the missing variant in committed results (expected until the rerun), all others pass.
- [ ] **Step 2:** commit the label (`feat: label the catalogue-trajectory ablation in the report tables`) so the reruns record a clean commit.
- [ ] **Step 3:** start the rerun:

```bash
tmux new-session -d -s rerun "cd $PWD && L=.superpowers/sdd/2026-10-08-local-search-blocks-delivery/rerun.txt; R='systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python'; { set -e; \
  \$R experiments/run_phase1.py --config configs/experiments/phase1.yaml; \
  for n in phase2_pilot phase2_main phase2_budget phase2_sensitivity phase2_design; do \$R experiments/run_phase2.py --config configs/experiments/\$n.yaml; done; \
  \$R experiments/tran_pilot.py; \
  \$R experiments/run_phase2.py --config configs/experiments/phase2_literature.yaml; \
  \$R experiments/run_phase2.py --config configs/experiments/phase2_case_study.yaml; \
  \$R experiments/case_study_trace.py; \
  uv run python experiments/phase2_statistics.py; uv run python experiments/phase2_literature_statistics.py; } > \$L 2>&1; echo exit=\$? >> \$L"
```

Expected: `exit=0`; each manifest `"status": "complete"`, `"failures": []`, `"git_dirty": false`, `git_commit` = the Step 2 commit. `tran_pilot.py` regenerates `configs/experiments/phase2_literature.yaml`; `git diff` on it must be empty (TRAN unchanged).

- [ ] **Step 4:** commit the results (`feat: rerun every experiment with the local search blocks`).

### Task 3: Report data and reproducibility

- [ ] Run `uv run python experiments/report_tables.py`, `report_tables_phase2.py`, `report_tables_extensions.py`; all exit 0.
- [ ] `uv run python experiments/reproduce.py --level tables` → `"passed": true`; commit `results/reproducibility/tables.json` and `docs/report/data/*`, `docs/report/tables/*`.
- [ ] Update `docs/reproducibility.md` (Sections 4–7: new pilot experiment `phase2_pilot_local`, new ablation, rerun date/commit) and README Pha 2 section; commit.

### Task 4: Report text

**Files:** `docs/report/sections/method.tex` (whole Sections "Block Search and Schedule Repair", figures `fig:pipeline`, `fig:tours`, `fig:repair` captions/labels as needed, `alg:bcd-repair`, complexity paragraph, design-rationale paragraph), `experiments.tex` (setup: parameters M, W, catalogue probability, pilot; results narrative; ablation discussion incl. `P_catalog_trajectory`; bounds as K-restricted with negative gaps), `analysis.tex` (where (P2)/(P3) are introduced: K-restricted wording), `discussion.tex`, `abstract.tex`, `introduction.tex` (contribution bullets), `docs/report/tables/params.tex` if parameters are listed there.

- [ ] Rewrite the method section: path block = neighbourhood of the incumbent path (kind flip, entry switch drawn ∝ 1/T(e) with residual backhaul capacity from the incumbent ledger, backhaul detour around the most congested link, unserve), alerts in urgency-weighted random order; trajectory block = segment shifts toward congested UAV links with exact feasible step, plus a catalogue tour with probability 0.1; one ledger call per block; Operation 2 uses the same neighbourhood; seeded RNG.
- [ ] Replace the tours figure by a figure of the two local moves (segment shift with the two boundary circles; path neighbourhood with the detour), drawn in TikZ in the style of the existing figures.
- [ ] Update the algorithm box and the per-round evaluation count: path block `1 + Σ_a |N(a)| ≤ 1 + (M + 3)|A|`, bandwidth `2|L_used|`, trajectory `1 + 2W + 1`, repair `1 + R(M + 3)`.
- [ ] Rewrite every results paragraph from the regenerated macros; add the ablation sentence on `P_catalog_trajectory`; state the pilot decision rule and the seed spread.
- [ ] Re-check every qualitative claim against the new numbers (Review Focus 4).

### Task 5: Build the PDF

- [ ] `cd docs/report && bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview`; no errors, no undefined references; inspect the preview PNGs of the method and results pages; commit `main.pdf` and sources.

### Task 6: Slides

- [ ] With the pptx skill: extract text of `docs/presentation/uav-alert-delivery.pptx`, update the method slides (block descriptions, pipeline figure text) and every result number, chart and table to the regenerated data; render to images and inspect; commit.

### Task 7: Final review and finish

- [ ] Whole-branch review by a fresh reviewer (most capable model) over Plan E.2's range; fix Critical/Important with tests.
- [ ] `uv run --extra test pytest -q`, `reproduce.py --level tables` pass; finish the branch (merge into `develop` locally, no push).
