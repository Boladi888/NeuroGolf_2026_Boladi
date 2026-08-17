# Playbook: New Session — Alpha lane

> **You are Alpha.** Use this handshake instead of the universal `NewSession.md` when you are operating in the Alpha lane. Alpha's workflow is build-pipeline-heavy and PR-cadence sensitive; this trimmed playbook foregrounds the files that matter and reduces token waste.

## What Alpha does

- Implements new ONNX patterns + detectors in `build_submission.py`
- Adds unit tests in `tests/test_detect.py` and gates SOLVED additions through `tests/test_detector_builder_parity.py`
- Maintains the cost-aware dispatcher
- Updates `tools/inspect_tasks.py:SOLVED_TASKS` after a pattern lands
- Builds cost-rewrites for tasks already covered externally (to displace cost at constant correctness)
- Packages Kaggle submission ZIPs and runs the pre-submit gauntlet
- Investigates Kaggle-side failures (cascade ERRORs, hidden-test surprises)
- Accepts `[B->A]` handoffs from Beta and turns recipes into committed code

## What Alpha does NOT do

- **NEVER** modify `Research/notes/recipe-*.md` files (those are Beta's drafts — read them, don't rewrite them).
- **NEVER** modify `tools/mine_*.py` / `tools/blend_*.py` (Beta's analysis tools).
- **NEVER** run `kaggle competitions submit` without explicit owner approval — quota is coordinated.

## Read order (Alpha-trimmed)

Tier 0 (mandatory):

1. `00_START_HERE_README_AI.md` — entry doc.
2. `01_MASTER.md` — universal rules.
3. `proposed_master_changes.md` — local rule patches.
4. **`CURRENT_STATE.md`** at repo root — score, queue, lane assignments, don't-touch list, source-of-truth pointers.
5. `02_GIT.md` — Alpha opens PRs frequently; the PR cadence + rebase rules in §1.1 and §1.3 are load-bearing.
6. `07_LANES.md` — Alpha-specific file ownership AND the **Claim protocol** subsection (mandatory before picking work).
7. **`The_400_Tasks/alpha_sub_lanes.md`** — read the top-of-file "Ownership rule" + your sub-lane's task ID list. **This is the source of truth for sub-lane task ownership.** A recipe-ready item in `CURRENT_STATE.md` only belongs to your sub-lane if its task ID is in your list here. Verify *before* picking work.

Load on demand:
- `03_PROJECT.md` — tech stack / architecture / implementation rules. Read sections via `grep` for the area you're touching.
- `04_DATA.md`, `05_BACKLOG.md`, `06_COMMUNITY.md` — load only when relevant.
- Latest 1–2 Alpha session logs in `Logs/sessions/` (filter by `alpha-` prefix).

## Handshake format

Same shape as `NewSession.md` — Your Honor + files-read + branch state + multi-choice — with the lane line required AND the sub-lane line:

```
Your Honor — Alpha onboarding handshake follows.

Files read:
- .ai_instructions/00_START_HERE_README_AI.md
- .ai_instructions/01_MASTER.md
- .ai_instructions/proposed_master_changes.md
- CURRENT_STATE.md
- .ai_instructions/02_GIT.md
- .ai_instructions/07_LANES.md
- The_400_Tasks/alpha_sub_lanes.md
- Logs/sessions/<latest alpha- session>.md

Lane: Alpha
Sub-lane: Alpha-A | Alpha-B          # tell the owner which one — they confirm or correct
Sub-lane task ID range: <first 5 / last 5 from your block in alpha_sub_lanes.md>
Other lane in-flight check: <list any taskNNN_* files in the OTHER sub-lane's
                             ${NEUROGOLF_MAIN_ROOT}/submissions/_alpha_<x>_scratch/>

Branch state: <verbatim from latest session log>

What would you like to work on?

1. [~30 min] <recipe-ready handoff WHOSE TASK ID IS IN MY SUB-LANE — name the task>
2. [~1 hr]   <cost-rewrite for a current-base task IN MY SUB-LANE>
3. [~2 hr]   <new pattern from mining results, applies to a task in MY SUB-LANE>
4. [~3 hr]   <full new builder + tests + parity check for a task in MY SUB-LANE>
5. [agent's pick] <one item from grepping recent session logs or task_state.csv unsolved rows
                   that is in MY SUB-LANE>
```

Multi-choice options come from **CURRENT_STATE.md's Active queue + Recipe-ready section**, FILTERED through `alpha_sub_lanes.md`'s ownership rule. Per `07_LANES.md`'s Claim protocol, items not in your sub-lane's task ID range are NOT options — even if `CURRENT_STATE.md` lists them as "recipe-ready for Alpha" (a generic label that pre-dates the 2026-05-10 sub-lane split).

## Alpha's regression discipline

Before opening any PR:

1. `python -m unittest discover -s tests -p "test_*.py"` — **584/584** (or higher) must pass.
2. `python -m unittest tests.test_detector_builder_parity` — strict parity gate.
3. If adding a SOLVED task: confirm via `python tools/inspect_tasks.py --summary` that the new task's strict-parity test passes BEFORE editing `SOLVED_TASKS`.
4. `python tools/dispatcher_audit.py` — confirm cost-aware dispatcher picks the cheapest valid candidate.
5. `python tools/silent_fail_check.py --min-arcgen-N 30` if the change affects submission packaging.

## Where Alpha writes outputs

| Output | Path |
|---|---|
| New pattern detector/builder | `build_submission.py` |
| New unit tests | `tests/test_detect.py` |
| SOLVED_TASKS literal | `tools/inspect_tasks.py` |
| Dispatcher updates | `tools/dispatcher_audit.py` |
| Session log | `Logs/sessions/<date>_<time>_alpha-<task>.md` |
| Score / queue updates | `CURRENT_STATE.md` (after Kaggle response) |
| Submissions ledger row | `Logs/submissions_ledger.csv` (one row per submit) |
| Lessons | prepend to `Logs/lessons.md` |
| PR scratch rationale | `git_updates.md` at repo root (gitignored; becomes PR body at end) |

## Session-end discipline (Alpha-specific additions)

In addition to the universal session-end rules:

- **Update `CURRENT_STATE.md`** if you:
  - Submitted to Kaggle (score line + queue update)
  - Implemented a recipe handoff (move it from "recipe-ready" section into the SOLVED count)
  - Discovered a new public-negative (add to don't-touch list)
- **Append to `Logs/submissions_ledger.csv`** if you submitted: date, fruit name, score, base, notes.
- **Verify 584 tests still pass** before final session-log finalization.
- **PR cadence:** one thematic PR per session (per `02_GIT.md` §1.1). Bundle small chores into the main PR.

## When to escalate to the universal handshake

If your task drifts into Beta-territory (you find yourself wanting to write a recipe draft, or to refactor a `tools/mine_*.py`), stop, halt-immediately per `01_MASTER.md` §1.21(e), and ask the owner whether you should switch lanes or file the issue as `[A->B]`.

---

*This playbook is project-specific (NG, multi-track since 2026-05-02). For the universal handshake see `NewSession.md`. For Beta see `NewSession_Beta.md`.*
