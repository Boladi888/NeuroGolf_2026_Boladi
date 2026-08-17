> LIVE — edit freely.
> Created: 2026-05-01. Last refresh: 2026-05-10.
>
> **🟦 New agents — read `CURRENT_STATE.md` at repo root FIRST.** That file is the single snapshot of score / queue / lane assignments / don't-touch / where-i-left-off. This file (`03_PROJECT.md`) is tech-stack + architecture + implementation rules + open bugs + workflow conventions. Closed-item history → `03_PROJECT_HISTORY.md`. Submission ledger → `Logs/submissions_ledger.csv`.
>
> **Doc hygiene note (2026-07-11).** This file still carries historical rationale, but live numbers and queues drift quickly. Treat hardcoded counts, scores, and old lane/staging examples below as historical unless re-verified against `CURRENT_STATE.md`, `AGENT_TASK_START.md`, `WHERE_DATA_LIVES.md`, `The_400_Tasks/Task_by_Task/`, and `Logs/submissions_ledger.csv`.

---

# 03_PROJECT.md — NG (NeuroGolf 2026 Kaggle Championship)

> **Anti-bloat discipline.** This file describes what is true RIGHT NOW about the project, plus pointers to the durable record. The session log is the durable record for "what happened this session." `Logs/lessons.md` is the durable record for "what I learned." Closed TODO items, fixed bugs, dismissed critiques: collapse to one-line entries with pointers — do NOT keep full prose here. See "End-of-session discipline" in `00_START_HERE_README_AI.md`.
>
> **Format constraint on entries.** Every TODO / closed-item / dismissed-critique / fixed-bug entry: short title + max 2 sentences. Longer rationale goes to `Logs/lessons.md` or the per-session log; link to it from here.

---

## Project description

The 2026 NeuroGolf Championship — Kaggle competition hosted by the Neurosynthetic Research Institute and the IJCAI-ECAI 2026 Competitions Track. Goal: build the smallest possible ONNX neural networks that solve the 400 public training-set tasks of ARC-AGI-1. Submissions are ONNX graphs scored by `onnx-tool` on computational graph size and memory footprint (1.44 MiB cap = 1,509,949 bytes binary). Successor to the NeurIPS 2025 Code-Golf Championship: same task suite, different submission medium (Python source → ONNX graphs).

Full reference: [`../Onnx_Advanced/00_COMPETITION_SPEC.md`](../Onnx_Advanced/00_COMPETITION_SPEC.md) (moved there from the retired root `Research/` in the 2026-06-10 restructure; the `NEW_RESEARCH_AGENT.md` topic roadmap was retired with it — the surviving topic record is [`../Research_Archive/_RESEARCHED_TOPICS.md`](../Research_Archive/_RESEARCHED_TOPICS.md)).

## GitHub URL, branch, license

- **Repo:** https://github.com/Boladi888/NeuroGolf_2026_Boladi
- **Default branch:** main
- **License:** Apache-2.0 (switched from MIT 2026-05-01 to align with the competition's winners'-license requirement and avoid relicensing churn at win-time).
- **Collaborators:** Owner (Boladi888) + named friend.
- **`.gitignore` convention:** shared-collaborator (blacklist) pattern per `02_GIT.md` §8.1.

## Tech stack

> **Data layout (2026-05-10 reorg; corrected 2026-06-06; rehomed 2026-08-17).** `.venv/` and `data/` in the main checkout are **Windows directory junctions** to `.notongit\NeuroGolf_nogit\<dir>\` — that payload now sits inside the repo, gitignored, instead of in a sibling folder. `submissions/` is a REAL in-repo directory (canonical — the `.notongit` copy is a stale snapshot, never restore from it). Full inventory + recovery in [`WHERE_DATA_LIVES.md`](../WHERE_DATA_LIVES.md) at repo root.

- **Python:** 3.14.4 in `.venv/` (junction → `.notongit\NeuroGolf_nogit\.venv\`). Gitignored. Claude-Code worktrees don't auto-get one — call via the junction at `.venv\Scripts\python.exe` or via `${NEUROGOLF_PYTHON}`. Mirrors Kaggle runtime.
- **Submission medium:** ONNX neural network graphs, opset 10 (NOT Python source).
- **Pinned deps:** `numpy==2.4.4`, `onnx==1.21.0`, `onnxruntime==1.24.4`, `onnx-tool==1.0.1` (per Kaggle starter notebook cell-0).
- **Dataset:** ARC-AGI-1 (400 public training tasks). Re-pull recipe in `04_DATA.md`.
- **Test:** `python -m pytest tests -q` — run it live; do not trust historical pass counts. **Do not use `unittest discover`**: 30 of the 37 modules are pytest-style and collect zero under it, so it reports OK on ~623 of 1,247 tests. `pytest.ini` pins the correct runner. Without the Kaggle dataset present, seven data-dependent modules skip themselves (`tests/conftest.py`) and 615 tests still run — that is what CI executes.
- **Build:** the root `build_submission.py` dispatcher is retired; the file now lives at `tools/build_submission.py` and is the 311-function ARC detector library that `tests/test_detect.py` exercises, not a build entry point. For one solved task's dispatcher-picked payload, use `python tools/build_dispatcher_task.py NNN`. This is an archived competition repo — it should not be used to rebuild or submit packages.
- **Deployment target:** Kaggle competition `neurogolf-2026`. See Workflows table below.

## File structure

```
K-NeuroGolf/
├── .ai_instructions/        ← agent rules (tracked)
├── .github/workflows/       ← CI: pytest on a bare checkout
├── .notongit/               ← local-only payload: venv, data, submission archives (gitignored)
├── .update/                 ← DONE_*.md insight packets awaiting Librarian integration
├── Agenters/                ← role playbooks for the multi-agent pipeline
├── docs/                    ← METHODOLOGY.md + project-history/ (archive records)
├── GUI/                     ← local task-grid viewer
├── Handoff/                 ← cross-agent handoff workspace (private/ is gitignored)
├── Logs/                    ← submissions ledger + durable lessons + reports
├── Onnx_Advanced/           ← technique library, scoring internals, worker protocol
├── Prompts/                 ← worker prompt packs
├── Research_Archive/        ← historical notes + source archive
├── The_400_Tasks/           ← per-task records, prompts, task-state map
├── data/                    ← competition tasks + competitor artifacts (junction, gitignored)
├── tools/                   ← verification gate, cost profiler, optimizers, detector library
├── tests/                   ← 37 modules / 1,247 tests
├── submissions/             ← submission.zip + manifests (per-task ONNXs gitignored)
├── ml/ngs/                  ← Neural-Guided ONNX program Search
├── README.md                ← project docs
├── LICENSE / NOTICE         ← Apache-2.0
└── .gitignore               ← shared-collaborator (blacklist) pattern
```

| File / Path | Purpose | Tracked? |
|---|---|---|
| `tools/build_submission.py` | The 311-function ARC detector library (19k lines) that `tests/test_detect.py` exercises. Moved off the root in the 2026-07 cleanup; not a build entry point. | Yes |
| `tests/test_detect.py` | Detector and builder unit tests | Yes |
| `tools/*.py` | Analysis + pre-submit + blend (full inventory in `04_DATA.md`) | Yes |
| `Onnx_Advanced/Research/` | Active research cards, completed-card ledger, and research deliverable lifecycle | Yes |
| `Research_Archive/` | Historical notes and older source/archive corpus | Yes |
| `The_400_Tasks/` | 400-task state map, candidate queue, normalized candidate-task metadata, preserved artifact manifest; source for `Logs/submission_candidate_digest_2026-05-08.*` and `Logs/orange5_ablation_queue_2026-05-08.*` | Yes |
| `Logs/sessions/`, `Logs/lessons.md` | Per-session logs + durable lessons | Yes |
| `Logs/runs.jsonl`, `Logs/*.csv` | Per-run logs + blend / silent-fail reports | Yes |
| `data/competitions/neurogolf-2026/` | Task JSONs + bundled `neurogolf_utils/`; junctioned local data | No (junction) |
| `data/competitor_artifacts/` | External ONNX/reference dumps; junctioned local data | No (junction) |
| `submissions/` | Real in-repo directory. `submission.zip` and selected manifests are tracked; per-task ONNXs, old zips, and scratch are gitignored. | Mixed |
| `local_profiler/` | Ignored home for ONNX Runtime profiler dumps | No |
| `.venv/` | Python 3.14.4 env (junction → `.notongit/...`) | No (junction) |

## Architecture and key design decisions

- **Cost-aware dispatcher (PR #21 D2).** `detect_pattern` builds + scores every matching detector and picks the cheapest verifying alternative. **Trade-off accepted:** more compute per dispatch in exchange for elimination of ordering-dependent regressions by construction. Source: `Logs/sessions/2026-05-02_2203_deep-audit-fixes.md`.
- **Ethical-play stance.** No use of the "positive-cost wrappers" exploit family or `Compress` / `ConstantOfShape`-with-data-derived-shape exploits — only mechanisms that produce the same answer at runtime as at scoring time. **Trade-off accepted:** clean-play ceiling around 4,036 LB (Infinitesimals reference) vs 6,000+ exploit-driven scores. Source: `Onnx_Advanced/scoring_internals.md#a2_cost_metric_exploits`.
- **Submission-message naming convention.** Every Kaggle `-m` message ends with a fruit name advancing alphabetically (apple, banana, cherry, ...); cycles back to A after Z. **Trade-off accepted:** cosmetic constraint, makes submission ledger trivially scannable. Standing queue is now closed; choose the next fruit only after building an isolated-ablation or organic+orange5 blend.
- **Worker staging workflow.** Workers read current prompts, build under `The_400_Tasks/Task_by_Task/taskNNN/prototypes/`, and stage gated candidates under task-local `staged_candidates/`; the captain ships. The old `staged_submits/` inbox was retired and removed 2026-08-17.
- **Default-arg lambda binding in `_collect_matches`.** Every detector's network factory binds runtime parameters as defaults to avoid late-binding capture bugs (names like `lut` / `in_h` / `M` / `F` are reused across handlers). **Trade-off accepted:** slightly more verbose builder syntax in exchange for correctness. Source: `Logs/lessons.md` 2026-05-02 "Late-binding lambda capture".

## Implementation rules

> Workflow / coding rules promoted from `Logs/lessons.md` because they govern recurring decisions. Each entry is short title + max 2 sentences. Detailed rationale lives in the lesson it was promoted from.

- **Pre-flight mining before implementing from a brief.** For any new pattern from a `[B→A]` handoff or research note, run a one-shot 400-task mining probe BEFORE writing builder/detector code. Brief estimates are systematically 5-15× optimistic; pre-flight catches 0-hit candidates in 30 sec.
- **Run `tools/inspect_tasks.py --summary` before designing detectors.** Confirms which detectors target real tasks (only 3/400 are all-30×30; 138/400 are shape-changing; 130/400 introduce new colors). Knowing the distribution prevents wasting effort on detectors that match nothing.
- **`verify_subset` on actual task pairs is the gate; structural checks are not.** `check_network` (op-set validity) and `score_network` (no banned ops, dims static) pass for any well-formed pipeline; only `neurogolf_utils.verify_subset(session, pairs)` against actual data tests cell-level correctness. Add it to every prototype script, not just integration tests.
- **Detector input validation at the dispatch boundary, not per-detector.** Validate input-shape contract (rectangular, non-empty pairs) ONCE in `_collect_matches`, return `[]` on malformed input; let detectors assume well-formed grids. Adding try/except around 28 individual detector calls would balloon the function and obscure flow.
- **Stamp/spread/rotation patterns must clip to `ch0`, not `is_active`.** Canvas-padding cells encode as "color 0" (ch0=1), same as in-grid bg cells. `is_active = ReduceSum` doesn't distinguish; only `input_ch0` (which is 1 only at active-grid color-0 cells) prevents stamps from leaking into padding. Source: `Logs/lessons.md` 2026-05-04 "Stamp/spread/rotation patterns must clip to ch0".
- **SOLVED_TASKS / Alpha-owned constants: import-with-fallback, never mirror.** Beta-owned tools that need SOLVED-status info should `from inspect_tasks import SOLVED_TASKS`. For worktrees lacking the heavy import chain, `mining_lib.py` ships a regex-fallback that parses SOLVED_TASKS literally from `inspect_tasks.py` source — single source of truth.
- **Detailed recipes (numpy verifier + ONNX sketch + cost) > code for cross-lane handoff.** When working in one lane and the next session is the OTHER lane, file a recipe instead of half-implementing. Per-hour return: ~30 min recipe × 2-4 hr saved by next-lane = 4-8× ROI vs implementing yourself.
- **Kaggle-faithful verification requires `convert_from_numpy`, not argmax.** The validator checks per-channel `value == 1` strictly and decodes multi-lit cells (multiple channels with value 1) to color 11. `--hybrid`'s `(result > 0.0)` and argmax-based checks are looser; only `nu.convert_from_numpy(out)` against `p['output']` matches Kaggle. Source: post-PR-#46 verification finding 2026-05-05.

## Owner workflow preferences

- **One thematic PR per session** (per lane in multi-track), bundled. Default cadence per `02_GIT.md` §1.1.
- **Pre-submit checklist** (codified in PR #21, exercised in PR #23): see `04_DATA.md` "Recipe: pre-submit checklist".
- **Worker flow:** prompt/context pack → task-local prototype → `ng_gate`/score gates → task-local `staged_candidates/` → captain ships via `tools/swap_and_submit.py`.
- **Daily Kaggle quota:** 5/day per agent, midnight UTC reset; coordinate before any submit. Owner-directed cadence (standing 500-pt threshold permanently met at 3,275.37 LB).

## Workflows / commands

| Action | Command | Notes |
|---|---|---|
| Run all tests | `python -m pytest tests -q` | Re-run live; historical pass counts are intentionally not copied here. `unittest discover` silently collects only ~half the suite — see "Test" under Tech stack. |
| Build one dispatcher task | `python tools/build_dispatcher_task.py 050` | Output: dispatcher-selected local `submissions/task050.onnx`; safer than the default single-task builder for solved tasks. |
| Build one task (per-task) | `python tools/build_dispatcher_task.py NNN` | Use this for per-task work; `--hybrid`/`--all` are disabled. |
| Pre-submit gauntlet | (see `04_DATA.md` "Recipe: pre-submit checklist") | 9-step sequence; gates submission. |
| Re-pull `data/` | `kaggle competitions download -c neurogolf-2026 -p data/competitions/neurogolf-2026 && cd data/competitions/neurogolf-2026 && unzip neurogolf-2026.zip && rm neurogolf-2026.zip && cd ../../..` | ~15 s. **NEVER junction `data/` into a worktree** (recursive-delete trap). |
| Submit | `cp submissions/blend.zip submissions/submission.zip && kaggle competitions submit -c neurogolf-2026 -f submissions/submission.zip -m "..."` | Kaggle CLI requires literal `submission.zip` filename. |
| Check scoring queue | Use `tools/agent_helpers/show_task.py --task NNN`, `The_400_Tasks/Task_by_Task/`, and the latest ledger/research cards | Avoid old date-stamped queue snapshots unless explicitly doing archaeology. |

## Lane attribution

NG now uses prompt/context-pack workers with task-local prototype and `staged_candidates/` folders. The `staged_submits/` inbox it replaced was removed 2026-08-17.

## Open TODO

> Live state — current Kaggle best, active queue, recipe-ready handoffs, and don't-touch list — is in **`CURRENT_STATE.md`** at repo root. The strategic backlog (tiered experiments / future patterns) is in **`05_BACKLOG.md`**.

## Closed-item ledger (audit trail)

> Moved to **`.ai_instructions/03_PROJECT_HISTORY.md`** (2026-05-10 reorg). Read on demand for historical context. Per-session narratives are in `Logs/sessions/`.

## Open bugs / known issues

- **🔴 KAGGLE CASCADE BUG (2026-05-04, host-confirmed).** Cross-task memory cascade in evaluator triggers `SubmissionStatus.ERROR` (entire submission rejected, not per-task 0). Triggered by both (a) zero-cost shape-only patterns (transpose, single-Gather color-perm — tasks 16/53/179/241/276/309/337) and (b) high-memory patterns (task1/2/20/98/143/243/251/346 etc.) and (c) certain externals. Historical notes claimed strawberry/xigua reached 736.61 via 39 stubs, but Kaggle CLI on 2026-05-07 reports those submissions as ERROR; current valid post-rule baseline is yumberry #25 at 3394.79. Detailed Beta-prep recipes filed: `Research_Archive/notes/recipe-zero-cost-cascade-fix.md`, `recipe-cost-dispatcher-update.md`, `recipe-memory-hog-audit.md`. Source: NAS DB thread 1 (Michael D. Moffitt 2026-05-04 18:57 UTC) + Kaggle web error message on raisin #18 + quince #17 + live CLI check 2026-05-07.
- **🔴 ALL submissions made 2026-05-01 → 2026-05-04 16:52 UTC are ERRORed under the new validator.** The 14 prior submissions in the Validation History table are HISTORICAL pre-rule-change scores, not current LB. Current valid best: banana #28 = 5002.95 LB.
- **`ConstantOfShape`-with-data-derived-shape is exploitable but not used.** Confirmed by `kq5y` in the Kaggle "Issues in onnx-tool" thread; ethical-play stance. See `06_COMMUNITY.md` Rejected patterns.
- **Silent-fail tasks exist on Kaggle's hidden private test set.** task363 passed local `verify_subset` but scored 0 on the hidden suite per the "Blended 401-Task" community notebook. **Mitigation:** pre-submit `silent_fail_check --min-arcgen-N 30`; gate on threshold.
<!-- closed 2026-05-03 by deep-read of LB-5126 notebook: April 21 update existence confirmed (host-announced rescoring fix); specifics still not detailed but the unknown is no longer "is there one?" -->
- ~~Two metric updates not one (2026-04-21 + 2026-04-28).~~ **Existence confirmed 2026-05-03** via deep-read of LB-5126 notebook — April 21 was a host-announced rescoring fix. Specifics still partial (full April 21 changelog not yet in any source we have access to); revisit if Kaggle discussion-forum access is restored. Source: `Research_Archive/notes/deep-read-lb-5126-bug-free-blend.md`.
- **task366 is on a "known-tricky" list** (5044-LB notebook explicitly notes "keeps task366 on the baseline branch"). Not in our SOLVED set; spot-check if ever targeted.
- **Twelve of our 67 SOLVED (16, 50, 53, 129, 266, 276, 294, 309, 331, 337, 373, 385) appear in competitors' skip lists** — original 4 (53/129/266/385) from Russian SKIP_TASKS; 5 added 2026-05-03 from `the-epic-sax-ensembler.ipynb` (016/050/276/309/337); 3 added 2026-05-03 from `acking-multiple-tasks-part-3.ipynb` (294/331/373). Submitted scores match local prediction within ~1.33% on every submission. Pattern-match robustness is competitor-specific. Run extra regression check on these 12 before any future submission. See `06_COMMUNITY.md` skip-list table + `Research_Archive/notes/deep-read-the-epic-sax-ensembler.md` + `Research_Archive/notes/deep-read-acking-multiple-tasks-part-3.md`. (**Verified 100% pre-jackfruit/kiwi 2026-05-04**.)
- **Strict detector/builder parity has zero known open failures.** `tests/test_detector_builder_parity.py` gates matched builders with Kaggle-faithful `convert_from_numpy` and fails if `tools.inspect_tasks.SOLVED_TASKS` falls out of sync with strict-clean detector coverage.

## Validation history

> Moved to **`Logs/submissions_ledger.csv`** (2026-05-10 reorg). One row per Kaggle submission with date, fruit, kind, local prediction, Kaggle score, verdict, and source pointer. Append a row after each `kaggle competitions submit`.

## Where I left off

> Moved to **`CURRENT_STATE.md`** at repo root (2026-05-10 reorg). Replace, don't stack — see anti-bloat rules in `00_START_HERE_README_AI.md`.

---

*Last updated: 2026-05-10 (layered reorg — see `Logs/sessions/2026-05-10_1700_layered-reorg.md`).*
