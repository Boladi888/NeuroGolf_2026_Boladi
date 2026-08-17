# AUDIT 2026-06-06 - agent context bloat and unnecessary complication

Scope: repo-wide audit for files, folders, docs, and workflows that waste agent context/tokens when assigning a single focused task.

Explicit exclusions:
- Ignored the pre-existing `Onnx_Advanced/Research/Todo/AUDIT_2026-06-06_further_optimizations.md`.
- Did not audit `.ai_instructions/`; owner will handle it later.
- Did not modify code or clean files. This is a triage report only.

Working-tree note: the repo is already dirty. `git status --porcelain -uall` showed 466 entries, including 366 untracked entries, mostly under `The_400_Tasks/`. Treat any cleanup below as owner-reviewed work, not something an agent should blindly delete.

## Executive summary

The biggest context drain is not one bad file; it is that the active workspace mixes durable source, agent instructions, historical experiments, generated binaries, scratch outputs, logs, frontend dependencies, and current competition payloads in the same visible tree.

Measured outside `.ai_instructions/` and excluding the pre-existing audit:

| Surface | Files | Size / lines | Context risk |
|---|---:|---:|---|
| Normal `rg --files` visible surface | 5,444 files | n/a | High: a default agent search sees thousands of active-looking files. |
| Tracked files | 5,244 files | ~239 MB | High: many generated artifacts are already tracked, so ignore rules do not hide them from git-aware tools. |
| Tracked text files | 4,982 files | ~787,669 lines | High: too much token-addressable material for single-task agents. |
| `The_400_Tasks/` | 8,704 files total; 3,323 tracked | ~295 MB total; ~548k tracked text lines | Critical: combines task notes, ledgers, prompts, prototypes, zips, ONNXs, and archives. |
| `build_submission.py` | 1 file | ~0.92 MB; ~19k+ lines; 311 `def`s | Critical: monolithic source/control surface. |
| `Logs/` | 912 files total; 842 tracked | ~33 MB total; up to ~258k local text lines | High: durable summaries mixed with huge output dumps. |
| `tools/` | 370 files total; 216 tracked | ~45k tracked text lines | Medium/high: many stale or automation-era tools remain visible. |
| Local ignored cache/scratch | 12,122 files | ~252 MB | Medium: ignored by normal search, but punishes unrestricted scans and broad file reads. |

## Ranked findings

### 1. `The_400_Tasks/Task_by_Task/` has become the main context sink

Evidence:
- Full filesystem: `The_400_Tasks/Task_by_Task/` has 8,476 files, ~291 MB.
- Task subdirectories alone account for 8,069 files, ~286 MB.
- Tracked prototype files under `Task_by_Task/**/prototypes/**`: 2,614 files, ~70 MB.
- Per-task Markdown: 401 `taskNNN.md` files, ~60,436 lines.

Worst prototype-heavy tasks by file count:

| Task | Files in `prototypes/` | Size | Notes |
|---|---:|---:|---|
| `task366` | 878 | 11.3 MB | 174 `.py`, 531 `.onnx`, 96 `.json`; many versioned structural attempts. |
| `task018` | 681 | 26.4 MB | 267 `.py`, 287 `.onnx`, 17 `.zip`, 17 `.json`. |
| `task233` | 322 | 18.6 MB | 105 `.py`, 207 `.onnx`; active note also very large. |
| `task157` | 289 | 21.8 MB | Includes large HTML/ONNX prototype payloads. |
| `task255` | 260 | 80.1 MB | Dominated by large `.npz` prototype context caches. |

Worst per-task notes by line count:

| File | Lines | Size |
|---|---:|---:|
| `The_400_Tasks/Task_by_Task/task233.md` | 2,355 | 186 KB |
| `The_400_Tasks/Task_by_Task/task018.md` | 1,077 | 239 KB |
| `The_400_Tasks/Task_by_Task/task255.md` | 786 | 77 KB |
| `The_400_Tasks/Task_by_Task/task110.md` | 595 | 55 KB |
| `The_400_Tasks/Task_by_Task/task286.md` | 594 | 59 KB |
| `The_400_Tasks/Task_by_Task/task366.md` | 443 | 138 KB |

Why this burns agent context:
- A single-task assignment should need one compact task card plus current best, not hundreds of old builders, ONNXs, candidate zips, and result dumps.
- Versioned prototypes are named like active candidates, so agents have to infer freshness from history instead of being routed to the live one.
- Big notes have become journals. Agents asked to "continue task233" will read stale attempts before they find the current rule, floor, current best, and next move.

Recommended cleanup:
1. For each task directory, create a strict active layout:
   - `taskNNN.md` = current compact brief only.
   - `prototypes/current/` = current candidate builder(s) and the latest ONNX.
   - `prototypes/archive/` or external archive = old versioned attempts.
2. Keep at most 3 active builders per task in normal search: current best, current experiment, and one fallback.
3. Move large generated prototype artifacts out of active task folders:
   - `.onnx`, `.zip`, `.npz`, `.html`, result JSON dumps.
   - Preserve a manifest row in the task note if historical provenance matters.
4. Add a "Current State" block at the top of every large `taskNNN.md`:
   - current best source/cost
   - current hypothesis
   - next exact experiment
   - known traps
   - active files to read
   - archive pointer

### 2. `build_submission.py` is too large for the role it plays

Evidence:
- Root `build_submission.py` is ~0.92 MB.
- Earlier line counting reported ~19,329 lines; `rg` line numbers reach ~21,992, so it is roughly a 20k-line file.
- It contains 311 top-level function definitions.
- It has 1,883 `onnx.helper` references and 1,332 `make_node` references.
- It mixes generic graph helpers, many task-specific network builders, detector logic, and CLI/build orchestration.

Why this burns agent context:
- Any task touching the builder forces agents into a huge root file.
- Search results inside the file are noisy because every task family is co-located.
- A single edit has high accidental-coupling risk because helper, detector, and task-specific code are interleaved.

Recommended cleanup:
1. Split by responsibility, not by arbitrary line count:
   - `neurogolf/build/core_graph.py` for `_make_model`, shape helpers, standard I/O wrappers.
   - `neurogolf/build/patterns/*.py` for reusable network builders.
   - `neurogolf/build/tasks/taskNNN.py` only for genuinely task-specific builders.
   - `neurogolf/build/dispatch.py` for detection/selection.
   - keep root `build_submission.py` as a thin CLI compatibility wrapper.
2. Do the split mechanically first, preserving imports and behavior.
3. After split, add a "single-task agent path":
   - `tools/agent_helpers/extract_task.py NNN`
   - `The_400_Tasks/Task_by_Task/taskNNN.md`
   - `neurogolf/build/tasks/taskNNN.py`
   - relevant pattern module only.

### 3. Generated artifacts are already tracked, so `.gitignore` does not fully protect agents

Evidence:
- Tracked extension summary outside `.ai_instructions/`:
  - `.json`: 638 files, ~94.9 MB
  - `.zip`: 74 files, ~57.8 MB
  - `.py`: 2,620 files, ~24.5 MB
  - `.md`: 1,279 files, ~11.4 MB
  - `.onnx`: 95 files, ~3.9 MB
- `.gitignore` correctly blocks many future generated payloads, including `*.onnx`, `*.npz`, `*.archive`, `Logs/*.out.log`, `work/`, `submissions/**`, and prototype result dumps.
- But tracked generated files remain visible to git, normal code review, and normal search.

Examples among largest tracked files:
- `The_400_Tasks/artifact_manifests/2026-05-07-preserved-files-manifest.txt` - 24,856 lines.
- Many tracked `The_400_Tasks/Task_by_Task/**/submission*.zip` files around ~0.9 MB each.
- GUI task JSONs under `GUI/client/public/tasks/` around ~1.0-1.4 MB each.
- Large JSON sweeps in `The_400_Tasks/Task_by_Task/task286/` and `Logs/`.

Why this burns agent context:
- Agents often trust tracked files as relevant source.
- Tracked generated payloads make the active repo look larger and more authoritative than it should.
- Ignore rules create a false sense of cleanup if already-tracked artifacts are not untracked.

Recommended cleanup:
1. Make an explicit "tracked artifact exception list":
   - exactly which zips/ONNXs/JSONs are intentionally versioned.
2. For everything else, use `git rm --cached` in a controlled cleanup branch/session, then rely on existing ignore rules.
3. Keep provenance as small Markdown/CSV manifests instead of binary payloads.
4. Add a CI/script check that fails if new `Task_by_Task/**/prototypes/*.zip`, `.onnx`, `.npz`, `.html`, or large result JSON files become tracked again unless allowlisted.

### 4. `Logs/` mixes durable knowledge with massive run dumps

Evidence:
- Full `Logs/`: 912 files, ~33 MB.
- Tracked `Logs/`: 842 files, ~10.5 MB and ~112.9k tracked text lines.
- Local text count including ignored logs/archives: 828 log-like text files, ~257.8k lines.
- Largest local log outputs:
  - `Logs/external_blend_2026-05-07T_try5500_jvlegend_safe_dry.out.log` - 47,554 lines, ~4.1 MB.
  - `Logs/external_blend_2026-05-07T_try5500_jvlegend_safe.out.log` - 47,549 lines, ~4.1 MB.
  - `Logs/runs.jsonl.2026-06-05T012735Z.archive` - 39,352 lines, ~6.4 MB.

Why this burns agent context:
- Agents need durable conclusions, not raw console output.
- Large logs make `Logs/` look more important than current task notes.
- Some logs are already ignored, but still present locally and visible to unrestricted scans.

Recommended cleanup:
1. Split `Logs/` into:
   - `Logs/current/` or root: small durable summaries only.
   - `Logs/archive/`: compact old summaries.
   - local-only external scratch path for raw dumps.
2. Replace large `.out.log`, `.archive`, and bulk JSON run dumps with 10-30 line summaries when the finding matters.
3. Make `tools/state.py` or a new `tools/brief.py --task NNN` the canonical way to retrieve current state, so agents do not browse logs manually.

### 5. Tool routing is stale and over-broad

Evidence:
- `tools/` has 370 files total, 216 tracked, ~45.7k tracked text lines.
- `tools/README.md` references several tools that do not exist in the current `tools/` folder:
  - `external_blend.py`
  - `blend_diff.py`
  - `audit_external_cascade.py`
  - `karnak_xref.py`
  - `mine_karnak_targeted.py`
- `tools/` also contains 4 malformed filenames that look like failed shell move commands, for example names containing `_archive... && mv C...`.

Why this burns agent context:
- Agents sent to a missing tool waste time searching, checking history, and guessing replacements.
- Malformed filenames are high-friction search noise and signal that cleanup operations partially failed.
- A 200+ script tool directory without a current capability index makes single-task routing expensive.

Recommended cleanup:
1. Replace `tools/README.md` with a generated, verified inventory:
   - tool name
   - status: active / archived / deprecated / missing
   - owner surface: build, submit gate, task triage, GUI, maintenance
   - one-line usage
2. Move deprecated automation-era tools into `tools/_archive/` or remove them from normal search.
3. Delete or quarantine malformed `tools/_archive... && mv C...` filenames after verifying their content is already preserved in `tools/_archive/`.
4. Add a script that checks `tools/README.md` references exist.

### 6. Multiple archive layers remain in the active checkout

Evidence:

| Archive / scratch path | Files | Size |
|---|---:|---:|
| `_hermes_archive/` | 1,858 | ~265.9 MB |
| `submissions/` | 1,533 | ~203.4 MB |
| `work/` | 1,102 | ~16.9 MB |
| `Onnx_Advanced/_archive/` | 69 | ~1.4 MB |
| `Logs/_archive/` | 171 | ~1.4 MB |
| `Research_Archive/` | 146 | ~0.8 MB |
| `The_400_Tasks/Task Prompts/` | 210 | ~0.7 MB |
| `tools/_archive/` | 42 | ~0.3 MB |

Why this burns agent context:
- There are too many "archive" concepts, so agents cannot tell whether history is relevant.
- `_hermes_archive/`, `work/`, and `submissions/` are large enough to punish broad scans even if mostly ignored.
- `Research_Archive/` and `Onnx_Advanced/Research/` coexist, which makes "research" routing ambiguous.

Recommended cleanup:
1. Define one active archive policy:
   - active docs live in normal folders
   - compact historical docs live under one `archive/` root
   - generated/binary archives live outside the repo or in `.notongit`
2. Move `_hermes_archive/` out of the active checkout if it is truly archived.
3. Keep `submissions/submission.zip` and `submissions/blend.zip` active; move old per-task payloads and bundle directories out of normal browsing.
4. Keep `work/` fully local-only and clear it on a cadence.

### 7. Root onboarding state is stale or misleading

Evidence:
- `CURRENT_STATE.md` says it was generated `2026-05-25 15:04 UTC`, but this audit ran on `2026-06-06`.
- `README.md` still says "Read .ai_instructions..." and mentions `Research/00_INDEX.md`; the visible repo has `Research_Archive/` and `Onnx_Advanced/Research/`, not a root `Research/` folder.
- `WHERE_DATA_LIVES.md` is current and useful, but it adds many warnings that agents may read before they know the task-specific path.

Why this burns agent context:
- Agents use root files first. A stale current-state snapshot sends them into obsolete scores, queues, and worker assumptions.
- Missing or renamed paths cause unnecessary file search.
- Onboarding docs are not separated by assignment type.

Recommended cleanup:
1. Replace stale `CURRENT_STATE.md` with either:
   - an auto-refreshed file generated at every state-changing operation, or
   - a tiny pointer that says to run `python tools/state.py --terse`.
2. Add a root `AGENT_TASK_START.md` outside `.ai_instructions/` for single-task agents:
   - read `WHERE_DATA_LIVES.md` only if touching filesystem layout
   - run `python tools/state.py --terse`
   - open `The_400_Tasks/Task_by_Task/taskNNN.md`
   - open only the active builder file for that task
   - never browse archives unless explicitly told
3. Fix root `README.md` paths that reference old/missing research locations.

### 8. Frontend dependencies are ignored, but still large locally

Evidence:
- `GUI/` total local surface: 9,809 files, ~288 MB.
- Most of that is ignored dependency/build output:
  - `GUI/client/node_modules/`
  - `GUI/client/dist/`
- `GUI/.gitignore` correctly ignores these paths.

Why this still matters:
- Normal `rg --files` does not show these files, so they are less dangerous than tracked task artifacts.
- But unrestricted scans (`rg -uu`, generic recursive listing, or naive agents) can dump thousands of dependency files into context.

Recommended cleanup:
1. Keep `GUI/.gitignore`; it is doing useful work.
2. Consider adding a root-level note or agent helper that says never use unrestricted recursive scans unless specifically auditing ignored files.
3. If frontend work is rare, document that `node_modules` can be removed and restored via `npm install`.

## Proposed cleanup sequence

### Phase 0 - guardrails before cleanup

Do these before moving or untracking anything:
1. Freeze a list of intentionally tracked generated artifacts.
2. Add a verification script for "new tracked generated payloads" under task prototypes, logs, and submissions.
3. Add a small root single-task routing doc outside `.ai_instructions/`.

### Phase 1 - high-signal, low-risk wins

1. Refresh or replace `CURRENT_STATE.md`.
2. Fix `tools/README.md` missing-tool references.
3. Quarantine malformed `tools/_archive... && mv C...` filenames.
4. Move large ignored local logs/archives out of `Logs/` or summarize them.
5. Add top-of-file "current state" blocks to oversized task notes: `task233`, `task018`, `task255`, `task110`, `task286`, `task366`.

### Phase 2 - reduce active file count

1. For each heavy task folder, move old prototypes into archive/external storage:
   - start with `task366`, `task018`, `task233`, `task157`, `task255`.
2. Untrack generated zips/ONNXs/large JSONs that are covered by ignore policy but already tracked.
3. Collapse prompt history so `The_400_Tasks/Task Prompts/` only holds active prompts and a compact index of already-ran prompts.

### Phase 3 - split the monolith

1. Extract generic ONNX graph helpers from `build_submission.py`.
2. Extract reusable pattern builders.
3. Extract task-specific builders.
4. Leave a thin root CLI wrapper.
5. Add tests after each mechanical split to preserve current behavior.

## Target end state for single-task agents

A focused task assignment should require only:

1. `AGENT_TASK_START.md` or equivalent 20-line root router.
2. `The_400_Tasks/Task_by_Task/taskNNN.md`, with a compact current-state block first.
3. One active builder module for task NNN, plus one reusable pattern module if needed.
4. `Onnx_Advanced/INDEX.md`, `core.md`, `cost_floor_reference.md`, and relevant `traps.md` sections only when ONNX details are actually needed.
5. `tools/ng_gate.py` or `tools/agent_helpers/*` via a specific command, not broad tool browsing.

The goal is not a smaller repo for its own sake. The goal is for an agent assigned "fix taskNNN" to avoid reading years of adjacent history, generated payloads, stale automation, and old logs before it can make one useful edit.
