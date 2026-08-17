# .ai_instructions Retirement Plan

Date: 2026-06-12
Mode: report-only planning. No files moved, deleted, or rewritten by this plan.

## Goal

Delete `.ai_instructions/` after preserving the still-useful material and updating live references so no active doc or tool points at a missing file.

## Current Inventory

`.ai_instructions/` is about 290 KB across 21 markdown files.

Top-level files:

- `00_START_HERE_README_AI.md` - old agent read order and anti-bloat/session discipline.
- `01_MASTER.md` - broad universal agent rules.
- `02_GIT.md` - universal git workflow rules.
- `03_PROJECT.md` - older project architecture/state/workflow notes.
- `03_PROJECT_HISTORY.md` - closed-item history.
- `04_DATA.md` - data/playbook recipes, including the pre-submit checklist.
- `05_BACKLOG.md` - older strategic backlog.
- `06_COMMUNITY.md` - external/community source catalog.
- `proposed_master_changes.md` - local rule proposals.

Playbooks:

- `agent_feedback_rules.md` - process/worker/captain rules, memory policy, anti-re-bloat charter.
- `NewSession.md` - old session-start/session-end protocol.
- `Research.md` - research filing workflow.
- `SpreadSweepWorker.md` - old spread-sweep worker prompt and manifest schema.
- `WorkerAgent_Setup.md` - archived worker transport doc.
- `handoffs/_README.md`, `handoffs/ResearchHandoff.md` - handoff templates.
- `_archive/*` - retired lane/session playbooks.

## Material Worth Preserving

Preserve by extracting the live facts, not by moving the whole folder wholesale.

| Source | Useful content | Proposed destination |
|---|---|---|
| `00_START_HERE_README_AI.md` | anti-bloat/session discipline that is still relevant | fold into `README.md` onboarding only if still needed; otherwise superseded by `AGENT_TASK_START.md` and `Agenters/*` role docs |
| `01_MASTER.md` | durable local safety rules not already covered by current Codex/Agenters rules | if any are still wanted, add short owner-facing rules to `CLAUDE.md` or an `Agenters/README.md` policy section |
| `02_GIT.md` | shared-collaborator `.gitignore`, PR/branch cautions | keep only repo-specific parts in `README.md` or `Agenters/README.md`; generic PR ritual is stale for current in-place workflow |
| `03_PROJECT.md` | project description, old architecture notes, `.gitignore` convention | current equivalents mostly exist in `README.md`, `CURRENT_STATE.md`, `WHERE_DATA_LIVES.md`, `AGENT_TASK_START.md`, and `The_400_Tasks/current_strategy.md` |
| `03_PROJECT_HISTORY.md` | closed-item history | move to `Logs/_archive/ai_instructions/03_PROJECT_HISTORY.md` or drop if fully redundant with session logs |
| `04_DATA.md` | data inventory, pre-submit checklist, cost/scoring notes | migrate live recipes to `submissions/README.md`, `tools/README.md`, or `WHERE_DATA_LIVES.md` depending on scope |
| `05_BACKLOG.md` | old strategic backlog | migrate only still-open items to `The_400_Tasks/current_strategy.md` or `Agenters/_proposals/`; archive the rest |
| `06_COMMUNITY.md` | external artifact/source catalog and licensing notes | move live source catalog to `Competitor_Notebooks/competitor_notebooks_2026-05/00_INDEX.md` or `Research_Archive/00_INDEX.md` |
| `agent_feedback_rules.md` | worker/captain process rules, memory policy, anti-re-bloat charter | split into `Agenters/README.md`, relevant role READMEs, `CLAUDE.md`, and `Onnx_Advanced/.update/README.md` where already cited |
| `NewSession.md` | session log format | if still used, move concise format to `Logs/README.md`; otherwise remove references |
| `Research.md`, `ResearchHandoff.md` | research workflow/templates | migrate to `Research_Archive/README.md` or `Research_Archive/00_INDEX.md` |
| `SpreadSweepWorker.md` | manifest/staging schema references | replace with `Onnx_Advanced/WORKER_PROTOCOL.md` plus `staged_submits/` manifest examples |
| `WorkerAgent_Setup.md`, `_archive/*` | retired transport/lane docs | archive only if historically valuable; not part of live read path |

## Active References To Update

These are outside `.ai_instructions/` and should be cleaned before deleting the folder.

Live docs:

- `README.md` onboarding list points directly at `.ai_instructions/00_START_HERE_README_AI.md`, `01_MASTER.md`, `02_GIT.md`, and `03_PROJECT.md`.
- `WHERE_DATA_LIVES.md` points to `.ai_instructions/03_PROJECT.md` for project context.
- `CLAUDE.md` cites `.ai_instructions/playbooks/agent_feedback_rules.md`.
- `Agenters/Librarian/README.md` cites `.ai_instructions/playbooks/agent_feedback_rules.md` charter section.
- `Logs/README.md` points to `.ai_instructions/playbooks/NewSession.md`.
- `submissions/README.md` points to `.ai_instructions/04_DATA.md` for the pre-submit checklist.
- `Onnx_Advanced/core.md` points to `.ai_instructions/playbooks/SpreadSweepWorker.md` and `agent_feedback_rules.md`.
- `Onnx_Advanced/.update/README.md` and `Onnx_Advanced/CHANGELOG.md` cite `agent_feedback_rules.md`.
- `The_400_Tasks/current_strategy.md` cites `agent_feedback_rules.md` for older submit-first doctrine.

Live code:

- `tools/repo_lint.py` includes `.ai_instructions/*.md` in active doc surfaces and allowlists `.ai_instructions/playbooks/WorkerAgent_Setup.md`.
- `tools/state.py` prints `.ai_instructions/03_PROJECT.md` as a next-read pointer.
- `tools/refactor_paths.py` includes `.ai_instructions` in `SEARCH_DIRS`.
- `Onnx_Advanced/.update/tools/lib_lint.py` resolves missing markdown references by also checking `repo / ".ai_instructions" / "playbooks" / rel`.

Historical references:

- Many `Logs/sessions/**`, `Logs/_archive/**`, `Research_Archive/**`, and audit files mention `03_PROJECT.md`, `04_DATA.md`, `05_BACKLOG.md`, `06_COMMUNITY.md`, or `.ai_instructions/playbooks/*`.
- These can remain as historical text if the retirement leaves a redirect/archive note somewhere stable, but direct markdown links to deleted files should be treated as dead references if the files are still in active lint scope.

## Proposed Replacement Map

Use these canonical homes after retirement:

- General onboarding: `README.md` + `AGENT_TASK_START.md`.
- Current score/state pointers: `CURRENT_STATE.md`.
- Filesystem/data layout: `WHERE_DATA_LIVES.md`.
- Worker protocol: `Onnx_Advanced/WORKER_PROTOCOL.md`.
- Agent role process: `Agenters/README.md` and `Agenters/<Role>/README.md`.
- Submit/checklist process: `Agenters/Submit_Captain/README.md` and `submissions/README.md`.
- Tool catalog: `tools/README.md`.
- Research and external sources: `Research_Archive/00_INDEX.md`, `Research_Archive/_RESEARCHED_TOPICS.md`, and `Competitor_Notebooks/competitor_notebooks_2026-05/00_INDEX.md`.
- Durable lessons: `Logs/lessons.md`.
- Session log format and storage: `Logs/README.md`.
- Historical retired `.ai_instructions` content, if kept: `Logs/_archive/ai_instructions_2026-06-12/`.

## Safe Execution Plan

1. Create an archive folder only if the owner wants historical retention:
   `Logs/_archive/ai_instructions_2026-06-12/`.

2. Extract live content into destination docs:
   - Copy the `04_DATA.md` pre-submit checklist into `submissions/README.md` or replace it with the current Submit Captain flow if obsolete.
   - Copy any still-current `agent_feedback_rules.md` anti-re-bloat rules into the owning role docs: Submit Captain rules to `Agenters/Submit_Captain/README.md`, Librarian/library rules to `Agenters/Librarian/README.md` and `Onnx_Advanced/.update/README.md`, general role boundaries to `Agenters/README.md`.
   - Move research workflow guidance from `Research.md` / handoff templates into `Research_Archive/00_INDEX.md` only if agents still use that workflow.
   - Move external source/catalog facts from `06_COMMUNITY.md` into `Research_Archive/00_INDEX.md` or the competitor notebooks index.
   - Promote any live backlog items from `05_BACKLOG.md` to `The_400_Tasks/current_strategy.md` or `Agenters/_proposals/`; archive or drop closed/stale backlog.

3. Update live docs:
   - Replace `README.md` read order with the current root/Agenters read order.
   - Replace `WHERE_DATA_LIVES.md` `.ai_instructions/03_PROJECT.md` pointer with `README.md`, `CURRENT_STATE.md`, and `The_400_Tasks/current_strategy.md`.
   - Replace `CLAUDE.md` and `Agenters/Librarian/README.md` citations with the new canonical destination.
   - Replace `Logs/README.md` session-format pointer with an inline short format or an `Agenters/`/`Logs/` doc.
   - Replace `submissions/README.md` pre-submit pointer with inline checklist or Submit Captain reference.
   - Replace `Onnx_Advanced/core.md` worker/process references with `Onnx_Advanced/WORKER_PROTOCOL.md` and `Agenters/` role docs.

4. Update live code:
   - In `tools/repo_lint.py`, remove `.ai_instructions` from active doc surfaces and remove the `.ai_instructions/playbooks/WorkerAgent_Setup.md` allowlist entry.
   - In `tools/state.py`, replace the printed `.ai_instructions/03_PROJECT.md` next-read pointer.
   - In `tools/refactor_paths.py`, remove `.ai_instructions` from `SEARCH_DIRS`.
   - In `Onnx_Advanced/.update/tools/lib_lint.py`, remove the `.ai_instructions/playbooks` fallback once all references are rewritten.

5. Handle references:
   - Run `rg -n "\.ai_instructions|00_START_HERE_README_AI|01_MASTER\.md|02_GIT\.md|03_PROJECT\.md|04_DATA\.md|05_BACKLOG\.md|06_COMMUNITY\.md|agent_feedback_rules\.md|SpreadSweepWorker\.md|NewSession\.md" --hidden -g !.git -g !.venv -g !data`.
   - Fix active references first.
   - Decide whether historical references remain acceptable as plain historical text. If not, point them at the archive folder or rewrite to "retired `.ai_instructions` archive".

6. Validate before deletion:
   - `python tools/repo_lint.py`
   - `python Onnx_Advanced/.update/tools/lib_lint.py --only dead-md-refs`
   - `rg -n "\.ai_instructions" --hidden -g !.git -g !.venv -g !data`
   - `git status --short -- .ai_instructions README.md WHERE_DATA_LIVES.md CLAUDE.md Agenters Logs submissions Onnx_Advanced tools The_400_Tasks Research_Archive Competitor_Notebooks`

7. Delete `.ai_instructions/` only after the live references are clean and the owner confirms whether historical content should be archived or discarded.

## Risk Notes

- Do not delete `.ai_instructions/` first. `tools/repo_lint.py`, `tools/state.py`, docs, and role files currently assume it exists.
- Do not blindly move all files into a new folder and call it done. Most of the folder is stale workflow history; moving it wholesale keeps the context bloat.
- Be careful with references like bare `03_PROJECT.md` in historical notes. Some are historical provenance, not live instructions.
- This repo already has newer replacements for much of the folder: `AGENT_TASK_START.md`, `Agenters/`, `CURRENT_STATE.md`, `WHERE_DATA_LIVES.md`, `tools/README.md`, and `Onnx_Advanced/WORKER_PROTOCOL.md`.

## Suggested First Target

Start with the active reference set:

1. `README.md`
2. `WHERE_DATA_LIVES.md`
3. `CLAUDE.md`
4. `Agenters/Librarian/README.md`
5. `Logs/README.md`
6. `submissions/README.md`
7. `Onnx_Advanced/core.md`
8. `tools/repo_lint.py`
9. `tools/state.py`
10. `tools/refactor_paths.py`
11. `Onnx_Advanced/.update/tools/lib_lint.py`

Once those are clean, the remaining work is mostly historical reference policy and optional archive retention.
