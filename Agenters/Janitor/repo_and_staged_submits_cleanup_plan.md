# `repo` And `staged_submits` Cleanup Plan

Date: 2026-06-12

Intent: decide whether the root `repo` folder can be removed and how to retire or relocate `staged_submits` into task-local homes.

No files have been moved or deleted by this plan.

## Short Answer

`repo` can probably be deleted. It is an ignored nested snapshot containing only a stale `submissions` copy, and the package bytes are already preserved elsewhere.

`staged_submits` should not be deleted or moved blindly. It is untracked staging data, but it was written as recently as 2026-06-11 and is still named in active worker/captain docs and tools. If the repo policy changes so candidates belong under their task folders, retire `staged_submits` by migrating candidates and updating the staging protocol at the same time.

## `repo` Inventory

Path:

```text
<REPO_ROOT>\repo
```

Observed contents:

```text
repo\submissions\submission.zip
repo\submissions\current_best_manifest.json
```

Size:

- 2 files
- about 614 KB

Git state:

- `repo/` is ignored by `.gitignore`.
- No tracked files were found under `repo`.

Reference check:

- Only a historical cleanup survey reference was found: `Logs\worktree_cleanup_survey_2026-05-18.json`.
- No active tool or doc reference to `repo\submissions` was found.

Package check:

- `repo\submissions\submission.zip` SHA256:
  - `C1C42D55ECD0FAF5F285BA1AEEADB696CD2975EE04A2067D776B3DF80DE53719`
- It matches:
  - `submissions\_requested_submits\task383_rule_rewrite_v1_on_6455_23\submission.zip`
- It does not match the current root `submissions\submission.zip`, which has a newer timestamp and a different hash.

Conclusion: `repo` is not the canonical submission location. It looks like a stale nested repo/output snapshot from the June 2 task383 sync. Since the same package bytes exist under `submissions\_requested_submits`, `repo` has no unique artifact value.

## `repo` Cleanup Plan

1. Reconfirm before deletion:

   ```powershell
   rg --files repo
   git status --short -- repo
   Get-FileHash -Algorithm SHA256 -LiteralPath repo\submissions\submission.zip, submissions\_requested_submits\task383_rule_rewrite_v1_on_6455_23\submission.zip
   ```

2. Delete `repo`.
3. Keep the `.gitignore` `repo/` rule for now. It prevents another accidental nested checkout/output snapshot from appearing in Git.
4. Optional later cleanup: remove or rewrite the historical `Logs\worktree_cleanup_survey_2026-05-18.json` reference only if the log is being normalized.

## `staged_submits` Inventory

Path:

```text
<REPO_ROOT>\staged_submits
```

Observed shape:

- 209 candidate directories
- 1,177 files
- about 3.9 MB
- recent writes through 2026-06-11

File types:

- 536 `.onnx`
- 574 `.json`
- 44 `.py`
- 18 `.md`
- 4 `.pyc`
- 1 `.npz`

Git state:

- `staged_submits\README.md` is tracked.
- Candidate directories and queue data are untracked.
- `.onnx` files are ignored by the repo-wide ONNX ignore rule.

Current local contract from `staged_submits\README.md`:

- Workers stage candidate ONNX + manifest here.
- The captain ships via `tools\swap_and_submit.py`.
- Old staged dirs are cleaned by the captain after shipping.

The queue file says it is temporary:

- `staged_submits\SUBMIT_QUEUE.md` says to delete it once the queue is drained.
- It also records a Kaggle daily-limit block from 2026-06-08, so it may be stale operational state rather than permanent repo memory.

## Active `staged_submits` References

Current docs and tools still treat `staged_submits` as active:

- `AGENT_TASK_START.md`
- `CURRENT_STATE.md`
- `Onnx_Advanced\WORKER_PROTOCOL.md`
- `Onnx_Advanced\core.md`
- `Agenters\Submit_Captain\README.md`
- `Agenters\Submit_Captain\PACKET_SCHEMA.md`
- `Agenters\Prompter\README.md`
- `tools\swap_and_submit.py`
- `tools\agent_helpers\context_pack.py`
- `tools\peephole_rewrite.py`
- `tools\repo_lint.py`

Some tools still refer to the older `.hermes\board\staged_submits` location. Those should be cleaned separately from the top-level `staged_submits` migration.

## Recommended Final State

Move durable candidate artifacts into task-local homes, and reserve root-level staging only for transient handoff if it remains necessary.

Recommended task-local layout:

```text
The_400_Tasks\Task_by_Task\taskNNN\staged_candidates\<candidate_dir>\
  manifest.json
  taskNNN.onnx
  build_*.py
  notes.md
```

Reasons:

- It matches the owner intuition: candidate history belongs with the task.
- It keeps `prototypes\` reserved for actual builders/prototypes instead of mixing in captain handoff state.
- It makes cleanup by task possible.
- It avoids losing manifests and gate notes that explain why a candidate mattered.

Do not move everything into `prototypes\` unless that becomes the agreed task-folder convention. Many task `prototypes\` folders are owner-maintained and should not absorb staging noise without a clear rule.

## `staged_submits` Migration Plan

1. Decide whether any current queue is still live.
   - Ask Submit Captain or check the live submit state.
   - If `SUBMIT_QUEUE.md` is stale, mark it retired before moving candidates.
2. For each candidate dir named `taskNNN_*`:
   - Create `The_400_Tasks\Task_by_Task\taskNNN\staged_candidates\<candidate_dir>\`.
   - Move the candidate's manifest, ONNX, builders, and notes into that directory.
   - Preserve the original candidate directory name.
3. For malformed or non-task directories:
   - Put them in `staged_candidates\_unmatched\` under a Janitor report first.
   - Do not guess task ownership from comments alone.
4. Generate or update a per-task index:

   ```text
   The_400_Tasks\Task_by_Task\taskNNN\staged_candidates\README.md
   ```

   Include candidate dir, cost, visible/truth/A-B gate status, SHA, and whether it was shipped, dominated, risky, or abandoned.
5. Update active workflow docs:
   - `AGENT_TASK_START.md`
   - `CURRENT_STATE.md`
   - `Onnx_Advanced\WORKER_PROTOCOL.md`
   - `Onnx_Advanced\core.md`
   - `Agenters\Submit_Captain\README.md`
   - `Agenters\Submit_Captain\PACKET_SCHEMA.md`
   - `Agenters\Prompter\README.md`
6. Update active tools:
   - `tools\swap_and_submit.py`
   - `tools\agent_helpers\context_pack.py`
   - `tools\peephole_rewrite.py`
   - `tools\repo_lint.py`
7. Add an explicit policy:
   - root `staged_submits` is deleted, or
   - root `staged_submits` remains as an ignored transient inbox that must be empty after each captain sweep.
8. Delete or archive the root `staged_submits` payload only after the migration and protocol update are complete.

## Validation

Before migration:

```powershell
git status --short -- staged_submits
Get-ChildItem -Directory -Force staged_submits | Measure-Object
rg -n "staged_submits" AGENT_TASK_START.md CURRENT_STATE.md Onnx_Advanced Agenters tools
```

After migration:

```powershell
rg -n "staged_submits" AGENT_TASK_START.md CURRENT_STATE.md Onnx_Advanced Agenters tools
python tools/repo_lint.py --only prompts
python tools/agent_helpers/context_pack.py --help
git status --short -- staged_submits The_400_Tasks\Task_by_Task
```

Acceptable remaining references after retirement should be:

- historical task notes,
- archived logs,
- Janitor cleanup reports,
- or a deliberate policy note saying the old root inbox was retired.

## Risk Notes

- Do not delete `staged_submits` while workers or Submit Captain still use it.
- Do not lose ONNX files just because they are ignored by Git.
- Do not flatten candidate directories; their names carry task, technique, cost, and sometimes risk/gate state.
- Do not treat old `.hermes\board\staged_submits` references as equivalent to current top-level `staged_submits`. They need separate retirement handling.

