# `.notongit` Rehome Plan

Date: 2026-06-12

Intent: move the NeuroGolf local-only payload out of the sibling workspace and into the repo-local ignored `.notongit` folder, without moving unrelated project data.

> **EXECUTED 2026-08-17.** The move below was carried out during the pre-publication pass.
> `<NOGIT_ROOT>` → `<NOGIT_ROOT>`,
> verified identical at 101,995 files / 10,076 dirs / 3,439,152,379 bytes. The `data` and
> `.venv` junctions were rebuilt against the new in-repo target and confirmed working
> (venv python resolves, 402 task JSONs reachable). The sibling `Py_cx\.notongit\` now holds
> only `Arc2_nogit`, which belongs to another project and was left alone.
>
> Deviations from the plan below: **no temporary compatibility junction was created** at the
> old sibling path — the point of the move was to retire that location, so the stale absolute
> paths were fixed instead (23 files; historical records left as-is). `Team_Best_Onnx` was not
> re-linked: the OnnxlyFans repo it pointed at is no longer on C:. See `WHERE_DATA_LIVES.md`
> for the current layout.

## Current State

- Repo: `<REPO_ROOT>`
- Existing repo-local ignored folder: `<REPO_ROOT>\.notongit`
- Current repo-local `.notongit` contents are tiny:
  - `.notongit\_t276_inspect\t276_dense_msparse.onnx`
  - `.notongit\_t276_inspect\t276_sparse.onnx`
  - `.notongit\_t276_inspect\task276.onnx`
- Sibling local-only store: `<WORKSPACE_ROOT>\.notongit`
- NeuroGolf payload to move: `<NOGIT_ROOT>`
- Do not move these sibling entries; they belong to other workspaces:
  - `<WORKSPACE_ROOT>\.notongit\.venv-ml`
  - `<WORKSPACE_ROOT>\.notongit\Arc2_nogit`
  - `<WORKSPACE_ROOT>\.notongit\Orbit Wars_nogit`

The root repo links currently point outside the repo:

- `data` -> `<NOGIT_ROOT>\data`
- `.venv` -> `<NOGIT_ROOT>\.venv`

The repo already ignores `.notongit/`, so the target location is suitable for local-only payloads.

## Payload Size

Approximate size of `NeuroGolf_nogit`: 3.3 GB.

Largest children observed:

- `worktrees`: about 900 MB
- `.venv`: about 682 MB
- `data`: about 623 MB
- `submissions`: about 552 MB
- `orchestrator`: about 500 MB
- `BatchPilotArchive`: about 22 MB

This is large enough to avoid copy/delete workflows unless a copy is explicitly desired. Because source and destination are both on `C:`, a directory move should be a metadata move.

## Proposed Target Layout

Move:

```text
<NOGIT_ROOT>
```

to:

```text
<NOGIT_ROOT>
```

Then retarget root junctions:

```text
<REPO_ROOT>\data
  -> <NOGIT_ROOT>\data

<REPO_ROOT>\.venv
  -> <NOGIT_ROOT>\.venv
```

Recommended temporary compatibility link:

```text
<NOGIT_ROOT>
  -> <NOGIT_ROOT>
```

That compatibility link gives existing hardcoded paths time to be cleaned up without breaking active scripts immediately.

## References To Update

Several active files still point at the sibling `.notongit\NeuroGolf_nogit` layout.

Docs and config:

- `.env.example`
- `.gitignore` comments
- `WHERE_DATA_LIVES.md`
- `submissions/README.md`
- `Onnx_Advanced/core.md`
- `tools/ng_harness/PLAN.md`

Active tools with hardcoded or default paths:

- `tools/verify_against_live.py`
- `tools/scan_scratch_onnxes.py`
- `tools/promote_task_work.py`
- `tools/onnx_opt_with_fold.py`
- `tools/onnx_opt_safe.py`
- `tools/neurogolf_submit_policy.py`
- `tools/catalog_organic_prototypes.py`
- `tools/enrich_task_state.py`
- `tools/dedup_onnx_inits.py`

There are many historical references under `The_400_Tasks`, `Logs`, and prototype notes. Do not bulk edit those unless they are used by current tooling. They are better handled with the temporary compatibility junction or left as historical records.

## Execution Plan

1. Stop running workers, notebooks, dev servers, and Python processes that may have files open in `.venv`, `data`, `worktrees`, or `submissions`.
2. Preflight:
   - Confirm old source exists.
   - Confirm new destination does not already exist.
   - Confirm `.notongit/` is ignored by Git.
   - Record file counts and sizes for source.
   - Record current `data` and `.venv` junction targets.
3. Move only the NeuroGolf payload:

   ```powershell
   Move-Item -LiteralPath "<NOGIT_ROOT>" -Destination "<NOGIT_ROOT>"
   ```

4. Recreate root junctions to the new in-repo local-only target.
   - Remove the old junction links with `Remove-Item -LiteralPath <link> -Force`.
   - Do not use `-Recurse` when removing junction links.
   - Create replacement junctions with `New-Item -ItemType Junction`.
5. Create the temporary compatibility junction at the old sibling path if hardcoded references have not yet been updated.
6. Update active docs and tools to prefer a repo-relative `.notongit\NeuroGolf_nogit` base, ideally through a shared path helper or environment variable.
7. Remove the compatibility junction only after active references have been updated and validated.

## Validation

Run after the move:

```powershell
Get-ChildItem -Force "<REPO_ROOT>" | Where-Object { $_.LinkType }
git status --ignored --short -- .notongit .venv data
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -c "import onnx; print(onnx.__version__)"
python tools/state.py --terse
```

If the data freshness helper exists and is still in use:

```powershell
python tools/agent_helpers/check_data_freshness.py
```

Also spot-check any workflow that currently reads from `data`, writes local scratch outputs, or stages submissions.

## Rollback

If the move has not happened yet, rollback means doing nothing.

If the move happened and something breaks:

1. Stop processes using the moved payload.
2. Recreate `data` and `.venv` junctions to whichever payload location is valid.
3. Prefer keeping the moved payload in the repo-local `.notongit` and adding the compatibility junction first.
4. Move the payload back to the sibling `.notongit` only if the compatibility junction cannot solve the breakage.

