# submissions/

> **This is a real in-repo directory and the canonical live submission home.** Do not replace it with the stale `.notongit` snapshot. See `../WHERE_DATA_LIVES.md` for the verified filesystem layout.

## What's tracked in git

- `submission.zip` — live Kaggle upload payload.
- `current_best_manifest.json` — manifest for the current best package.
- `README.md` (this file).

## What's gitignored

- `task*.onnx` — per-task ONNX builds output by `build_submission.py`.
- `<fruit_or_topic>_<date>/` — bundled per-submission artifact dirs (e.g. `afr1ste_5377_grape/`, `_handoff_alpha_*`).
- `_*` scratch dirs from autonomous mining sessions (legacy — new mining scratch goes to `../Logs/scratch/`, not here).

## Don't do this

- ❌ Drop ad-hoc `_my_experiment/` dirs here. Mining/probe scratch belongs in `../Logs/scratch/<topic>_<date>/`.
- ❌ Rebuild `task*.onnx` if they already exist — `python build_submission.py <task>` is idempotent for the same input.
- ❌ Commit individual `*.onnx` files. The global `.gitignore` `*.onnx` rule blocks them anyway, but don't try to force them through.

## Pre-submit gauntlet

Before any `kaggle competitions submit`, run the pre-submit checklist in `../.ai_instructions/04_DATA.md` "Recipe: pre-submit checklist". The Kaggle CLI requires the literal filename `submission.zip`:

```bash
cp submissions/<fruit_bundle>/submission.zip submissions/submission.zip
kaggle competitions submit -c neurogolf-2026 -f submissions/submission.zip -m "fruit_name #N"
```

After a public-positive submission completes, promote the exact package through
`tools/neurogolf_sync_current_best.py` or let `tools/neurogolf_auto_sync_current_best.py`
do it from the submit queue. That promotion path updates the ledger, current-best
manifest, `CURRENT_STATE.md` inputs, `The_400_Tasks/task_state.csv`,
`NeuroGolf_Task_Status.xlsx`, and the dated latest-package cost cache. Do not
hand-edit the task-state matrix except to repair old history.
