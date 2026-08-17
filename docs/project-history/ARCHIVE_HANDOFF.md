# ARCHIVE HANDOFF — what's left for the owner

> 2026-08-17 update: **all three actions below are done.** The upstream lift (Action 1) was
> completed during the 2026-07-16 archive pass; the two live processes (Action 2) are no
> longer running; the move (Action 3) happened, and the checkout has since been restored
> from the NAS archive back to `<REPO_ROOT>`. The local-only
> payload now lives *inside* the repo at `.notongit\NeuroGolf_nogit\` (gitignored), so the
> junction table below is historical — only `.venv` and `data` are junctions now, both
> pointing in-repo. Current layout: `WHERE_DATA_LIVES.md`. This file is kept as the record
> of the cleanup, not as a to-do list.

> 2026-07-16 update: the project has been copied to the V: archive, and the former
> junction target `<NOGIT_ROOT>` has been
> materialized under this archive at `.notongit\NeuroGolf_nogit`. Read
> `ARCHIVE_COMPLETED_2026-07-16.md` for the current archive status before acting on
> the older action list below.

Cleanup completed 2026-07-15.

## State

| | Before (`0f15a1ac`) | After |
|---|---|---|
| Tracked files | 16,770 | 2,758 |
| `python tools/repo_lint.py` | 32 errors / 869 warnings | 0 errors |
| `python Onnx_Advanced/.update/tools/lib_lint.py` | 0 errors / 573 warnings | 0 errors / 99 warnings |
| `.venv/Scripts/python.exe -m pytest tests/ -q` | 1244 passed / 9 failed | 1253 passed / 0 failed |
| `.git/` | 414 MB | 125 MB |

Six agent lanes (A–F) merged, then one integration commit fixed the cross-lane gaps no
worktree agent could see. All agent branches, worktrees, snapshot refs and stashes are
gone; `master` is the only branch. Nothing was pushed — `origin/master` is untouched and
`master` is 29 commits ahead of it.

Per-lane reports: `Logs/_archive/cleanup_2026-07-15/`.

## ACTION 1 — copy 5 files upstream (do this BEFORE archiving)

`.ai_instructions/playbooks/agent_feedback_rules.md` **exists nowhere else on disk.** Its own
header says it was moved out of `.claude/memory/` precisely so it wouldn't be trapped in one
agent's private storage. Archiving this repo without lifting it would repeat that failure.

Source: `_cleanup/UPSTREAM_LIFT/` — Destination: `<ARCHIVE_ROOT>\AI_Instructions\`

All five are **adds** upstream (verified by SHA — nothing is overwritten):

| From `_cleanup/UPSTREAM_LIFT/` | To `Archive\AI_Instructions\` |
|---|---|
| `04_DATA.md` | `04_DATA.md` |
| `05_BACKLOG.md` | `05_BACKLOG.md` |
| `playbooks/agent_feedback_rules.md` | `playbooks/agent_feedback_rules.md` |
| `playbooks/SpreadSweepWorker.md` | `playbooks/SpreadSweepWorker.md` |
| `playbooks/WorkerAgent_Setup.md` | `playbooks/WorkerAgent_Setup.md` |

That upstream repo is itself disorganised (four parallel `Current/.ai_instructions{,1,2,3}/`
copies plus a template). The root layer is the canonical/newest one — that's where these go.
Worth its own cleanup pass someday; not this project's problem.

Then delete `_cleanup/` and this file.

## ACTION 2 — stop the two live processes

Two processes were still running against this repo during cleanup:

- `hermes_cli.main gateway` (Python 3.11) — writes `.hermes/board/submit_credit_state.json`
- `tools/ng_gate_queue_window...` (Python 3.14) — the ng_gate queue watcher

**Hermes is NOT retired**, despite `CURRENT_STATE.md` claiming it was archived to
`_hermes_archive/` on 2026-06-05 (that folder does not exist; `.hermes/` does). 52 of the
223 files in `tools/` reference it. Nothing hermes-related was deleted for this reason.
Stop both before moving the folder, or the move will fail on locked files.

## ACTION 3 — the move

Move `<REPO_ROOT>` to your archive location.

**Junctions — read this before moving.** Four directories are Windows junctions pointing
OUTSIDE this repo. `Remove-Item -Recurse` on any of them destroys the TARGET, not the link:

| Junction | Target |
|---|---|
| `.venv/` | `..\.notongit\NeuroGolf_nogit\.venv` |
| `data/` | `..\.notongit\NeuroGolf_nogit\data` (670 MB Kaggle source) |
| `Team_Best_Onnx/` | `..\K-NeuroGolf-OnnxlyFans\...` — **a different repo** |
| `Handoff\skills\handoff-workflows\` | `..\Boladi_Skillz\HandOffz\...` — **your skills library** |

They contain no repo data — they only point at it. Their targets are absolute paths, so the
junctions keep resolving after a move; they'll just reach back to the original locations.
If you want a self-contained archive, delete the LINKS ONLY:

```powershell
# removes the link, NOT the target — verify LinkType first, never use -Recurse here
foreach ($j in '.venv','data','Team_Best_Onnx','Handoff\skills\handoff-workflows') {
  $p = Join-Path '<REPO_ROOT>' $j
  $i = Get-Item -LiteralPath $p -Force
  if ($i.LinkType) { cmd /c rmdir "$p" }   # rmdir on a junction removes only the link
}
```

`submissions/submission.zip` is the live Kaggle payload and the real artifact of this
project — it and the score-labelled `_canonical_*_backup.zip` files are kept deliberately.

## What was deliberately NOT done

- **`.hermes/` and the 52 hermes-coupled tools** — kept; hermes is live (see Action 2).
- **`ml/ngs/`** — kept, untouched. It is the freshest code in the repo (last commit
  2026-07-15, the day of this cleanup): a program-synthesis engine that solved 23 tasks
  byte-exact on 512 held-out ARC-GEN pairs. Not classified as dead without your say-so.
- **`Handoff/private/`** (144 files, 14 MB) and `.env.handoff` — untouched; they hold live
  credentials/API captures. Do not archive secrets without a look.
- **~1.2 GB of gitignored regenerable files** — see the note in the cleanup reports. Deleting
  them is unrecoverable (never in git), so it needs an explicit owner decision.

## Known truths worth keeping

- `build_submission.py` moved to `tools/build_submission.py`. It is NOT a retired dispatcher —
  it is the 311-function ARC detector library that `tests/test_detect.py` exercises. The
  README's "--all/--hybrid are disabled" warning refers to the wholesale-rebuild entry point
  only. A cleanup agent misread that and deleted it; it's restored and test-verified.
- The test suite's documented runner **under-reports**: `unittest discover` collects 623 and
  reports OK, because 30 of 37 modules are pytest-style and collect zero. `pytest` collects
  1253. A `pytest.ini` now pins this. The same pattern bit `Logs/lint_last.md`, whose
  committed copy claimed "0 errors / 0 warnings" because it was generated with one check on.
- `Competitor_Notebooks/` held 14 third-party Kaggle notebooks with no license metadata under
  an Apache-2.0 NOTICE claiming the whole tree. The notebooks are gone; the analytical
  `00_INDEX.md` survives. `NOTICE` now carries onnx-tool's MIT attribution for the vendored
  reference under `tools/onnx_tool_1.0.1_reference/`.
