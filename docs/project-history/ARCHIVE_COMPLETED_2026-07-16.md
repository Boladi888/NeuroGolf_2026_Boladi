# NeuroGolf archive completion - 2026-07-16

This archive copy is the durable home for the personal NeuroGolf project after the competition.

> **SUPERSEDED 2026-08-17 — kept as the record of the archive pass.** Two things below are no
> longer current. (1) The archive paths were reorganised on the NAS; the archive now lives at
> `V:\Archive_CX_202607\Tech\Kaggle\K-NeuroGolf` (team copy: `...\K-NeuroGolf-OnnxlyFans`).
> (2) The C: checkout has been restored from that archive and is live again at
> `<REPO_ROOT>`, with the local-only payload moved *inside* the
> repo at `.notongit\NeuroGolf_nogit\` (gitignored) rather than in the sibling folder. Junction
> targets recorded below describe the 2026-07-16 state, not today's. Current layout:
> `WHERE_DATA_LIVES.md`.

## Archive roots

- Personal project archive: `V:\Archive_CX_202406_a\Docs\Py_cx\K-NeuroGolf`
- Team project archive: `V:\Archive_CX_202406_a\Docs\Py_cx\K-NeuroGolf-OnnxlyFans`

## Junction fix

The original C: checkout did not move cleanly because the remaining root entries were Windows
directory junctions:

- `.venv` -> `<NOGIT_ROOT>\.venv`
- `data` -> `<NOGIT_ROOT>\data`
- `Team_Best_Onnx` -> `<TEAM_REPO>\Team_Submission\Backend\Team_Best_Onnx`

The V: archive intentionally does not recreate those root junctions.

## Materialized off-repo payload

The former off-repo target has been copied into the V: archive at:

`V:\Archive_CX_202406_a\Docs\Py_cx\K-NeuroGolf\.notongit\NeuroGolf_nogit`

Verification from `robocopy /L /E /XJ /BYTES` after copy:

- Source: 10,077 dirs, 101,995 files, 3,439,152,379 bytes
- Destination: 10,077 dirs, 101,995 files, 3,439,152,379 bytes
- Dry-run source-to-destination: 0 files left to copy, 0 failed
- Reparse-point scan under the copied target: no reparse points found

That means the C: `.notongit\NeuroGolf_nogit` payload is now preserved on V:.

## Team archive boundary

The team best ONNX set is archived in the sibling team folder:

`V:\Archive_CX_202406_a\Docs\Py_cx\K-NeuroGolf-OnnxlyFans\Team_Submission\Backend\Team_Best_Onnx`

Do not push or reconnect the team repo remote from this archive pass. The team archive currently has
no configured Git remote.

## Upstream lift

The handoff's five AI instruction files were copied from `_cleanup\UPSTREAM_LIFT` into:

`<ARCHIVE_ROOT>\AI_Instructions`

Copied files:

- `04_DATA.md`
- `05_BACKLOG.md`
- `playbooks\agent_feedback_rules.md`
- `playbooks\SpreadSweepWorker.md`
- `playbooks\WorkerAgent_Setup.md`

## C: cleanup note

After the archive was verified, `<REPO_ROOT>` contained only the three
junction stubs listed above plus a zero-byte `NUL` file. Codex Desktop blocked deletion of even
those verified stubs, so this archive note records the exact state instead of bypassing the safety
layer.

When removing the leftover C: folder manually, verify `LinkType : Junction` first and remove the
links without recursion. Do not use a recursive delete on those junction paths.
