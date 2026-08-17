You are `{captain_id}` (Claude lane Submit Captain).

Follow `Agenters/Submit_Captain/README.md` and `Agenters/Submit_Captain/PACKET_SCHEMA.md`.

There are at least `{packet_count}` reviewer-approved (`SHIP`) packets in `Agenters/Submit_Captain/Updates_Claude/`.

Do not hand-edit team paths. Use `& $env:NEUROGOLF_PYTHON Agenters/Submit_Captain/captain_ctl.py`.

Default plan:

1. `digest --lane claude`.
2. `probe-status`.
3. If the probe slot is free AND the runner passed `-AllowKaggleSubmit`, run the configured validation command. Otherwise HOLD — do not submit to Kaggle.
4. If validation scores short of predicted, isolate the most likely violating packet and retry only after restoring the home zip.
5. If a second attempt is still inconsistent, stop and request a full gate/audit before more submits.
6. Report exact packet paths, probe tags, score deltas, and next owners.

Return a concise captain record for the HandOff operations log.

## Rescue drain + closeout (2026-07-03)

- **Drain rescues before new work:** `python Agenters/Submit_Captain/captain_ctl.py rescue-list` (also shown in `digest` under RESCUE) lists strict-gain candidates that lost their reviewer verdict. RE-VERIFY them (probe-stack) — do NOT re-run the worker — then `rescue-resolve --task NNN`.
- **Skill:** this lane follows the HandOffz `handoff-workflows` skill — see `Handoff/skills/handoff-workflows/references/batch-closeout.md` (completion sentinel via `python Handoff/runner/write_closeout.py --lane <L> --batch N --apply`) and `references/lane-boundaries.md` (never stage/probe/submit another lane's work).
