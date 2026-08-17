# Orchestrator run folders

Each concurrent Orchestrator instance gets a private run folder:

```text
Agenters/Orchestrator/runs/<orchestrator_id>/
```

Write ownership:

- The matching Orchestrator instance may write its own run folder.
- Sibling Orchestrators may read this folder to detect task/reviewer collisions.
- No Orchestrator may write a sibling run folder unless the owner explicitly assigns cleanup or recovery.

Minimum files per run:

- `task_claims.csv`
- `reviewer_leases.csv`
- `final_report.csv`

Use the `_template/` CSV headers when starting a new run.

`final_report.csv` is the terminal rollup. Keep one row per owner-assigned task and update it at the end of the wave
or when a task becomes terminal. Schema:

```text
task,status,worker_slot,reviewer_slot,captain_status,staged_path,sha256,static_cost,profile_cost,expected_lb_delta,lesson_packet,blocker
```

Status values should stay simple: `staged`, `captain_packet`, `drop`, `blocked`, or `no_candidate`. For a
floor-confirmed no-ship (nothing shippable below team-best), prefer `drop` or `no_candidate`. `run_audit.py` also
accepts the legacy synonyms `wall`/`floor` as terminal DROP, but emit the schema words going forward.
`captain_status` should say whether the packet is `held`, `probing`, `probe_ok`, `staged`, `rejected`,
`external_owner`, or `not_sent`.

`held` is not terminal for a positive expected delta. A positive candidate may close only when it is staged/probe-ok,
rejected/dropped, actively probing, or explicitly owned by another lane/agent with the next action recorded.

## Setup & status tooling

Three stdlib-only helpers live in `Agenters/Orchestrator/` (run from any CWD; paths derive from `__file__`):

- **`prep_run.py`** — deterministic Part-A scaffolder. One command parses task IDs from the prompt folder(s), runs
  the collision scan, picks a unique run id (`--bump` on collision), copies the ledgers, snapshots the manifest to
  `team_best.csv`, builds the EV table, and renders the launch packet + lane-correct launch scripts from the
  `_template/*.template` files. Lane shape is enforced (Codex omits `-m` + uses danger-full-access; Claude pins
  `--model` + forces subscription auth — both validated before it reports success). It removes the mechanical
  setup, not the judgement — REVIEW the generated packet (EV table / model policy / probe boundary) before launch.
  `python prep_run.py --run-id <id> --lane Claude|Codex --prompt-folder "<path>" [--prompt-folder ...] [--exclude task042] [--bump] [--dry-run]`

- **`runs_status.py`** — liveness + collision signal across ALL run folders, so the active-run check is one command
  instead of a ~30-folder grep + mtime forensics. A run is ACTIVE if it has no `RUN_CLOSED.md` and was touched
  within the fresh window (default 2h), else STALE (audit it, then stamp `RUN_CLOSED.md`) or CLOSED.
  `runs_status.py active` (what's running now) | `list` (full table) | `collisions --tasks task...,task...`

- **`run_audit.py`** — single-run pre-closure consistency gate (task_claims ↔ final_report agreement, no unowned
  positive delta, staged-row provenance). Run it before writing `RUN_CLOSED.md`. Its terminal vocabulary is shared
  with `runs_status.py` (imported, single-sourced).

`_template/` holds the 3 ledger CSVs plus `launch_packet.template.md`, `launch.sh.template` (Claude),
`launch.ps1.template` + `start_orchestrator.ps1.template` (Codex) — the placeholder sources `prep_run.py` fills.
