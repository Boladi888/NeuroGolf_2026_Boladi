# Review Checklist Before Implementation

## Scope

- Confirm this should live under `tools/` plus `Handoff/scripts/`, not as a new hidden service.
- Confirm v1 should be read-only and visual only.
- Confirm no auto-kill, no queue mutation, no cap changes, and no `--no-throttle` behavior.

## Existing Tool Reuse

- Review `tools/ng_gate_status.py`; decide whether to extend it or create `tools/ng_gate_queue_snapshot.py`.
- If extending, keep the existing simple "active lease" behavior stable for reviewers who already use it.
- Avoid `throttle.status()` for the dashboard core if it can reclaim/delete locks indirectly. Prefer direct lockfile reads.

## Process Tree Accuracy

- Test grouping on these shapes:
  - `python tools/ng_gate.py ...`
  - `$env:NEUROGOLF_PYTHON tools/ng_gate.py ...`
  - PowerShell wrapper -> venv launcher -> Python child
  - `--new-window`
  - parent process exits while child remains alive
- Ensure duplicate command lines in a tree count as one logical gate run.

## Queue Age Semantics

- For read-only v1, label wait age as "process age" or "approx wait".
- Do not claim exact queue age until explicit wait records exist.
- If wait records are added later, verify they are created before the acquire loop and removed after acquire/exit.

## Orphan Semantics

- Define "orphan suspect" as a warning only.
- Require at least two or three evidence points before showing a high severity orphan:
  - parent chain broken;
  - no runner state match;
  - terminal runner state while process remains alive;
  - lock age or process age above threshold;
  - low CPU delta over repeated polls.
- Never kill from v1.

## Visual Launcher

- Use VizTerms visible-terminal rules:
  - visible title;
  - clear banner;
  - close window stops watcher only;
  - watcher does not own or kill gate processes;
  - logs/snapshots optional but visible.
- Prefer no extra dependencies for v1. A plain PowerShell or standard-library Python loop is enough.
- If a browser dashboard is later desired, launch the server from a visible terminal and bind to `127.0.0.1`.

## Validation Plan

Non-invasive tests:

1. Run snapshot while no full gate is waiting.
2. Run snapshot while one full gate is active.
3. Run snapshot while one full gate is active and one full gate is waiting.
4. Compare active holders with `python tools/ng_gate_status.py --json`.
5. Compare task labels with `Handoff/private/runner_state_*.json`.
6. Compare open backlog with `Handoff/private/review_rescue_queue.jsonl`.
7. Confirm the watcher can be closed without touching gate processes.

Implementation tests that require deliberate setup:

- Start a tiny mock process command line containing `tools/ng_gate.py` but not actually running a gate, to test grouping safely.
- Or add a fixture-based parser test that does not start any live `ng_gate`.
- Do not start a real full `ng_gate --full` just to test the UI while the machine is busy.

## Open Questions

- Should the watch include both Codex2 and Claude lanes by default, or all detected runner states?
- Should open rescue rows be grouped/deduplicated by latest row per task?
- What age threshold should color a heavy gate as suspicious: 30 min, 60 min, or mode/task dependent?
- Should `ng_gate.py --new-window` print a wait status line while blocked, or should all visibility live outside it?
- Should future wait-record files live in `%TEMP%\ng_throttle` next to locks, or under `Handoff/private/gate_queue/` for easier repo-local inspection?

