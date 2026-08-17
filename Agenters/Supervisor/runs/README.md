# Supervisor runs

Supervisor instances may keep local state here when the owner assigns a task wave.

Suggested instance folder:

```text
Agenters/Supervisor/runs/<supervisor_id>/
```

Suggested files:

- assignment note - exact owner assignment, prompt folder, task IDs, and worker count
- orchestrator note - Orchestrator ID, task allowlist, run dir, and status
- status note - compact owner-facing rollups
- process notes - observations that may become `_proposals` entries

Do not use this folder as a shared queue for workers. Workers and the Orchestrator have their own role-local state.
