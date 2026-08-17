# task_outcome_registry.json schema additions — v1.1 (additive, backward-compatible)

Submit-gate redesign adds three per-task fields to entries in `.hermes/board/task_outcome_registry.json`. These extend the existing `neurogolf.task_outcome_registry.v1` schema **additively**: any reader that ignores unknown keys continues to work. No migration required — fields default to `0` / `[]` when missing.

## New per-task fields

| Field                    | Type            | Default | Purpose                                                                  |
|--------------------------|-----------------|---------|--------------------------------------------------------------------------|
| `attempts_this_pass`     | int             | `0`     | Count of worker handoffs submit_gate has processed for this task in the current autoforeman pass. Reset to `0` when `pass_count` in `.hermes/board/state.json` increments. |
| `attempted_families`     | list of strings | `[]`    | Algorithm families tried this pass (normalized via algorithm_family_directive). Cleared when `pass_count` increments. Used to build the "don't repeat these families" retry directive. |
| `last_attempt_pass`      | int             | `0`     | The `pass_count` value at the most recent attempt. submit_gate / autoforeman compare against current `pass_count` to detect a new pass and reset the counters. |

## Reading rule (for submit_gate.py and autoforeman)

```python
def _read_pass_state(registry, task_id, current_pass):
    entry = registry.get("tasks", {}).get(task_id, {})
    last_pass = entry.get("last_attempt_pass", 0)
    if last_pass != current_pass:
        # New pass — counters are stale, treat as fresh
        return {"attempts_this_pass": 0, "attempted_families": []}
    return {
        "attempts_this_pass": entry.get("attempts_this_pass", 0),
        "attempted_families": entry.get("attempted_families", []),
    }
```

This means a missing entry, a missing field, or an entry from a prior pass all return the same "fresh" state — no special-casing needed at call sites.

## Writing rule

```python
def _bump_pass_state(registry, task_id, current_pass, family):
    entry = registry["tasks"].setdefault(task_id, {})
    if entry.get("last_attempt_pass", 0) != current_pass:
        entry["attempts_this_pass"] = 0
        entry["attempted_families"] = []
    entry["attempts_this_pass"] += 1
    entry["last_attempt_pass"] = current_pass
    if family and family not in entry["attempted_families"]:
        entry["attempted_families"].append(family)
```

Both helpers live in `tools/task_outcome_registry_io.py` (created in the next subtask alongside `algorithm_family_directive.py`).

## Existing fields (unchanged)

`status`, `current_cost`, `current_src`, `best_known_status`, `summary`, `known_attempts` (the cumulative history list) — all untouched. submit_gate writes a new `known_attempts` entry only when an attempt actually executed (mechanical pass + submit OR retry that consumed an attempt); the new short-cycle fields above are for *this-pass* state.
