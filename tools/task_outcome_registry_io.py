"""Atomic read/write helpers for `.hermes/board/task_outcome_registry.json`.

The submit-gate redesign adds three per-task fields (`attempts_this_pass`,
`attempted_families`, `last_attempt_pass`) that are short-cycle state — they
reset on autoforeman pass increment. The cumulative `known_attempts` list and
the per-task summary fields (`status`, `summary`, `best_known_status`, etc.)
are untouched.

All writes are atomic: write to `<path>.tmp`, fsync, then replace. A SIGKILL
mid-write must never corrupt the registry. See the same pattern in
``tools/auto_triage_blocked.py::apply_bank`` (the candidate_registry write).

This module intentionally has no kanban or subprocess dependencies — it's pure
JSON IO so the test suite can exercise it without mocking.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    """Load the registry, returning an empty skeleton if missing/corrupt.

    Missing file or unparseable JSON both return the same v1 skeleton — this
    keeps callers' logic simple (always work against a dict, never check for
    None). Corruption is loud-logged via FileNotFoundError chain on write.
    """
    if not path.exists():
        return _empty_skeleton()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_skeleton()
    if not isinstance(data, dict):
        return _empty_skeleton()
    data.setdefault("tasks", {})
    return data


def save_registry(path: Path, data: dict[str, Any]) -> None:
    """Atomic write: serialize → write tmp → fsync → replace.

    Raises any IO error encountered. Callers wrapping this in a rollback path
    should snapshot the prior file contents before invoking.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_pass_state(
    registry: dict[str, Any],
    task_id: str,
    current_pass: int,
) -> dict[str, Any]:
    """Return ``{"attempts_this_pass": int, "attempted_families": list}``.

    When the registry's ``last_attempt_pass`` for this task doesn't match
    ``current_pass``, the counters are stale and we report fresh-pass state
    (`{0, []}`) without mutating the registry. The eventual write via
    ``bump_pass_state`` is what resets the stored values.

    A missing task, a missing entry, or an entry without these fields all
    produce the same fresh state — callers don't need to special-case.
    """
    entry = registry.get("tasks", {}).get(task_id, {}) or {}
    last_pass = _coerce_int(entry.get("last_attempt_pass"), default=0)
    if last_pass != current_pass:
        return {"attempts_this_pass": 0, "attempted_families": []}
    return {
        "attempts_this_pass": _coerce_int(entry.get("attempts_this_pass"), default=0),
        "attempted_families": list(entry.get("attempted_families") or []),
    }


def bump_pass_state(
    registry: dict[str, Any],
    task_id: str,
    current_pass: int,
    family: str = "",
) -> dict[str, Any]:
    """Increment attempts_this_pass and append ``family`` if new.

    Mutates ``registry`` in place and returns the updated per-task entry.
    Resets the counters first if this is a new pass.

    Family de-duplication: an empty/whitespace ``family`` is recorded as
    ``"unspecified"`` so the blacklist still grows. Duplicates of an existing
    entry are silently skipped (case-sensitive — normalize upstream via
    ``algorithm_family_directive.normalize_family``).
    """
    tasks = registry.setdefault("tasks", {})
    entry = tasks.setdefault(task_id, {})
    if _coerce_int(entry.get("last_attempt_pass"), default=0) != current_pass:
        entry["attempts_this_pass"] = 0
        entry["attempted_families"] = []
    entry["attempts_this_pass"] = _coerce_int(entry.get("attempts_this_pass"), default=0) + 1
    entry["last_attempt_pass"] = current_pass
    families = entry.setdefault("attempted_families", [])
    family_to_record = family.strip() if family and family.strip() else "unspecified"
    if family_to_record not in families:
        families.append(family_to_record)
    return entry


def record_known_attempt(
    registry: dict[str, Any],
    task_id: str,
    *,
    candidate: str,
    result: str,
    evidence: str = "",
) -> dict[str, Any]:
    """Append a permanent record to the cumulative known_attempts list.

    Distinct from ``bump_pass_state`` which tracks short-cycle this-pass state.
    ``known_attempts`` is the long-lived history field that survives pass
    increments.
    """
    tasks = registry.setdefault("tasks", {})
    entry = tasks.setdefault(task_id, {})
    attempts = entry.setdefault("known_attempts", [])
    attempts.append({
        "candidate": candidate,
        "result": result,
        "evidence": evidence,
    })
    return entry


def _empty_skeleton() -> dict[str, Any]:
    return {
        "schema": "neurogolf.task_outcome_registry.v1",
        "tasks": {},
    }


def _coerce_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        # bool is an int subclass — explicit reject
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default
