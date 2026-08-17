"""Tests for tools/task_outcome_registry_io.py.

Pure JSON IO module — no kanban / subprocess deps, so these tests run fast
and exercise every branch deterministically.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root on path (mirrors other tests in this suite).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.task_outcome_registry_io import (
    bump_pass_state,
    load_registry,
    read_pass_state,
    record_known_attempt,
    save_registry,
)


# ----- load_registry -------------------------------------------------------


def test_load_registry_returns_skeleton_when_file_missing(tmp_path):
    """A fresh install with no registry file yet must not crash submit_gate.
    Callers always get a dict with a 'tasks' key."""
    registry = load_registry(tmp_path / "missing.json")
    assert registry["schema"] == "neurogolf.task_outcome_registry.v1"
    assert registry["tasks"] == {}


def test_load_registry_returns_skeleton_on_corrupt_json(tmp_path):
    """A SIGKILL mid-write could leave a half-written file. Don't crash —
    return a fresh skeleton and let the next write rebuild."""
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    registry = load_registry(path)
    assert registry["tasks"] == {}


def test_load_registry_returns_skeleton_when_root_is_not_dict(tmp_path):
    """Defensive: someone hand-edited the registry into a list."""
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    registry = load_registry(path)
    assert registry["tasks"] == {}


def test_load_registry_preserves_existing_data(tmp_path):
    path = tmp_path / "good.json"
    payload = {
        "schema": "neurogolf.task_outcome_registry.v1",
        "tasks": {"task042": {"status": "ok", "current_cost": 12345}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = load_registry(path)
    assert registry["tasks"]["task042"]["status"] == "ok"
    assert registry["tasks"]["task042"]["current_cost"] == 12345


def test_load_registry_adds_missing_tasks_key(tmp_path):
    """An older schema dump that lacks the 'tasks' key still loads cleanly."""
    path = tmp_path / "no_tasks.json"
    path.write_text(json.dumps({"schema": "old"}), encoding="utf-8")
    registry = load_registry(path)
    assert registry["tasks"] == {}


# ----- save_registry -------------------------------------------------------


def test_save_registry_round_trips(tmp_path):
    path = tmp_path / "rt.json"
    save_registry(path, {"schema": "x", "tasks": {"task001": {"k": "v"}}})
    loaded = load_registry(path)
    assert loaded["tasks"]["task001"]["k"] == "v"


def test_save_registry_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "registry.json"
    save_registry(path, {"schema": "x", "tasks": {}})
    assert path.exists()


def test_save_registry_atomic_uses_tmp_then_rename(tmp_path):
    """Verify the write actually goes through a .tmp + replace path rather
    than truncating the destination directly. A SIGKILL between truncate and
    write would leak an empty file otherwise."""
    path = tmp_path / "atomic.json"
    save_registry(path, {"schema": "x", "tasks": {"task001": {"a": 1}}})

    seen_replace = {"count": 0, "from": None, "to": None}
    original_replace = os.replace

    def tracking_replace(src, dst):
        seen_replace["count"] += 1
        seen_replace["from"] = str(src)
        seen_replace["to"] = str(dst)
        return original_replace(src, dst)

    with patch("tools.task_outcome_registry_io.os.replace", side_effect=tracking_replace):
        save_registry(path, {"schema": "x", "tasks": {"task001": {"a": 2}}})

    assert seen_replace["count"] == 1
    assert seen_replace["from"].endswith(".tmp")
    assert seen_replace["to"] == str(path)
    assert load_registry(path)["tasks"]["task001"]["a"] == 2


def test_save_registry_preserves_unicode(tmp_path):
    """Worker summaries can contain em-dashes / curly quotes. The cp1252
    decode bug we already hit on Windows means we MUST write UTF-8 with no
    BOM, and we MUST preserve non-ASCII bytes verbatim."""
    path = tmp_path / "unicode.json"
    note = "rebuild — preserves the ‘where-select' family"
    save_registry(path, {"schema": "x", "tasks": {"task001": {"summary": note}}})
    loaded = load_registry(path)
    assert loaded["tasks"]["task001"]["summary"] == note


# ----- read_pass_state -----------------------------------------------------


def test_read_pass_state_fresh_for_missing_task():
    """Default fresh state for any task the registry has never seen."""
    registry = {"tasks": {}}
    state = read_pass_state(registry, "task100", current_pass=2)
    assert state == {"attempts_this_pass": 0, "attempted_families": []}


def test_read_pass_state_fresh_when_pass_advanced():
    """Counters from a prior pass MUST present as fresh — submit_gate should
    not think 3 attempts have already happened in the current pass when those
    attempts were actually from pass N-1."""
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 3,
                "attempted_families": ["where-select", "reduce-broadcast", "conv-stamp"],
                "last_attempt_pass": 1,
            }
        }
    }
    state = read_pass_state(registry, "task100", current_pass=2)
    assert state == {"attempts_this_pass": 0, "attempted_families": []}


def test_read_pass_state_returns_current_when_pass_matches():
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 2,
                "attempted_families": ["where-select", "reduce-broadcast"],
                "last_attempt_pass": 5,
            }
        }
    }
    state = read_pass_state(registry, "task100", current_pass=5)
    assert state["attempts_this_pass"] == 2
    assert state["attempted_families"] == ["where-select", "reduce-broadcast"]


def test_read_pass_state_tolerates_missing_subfields():
    """An older entry might have last_attempt_pass but never had the
    counter / family fields populated."""
    registry = {"tasks": {"task100": {"last_attempt_pass": 5}}}
    state = read_pass_state(registry, "task100", current_pass=5)
    assert state == {"attempts_this_pass": 0, "attempted_families": []}


def test_read_pass_state_does_not_mutate_registry():
    """read is read-only — bump_pass_state is the writer."""
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 3,
                "attempted_families": ["where-select"],
                "last_attempt_pass": 1,
            }
        }
    }
    snapshot = json.dumps(registry, sort_keys=True)
    read_pass_state(registry, "task100", current_pass=2)
    assert json.dumps(registry, sort_keys=True) == snapshot


def test_read_pass_state_coerces_string_counters():
    """Defensive against hand-edited registry where someone wrote '3' instead of 3."""
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": "2",
                "attempted_families": ["where-select"],
                "last_attempt_pass": "5",
            }
        }
    }
    state = read_pass_state(registry, "task100", current_pass=5)
    assert state["attempts_this_pass"] == 2


# ----- bump_pass_state -----------------------------------------------------


def test_bump_pass_state_initializes_new_entry():
    registry = {"tasks": {}}
    entry = bump_pass_state(registry, "task100", current_pass=2, family="where-select")
    assert entry["attempts_this_pass"] == 1
    assert entry["attempted_families"] == ["where-select"]
    assert entry["last_attempt_pass"] == 2
    # And the registry was mutated in place
    assert registry["tasks"]["task100"]["attempts_this_pass"] == 1


def test_bump_pass_state_resets_on_pass_increment():
    """The whole point — when current_pass differs from last_attempt_pass,
    counters reset before incrementing."""
    registry = {
        "tasks": {
            "task100": {
                "attempts_this_pass": 3,
                "attempted_families": ["a", "b", "c"],
                "last_attempt_pass": 1,
            }
        }
    }
    entry = bump_pass_state(registry, "task100", current_pass=2, family="d")
    assert entry["attempts_this_pass"] == 1
    assert entry["attempted_families"] == ["d"]
    assert entry["last_attempt_pass"] == 2


def test_bump_pass_state_appends_within_same_pass():
    registry = {"tasks": {}}
    bump_pass_state(registry, "task100", current_pass=2, family="a")
    bump_pass_state(registry, "task100", current_pass=2, family="b")
    bump_pass_state(registry, "task100", current_pass=2, family="c")
    entry = registry["tasks"]["task100"]
    assert entry["attempts_this_pass"] == 3
    assert entry["attempted_families"] == ["a", "b", "c"]


def test_bump_pass_state_dedupes_within_same_pass():
    """If a worker retries with the same family (shouldn't, but might), we
    don't double-record the family — just bump the attempt counter."""
    registry = {"tasks": {}}
    bump_pass_state(registry, "task100", current_pass=2, family="where-select")
    bump_pass_state(registry, "task100", current_pass=2, family="where-select")
    entry = registry["tasks"]["task100"]
    assert entry["attempts_this_pass"] == 2
    assert entry["attempted_families"] == ["where-select"]


def test_bump_pass_state_empty_family_recorded_as_unspecified():
    """Workers that pre-date the algorithm_family field still consume attempts.
    The blacklist needs to grow regardless so the retry directive can say
    'try anything but unspecified' (effectively forcing the worker to label)."""
    registry = {"tasks": {}}
    bump_pass_state(registry, "task100", current_pass=2, family="")
    entry = registry["tasks"]["task100"]
    assert entry["attempted_families"] == ["unspecified"]


def test_bump_pass_state_whitespace_family_recorded_as_unspecified():
    registry = {"tasks": {}}
    bump_pass_state(registry, "task100", current_pass=2, family="   ")
    assert registry["tasks"]["task100"]["attempted_families"] == ["unspecified"]


def test_bump_pass_state_preserves_other_entry_fields():
    """Existing fields (status, summary, current_cost, known_attempts) must
    survive the per-pass counter mutation."""
    registry = {
        "tasks": {
            "task100": {
                "status": "known_public_flat",
                "summary": "prior attempts did not move public score",
                "current_cost": 12345,
                "known_attempts": [{"candidate": "x", "result": "y", "evidence": "z"}],
            }
        }
    }
    bump_pass_state(registry, "task100", current_pass=2, family="where-select")
    entry = registry["tasks"]["task100"]
    assert entry["status"] == "known_public_flat"
    assert entry["summary"] == "prior attempts did not move public score"
    assert entry["current_cost"] == 12345
    assert entry["known_attempts"] == [{"candidate": "x", "result": "y", "evidence": "z"}]


# ----- record_known_attempt -----------------------------------------------


def test_record_known_attempt_appends_to_cumulative_list():
    registry = {"tasks": {}}
    record_known_attempt(
        registry,
        "task100",
        candidate="task100_v1",
        result="public score unchanged",
        evidence="Logs/submissions_ledger.csv",
    )
    record_known_attempt(
        registry,
        "task100",
        candidate="task100_v2",
        result="public regressed -2pt",
        evidence="ledger row 87",
    )
    attempts = registry["tasks"]["task100"]["known_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["candidate"] == "task100_v1"
    assert attempts[1]["result"] == "public regressed -2pt"


def test_record_known_attempt_survives_pass_increment():
    """known_attempts is the LONG-LIVED history. Unlike per-pass counters
    it must NEVER be cleared by bump_pass_state."""
    registry = {"tasks": {}}
    record_known_attempt(registry, "task100", candidate="v1", result="r1")
    bump_pass_state(registry, "task100", current_pass=1, family="a")
    bump_pass_state(registry, "task100", current_pass=2, family="b")  # pass advance
    assert len(registry["tasks"]["task100"]["known_attempts"]) == 1
    assert registry["tasks"]["task100"]["known_attempts"][0]["candidate"] == "v1"
