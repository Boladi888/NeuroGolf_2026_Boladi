"""state.py — one-shot bootstrap snapshot for a new agent.

Prints ~10 lines summarizing the current scoring state, task counts, and
pointers to authoritative files. Run when starting a session to skip reading
five separate files.

Pulls facts from authoritative sources:
- SOLVED count: tools/inspect_tasks.py:SOLVED_TASKS
- Score line: parses CURRENT_STATE.md if present
- Task state: The_400_Tasks/task_state.csv
- Submit credit state: .hermes/board/submit_credit_state.json
- Latest digest: Logs/submission_candidate_digest_<latest>.md
- Latest probe queue: Logs/isolated_probe_recommendations_<latest>.md


Usage:
    python tools/state.py            # full snapshot
    python tools/state.py --terse    # one-line score + counts only
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _solved_count() -> int | None:
    """Import SOLVED_TASKS from the canonical source."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from inspect_tasks import SOLVED_TASKS  # type: ignore
        return len(SOLVED_TASKS)
    except Exception as exc:
        print(f"  (warn: could not import SOLVED_TASKS: {exc})", file=sys.stderr)
        return None


def _parse_score_from_current_state() -> str | None:
    """Pull the 'Current valid Kaggle best' line out of CURRENT_STATE.md."""
    p = REPO_ROOT / "CURRENT_STATE.md"
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(r"\*\*Current valid Kaggle best:\*\*\s*(.+?)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def _task_state_counts() -> dict:
    """Tabulate task_state.csv statuses."""
    p = REPO_ROOT / "The_400_Tasks" / "task_state.csv"
    counts: dict = {"total": 0, "by_status": {}}
    if not p.exists():
        return counts
    try:
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                counts["total"] += 1
                status = (row.get("best_known_status") or "").strip() or "(empty)"
                counts["by_status"][status] = counts["by_status"].get(status, 0) + 1
    except Exception as exc:
        print(f"  (warn: could not read task_state.csv: {exc})", file=sys.stderr)
    return counts


def _latest_digest_path() -> Path | None:
    logs = REPO_ROOT / "Logs"
    if not logs.exists():
        return None
    candidates = sorted(logs.glob("submission_candidate_digest_*.md"))
    return candidates[-1] if candidates else None


def _latest_probe_path() -> Path | None:
    logs = REPO_ROOT / "Logs"
    if not logs.exists():
        return None
    candidates = sorted(
        list(logs.glob("isolated_probe_recommendations_*.md"))
        + list(logs.glob("next_isolated_probe_recommendations_*.md"))
    )
    return candidates[-1] if candidates else None


def _latest_session_log() -> Path | None:
    sessions = REPO_ROOT / "Logs" / "sessions"
    if not sessions.exists():
        return None
    candidates = sorted(sessions.glob("*.md"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--terse", action="store_true", help="One-line summary only")
    args = parser.parse_args(argv)

    score = _parse_score_from_current_state()
    solved = _solved_count()
    state = _task_state_counts()

    if args.terse:
        s = score or "(score unknown — CURRENT_STATE.md missing or unparseable)"
        n = solved if solved is not None else "?"
        print(f"score={s}  solved={n}  tasks={state['total']}")
        return 0

    print("=" * 70)
    print(" NeuroGolf 2026 — current state snapshot")
    print("=" * 70)
    print()
    print(f"  Score:           {score or '(unknown; CURRENT_STATE.md not parseable)'}")
    print(f"  SOLVED organic:  {solved if solved is not None else '(import failed)'}")
    print(f"  task_state rows: {state['total']}")
    if state["by_status"]:
        for status, n in sorted(state["by_status"].items(), key=lambda x: -x[1])[:6]:
            print(f"    {status:<32} {n}")
    print()

    digest = _latest_digest_path()
    probe = _latest_probe_path()
    session = _latest_session_log()
    credit = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"
    print("  Latest digest:   {}".format(digest.relative_to(REPO_ROOT) if digest else "(none)"))
    print("  Latest probe q:  {}".format(probe.relative_to(REPO_ROOT) if probe else "(none)"))
    print("  Latest session:  {}".format(session.relative_to(REPO_ROOT) if session else "(none)"))
    print("  Submit credits:  {}".format(credit.relative_to(REPO_ROOT) if credit.exists() else "(none)"))

    print("  Read these next for context:")
    print("    1. CURRENT_STATE.md          — top-level snapshot")
    print("    2. WHERE_DATA_LIVES.md       — junction layout / data locations")
    print("    3. .ai_instructions/03_PROJECT.md  — tech stack / architecture")
    print("    4. The latest session log    — what just happened")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
