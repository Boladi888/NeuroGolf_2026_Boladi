#!/usr/bin/env python3
"""Build one or more tasks through detect_pattern's dispatcher-selected network.

This is the safe single-task companion to ``python build_submission.py --hybrid``.
It materializes ``submissions/taskNNN.onnx`` via ``build_one(..., network_fn=...)``.
``tools/dispatcher_audit.py`` verifies the dispatcher choice, but intentionally
does not write the chosen ONNX payload into ``submissions/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_submission as bs  # noqa: E402


def _parse_tasks(raw: str) -> list[int]:
    tasks: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tasks.append(int(part))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks",
        help="Comma-separated NeuroGolf task ids, e.g. 249 or 003,020,249.",
    )
    args = parser.parse_args()

    for task_num in _parse_tasks(args.tasks):
        examples = bs.neurogolf_utils.load_examples(task_num)
        label, network_fn = bs.detect_pattern(examples)
        if network_fn is None:
            print(f"task{task_num:03d}: no dispatcher match", file=sys.stderr)
            return 1
        result = bs.build_one(task_num, network_fn=network_fn)
        status = result.get("status", "")
        print(
            f"task{task_num:03d}: {label} -> {status}; "
            f"cost={result.get('cost')} file={result.get('filename')}"
        )
        if "CORRECT" not in status:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
