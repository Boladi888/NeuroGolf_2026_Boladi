#!/usr/bin/env python3
"""Run score_network_v2 on one ONNX file and print JSON to stdout.

Used by spread_scan.py to isolate scoring from C-level crashes
(e.g. ORT raising SIGFPE on certain fp16-converted graphs). The driver
calls this script via subprocess; if it crashes (non-zero exit), the
driver records the task/pattern as ``score-failed`` and continues.

Usage::

    python tools/spread_score_isolated.py <path-to-onnx>

Emits to stdout: ``{"cost": int, "memory": int, "params": int, "macs": int, "points": float}``
or ``{"error": "<reason>"}`` on failure. Always exits 0 except on
unhandled crashes (where the parent reads the non-zero exit code).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools._score_v2 import score_network_v2  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(json.dumps({"error": "usage: spread_score_isolated.py <onnx-path>"}))
        return 2
    path = argv[1]
    try:
        result = score_network_v2(path)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 0
    if result is None:
        print(json.dumps({"error": "score_network_v2 returned None"}))
        return 0
    macs, memory, params = result
    cost = int(memory) + int(params)
    points = max(1.0, 25.0 - math.log(max(1.0, cost)))
    print(json.dumps({
        "macs": int(macs),
        "memory": int(memory),
        "params": int(params),
        "cost": cost,
        "points": round(points, 6),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
