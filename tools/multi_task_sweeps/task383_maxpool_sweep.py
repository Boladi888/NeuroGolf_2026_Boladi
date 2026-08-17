#!/usr/bin/env python3
"""task383 directional Pad+Slice → MaxPool window sweep.

Pattern: a Slice + Pad pair forming a directional proximity probe:
    Slice(input, start=K, end=W, axis=W_axis) -> shifted
    Pad(shifted, pads=[..., K, ...]) -> probe_at_distance_K

Replaceable with a single MaxPool with asymmetric pad on the same axis.
Per task383 +1.01 LB Kaggle confirmed.

Detection heuristic: tasks with 4+ Slice operations that have similar
attribute patterns (different start/end indices on same axis). Auto-fold
each Pad+Slice pair into a MaxPool window. Verify against LIVE.

Conservative: only apply if we can definitively detect the directional
probe pattern; skip otherwise (false-positive risk).

Usage:
    .venv/Scripts/python tools/multi_task_sweeps/task383_maxpool_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

from tools.multi_task_sweeps._common import (
    iter_canonical_tasks, sweep_task, write_report,
)


def _get_initializer(g, name):
    for init in g.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def transform_task383_pattern(model: onnx.ModelProto) -> onnx.ModelProto | None:
    """Detect directional Pad+Slice probe chains.

    Conservative implementation: count Pad+Slice pairs that probe the same
    input tensor. If count >= 4 AND a clear directional pattern emerges,
    attempt the rewrite. Otherwise skip.

    Note: full task383 rewrite is complex graph surgery. This pass identifies
    candidates; final transform is left as a TODO marker — for now we just
    return None on every task (advisory-only). A future iteration can implement
    the actual MaxPool rewrite for matching tasks.
    """
    g = model.graph
    # Count Slice + Pad pairs
    slice_pad_pairs = 0
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    for n in g.node:
        if n.op_type != "Slice":
            continue
        slice_out = n.output[0]
        for c in consumers.get(slice_out, []):
            if c.op_type == "Pad":
                slice_pad_pairs += 1
                break
    if slice_pad_pairs < 4:
        return None
    # Detection only — full rewrite is task-specific. Mark as advisory.
    return None


def main():
    # Detection-only mode: log candidates that COULD benefit from task383 transfer.
    print("task383 MaxPool sweep: detection-only mode")
    print("Tasks with 4+ Slice+Pad directional probe pairs:")
    print()
    candidates = []
    for tid, onnx_bytes in iter_canonical_tasks():
        try:
            m = onnx.load_model_from_string(onnx_bytes)
        except Exception:
            continue
        g = m.graph
        consumers = {}
        for n in g.node:
            for inp in n.input:
                consumers.setdefault(inp, []).append(n)
        pairs = 0
        for n in g.node:
            if n.op_type != "Slice":
                continue
            for c in consumers.get(n.output[0], []):
                if c.op_type == "Pad":
                    pairs += 1
                    break
        if pairs >= 4:
            candidates.append((tid, pairs))
    candidates.sort(key=lambda x: -x[1])
    md_lines = [
        "# task383 MaxPool-window transferer — candidates (detection-only)",
        "",
        f"**{len(candidates)} tasks** have 4+ Slice→Pad directional probe pairs and may benefit from the task383 MaxPool-window technique.",
        "",
        "Full automated rewrite is task-specific (requires asymmetric MaxPool kernel + pad construction). Workers can manually apply task383's technique to top candidates.",
        "",
        "| task | Slice→Pad pairs | priority |",
        "|---|---:|---|",
    ]
    for tid, n in candidates:
        priority = "HIGH" if n >= 15 else "MED" if n >= 8 else "low"
        md_lines.append(f"| {tid} | {n} | {priority} |")
        print(f"  {tid}: {n} Slice+Pad pairs ({priority})")
    out_path = REPO_ROOT / "Logs" / "multi_task_sweeps" / "task383_maxpool_candidates.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nReport: {out_path}")
    print(f"Total candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
