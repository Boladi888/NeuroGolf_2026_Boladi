#!/usr/bin/env python3
"""task175 v12 variadic-Max-over-slices sweep.

Pattern: `Concat(slice0, slice1, ..., sliceN) -> ReduceMax(axis=concat_axis)`
where slices share dtype. Replace with variadic `Max(slice0, ..., sliceN)`.

Saves the Concat intermediate. Per task175 v12 ship +2.98 LB Kaggle confirmed.

Usage:
    .venv/Scripts/python tools/multi_task_sweeps/variadic_max_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import onnx
from onnx import TensorProto, helper

from tools.multi_task_sweeps._common import (
    iter_canonical_tasks, sweep_task, write_report,
)


def transform_variadic_max(model: onnx.ModelProto) -> onnx.ModelProto | None:
    """Find Concat→ReduceMax(axis=concat-axis) patterns; replace with variadic Max.

    Only applies when:
      1. ReduceMax reduces along the SAME axis as the Concat
      2. ReduceMax has keepdims=0 (single axis collapse → all inputs same shape)
      3. Concat output not consumed by anything else
    """
    g = model.graph
    # Build consumer map
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    found = []
    for i, n in enumerate(g.node):
        if n.op_type != "ReduceMax":
            continue
        red_input = n.input[0]
        # Find Concat producer
        concat = next((p for p in g.node if red_input in p.output), None)
        if concat is None or concat.op_type != "Concat":
            continue
        # Concat must have multiple inputs
        if len(concat.input) < 2:
            continue
        # Concat axis vs ReduceMax axes
        concat_axis_attr = next((a for a in concat.attribute if a.name == "axis"), None)
        if concat_axis_attr is None:
            continue
        c_axis = concat_axis_attr.i
        red_axes_attr = next((a for a in n.attribute if a.name == "axes"), None)
        red_keepdims_attr = next((a for a in n.attribute if a.name == "keepdims"), None)
        if red_axes_attr is None:
            continue
        red_axes = list(red_axes_attr.ints)
        keepdims = red_keepdims_attr.i if red_keepdims_attr else 1
        if keepdims == 1 or len(red_axes) != 1 or red_axes[0] != c_axis:
            continue
        # Concat output should only be consumed by this ReduceMax
        if len(consumers.get(red_input, [])) > 1:
            continue
        found.append((i, n, concat))

    if not found:
        return None

    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    new_g = new_model.graph
    new_nodes = list(new_g.node)

    # For each match, replace ReduceMax with variadic Max + drop Concat
    # Process in reverse order so indices remain valid
    for i, reduce_node, concat_node in sorted(found, key=lambda x: -x[0]):
        # New Max op: same output as reduce, inputs from concat
        new_max = helper.make_node("Max", concat_node.input, reduce_node.output,
                                   name=reduce_node.name + "_variadic")
        # Find indices
        red_idx = next(j for j, n in enumerate(new_nodes) if n is new_g.node[i])
        # Hmm — proto references are tricky; let me find by output name instead
        red_idx = None
        concat_idx = None
        for j, n in enumerate(new_nodes):
            if list(n.output) == list(reduce_node.output):
                red_idx = j
            if list(n.output) == list(concat_node.output):
                concat_idx = j
        if red_idx is None or concat_idx is None:
            continue
        # Remove Concat (it's no longer needed since variadic Max consumes its inputs)
        new_nodes[red_idx] = new_max
        new_nodes = [n for j, n in enumerate(new_nodes) if j != concat_idx]

    del new_g.node[:]
    new_g.node.extend(new_nodes)
    new_g.ClearField("value_info")
    return new_model


def main():
    results = []
    skipped = 0
    for tid, onnx_bytes in iter_canonical_tasks():
        r = sweep_task(tid, onnx_bytes, transform_variadic_max, "variadic_max")
        if r is not None:
            results.append(r)
            print(f"  WIN  {tid}: {r.base_cost:,} -> {r.candidate_cost:,} "
                  f"(+{r.raw_log_e_gain:.3f} LB)")
        else:
            skipped += 1
    write_report("variadic_max", results)
    print(f"\nvariadic_max sweep: {len(results)} winners / {len(results) + skipped} scanned")


if __name__ == "__main__":
    main()
