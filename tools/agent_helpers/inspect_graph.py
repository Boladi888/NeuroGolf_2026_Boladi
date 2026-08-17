"""Print the full graph structure of an ONNX: nodes, initializers, value_info, flags.

Usage:
    python tools/agent_helpers/inspect_graph.py path/to/model.onnx
    python tools/agent_helpers/inspect_graph.py path/to/model.onnx --tasks-only
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

DTYPE_INFO = {1: ("F32", 4), 2: ("UINT8", 1), 3: ("INT8", 1), 4: ("UINT16", 2),
               5: ("INT16", 2), 6: ("INT32", 4), 7: ("INT64", 8), 9: ("BOOL", 1),
               10: ("F16", 2), 11: ("F64", 8)}


def dtype_short(e):
    return DTYPE_INFO.get(e, (f"dtype{e}",))[0]


def dtype_size(e):
    return DTYPE_INFO.get(e, (None, 4))[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_path")
    args = p.parse_args()

    model = onnx.load(args.model_path)
    g = model.graph

    print(f"=== ONNX: {args.model_path} ===")
    print(f"opset_import: {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print(f"ir_version: {model.ir_version}")
    print(f"graph name: {g.name}")
    print(f"inputs: {[(i.name, dtype_short(i.type.tensor_type.elem_type), [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]}")
    print(f"outputs: {[(o.name, dtype_short(o.type.tensor_type.elem_type), [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]}")
    print(f"n_nodes: {len(g.node)}, n_inits: {len(g.initializer)}, n_value_info: {len(g.value_info)}")

    print(f"\n=== Op-type histogram ===")
    c = Counter(n.op_type for n in g.node)
    for op, n in c.most_common():
        print(f"  {op:<20} {n}")

    print(f"\n=== Initializers (top 20 by params) ===")
    inits = []
    for init in g.initializer:
        params = 1
        for d in init.dims:
            params *= d
        inits.append((params, init.name, dtype_short(init.data_type), list(init.dims)))
    inits.sort(reverse=True)
    for params, name, dtype, dims in inits[:20]:
        print(f"  {name:<40} {dtype:<6} {str(dims):<20} params={params}")
    if len(inits) > 20:
        print(f"  ... ({len(inits) - 20} more)")

    print(f"\n=== value_info intermediates (top 20 by static bytes) ===")
    vis = []
    for vi in g.value_info:
        tt = vi.type.tensor_type
        if not tt.shape.dim:
            continue
        n_elem = 1
        ok = True
        for d in tt.shape.dim:
            if d.dim_value <= 0:
                ok = False; break
            n_elem *= d.dim_value
        if not ok:
            continue
        bytes_ = n_elem * dtype_size(tt.elem_type)
        vis.append((bytes_, vi.name, dtype_short(tt.elem_type), [d.dim_value for d in tt.shape.dim]))
    vis.sort(reverse=True)
    for bytes_, name, dtype, dims in vis[:20]:
        print(f"  {name:<40} {dtype:<6} {str(dims):<20} bytes={bytes_:,}")
    if len(vis) > 20:
        print(f"  ... ({len(vis) - 20} more)")

    print(f"\n=== Nodes (in order) ===")
    for i, n in enumerate(g.node):
        name = n.name or "(unnamed)"
        ins = ", ".join(n.input)[:60]
        outs = ", ".join(n.output)[:40]
        attrs = " ".join(f"{a.name}={a.i if a.type == 2 else (a.f if a.type == 1 else '..')}" for a in n.attribute if a.name)[:30]
        print(f"  [{i:>3}] {n.op_type:<15} {name:<20} | in: {ins} | out: {outs} | {attrs}")


if __name__ == "__main__":
    raise SystemExit(main())
