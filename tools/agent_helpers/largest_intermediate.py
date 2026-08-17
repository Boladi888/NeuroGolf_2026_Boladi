"""Print top-N intermediates by static byte size.

Use this to identify where memory is concentrated — the biggest intermediate
is usually the highest-leverage target for dtype-shrinking or rank reduction.

Usage:
    python tools/agent_helpers/largest_intermediate.py model.onnx
    python tools/agent_helpers/largest_intermediate.py model.onnx --top 20
"""
import argparse
import sys
from pathlib import Path

import onnx

DTYPE_INFO = {1: ("F32", 4), 2: ("UINT8", 1), 3: ("INT8", 1), 4: ("UINT16", 2),
               5: ("INT16", 2), 6: ("INT32", 4), 7: ("INT64", 8), 9: ("BOOL", 1),
               10: ("F16", 2), 11: ("F64", 8)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_path")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    model = onnx.load(args.model_path)
    # If value_info is empty (worker did ClearField), run shape_inference to get intermediate shapes
    if len(model.graph.value_info) == 0:
        try:
            model = onnx.shape_inference.infer_shapes(model)
            print(f"(value_info was empty; ran shape_inference, got {len(model.graph.value_info)} intermediates)")
        except Exception as exc:
            print(f"(WARNING: shape_inference failed: {exc})")
    g = model.graph

    # Map output name → producing node op
    producer_op = {out: n.op_type for n in g.node for out in n.output if out}

    intermediates = []
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
        info = DTYPE_INFO.get(tt.elem_type, (f"dtype{tt.elem_type}", 4))
        dtype_name, dtype_bytes = info
        bytes_ = n_elem * dtype_bytes
        intermediates.append({
            "name": vi.name,
            "dtype": dtype_name,
            "dims": [d.dim_value for d in tt.shape.dim],
            "elements": n_elem,
            "bytes": bytes_,
            "producer_op": producer_op.get(vi.name, "?"),
        })

    intermediates.sort(key=lambda x: -x["bytes"])
    total = sum(it["bytes"] for it in intermediates)

    print(f"=== {args.model_path} — top {args.top} intermediates ===")
    print(f"Total intermediate bytes (static value_info): {total:,}")
    print(f"{'rank':>4}  {'bytes':>12}  {'pct':>5}  {'dtype':>6}  {'shape':<25}  {'producer':<15}  name")
    for i, it in enumerate(intermediates[:args.top], 1):
        pct = 100.0 * it["bytes"] / max(total, 1)
        print(f"{i:>4}  {it['bytes']:>12,}  {pct:>5.1f}  {it['dtype']:>6}  {str(it['dims']):<25}  {it['producer_op']:<15}  {it['name']}")

    if not intermediates:
        print("(no static-shape intermediates; check `onnx.shape_inference.infer_shapes(model)` was applied)")


if __name__ == "__main__":
    raise SystemExit(main())
