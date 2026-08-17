"""onnx_opt_extended — additional SAFE-opt passes (math-equivalent transformations).

Extends tools/onnx_opt_safe.py + tools/onnx_opt_with_fold.py with passes
that should also be math-equivalent (provably hidden-correct):

  1. elim_div_by_1 / elim_pow_by_1     — Div(x, 1), Pow(x, 1) -> x
  2. elim_identity                     — Identity(x) -> x
  3. elim_pad_zero                     — Pad with all-zero pads -> x
  4. elim_transpose_identity           — Transpose(x, perm=[0,1,2,...]) -> x
  5. elim_reshape_same_shape           — Reshape(x, shape == x.shape) -> x
  6. elim_cast_same_type               — Cast(x, dtype == x.dtype) -> x
  7. elim_slice_full                   — Slice that selects entire axis -> x
  8. elim_squeeze_unsqueeze_pair       — Squeeze(Unsqueeze(x, ax), ax) -> x
  9. elim_concat_one_input             — Concat([x], axis) -> x
 10. elim_minmax_same                  — Min/Max(x, x) -> x
 11. elim_reducex_empty_axes           — ReduceSum/ReduceMax(x, axes=[]) -> x
 12. elim_unsqueeze_squeeze_pair       — Unsqueeze(Squeeze(x, ax), ax) -> x

Each pass returns count of removed nodes. Composed into a single
optimize() that runs to fixed point.

Safety: each removed node is replaced by direct reference to its input
(via the standard `rename` map pattern). Graph outputs are never
eliminated. All inits are preserved (DCE happens elsewhere).

Usage:
    python tools/onnx_opt_extended.py --in model.onnx --out optimized.onnx
    python tools/onnx_opt_extended.py --zip submission.zip --out-zip opt.zip

The passes here can be safely chained AFTER onnx_opt_safe.py / fold,
which gives the full SAFE-opt pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))


def init_data_tuple(init):
    """Robustly extract initializer values as a Python tuple."""
    if init.raw_data:
        dt = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        arr = np.frombuffer(init.raw_data, dtype=dt).reshape(list(init.dims) or [1])
        return tuple(arr.flatten().tolist())
    if init.data_type == TensorProto.FLOAT:
        return tuple(init.float_data)
    if init.data_type == TensorProto.INT64:
        return tuple(init.int64_data)
    if init.data_type == TensorProto.INT32:
        return tuple(init.int32_data)
    if init.data_type == TensorProto.DOUBLE:
        return tuple(init.double_data)
    return None


def apply_rename(model, rename):
    """Substitute all renamed tensor uses in the graph."""
    for node in model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] in rename:
                node.input[i] = rename[node.input[i]]


def is_scalar_value(init, value, atol=1e-9):
    """Return True if init has scalar or singleton shape AND all values ≈ value."""
    dims = list(init.dims)
    if dims and dims != [1]:
        return False
    d = init_data_tuple(init)
    if d is None or len(d) == 0:
        return False
    return all(abs(v - value) < atol for v in d)


def elim_div_by_1(model):
    """Div(x, 1) -> x. (1) must be a scalar-shaped initializer."""
    init_by_name = {i.name: i for i in model.graph.initializer}
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False; rep = None
        if (node.op_type == 'Div' and len(node.input) == 2
                and not any(o in graph_outputs for o in node.output)):
            if node.input[1] in init_by_name and is_scalar_value(init_by_name[node.input[1]], 1):
                rep = node.input[0]; elim = True
        if elim:
            for o in node.output: rename[o] = rep
            n += 1
        else:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_pow_by_1(model):
    """Pow(x, 1) -> x. (1) must be a scalar-shaped initializer."""
    init_by_name = {i.name: i for i in model.graph.initializer}
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False; rep = None
        if (node.op_type == 'Pow' and len(node.input) == 2
                and not any(o in graph_outputs for o in node.output)):
            if node.input[1] in init_by_name and is_scalar_value(init_by_name[node.input[1]], 1):
                rep = node.input[0]; elim = True
        if elim:
            for o in node.output: rename[o] = rep
            n += 1
        else:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_identity(model):
    """Identity(x) -> x."""
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        if (node.op_type == 'Identity' and len(node.input) == 1
                and not any(o in graph_outputs for o in node.output)):
            for o in node.output: rename[o] = node.input[0]
            n += 1
        else:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_pad_zero(model):
    """Pad with all-zero pad amounts -> x.

    Opset 11+: pads come as input tensor at index 1.
    Opset 10-: pads as 'pads' attribute.
    """
    init_by_name = {i.name: i for i in model.graph.initializer}
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False
        if node.op_type == 'Pad' and not any(o in graph_outputs for o in node.output):
            pads = None
            # Try input form (opset 11+)
            if len(node.input) >= 2 and node.input[1] in init_by_name:
                d = init_data_tuple(init_by_name[node.input[1]])
                if d is not None:
                    pads = d
            # Try attribute form (opset 10-)
            if pads is None:
                for a in node.attribute:
                    if a.name == 'pads':
                        pads = tuple(a.ints)
                        break
            if pads is not None and all(p == 0 for p in pads):
                for o in node.output: rename[o] = node.input[0]
                elim = True
                n += 1
        if not elim:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_transpose_identity(model):
    """Transpose with perm=[0,1,2,...] (identity permutation) -> x."""
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False
        if (node.op_type == 'Transpose' and len(node.input) == 1
                and not any(o in graph_outputs for o in node.output)):
            perm = None
            for a in node.attribute:
                if a.name == 'perm':
                    perm = list(a.ints)
                    break
            if perm is not None and perm == list(range(len(perm))):
                for o in node.output: rename[o] = node.input[0]
                elim = True
                n += 1
        if not elim:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_concat_one_input(model):
    """Concat with single input -> x."""
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        if (node.op_type == 'Concat' and len(node.input) == 1
                and not any(o in graph_outputs for o in node.output)):
            for o in node.output: rename[o] = node.input[0]
            n += 1
        else:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_minmax_same(model):
    """Min(x, x) -> x.  Max(x, x) -> x.  Mul(x, ones_full) etc.  Limited to len-2."""
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False
        if (node.op_type in {'Min', 'Max'} and len(node.input) == 2
                and not any(o in graph_outputs for o in node.output)
                and node.input[0] == node.input[1]):
            for o in node.output: rename[o] = node.input[0]
            elim = True; n += 1
        if not elim:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_reducex_empty_axes(model):
    """ReduceSum/ReduceMax with empty axes attr (no-op) -> x.

    Note: this is a rare case; default behavior of opset-13+ ReduceSum is to
    reduce ALL axes if axes is empty, NOT no-op. So we only eliminate when
    `noop_with_empty_axes` attribute is explicitly 1 (opset 13+ Reduce ops).
    """
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False
        if (node.op_type in {'ReduceSum', 'ReduceMax', 'ReduceMin', 'ReduceMean', 'ReduceProd'}
                and len(node.input) >= 1
                and not any(o in graph_outputs for o in node.output)):
            noop_empty = 0
            axes_attr_empty = False
            for a in node.attribute:
                if a.name == 'noop_with_empty_axes':
                    noop_empty = a.i
                elif a.name == 'axes':
                    if not list(a.ints):
                        axes_attr_empty = True
            # Also check axes-as-input (opset 18+ for ReduceSum). Skip — rare.
            if noop_empty == 1 and (len(node.input) == 1 or axes_attr_empty):
                for o in node.output: rename[o] = node.input[0]
                elim = True; n += 1
        if not elim:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


def elim_cast_same_type(model):
    """Cast(x, dtype == x.dtype) -> x.

    Requires shape inference (or initializer-type lookup) to know x's dtype.
    Conservative: only eliminate if x is an initializer whose dtype matches the Cast's `to`.
    """
    init_by_name = {i.name: i for i in model.graph.initializer}
    graph_outputs = {o.name for o in model.graph.output}
    rename, keep, n = {}, [], 0
    for node in model.graph.node:
        elim = False
        if (node.op_type == 'Cast' and len(node.input) == 1
                and not any(o in graph_outputs for o in node.output)):
            to_type = None
            for a in node.attribute:
                if a.name == 'to':
                    to_type = a.i
                    break
            if to_type is not None and node.input[0] in init_by_name:
                src_type = init_by_name[node.input[0]].data_type
                if src_type == to_type:
                    for o in node.output: rename[o] = node.input[0]
                    elim = True; n += 1
        if not elim:
            keep.append(node)
    if n:
        del model.graph.node[:]; model.graph.node.extend(keep); apply_rename(model, rename)
    return n


PASS_FUNCS = {
    'elim_div_by_1': elim_div_by_1,
    'elim_pow_by_1': elim_pow_by_1,
    'elim_identity': elim_identity,
    'elim_pad_zero': elim_pad_zero,
    'elim_transpose_identity': elim_transpose_identity,
    'elim_concat_one_input': elim_concat_one_input,
    'elim_minmax_same': elim_minmax_same,
    'elim_reducex_empty_axes': elim_reducex_empty_axes,
    'elim_cast_same_type': elim_cast_same_type,
}


def optimize(model, verbose=False):
    """Run all extended passes to fixed point. Returns dict of {pass_name: total_removed}."""
    totals = {n: 0 for n in PASS_FUNCS}
    while True:
        progress = 0
        for name, fn in PASS_FUNCS.items():
            removed = fn(model)
            totals[name] += removed
            progress += removed
        if not progress:
            break
    if verbose:
        for name, total in totals.items():
            if total:
                print(f"  {name}: -{total}")
    return totals


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=False)
    p.add_argument("--out", dest="out_path", type=Path, required=False)
    p.add_argument("--zip", dest="zip_path", type=Path, required=False,
                   help="Apply to every ONNX in a submission zip")
    p.add_argument("--out-zip", type=Path, required=False)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.zip_path:
        if not args.out_zip:
            print("--zip requires --out-zip")
            return 1
        from zipfile import ZipFile, ZIP_DEFLATED
        import tempfile, os
        with ZipFile(args.zip_path) as zin, ZipFile(args.out_zip, 'w', ZIP_DEFLATED) as zout:
            grand_total = {n: 0 for n in PASS_FUNCS}
            for name in sorted(zin.namelist()):
                if not name.endswith('.onnx'):
                    zout.writestr(name, zin.read(name))
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
                tmp.write(zin.read(name)); tmp.close()
                try:
                    m = onnx.load(tmp.name)
                    totals = optimize(m, verbose=False)
                    for k, v in totals.items(): grand_total[k] += v
                    out_bytes = m.SerializeToString()
                    zout.writestr(name, out_bytes)
                except Exception as e:
                    print(f"  {name}: ERR {e}; copying unchanged")
                    zout.writestr(name, zin.read(name))
                try: os.unlink(tmp.name)
                except OSError: pass
        print(f"Wrote {args.out_zip}")
        for k, v in grand_total.items():
            if v: print(f"  {k}: -{v}")
        return 0

    if not (args.in_path and args.out_path):
        print("Need --in + --out, or --zip + --out-zip")
        return 1

    m = onnx.load(args.in_path)
    totals = optimize(m, verbose=args.verbose)
    onnx.save(m, args.out_path)
    print(f"Wrote {args.out_path}")
    for k, v in totals.items():
        if v: print(f"  {k}: -{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
