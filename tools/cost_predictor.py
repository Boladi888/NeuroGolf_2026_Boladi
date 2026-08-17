"""Fast standalone (macs, memory, params, cost, pts) estimator for an ONNX model.

Reproduces the NeuroGolf scoring formula without invoking `onnx-tool`'s
profiler. Intended for prototype-iteration: ~1 ms per call vs `score_network`
~100 ms. Re-implements the three accounting rules the validator + onnx-tool
1.0.1 use; run `--solved --verify` to re-check against `score_network`
across the live `SOLVED_TASKS` set:

  * macs accounting:
    - Conv / Gemm / MatMul          : documented formula
    - Reduce* (Sum/Max/Min/Mean/...): 1 MAC per INPUT element
    - Element-wise ops              : 1 MAC per OUTPUT element. Includes
        Add, Sub, Mul, Mod, Equal, Greater, Less, And, Or, Xor, Not,
        Hardmax, Relu, Sigmoid, Tanh, Abs, Neg, Sign, Floor, Ceil, Round, Clip
    - Where, OneHot, Gather, Slice, Pad, Reshape, Concat, Transpose,
      Cast, Tile, Resize, Identity : 0 MACs (verified in onnx-tool source)

  * memory: sum of intermediate `value_info` tensor byte sizes, EXCLUDING
    tensors named "input"/"output". Matches neurogolf_utils.calculate_memory.
    Initializers do NOT count toward memory.

  * params: element count (not bytes) of unique initializers (deduped by
    name; matches onnx-tool's params_flag_map). Constant nodes' folded
    outputs are NOT counted directly; if a network uses many Constant
    nodes, run `onnx.shape_inference` (already done) catches the shapes,
    but the constant-folding step that score_network performs may move
    Constants into initializers and shift the count. Re-run `--verify`
    if you suspect drift.

Cost = memory + params (May-6 scorer). Points = max(1.0, 25.0 - log(max(1.0, cost))).
The MACs term is no longer part of the official cost since the 2026-05-06
scoring change; this predictor still computes MACs for diagnostic purposes
but excludes them from `cost`.

Usage:
    python tools/cost_predictor.py PATH/TO/model.onnx
    python tools/cost_predictor.py PATH --json       # JSON output (machine-readable)
    python tools/cost_predictor.py --solved          # check every SOLVED_TASK
    python tools/cost_predictor.py --solved --verify # also call score_network and compute delta
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = NG_ROOT / "submissions"
sys.path.insert(0, str(NG_ROOT / "tools"))

import onnx  # noqa: E402

# ONNX TensorProto data type → bytes per element
_BYTES_PER_ELEM = {
    1: 4,   # FLOAT
    2: 1,   # UINT8
    3: 1,   # INT8
    4: 2,   # UINT16
    5: 2,   # INT16
    6: 4,   # INT32
    7: 8,   # INT64
    9: 1,   # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    13: 8,  # UINT64
    16: 2,  # BFLOAT16
}


def _shape_dims(shape_proto):
    """Return list of ints, or None if any dim is dynamic / missing."""
    dims = []
    for dim in shape_proto.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            return None
    return dims


def predict(onnx_path):
    """Return dict {macs, memory, params, cost, points, ops} for an ONNX file.

    Returns dict with 'error' if shape inference or parsing fails.
    """
    raw_path = pathlib.Path(onnx_path)
    if not raw_path.exists():
        return {"error": f"file not found: {raw_path}"}

    try:
        model = onnx.load(str(raw_path))
        # strict_mode=True matches the validator's calculate_memory shape inference
        # (neurogolf_utils.py:140). Without it, some intermediate tensors lack
        # value_info entries -> memory undercount on graphs with chained reductions
        # (e.g. bbox-fill: -3,600 B; enclosed-region-fill: -27,000 B).
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load/shape_infer: {e}"}

    # 1. params: unique initializer element-count sum.
    # Match onnx-tool's `volume([]) == 0` for scalar tensors (empty shape),
    # not Python's `math.prod([]) == 1`. Otherwise we overcount by +1 per
    # scalar initializer (e.g. OneHot's `depth`). See onnx-tool's
    # `tensor.py:volume`.
    params = 0
    seen_init = set()
    init_shapes = {}
    for init in model.graph.initializer:
        if init.name in seen_init:
            continue
        seen_init.add(init.name)
        dims = list(init.dims)
        n = math.prod(dims) if dims else 0
        params += n
        init_shapes[init.name] = dims

    # 2. memory: intermediate value_info byte sizes (exclude input/output names)
    memory = 0
    all_shapes = dict(init_shapes)
    for inp in model.graph.input:
        d = _shape_dims(inp.type.tensor_type.shape)
        if d is not None:
            all_shapes[inp.name] = d
    for out in model.graph.output:
        d = _shape_dims(out.type.tensor_type.shape)
        if d is not None:
            all_shapes[out.name] = d

    for vi in model.graph.value_info:
        if vi.name in ("input", "output"):
            continue
        d = _shape_dims(vi.type.tensor_type.shape)
        if d is None:
            continue
        all_shapes[vi.name] = d
        # Same scalar-shape convention as params: empty-shape tensors are 0 bytes.
        n = math.prod(d) if d else 0
        bpe = _BYTES_PER_ELEM.get(vi.type.tensor_type.elem_type, 4)
        memory += n * bpe

    # 3. macs accounting (per Research_Archive/notes/_cluster-summary-empirical-findings.md
    # + onnx-tool 1.0.1 source at Research_Archive/sources/onnx_tool_1.0.1_node.py):
    #   Conv / Gemm / MatMul                — documented formula
    #   Reduce* (Sum/Max/Min/Mean)          — 1 MAC per INPUT element
    #   MaxPool / AveragePool               — output_volume * kernel_h * kernel_w (PoolBase.profile)
    #   Element-wise                        — output_volume * mac_multiplier (PWNode/NpMathBase.profile)
    #         multiplier = 1 for most: Add/Sub/Mul/Mod/Equal/Greater/Less/And/Or/Xor/Not/
    #               Hardmax/Relu/Abs/Neg/Sign/Floor/Ceil/Round/Min/Max
    #         multiplier = 2 for Clip   (CMP_MACS * 2 in ClipNode)
    #         multiplier = 4 for Div    (DIV_MACS in DivNode)
    #         multiplier = 32 for Exp/Sigmoid/Tanh/Sqrt   (EXP_MACS in ExpNode/derived)
    #         multiplier = 43 for Log   (LOG_MACS in LogNode)
    #         multiplier = 36 for Softmax  (Exp + Div = 32 + 4)
    #   Where, OneHot, Gather, Slice, Pad, Reshape, Concat, Transpose, Cast, Tile,
    #     Resize, Identity, ArgMax, TopK, ScatterND, GatherND, GatherElements — 0 MACs
    _ELEM_OPS = {
        "Add": 1, "Sub": 1, "Mul": 1, "Mod": 1,
        "Equal": 1, "Greater": 1, "Less": 1, "GreaterOrEqual": 1, "LessOrEqual": 1,
        "And": 1, "Or": 1, "Xor": 1, "Not": 1,
        "Hardmax": 1, "Relu": 1, "Abs": 1, "Neg": 1, "Sign": 1,
        "Floor": 1, "Ceil": 1, "Round": 1,
        "Min": 1, "Max": 1,
        "Clip": 2,
        "Div": 4,
        "Exp": 32, "Sigmoid": 32, "Tanh": 32, "Sqrt": 32,
        "Log": 43,
        "Softmax": 36,
    }
    _REDUCE_OPS = {"ReduceSum", "ReduceMax", "ReduceMin", "ReduceMean", "ReduceProd", "ReduceL1", "ReduceL2"}
    _POOL_OPS = {"MaxPool", "AveragePool"}

    macs = 0
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        if node.op_type == "Conv" and len(node.input) >= 2 and node.output:
            out_name = node.output[0]
            w_name = node.input[1]
            if out_name in all_shapes and w_name in all_shapes:
                out_shape = all_shapes[out_name]
                w_shape = all_shapes[w_name]
                if len(out_shape) == 4 and len(w_shape) == 4:
                    _, c_out, h_out, w_out = out_shape
                    _, c_in, k_h, k_w = w_shape
                    macs += c_out * c_in * k_h * k_w * h_out * w_out
        elif node.op_type in ("Gemm", "MatMul") and len(node.input) >= 2 and node.output:
            a_name, b_name = node.input[0], node.input[1]
            if a_name in all_shapes and b_name in all_shapes:
                a_shape, b_shape = all_shapes[a_name], all_shapes[b_name]
                if len(a_shape) >= 2 and len(b_shape) >= 2:
                    m = math.prod(a_shape[:-1])
                    k = a_shape[-1]
                    n = b_shape[-1]
                    macs += m * k * n
        elif node.op_type in _REDUCE_OPS and node.input and node.input[0] in all_shapes:
            macs += math.prod(all_shapes[node.input[0]])
        elif node.op_type in _ELEM_OPS and node.output and node.output[0] in all_shapes:
            macs += math.prod(all_shapes[node.output[0]]) * _ELEM_OPS[node.op_type]
        elif node.op_type in _POOL_OPS and node.output and node.output[0] in all_shapes:
            kh, kw = 3, 3
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    ks = list(attr.ints)
                    if len(ks) >= 1:
                        kh = ks[0]
                    if len(ks) >= 2:
                        kw = ks[1]
            macs += math.prod(all_shapes[node.output[0]]) * kh * kw

    # May-6 scorer: cost = memory + params only (MACs excluded). MACs is still
    # computed above for diagnostic output but does NOT enter the cost formula.
    cost_raw = memory + params  # may be 0 for pure-shape graphs
    points = round(max(1.0, 25.0 - math.log(max(1.0, cost_raw))), 4)
    cost = cost_raw
    
    risk = cost_raw < 1000

    return {
        "path": str(raw_path),
        "macs": int(macs),
        "memory": int(memory),
        "params": int(params),
        "cost": int(cost),
        "points": points,
        "cascade_risk": risk,
        "ops": dict(sorted(op_counts.items())),
        "error": None,
    }


def _verify_against_score_network(result, onnx_path):
    """Call neurogolf_utils.score_network for delta; require data/ + venv. Add deltas to result."""
    import sys
    import types
    for n in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["IPython.display"].display = lambda *a, **kw: None
    sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
    sys.modules["IPython"].display = sys.modules["IPython.display"]
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

    DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
    sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
    import neurogolf_utils  # noqa: E402
    neurogolf_utils._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

    try:
        sys.path.insert(0, str(NG_ROOT / "tools"))
        from _score_v2 import score_network_v2  # noqa: E402
        sn_result = score_network_v2(str(onnx_path))
        if sn_result is None or sn_result[0] is None:
            result["score_network"] = {"error": "score_network rejected"}
            return
        sn_macs, sn_mem, sn_params = sn_result
        # 2026-05-04: cost = memory + params (MACs removed from objective).
        sn_cost = sn_mem + sn_params
        result["score_network"] = {
            "macs": int(sn_macs),
            "memory": int(sn_mem),
            "params": int(sn_params),
            "cost": int(sn_cost),
            "delta_macs": result["macs"] - int(sn_macs),
            "delta_memory": result["memory"] - int(sn_mem),
            "delta_params": result["params"] - int(sn_params),
            "delta_cost": result["cost"] - int(sn_cost),
            "delta_pct": round(100.0 * (result["cost"] - sn_cost) / max(1, sn_cost), 3),
        }
    except Exception as e:  # noqa: BLE001
        result["score_network"] = {"error": str(e)}


def _check_solved(verify=False):
    from inspect_tasks import SOLVED_TASKS  # noqa: E402

    print(f"Cost-prediction self-check on {len(SOLVED_TASKS)} SOLVED_TASKS")
    if verify:
        print("(verifying against neurogolf_utils.score_network)")
    print()
    print(f"{'task':<5} {'pred_cost':>10} {'pred_pts':>9}  ", end="")
    if verify:
        print(f"{'sn_cost':>10} {'sn_pts':>8}  {'d_cost':>9} {'d_pct':>7}", end="")
    print(f"  ops_compact")
    print("-" * (110 if verify else 60))

    total_dt = 0.0
    deltas = []
    for tn in sorted(SOLVED_TASKS):
        path = SUBMISSIONS_DIR / f"task{tn:03d}.onnx"
        t0 = time.perf_counter()
        r = predict(path)
        dt = time.perf_counter() - t0
        total_dt += dt
        if r.get("error"):
            print(f"task{tn:03d}  ERROR: {r['error']}")
            continue
        ops_str = ",".join(f"{k}{v}" if v > 1 else k for k, v in list(r["ops"].items())[:6])
        risk_marker = "⚠️ " if r["cascade_risk"] else "  "
        line = f"task{tn:03d}{risk_marker}{r['cost']:>10} {r['points']:>8.4f}  "
        if verify:
            _verify_against_score_network(r, path)
            sn = r.get("score_network", {})
            if "error" in sn:
                line += f"{'ERR':>10} {'-':>8}  {'-':>9} {'-':>7}"
            else:
                deltas.append(sn["delta_pct"])
                line += f"{sn['cost']:>10} {round(max(1.0, 25.0 - math.log(max(1, sn['cost']))), 4):>8.4f}  {sn['delta_cost']:>+9} {sn['delta_pct']:>+6.2f}%"
        print(line + f"  {ops_str}")
    print()
    print(f"Total predict() time: {total_dt*1000:.1f} ms ({total_dt*1000/len(SOLVED_TASKS):.2f} ms/task)")
    if verify and deltas:
        print(f"Cost delta vs score_network: avg {sum(deltas)/len(deltas):+.2f}%, max abs {max(abs(d) for d in deltas):.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", nargs="?", help="ONNX file path (omitted with --solved)")
    parser.add_argument("--solved", action="store_true", help="Predict for every SOLVED_TASK in submissions/")
    parser.add_argument("--verify", action="store_true", help="With --solved: cross-check vs neurogolf_utils.score_network")
    parser.add_argument("--json", action="store_true", help="Single-file mode: emit JSON instead of table")
    args = parser.parse_args()

    if args.solved:
        _check_solved(verify=args.verify)
        return

    if not args.path:
        parser.print_help()
        sys.exit(2)

    r = predict(args.path)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    if r.get("error"):
        print(f"ERROR: {r['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"path:    {r['path']}")
    print(f"macs:    {r['macs']:>12,}")
    print(f"memory:  {r['memory']:>12,}")
    print(f"params:  {r['params']:>12,}")
    print(f"cost:    {r['cost']:>12,}")
    print(f"points:  {r['points']:>12.4f}")
    if r["cascade_risk"]:
        print("WARNING: CASCADE-RISK (cost < 1000)")
    print(f"ops:     {r['ops']}")


if __name__ == "__main__":
    main()
