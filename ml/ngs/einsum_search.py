"""einsum_search.py — the cost-0 method: search zero-parameter Einsum equations.

Why zero-param Einsum is THE lever for cost-0/1 tasks (points = 25 = the ceiling): an Einsum whose
operands are all copies of the input one-hot carries NO initializers, and its output IS the graph
output (both input and output are exempt from the cost) -> cost 0. A surprising range of transforms
is expressible this way (task067 ships one: `abcd,aedf->abcd`, a self-gating contraction), and finding
the right equation is a *search* over a discrete grammar — exactly what ML/guided search is good at.

This module enumerates a bounded grammar (axis permutations + the 1..k-operand self-gating family that
covers task067), pre-filters each equation NUMERICALLY on a batch of ARC-GEN pairs (fast: one np.einsum
over the whole batch, `>0` decode, byte-compare), then confirms survivors with the official ONNX path +
the argen truth gate and reports cost (0 for a pure self-Einsum). It plugs into `longrun`'s solver
cascade as the `einsum` stage.

Grammar is intentionally small (v1) so the reach is honest and the run is fast; it is the seed the
"long run" grows (more operands / label patterns / a mixed-radix batched sweep — see longrun/batched).
"""
import itertools
import numpy as np
import onnx
from onnx import helper

from . import spec, score, argen

OUT = "abcd"                       # output axes: a=batch(1) b=channel(10) c=H(30) d=W(30)
_SIZES = {"a": 1, "b": spec.CHANNELS, "c": spec.CANVAS, "d": spec.CANVAS}


def _perm_equations():
    """Single-operand permutations that keep the output a valid [1,10,30,30]."""
    eqs = []
    for perm in ("abcd", "abdc"):         # identity, transpose(H,W). others change axis sizes -> invalid
        eqs.append(f"abcd->{perm}")
    return eqs


def _gating_equations():
    """Two-operand self-gating: out[a,b,c,d] = in[a,b,c,d] * (a contraction of a 2nd input copy).
    op2's 4 axes (sizes 1,10,30,30) each either BIND to a same-size output label or CONTRACT (fresh
    label). Sizes: op2 ch(10)->{b|contract}; op2 H(30),W(30)->{c|d|contract}. Covers task067
    (`abcd,aedf->abcd`: op2 H binds to output W, ch+W contract)."""
    fresh = iter("efghijkl")
    eqs = set()
    ch_opts = ["b", "e"]                    # bind channel, or contract
    hw_opts = ["c", "d", "f"], ["c", "d", "g"]
    for ch, h2, w2 in itertools.product(ch_opts, hw_opts[0], hw_opts[1]):
        op2 = "a" + ch + h2 + w2
        if op2 == "abcd":
            continue                        # in*in == in for a one-hot (identity) — skip
        eqs.add(f"abcd,{op2}->abcd")
    # op1 transposed variant (gate the transpose): out is still abcd but op1 reads abdc
    for e in list(eqs):
        body = e.split("->")[0]
        eqs.add(f"abdc,{body.split(',')[1]}->abcd")
    return sorted(eqs)


def _gating3_equations():
    """Three-operand self-gating: out = in * stat_A * stat_B (row-count AND col-count). Measured to add
    ZERO reach over the 2-operand family AND to build explosive numpy intermediates — kept for reference
    but NOT in the default grammar. Zero-param Einsum is structurally limited to masking/permutation, so
    this is near the reach ceiling; growing cost-0 reach needs a different lever (see LONGRUN.md)."""
    row, col = ["aefc", "aefd"], ["aecf", "aedf"]
    return sorted({f"abcd,{op2},{op3}->abcd" for op2 in row for op3 in col})


def all_equations():
    # 2-operand self-gating + permutations. (3-operand gating adds no solves; excluded — see above.)
    return _perm_equations() + _gating_equations()


def _numpy_ok(eq, X, Y):
    """Vectorized pre-filter: run the einsum over the whole batch (a=batch), >0 decode, byte-compare.
    X,Y: [N,10,30,30] input/expected one-hot. Returns True iff every sample matches exactly."""
    k = eq.count(",") + 1
    try:
        out = np.einsum(eq, *([X] * k), optimize=True)   # optimize avoids explosive intermediates
    except Exception:
        return False
    if out.shape != Y.shape:
        return False
    pred = (out > 0.0)
    return bool(np.all(pred == (Y > 0.0)))


def _build_model(eq):
    k = eq.count(",") + 1
    node = helper.make_node("Einsum", [spec.IN_NAME] * k, [spec.OUT_NAME], equation=eq)
    graph = helper.make_graph(
        [node], "einsum0",
        inputs=[helper.make_tensor_value_info(spec.IN_NAME, spec.FLOAT, list(spec.IN_SHAPE))],
        outputs=[helper.make_tensor_value_info(spec.OUT_NAME, spec.FLOAT, list(spec.IN_SHAPE))])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    return m


def solve_einsum(tid, pairs=None, infer=None, verify=None, budget_s=None, n_filter=64, n_gate=512):
    """Search zero-param Einsum equations for a cost-0 solve of `tid`. Returns a longrun result
    dict {cost, points, label, method, argen} or None. Pre-filters on ARC-GEN pairs (numpy), then
    confirms the winner with the official ONNX path + the argen gate."""
    s = argen.sample(tid, n_filter)
    if s is None:
        return None
    X = s["X"]
    Y = argen.encode_batch(s["out_idx"], s["out_shapes"])
    for eq in all_equations():
        if not _numpy_ok(eq, X, Y):
            continue
        model = _build_model(eq)
        try:
            onnx.checker.check_model(model)
        except Exception:
            continue
        p, t = score.argen_gate(model, tid, n=n_gate)
        if t > 0 and p == t:
            cost, kind = score.exact_cost(model)
            return {"cost": cost, "points": spec.points(cost), "label": f"einsum[{eq}]",
                    "method": "einsum", "argen": (p, t)}
    return None


def scan(lo, hi, n_filter=64):
    """CLI: which tasks does the zero-param Einsum grammar solve (and at what cost)?"""
    hits = []
    for tid in range(lo, hi + 1):
        if not argen.available(tid):
            continue
        r = solve_einsum(tid, n_filter=n_filter)
        if r:
            hits.append((tid, r))
            print(f"  task{tid:03d}: {r['label']} cost={r['cost']} pts={r['points']:.2f} argen {r['argen'][0]}/{r['argen'][1]}", flush=True)
    print(f"\nEINSUM scan {lo}-{hi}: {len(hits)} solve(s); "
          f"{sum(1 for _, r in hits if r['cost'] <= 1)} at cost<=1")
    return hits


if __name__ == "__main__":
    import sys
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (1, 400)
    print(f"zero-param Einsum grammar: {len(all_equations())} equations")
    scan(lo, hi)
