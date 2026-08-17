"""fit.py — fit-based solvers: derive a small ONNX operator's parameters from the ARC-GEN examples
by least-squares, discretize to ternary, and verify with the official per-channel `>0` gate.

This is the generalizable half of "cheap Einsum/Conv dominate the cheap tier" (Agent research): the
team's sub-100 graphs are mostly a single op with a SMALL LEARNED tensor. We can recover many of them
generally by FITTING that tensor to the corpus instead of hand-crafting per task.

Implemented families (cheapest-first):
  * color-map Einsum  `nchw,dc->ndhw`  — general recolor incl. many-to-one merges (bijections are
    already cheaper via ColorRemap's Gather, so this only fires for non-bijective maps).
  * depthwise Conv    `[10,1,kh,kw]`   — per-colour neighbourhood rule (dilation/edge/shift). Fitting
    over ALL windows (incl. the 0-padded boundary) learns correct boundary behaviour, so unlike a blind
    kernel sweep it does not bleed into the padding.
  * cross-channel Conv `[10,10,kh,kw]` — general local rule mixing colours (higher cost floor 900).

Verification is always the harness contract: `(Conv/Einsum(input) > 0) == encode(output)` on held-out
ARC-GEN pairs (the argen truth gate). Numpy-only fit; runs in the project venv.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper

from . import spec, score, argen

CV, C = spec.CANVAS, spec.CHANNELS


# ---- least-squares helpers ---------------------------------------------------------------
def _windows(oh, kh, kw):
    """One-hot [ch,30,30] -> im2col [30*30, ch*kh*kw] of kh×kw neighbourhoods (0-padded)."""
    ch = oh.shape[0]
    ph, pw = kh // 2, kw // 2
    P = np.pad(oh, ((0, 0), (ph, ph), (pw, pw)))
    cols = []
    for dr in range(kh):
        for dc in range(kw):
            cols.append(P[:, dr:dr + CV, dc:dc + CV])          # [ch,30,30]
    return np.stack(cols, -1).reshape(ch, CV * CV, kh * kw).transpose(1, 0, 2).reshape(CV * CV, ch * kh * kw)


def _ternarize(w, thr=0.3):
    t = np.zeros_like(w)
    t[w > thr] = 1.0
    t[w < -thr] = -1.0
    return t


def _encode_batch_oh(pairs):
    """pairs of grids -> (Xoh [n,C,30,30], Yoh [n,C,30,30]) one-hot, via spec.encode."""
    X = np.concatenate([spec.encode(a) for a, _ in pairs], 0)
    Y = np.concatenate([spec.encode(b) for _, b in pairs], 0)
    return X, Y


# ---- fitters (return an onnx.ModelProto or None) -----------------------------------------
def fit_color_map(pairs):
    """Non-bijective cell-to-cell recolor -> Einsum `nchw,dc->ndhw`, W[10,10] binary (cost 100)."""
    m = {}
    for a, b in pairs:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return None
        for x, y in zip(a.flatten().tolist(), b.flatten().tolist()):
            if x in m and m[x] != y:
                return None
            m[x] = y
    if not m or len(set(m.values())) == len(m):
        return None                                            # empty or bijection (ColorRemap does it)
    W = np.zeros((C, C), np.float32)
    for c in range(C):
        W[m.get(c, c), c] = 1.0
    Wt = numpy_helper.from_array(W, "W")
    node = helper.make_node("Einsum", [spec.IN_NAME, "W"], [spec.OUT_NAME], equation="nchw,dc->ndhw")
    return _model([node], [Wt])


def fit_conv(pairs, depthwise=True, ksizes=((1, 1), (3, 1), (1, 3), (3, 3))):
    """Fit a Conv kernel by lstsq over all windows, ternarize, and (caller) verify.
    depthwise: per-colour [10,1,kh,kw] (cheap); else cross-channel [10,10,kh,kw] (floor 900).
    Returns list of candidate models (cheapest ksize first)."""
    X, Y = _encode_batch_oh(pairs)                             # [n,C,30,30]
    n = X.shape[0]
    Yb = (Y > 0).astype(np.float32)
    out = []
    for kh, kw in ksizes:
        if depthwise:
            W = np.zeros((C, 1, kh, kw), np.float32)
            for c in range(C):
                A = np.concatenate([_windows(X[i, c:c + 1], kh, kw) for i in range(n)], 0)  # [n*900, kh*kw]
                B = Yb[:, c].reshape(-1)                        # [n*900]
                sol, *_ = np.linalg.lstsq(A, B, rcond=None)
                W[c, 0] = _ternarize(sol.reshape(kh, kw))
            grp = C
        else:
            A = np.concatenate([_windows(X[i], kh, kw) for i in range(n)], 0)               # [n*900, C*kh*kw]
            W = np.zeros((C, C, kh, kw), np.float32)
            for d in range(C):
                B = Yb[:, d].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, B, rcond=None)
                W[d] = _ternarize(sol.reshape(C, kh, kw))
            grp = 1
        if not W.any():
            continue
        weight = numpy_helper.from_array(W, "K")
        node = helper.make_node("Conv", [spec.IN_NAME, "K"], [spec.OUT_NAME],
                                group=grp, pads=[kh // 2, kw // 2, kh // 2, kw // 2], kernel_shape=[kh, kw])
        out.append(_model([node], [weight]))
    return out


def _model(nodes, inits):
    g = helper.make_graph(nodes, "fit",
                          [helper.make_tensor_value_info(spec.IN_NAME, spec.FLOAT, list(spec.IN_SHAPE))],
                          [helper.make_tensor_value_info(spec.OUT_NAME, spec.FLOAT, list(spec.IN_SHAPE))], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
    return m


# ---- solver (longrun cascade compatible) -------------------------------------------------
def solve_fit(tid, pairs=None, infer=None, verify=None, budget_s=None, n_fit=24, n_gate=512):
    """Try the fit families cheapest-first; return the cheapest argen-confirmed result or None."""
    fp = argen.load_grids(tid, n_fit)
    if not fp:
        return None
    cands = []
    cm = fit_color_map(fp)
    if cm is not None:
        cands.append(cm)
    cands += fit_conv(fp, depthwise=True)
    cands += fit_conv(fp, depthwise=False)
    best = None
    for m in cands:
        try:
            onnx.checker.check_model(m)
        except Exception:
            continue
        p, t = score.argen_gate(m, tid, n=n_gate)
        if t > 0 and p == t:
            cost, _ = score.exact_cost(m)
            if best is None or cost < best["cost"]:
                op = m.graph.node[0].op_type
                best = {"cost": cost, "points": spec.points(cost),
                        "label": f"fit[{op}]", "method": "fit", "argen": (p, t)}
    return best


def scan(lo, hi, n_fit=24):
    hits = []
    for tid in range(lo, hi + 1):
        if not argen.available(tid):
            continue
        r = solve_fit(tid, n_fit=n_fit)
        if r:
            hits.append((tid, r))
            print(f"  task{tid:03d}: {r['label']} cost={r['cost']} pts={r['points']:.2f} argen {r['argen'][0]}/{r['argen'][1]}", flush=True)
    print(f"\nFIT scan {lo}-{hi}: {len(hits)} solve(s)")
    return hits


if __name__ == "__main__":
    import sys
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (1, 100)
    scan(lo, hi)
