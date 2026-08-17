"""compress.py — SAFE, byte-equal cost compression of an existing ONNX graph.

Not out-solving — the team's own staging_log wins are lossless shrinks of graphs that already work
(share a duplicate initializer, drop a dead node, bake a counted constant into a free attribute).
This applies those transforms automatically to every Team_Best_Onnx, re-verifies the output is
byte-IDENTICAL to the original on a large ARC-GEN sample, and reports only graphs that get strictly
cheaper (via the canonical tools/_score_v2 scorer). Every transform preserves semantics exactly:

  dedup_initializers  — point every reference at one copy of each byte-identical initializer, drop the
                        rest (params charges element count per initializer, so duplicates are counted
                        twice; merging removes the copies). Output unchanged by construction.
  drop_unreachable    — remove nodes + their initializers not reachable from `output` (dead code).

Verification is double: (1) the scorer cost must strictly drop, (2) the compressed graph must reproduce
the ORIGINAL graph's decoded output on N ARC-GEN inputs (a drop-in replacement, not merely "correct").
"""
import copy
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from . import spec, score, argen

ort.set_default_logger_severity(3)


def dedup_initializers(model):
    """Merge byte-identical initializers (same dtype+dims+bytes). Returns a new model or None if none."""
    m = copy.deepcopy(model)
    g = m.graph
    canon, rename, keep = {}, {}, []
    for i in g.initializer:
        b = i.raw_data if i.raw_data else numpy_helper.to_array(i).tobytes()
        key = (i.data_type, tuple(i.dims), b)
        if key in canon:
            rename[i.name] = canon[key]
        else:
            canon[key] = i.name
            keep.append(i)
    if not rename:
        return None
    for n in g.node:
        n.input[:] = [rename.get(x, x) for x in n.input]
    del g.initializer[:]
    g.initializer.extend(keep)
    return m


def drop_unreachable(model):
    """Remove nodes + initializers not reachable backward from the graph outputs. New model or None."""
    m = copy.deepcopy(model)
    g = m.graph
    prod = {o: idx for idx, n in enumerate(g.node) for o in n.output}
    need = [o.name for o in g.output]
    reach = set()
    while need:
        nm = need.pop()
        if nm in prod and prod[nm] not in reach:
            idx = prod[nm]
            reach.add(idx)
            need.extend(g.node[idx].input)
    if len(reach) == len(g.node):
        return None
    kept_nodes = [n for idx, n in enumerate(g.node) if idx in reach]
    used = {x for n in kept_nodes for x in n.input}
    kept_inits = [i for i in g.initializer if i.name in used]
    del g.node[:]; g.node.extend(kept_nodes)
    del g.initializer[:]; g.initializer.extend(kept_inits)
    return m


TRANSFORMS = [("dedup", dedup_initializers), ("dropdead", drop_unreachable)]


def _decode_outputs(model, X):
    """Run model on batched X [n,10,30,30]; return the >0-decoded bool outputs [n,10,30,30] or None."""
    try:
        m = score._batch_model(model, X.shape[0])
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        r = np.asarray(sess.run([spec.OUT_NAME], {spec.IN_NAME: X})[0])
        return (r > 0.0) if r.shape == X.shape else None
    except Exception:
        return None


def verify_identical(orig, new, tid, n=512):
    """New graph must decode-identically to ORIG on n ARC-GEN inputs (drop-in replacement)."""
    s = argen.sample(tid, n)
    if s is None:
        return False
    X = s["X"]
    a, b = _decode_outputs(orig, X), _decode_outputs(new, X)
    if a is None or b is None:
        # fall back to per-sample (batched rewrite can fail on exotic graphs e.g. runtime-weight)
        return _persample_identical(orig, new, X)
    return bool(np.array_equal(a, b))


def _persample_identical(orig, new, X):
    try:
        so = ort.InferenceSession(orig.SerializeToString(), providers=["CPUExecutionProvider"])
        sn = ort.InferenceSession(new.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception:
        return False
    for i in range(X.shape[0]):
        try:
            ro = (np.asarray(so.run([spec.OUT_NAME], {spec.IN_NAME: X[i:i + 1]})[0]) > 0.0)
            rn = (np.asarray(sn.run([spec.OUT_NAME], {spec.IN_NAME: X[i:i + 1]})[0]) > 0.0)
        except Exception:
            return False
        if not np.array_equal(ro, rn):
            return False
    return True


def compress(model, tid, n_verify=512):
    """Apply the transform chain; return (best_model, orig_cost, new_cost, applied) if a strictly cheaper
    byte-identical graph is found, else None."""
    orig_cost, _ = score.exact_cost(model)
    cur, cur_cost, applied = model, orig_cost, []
    for name, fn in TRANSFORMS:
        cand = fn(cur)
        if cand is None:
            continue
        c, _ = score.exact_cost(cand)
        if c < cur_cost and verify_identical(model, cand, tid, n_verify):
            cur, cur_cost, applied = cand, c, applied + [name]
    if cur_cost < orig_cost:
        return cur, orig_cost, cur_cost, applied
    return None
