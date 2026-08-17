"""ops.py — the DSL: ops the search composes into ONNX graphs (now DAG-capable).

A graph state (GState) holds the SET of tensors built so far (tensors[0] is always the input).
Each Op declares `candidates(g)` — the bounded set of operand tuples it would try given the
current tensors — and `build(g, operands, pairs)` which appends node(s) and returns a new GState.
This lets the search build BRANCHES (e.g. a mask from input + a recolor from input) and merge them
(Where/Add/...), instead of a single straight pipe. Data-aware ops (ColorRemap) read the task pairs
to propose arguments. Downstream (builder/score/search/policy) stays vocabulary-agnostic.
"""
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
from onnx import helper, numpy_helper

from . import spec

MAX_REFS = 3          # operand pool = input + last MAX_REFS tensors (bounds the DAG branching factor)


@dataclass(frozen=True)
class TensorRef:
    name: str
    shape: tuple
    dtype: int


@dataclass(frozen=True)
class GState:
    tensors: tuple
    nodes: tuple = ()
    inits: tuple = ()
    ctr: int = 0

    @staticmethod
    def start():
        return GState(tensors=(TensorRef(spec.IN_NAME, spec.IN_SHAPE, spec.FLOAT),))

    @property
    def cur(self):
        return self.tensors[-1]

    @property
    def inp(self):
        return self.tensors[0]

    def _next(self):
        return f"t{self.ctr + 1}"


def pool(g):
    """Bounded operand candidates: the input + the last MAX_REFS tensors (deduped)."""
    sel = [g.tensors[0]] + list(g.tensors[-MAX_REFS:])
    out, seen = [], set()
    for r in sel:
        if r.name not in seen:
            seen.add(r.name); out.append(r)
    return out


def _emit(g, node, shape, dtype, inits=()):
    tr = TensorRef(node.output[0], tuple(shape), dtype)
    return replace(g, tensors=g.tensors + (tr,), nodes=g.nodes + (node,), inits=g.inits + tuple(inits), ctr=g.ctr + 1)


def _bcast(s1, s2):
    if len(s1) != len(s2):
        return None
    out = []
    for a, b in zip(s1, s2):
        if a == b or a == 1 or b == 1:
            out.append(max(a, b))
        else:
            return None
    return tuple(out)


class Op:
    label = "op"
    def candidates(self, g):
        return [(g.cur,)]                       # default: unary on the latest tensor
    def build(self, g, operands, pairs=None):
        raise NotImplementedError


def _unary_sources(g):
    return [g.cur] if g.cur.name == g.inp.name else [g.cur, g.inp]


# ----- unary structural ops -----------------------------------------------------------
class Transpose(Op):
    def __init__(self, perm):
        self.perm = tuple(perm); self.label = "transpose" + "".join(map(str, perm))
    def candidates(self, g):
        return [(t,) for t in _unary_sources(g)]
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if len(self.perm) != len(a.shape):
            return None
        node = helper.make_node("Transpose", [a.name], [g._next()], perm=list(self.perm))
        return _emit(g, node, tuple(a.shape[i] for i in self.perm), a.dtype)


class Cast(Op):
    def __init__(self, to):
        self.to = to; self.label = "cast_" + {spec.BOOL: "bool", spec.FLOAT16: "f16", spec.FLOAT: "f32"}.get(to, str(to))
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if a.dtype == self.to:
            return None
        return _emit(g, helper.make_node("Cast", [a.name], [g._next()], to=self.to), a.shape, self.to)


class GatherReverse(Op):
    def __init__(self, axis):
        self.axis = axis; self.label = f"flip{axis}"
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if self.axis >= len(a.shape):
            return None
        idx = numpy_helper.from_array(np.arange(a.shape[self.axis] - 1, -1, -1, dtype=np.int64), f"flip{self.axis}_{g.ctr}")
        node = helper.make_node("Gather", [a.name, idx.name], [g._next()], axis=self.axis)
        return _emit(g, node, a.shape, a.dtype, (idx,))


class _ReduceCh(Op):
    onnx_op = "ReduceSum"
    def candidates(self, g):
        return [(t,) for t in _unary_sources(g)]
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if len(a.shape) != 4 or a.shape[1] == 1 or a.dtype not in (spec.FLOAT, spec.FLOAT16, spec.INT32):
            return None
        axes = numpy_helper.from_array(np.array([1], dtype=np.int64), f"ax1_{g.ctr}")
        node = helper.make_node(self.onnx_op, [a.name, axes.name], [g._next()], keepdims=1)
        return _emit(g, node, (a.shape[0], 1, a.shape[2], a.shape[3]), a.dtype, (axes,))


class ReduceSumCh(_ReduceCh):
    onnx_op = "ReduceSum"; label = "reducesum_ch"


class ReduceMaxCh(_ReduceCh):
    onnx_op = "ReduceMax"; label = "reducemax_ch"


class Greater0(Op):
    label = "greater0"
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if a.dtype not in (spec.FLOAT, spec.FLOAT16, spec.INT32):
            return None
        npd = {spec.FLOAT: np.float32, spec.FLOAT16: np.float16, spec.INT32: np.int32}[a.dtype]
        z = numpy_helper.from_array(np.array(0, dtype=npd), f"z_{g.ctr}")
        return _emit(g, helper.make_node("Greater", [a.name, z.name], [g._next()]), a.shape, spec.BOOL, (z,))


class Conv3x3(Op):
    """Depthwise 3x3 same-pad convolution with a fixed kernel (per-channel neighbour aggregation).
    `ones` = 3x3 neighbour count (dilate/CA building block). Heavy op -> real spatial compute."""
    def __init__(self, kind="ones"):
        self.kind = kind; self.label = f"conv3x3_{kind}"
    def candidates(self, g):
        return [(t,) for t in _unary_sources(g)]
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if len(a.shape) != 4 or a.dtype not in (spec.FLOAT, spec.FLOAT16):
            return None
        c = a.shape[1]
        k = np.ones((c, 1, 3, 3), dtype=np.float32)
        if self.kind == "center0":
            k[:, :, 1, 1] = 0.0                       # neighbours only (exclude self)
        w = numpy_helper.from_array(k.astype(np.float32 if a.dtype == spec.FLOAT else np.float16), f"k_{g.ctr}")
        node = helper.make_node("Conv", [a.name, w.name], [g._next()], group=c, pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        return _emit(g, node, a.shape, a.dtype, (w,))


# ----- binary / merge ops (the DAG-combine core) --------------------------------------
class _Binary(Op):
    onnx_op = "Add"; commutative = True
    def candidates(self, g):
        outs = []
        for b in pool(g):
            if b.name == g.cur.name:
                continue
            if _bcast(g.cur.shape, b.shape) is None or g.cur.dtype != b.dtype:
                continue
            outs.append((g.cur, b))
            if not self.commutative:
                outs.append((b, g.cur))
        return outs
    def _out_dtype(self, a):
        return a.dtype
    def build(self, g, operands, pairs=None):
        a, b = operands
        sh = _bcast(a.shape, b.shape)
        if sh is None or a.dtype != b.dtype:
            return None
        node = helper.make_node(self.onnx_op, [a.name, b.name], [g._next()])
        return _emit(g, node, sh, self._out_dtype(a))


class Add(_Binary):
    onnx_op = "Add"; label = "add"


class Mul(_Binary):
    onnx_op = "Mul"; label = "mul"


class Max(_Binary):
    onnx_op = "Max"; label = "max"


class Min(_Binary):
    onnx_op = "Min"; label = "min"


class Sub(_Binary):
    onnx_op = "Sub"; label = "sub"; commutative = False


class Equal(_Binary):
    onnx_op = "Equal"; label = "equal"
    def _out_dtype(self, a):
        return spec.BOOL


class WhereZero(Op):
    """Where(cond, x, 0): keep x where the bool cond is true, else 0. The merge primitive — cond is any
    bool tensor, x is any tensor (a derived branch or the input). Generalizes the old mask-gate."""
    label = "where0"
    def candidates(self, g):
        conds = [t for t in pool(g) if t.dtype == spec.BOOL]
        xs = [t for t in pool(g) if t.dtype in (spec.FLOAT, spec.FLOAT16)]
        outs = []
        for cnd in conds:
            for x in xs:
                if _bcast(cnd.shape, x.shape) is not None:
                    outs.append((cnd, x))
        return outs
    def build(self, g, operands, pairs=None):
        cnd, x = operands
        sh = _bcast(cnd.shape, x.shape)
        if sh is None or cnd.dtype != spec.BOOL:
            return None
        npd = np.float32 if x.dtype == spec.FLOAT else np.float16
        z = numpy_helper.from_array(np.array(0, dtype=npd), f"wz_{g.ctr}")
        node = helper.make_node("Where", [cnd.name, x.name, z.name], [g._next()])
        return _emit(g, node, sh, x.dtype, (z,))


# ----- data-aware (arg-proposing) ops -------------------------------------------------
def _infer_color_perm(pairs):
    cmap = {}
    for ig, og in (pairs or []):
        ig = np.asarray(ig); og = np.asarray(og)
        if ig.shape != og.shape:
            return None
        for a, b in zip(ig.flatten().tolist(), og.flatten().tolist()):
            if a in cmap and cmap[a] != b:
                return None
            cmap[a] = b
    if not cmap or len(set(cmap.values())) != len(cmap):
        return None
    perm = [None] * 10; used = set()
    for a, b in cmap.items():
        if not (0 <= a < 10 and 0 <= b < 10):
            return None
        perm[b] = a; used.add(a)
    free = [c for c in range(10) if c not in used]
    for c in range(10):
        if perm[c] is None:
            perm[c] = free.pop(0)
    return perm


class ColorRemap(Op):
    label = "colorremap"
    def candidates(self, g):
        return [(g.inp,)]                       # only meaningful on the raw input one-hot
    def build(self, g, operands, pairs=None):
        a = operands[0]
        if a.name != spec.IN_NAME or a.shape != spec.IN_SHAPE:
            return None
        perm = _infer_color_perm(pairs)
        if perm is None or perm == list(range(10)):
            return None
        idx = numpy_helper.from_array(np.array(perm, dtype=np.int64), f"cperm_{g.ctr}")
        node = helper.make_node("Gather", [a.name, idx.name], [g._next()], axis=1)
        return _emit(g, node, a.shape, a.dtype, (idx,))


def _infer_separable(pairs, transpose):
    """Infer a per-axis coordinate map so out[r,c] == in[idx_r[r], idx_c[c]] across ALL pairs
    (optionally after a leading transpose). Returns (idx_r, idx_c, Hi, Wi) or None, where Hi,Wi is
    the constant input grid size (needed to tell a true axis no-op / crop from a full-canvas index).

    Method: with K example pairs, fingerprint each cell by its value-vector across pairs. A
    fingerprint shared by >1 input cell is a "wildcard" (cells that are always background-0) — it
    can't pin an axis but doesn't disprove a rearrangement, so it's skipped; a fingerprint absent
    from the input is a genuine MISS (an output value never present in the input -> not a pure
    rearrangement). The unique cells over-determine both axis maps. Const-geometry only (a static
    index vector can't track a grid whose size varies pair-to-pair)."""
    ins = [np.asarray(a) for a, _ in pairs]
    outs = [np.asarray(b) for _, b in pairs]
    if len({a.shape for a in ins}) != 1 or len({b.shape for b in outs}) != 1:
        return None
    if transpose:
        ins = [a.T for a in ins]
    Hi, Wi = ins[0].shape
    Ho, Wo = outs[0].shape
    if max(Hi, Wi, Ho, Wo) > spec.CANVAS or min(Hi, Wi, Ho, Wo) < 1:
        return None
    in_stack, out_stack = np.stack(ins), np.stack(outs)
    AMBIG = object()
    fp = {}
    for i in range(Hi):
        for j in range(Wi):
            key = in_stack[:, i, j].tobytes()
            fp[key] = AMBIG if key in fp else (i, j)
    idx_r, idx_c = [None] * Ho, [None] * Wo
    for r in range(Ho):
        for c in range(Wo):
            key = out_stack[:, r, c].tobytes()
            if key not in fp:
                return None                     # miss -> not a rearrangement
            v = fp[key]
            if v is AMBIG:
                continue                        # wildcard cell -> skip
            i, j = v
            if idx_r[r] is None:
                idx_r[r] = i
            elif idx_r[r] != i:
                return None
            if idx_c[c] is None:
                idx_c[c] = j
            elif idx_c[c] != j:
                return None
    if any(v is None for v in idx_r) or any(v is None for v in idx_c):
        return None
    # a pure no-op / pure transpose is expressed more cheaply by identity / the Transpose op
    if idx_r == list(range(Hi)) and idx_c == list(range(Wi)) and (Ho, Wo) == (Hi, Wi):
        return None
    return idx_r, idx_c, Hi, Wi


def _contig(idx):
    """(start, step) if idx is an arithmetic run with step in {+1,-1} (a Slice-able window), else None."""
    if len(idx) == 1:
        return (idx[0], 1)
    step = idx[1] - idx[0]
    if step not in (1, -1) or any(b - a != step for a, b in zip(idx, idx[1:])):
        return None
    return (idx[0], step)


def _slice_bounds(start, step, n, dim):
    """(starts, ends) for taking `n` elements from `start` by `step` along an axis of length `dim`.
    For a reverse (step -1) reaching index 0, ends uses the -(dim+1) sentinel ONNX needs to include 0."""
    if step == 1:
        return start, start + n
    return start, (start - n if start - n >= 0 else -(dim + 1))


class CoordRemap(Op):
    """Data-aware separable pixel rearrangement (const-geometry). Infers per-axis index vectors from
    the examples, then emits the CHEAPEST correct graph for the case (all re-embedded top-left in
    [1,10,30,30]; a Slice-as-output is grader-dead):
      A both axes reindexed & contiguous -> one 2-axis Slice + Pad     (flip / rot180 / crop)
      B exactly one axis reindexed        -> one canvas Gather(idx[30])  (single-axis mirror/shift; the
                                             untouched axis keeps the input, padding stays 0 -> no Pad)
      C both axes reindexed, not contiguous -> crop Slice + per-axis Gather/Slice + Pad  (mirror-tile)
    A leading Transpose handles axis-swaps (rot90). One op covers the whole flip/rot/crop/mirror family
    when the geometry is fixed across pairs (the ARC-GEN corpus makes the inference robust)."""
    label = "coordremap"

    def candidates(self, g):
        return [(g.inp,)] if len(g.nodes) == 0 else []      # only on the raw input

    def build(self, g, operands, pairs=None):
        a = operands[0]
        if a.name != spec.IN_NAME or a.shape != spec.IN_SHAPE or not pairs:
            return None
        CV = spec.CANVAS
        for tpose in (False, True):
            res = _infer_separable(pairs, tpose)
            if res is None:
                continue
            idx_r, idx_c, Hi, Wi = res
            Ho, Wo = len(idx_r), len(idx_c)
            gg, cur, salt = g, a, [0]

            def _iv(vals):
                salt[0] += 1
                return numpy_helper.from_array(np.array(vals, np.int64), f"crv{salt[0]}")

            def _emit_node(op_type, in_extra, out_shape, inits, **attrs):
                nonlocal gg, cur
                node = helper.make_node(op_type, [cur.name] + [t.name for t in in_extra],
                                        [gg._next()], **attrs)
                gg = _emit(gg, node, tuple(out_shape), cur.dtype, inits)
                cur = gg.cur

            if tpose:
                _emit_node("Transpose", [], (cur.shape[0], cur.shape[1], cur.shape[3], cur.shape[2]),
                           (), perm=[0, 1, 3, 2])
                Hi, Wi = Wi, Hi
            r_un = idx_r == list(range(Hi))
            c_un = idx_c == list(range(Wi))
            cr, cc = _contig(idx_r), _contig(idx_c)
            pad = True

            if not r_un and not c_un and cr and cc:                       # case A
                rs, re = _slice_bounds(*cr, Ho, CV)
                cs, ce = _slice_bounds(*cc, Wo, CV)
                ii = [_iv([rs, cs]), _iv([re, ce]), _iv([2, 3]), _iv([cr[1], cc[1]])]
                _emit_node("Slice", ii, (cur.shape[0], cur.shape[1], Ho, Wo), ii)
            elif r_un != c_un:                                            # case B
                axis, idx, gdim = (2, idx_r, Hi) if c_un else (3, idx_c, Wi)
                zero = CV - 1 if gdim < CV else (0 if len(idx) == CV else CV - 1)
                it = _iv(list(idx) + [zero] * (CV - len(idx)))
                sh = list(cur.shape); sh[axis] = CV
                _emit_node("Gather", [it], sh, (it,), axis=axis)
                pad = False
            else:                                                         # case C
                ii = [_iv([0, 0]), _iv([Hi, Wi]), _iv([2, 3]), _iv([1, 1])]
                _emit_node("Slice", ii, (cur.shape[0], cur.shape[1], Hi, Wi), ii)
                for axis, idx, cont, dim in ((2, idx_r, cr, Hi), (3, idx_c, cc, Wi)):
                    if idx == list(range(dim)):
                        continue
                    sh = list(cur.shape); sh[axis] = len(idx)
                    if cont:
                        s0, e0 = _slice_bounds(*cont, len(idx), dim)
                        ii = [_iv([s0]), _iv([e0]), _iv([axis]), _iv([cont[1]])]
                        _emit_node("Slice", ii, sh, ii)
                    else:
                        it = _iv(list(idx))
                        _emit_node("Gather", [it], sh, (it,), axis=axis)

            if pad and (Ho, Wo) != (CV, CV):
                pads = _iv([0, 0, 0, 0, 0, 0, CV - Ho, CV - Wo])
                _emit_node("Pad", [pads], (cur.shape[0], cur.shape[1], CV, CV), (pads,), mode="constant")
            return gg
        return None


def base_vocab():
    return [
        ColorRemap(),
        CoordRemap(),
        Transpose((0, 1, 3, 2)),
        ReduceSumCh(), ReduceMaxCh(), Greater0(),
        Conv3x3("ones"), Conv3x3("center0"),
        Add(), Sub(), Mul(), Max(), Min(), Equal(), WhereZero(),
        Cast(spec.BOOL), Cast(spec.FLOAT16),
        GatherReverse(2), GatherReverse(3),
    ]
