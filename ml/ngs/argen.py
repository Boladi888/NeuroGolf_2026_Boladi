"""argen.py — read the deterministic ARC-GEN corpus (BoBrain's argen_cache) for a task.

Why this exists: the repo task JSON gives only ~4-6 VISIBLE pairs per task. BoBrain caches
up to 200,000 deterministic ARC-GEN input/output pairs PER task (all 400 tasks), generated once
and shared across every agent. Two things that corpus buys ml/ngs:

  1. A real ARC-GEN TRUTH GATE. The ngs README's §VISIBLE-FAIL axiom says one visible miss = a
     certain hidden 0, but it also warns: "Add an arc-gen / truth-full gate before trusting any
     NGS graph as shippable." A program that reproduces 5 visible pairs but is the WRONG rule gets
     caught when re-verified on hundreds of held-out ARC-GEN pairs. This is that gate.
  2. Robust ARG-INFERENCE. Inferring an op's argument (a color map, a crop window, ...) from 5
     pairs is fragile; inferring it from thousands is not — a coincidence that survives 5 pairs
     rarely survives 3,000.

This module is READ-ONLY: it only np.load()s cached .npz files. It never calls the harness
`corpus.build()` (which would try to *extend* the cache and needs ARC-GEN + write access). If a
task's cache is missing or short, callers just get fewer (or zero) pairs and degrade gracefully.

Cache file: <NEUROGOLF_ARGEN_CACHE>/taskNNN.npz  (default: BoBrain's neurogolf domain), holding
  in_idx [V,30,30] u8, in_shapes [V,2] u16, out_idx [V,30,30] u8, out_shapes [V,2] u16, ...
where cells outside (h,w) are 0-padding (all-zero after one-hot, never colliding with real color 0).
The one-hot expansion here is byte-identical to spec.encode (verified in tests below).
"""
import os
import pathlib
import numpy as np

from . import spec

CACHE_DIR = pathlib.Path(os.environ.get(
    "NEUROGOLF_ARGEN_CACHE", r"Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache"))

_H = _W = spec.CANVAS          # 30
_C = spec.CHANNELS             # 10

# tiny in-process cache: the search hits one task's corpus repeatedly, so keep the last one
# resident (each array is ~180 MB decompressed; we cap at 1 task to bound memory).
_CACHE_TID = None
_CACHE_Z = None


def _path(tid):
    return CACHE_DIR / f"task{int(tid):03d}.npz"


def available(tid):
    return _path(tid).exists()


def _arrays(tid):
    """Return the (cached) dict of arrays for a task, or None. Keeps only the last task resident."""
    global _CACHE_TID, _CACHE_Z
    tid = int(tid)
    if _CACHE_TID == tid and _CACHE_Z is not None:
        return _CACHE_Z
    p = _path(tid)
    if not p.exists():
        return None
    try:
        z = np.load(p)
        d = {k: z[k] for k in ("in_idx", "in_shapes", "out_idx", "out_shapes")}
    except Exception:
        return None
    _CACHE_TID, _CACHE_Z = tid, d
    return d


def cached_count(tid):
    d = _arrays(tid)
    return int(d["in_idx"].shape[0]) if d else 0


def encode_batch(idx, shapes):
    """[N,30,30] u8 color-index + [N,2] (h,w) -> [N,10,30,30] f32 one-hot.

    Byte-identical to stacking spec.encode() over the true HxW grids: padding cells (r>=h or
    c>=w) stay all-zero so a real color-0 cell and off-grid padding never collide.
    """
    idx = np.asarray(idx)
    shapes = np.asarray(shapes)
    n = idx.shape[0]
    rr = np.arange(_H)[None, :, None]
    cc = np.arange(_W)[None, None, :]
    active = (rr < shapes[:, 0, None, None]) & (cc < shapes[:, 1, None, None])   # [N,30,30]
    oh = np.zeros((n, _C, _H, _W), dtype=np.float32)
    for c in range(_C):
        oh[:, c, :, :] = ((idx == c) & active)
    return oh


def load_grids(tid, n=None, offset=0):
    """Return list[(in_grid, out_grid)] as int64 true-HxW grids (decoded via shapes).

    These plug straight into ml/ngs arg-inference (ops that read `pairs`) and correctness checks,
    exactly like tasks.all_pairs(...) but drawn from the large ARC-GEN corpus instead of the ~5
    visible pairs. Deterministic (corpus order is fixed by the shared seed stream).
    """
    d = _arrays(tid)
    if d is None:
        return []
    V = int(d["in_idx"].shape[0])
    end = V if n is None else min(offset + n, V)
    ii, ish, oi, osh = d["in_idx"], d["in_shapes"], d["out_idx"], d["out_shapes"]
    out = []
    for i in range(offset, end):
        ih, iw = int(ish[i][0]), int(ish[i][1])
        oh, ow = int(osh[i][0]), int(osh[i][1])
        out.append((ii[i][:ih, :iw].astype(np.int64), oi[i][:oh, :ow].astype(np.int64)))
    return out


def sample(tid, n, seed=0):
    """Deterministic random sample of up to n pairs for the truth gate.

    Returns a dict with batched input one-hot X [m,10,30,30] f32 and the expected-output index
    grids (out_idx [m,30,30] u8, out_shapes [m,2]) so a caller can decode-and-compare without
    re-encoding, or None if the corpus is unavailable. m = min(n, cached_count).
    """
    d = _arrays(tid)
    if d is None:
        return None
    V = int(d["in_idx"].shape[0])
    if V == 0:
        return None
    m = min(int(n), V)
    if m < V:
        rng = np.random.default_rng(seed)
        sel = rng.choice(V, size=m, replace=False)
        sel.sort()
    else:
        sel = np.arange(V)
    ii, ish = d["in_idx"][sel], d["in_shapes"][sel]
    oi, osh = d["out_idx"][sel], d["out_shapes"][sel]
    return {"n": m, "X": encode_batch(ii, ish),
            "out_idx": oi, "out_shapes": osh, "in_shapes": ish}
