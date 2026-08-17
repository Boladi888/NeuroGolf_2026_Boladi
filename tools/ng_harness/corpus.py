"""Cached, deterministic ARC-GEN corpus — generate once, every agent reuses.

ARC-GEN generate() is deterministic given random.seed(SEED_OFFSET + raw_id), so the
corpus is identical across agents/runs. We cache compact uint8 color-index grids
(input+output) per task on the dedicated cache drive and expand to one-hot on load.

Cache file: <NEUROGOLF_ARGEN_CACHE>/taskNNN.npz
  in_idx [V,30,30] u8, in_shapes [V,2] u16, out_idx, out_shapes, n_raw (i64), seed_offset
where V = number of VALID (<=30x30) pairs, n_raw = raw example_ids consumed to get them.
Lazy + incremental: extends the cache (continuing the deterministic seed stream) when
more pairs are requested than cached.
"""
from __future__ import annotations

import os
import pathlib
import random
import shutil

import numpy as np

from . import encode, registry

SEED_OFFSET = 2026          # canonical — fixes the shared corpus across all agents
MAX_N = 200_000             # per-task cap (covers the heaviest sweeps)
_MIN_FREE_GB = 2.0          # disk guard

CACHE_DIR = pathlib.Path(os.environ.get(
    "NEUROGOLF_ARGEN_CACHE", r"Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache"))


def _cache_path(task_num: int) -> pathlib.Path:
    return CACHE_DIR / f"task{task_num:03d}.npz"


def _load(task_num: int):
    p = _cache_path(task_num)
    if not p.exists():
        return None
    try:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    except Exception:  # noqa: BLE001
        return None


def _free_gb(path: pathlib.Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except Exception:  # noqa: BLE001
        return float("inf")


def _save(task_num: int, d: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _free_gb(CACHE_DIR) < _MIN_FREE_GB:
        raise RuntimeError(f"corpus cache drive low on space (<{_MIN_FREE_GB} GB free at {CACHE_DIR})")
    tmp = CACHE_DIR / f"task{task_num:03d}.tmp.npz"
    with open(tmp, "wb") as f:        # file handle -> numpy won't munge the name
        np.savez_compressed(f, **d)
    tmp.replace(_cache_path(task_num))


def build(task_num: int, want: int, verbose: bool = False,
          max_seconds: float | None = None, checkpoint_every: int = 25_000) -> dict:
    """Ensure the cache holds >= `want` valid pairs; return the arrays dict.
    Generates/extends deterministically. Capped at MAX_N.

    Robust to interruption + pathologically slow generators:
      - checkpoints the cache every `checkpoint_every` valid pairs, so a kill /
        window-close / crash never loses more than that many pairs (resume = re-run).
      - `max_seconds` budget: if exceeded, saves the partial corpus and returns
        (valid < want) instead of running for hours. Some ARC-GEN generators are
        ~3 pairs/sec (200k = ~19h); a budget caps them at a usable partial."""
    import time as _t
    want = min(int(want), MAX_N)
    d = _load(task_num)
    in_idx = list(d["in_idx"]) if d else []
    in_sh = list(d["in_shapes"]) if d else []
    out_idx = list(d["out_idx"]) if d else []
    out_sh = list(d["out_shapes"]) if d else []
    n_raw = int(d["n_raw"]) if d else 0
    valid = len(in_idx)
    if valid >= want:
        return d
    gen, err = registry.generator(task_num)
    if err:
        raise RuntimeError(f"task{task_num:03d} generator unavailable: {err}")

    def _pack():
        return {
            "in_idx": np.asarray(in_idx, dtype=np.uint8),
            "in_shapes": np.asarray(in_sh, dtype=np.uint16),
            "out_idx": np.asarray(out_idx, dtype=np.uint8),
            "out_shapes": np.asarray(out_sh, dtype=np.uint16),
            "n_raw": np.int64(n_raw),
            "seed_offset": np.int64(SEED_OFFSET),
        }

    # Robust per-seed loop: seed = SEED_OFFSET + raw_id (same mapping as before, so
    # determinism is preserved), but a seed whose generator RAISES (upstream ARC-GEN
    # bug on a rare RNG state) or yields a >30x30 grid is SKIPPED (still counted in
    # n_raw). Checkpoint periodically; honor a time budget; never run unboundedly.
    skipped_exc = skipped_big = 0
    max_raw = n_raw + (want - valid) * 8 + 50_000
    t0 = _t.time(); last_ckpt = valid
    while valid < want:
        if max_seconds is not None and (_t.time() - t0) > max_seconds:
            if verbose:
                print(f"  task{task_num:03d}: time budget {max_seconds}s hit at {valid}/{want}", flush=True)
            break
        if n_raw >= max_raw:
            if verbose:
                print(f"  task{task_num:03d}: max_raw hit at {valid}/{want} "
                      f"(exc={skipped_exc}, oversize={skipped_big})", flush=True)
            break
        random.seed(SEED_OFFSET + n_raw)
        n_raw += 1
        try:
            ex = gen()
        except Exception:  # noqa: BLE001 -- upstream generator crash on this seed
            skipped_exc += 1
            continue
        if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
            skipped_exc += 1
            continue
        ii, ish = encode.grid_to_index(ex["input"])
        oi, osh = encode.grid_to_index(ex["output"])
        if ii is None or oi is None:
            skipped_big += 1            # >30x30 (matches official convert_to_numpy)
            continue
        in_idx.append(ii); in_sh.append(ish); out_idx.append(oi); out_sh.append(osh)
        valid += 1
        if valid - last_ckpt >= checkpoint_every:   # crash/interrupt-safe checkpoint
            _save(task_num, _pack()); last_ckpt = valid
            if verbose:
                print(f"  task{task_num:03d}: checkpoint {valid}/{want} ({n_raw} raw)", flush=True)
    _save(task_num, _pack())
    return _pack()


def get_tensors(task_num: int, n: int, verbose: bool = False):
    """Return (X [n,10,30,30] f32, Y [n,10,30,30] f32) input+output one-hot.
    Builds/extends the cache as needed. NOTE: materializes all n one-hot tensors
    (~36 KB/pair*2); for large n (>~20k) use iter_tensors to bound memory."""
    d = build(task_num, n, verbose=verbose)
    n = min(n, len(d["in_idx"]))
    X = encode.encode_batch_index(d["in_idx"][:n], d["in_shapes"][:n])
    Y = encode.encode_batch_index(d["out_idx"][:n], d["out_shapes"][:n])
    return X, Y


def iter_tensors(task_num: int, n: int, batch: int = 4096, verbose: bool = False):
    """Stream (X_chunk, Y_chunk, start_idx) one-hot batches — bounds peak memory.
    A 200k truth/ab sweep one-shot would be ~7 GB of f32 one-hot; this keeps it to
    ~batch*72 KB. The compact uint8 corpus stays resident (uint8, ~1.8 KB/pair)."""
    d = build(task_num, n, verbose=verbose)
    n = min(n, len(d["in_idx"]))
    for s in range(0, n, batch):
        e = min(s + batch, n)
        X = encode.encode_batch_index(d["in_idx"][s:e], d["in_shapes"][s:e])
        Y = encode.encode_batch_index(d["out_idx"][s:e], d["out_shapes"][s:e])
        yield X, Y, s


def cache_info(task_num: int) -> dict:
    d = _load(task_num)
    p = _cache_path(task_num)
    if d is None:
        return {"task": task_num, "cached": 0, "exists": False}
    return {"task": task_num, "cached": int(len(d["in_idx"])), "n_raw": int(d["n_raw"]),
            "exists": True, "size_mb": round(p.stat().st_size / 1e6, 2),
            "seed_offset": int(d["seed_offset"])}
