# Transferable insight — structural rewrite by (a) INT8 detection domain + (b) reuse LIVE's render tail

**Source**: task090 batch 2026-07-08 (`bo_ng_cld_worker3`). LIVE 2752 → staged 2465 (−287 B, +0.110 pts), visible 267/267, truth 99.994% (more-correct-than-LIVE, DUAL-TRUTH shippable).

## Lever 1 — split the graph into a UINT32 "bit domain" and an INT8 "arithmetic domain"
For tasks that bit-pack a mask and then do integer selection (largest-rectangle, run-length, coordinate pick), ORT's kernel coverage forces a dtype split:
- **Bit ops** (`BitShift`, `BitwiseAnd/Or`, `Greater`) need **UINT32** for 30-bit masks — fine, ORT has these.
- **Arithmetic/selection** (`Mul`, `ArgMax`, `Where`, `Add`, `Sub`) has **NO uint32 kernel** → must leave U32. Do them in **INT8** when every value <128 (areas ≤ H·Wmax, coords ≤ 29). 
- This is not just legality — it is the COST lever: memory = Σ elements×dtype_bytes, so INT8 intermediates are **4× cheaper than INT32**. A per-width maxrun accumulation (`Cast`+`Add` per width over `[ng]` spans) in INT32 scored **3290 (WORSE than the 2752 LIVE)**; the identical graph in INT8 scored **2465**. INT8 was the whole win.
- **ORT gotchas proven this session**: no uint32 `Mul`/`ArgMax`/`Where`; no int8/int16 `ReduceSum`; `Gather` **indices** must be int32/int64 (int8 index → INVALID_GRAPH); `ArgMax` OK on int8/uint8/int32 but NOT int16. Params count ELEMENTS (dtype-agnostic) so INT8 helps memory, not params — dedup duplicate int64 slice-bound initializers to cut params (−27 here).

## Lever 2 — reuse the LIVE render tail; only rebuild the cheap-to-beat front-end
A "STRUCTURAL from-scratch rewrite" does NOT mean rebuild the whole graph. When cost is concentrated in ONE stage (here: detection ~2590 of 2752; the Einsum render was only ~160), rebuild that stage and **splice the proven tail**:
1. Instrument LIVE: add its render-input tensors as extra graph outputs, run on visible+corpus, and read off the **exact interface contract** (task090: render consumes 4 scalars `cr=2y0+H−1, cx=2x0+W−1, h+1, w+1` + `input`).
2. Build the new front-end to emit those SAME tensors; **verify the front-end's outputs equal a numpy oracle on 200k** before wiring the tail.
3. Copy the tail nodes + its initializer VALUES verbatim (dump via `numpy_helper.to_array`). Render is then byte-identical to LIVE by construction — you inherit its correctness AND its limitations for free (here both undershoot large-rectangle corners identically → not a regression).

## Lever 3 — the generator file is the spec; size caps to the cached corpus, not a sample
`external/ARC-GEN/tasks/task_*.py` is ground truth. task090's docs/prompt claimed a "canonical shape set (W≤8)"; the generator's `find_most` is a **generic** largest-rectangle (any W up to grid width; made unique by rejection). A shape-capped detector passed 267 visible but diverged on random inputs. Correct move: read the generator, then measure per-parameter maxima over the **exact cached 200k truth corpus** (`Q:\…\argen_cache\taskNNN.npz`), not a random re-sample. Per-height width caps = corpus max (h2≤15,…) give provable 100% truth via "underestimating a NON-winner's run only lowers its area (safe); the winner of height h has W ≤ corpus-max(h) so is never capped."

## Bonus — verify against the throttle-serialized gate with your own closed-form corpus run
Full `ng_gate truth/ab` is heavy-gate serialized (one slot across all agents) and timed out repeatedly under contention. Loading the cached `.npz` and running the candidate over it yourself (binarize `>0`, compare to stored output grids) is byte-identical to the official gate, ~instant, and gives full DUAL-TRUTH attribution (mine-wrong vs LIVE-wrong per case). Confirm the official gate on a small `--n` for the record.

**Applies to**: any bit-pack + integer-select task (largest-rect, run-length, arg-select coordinate render); any task where cost concentrates in one stage beside a reusable render.
