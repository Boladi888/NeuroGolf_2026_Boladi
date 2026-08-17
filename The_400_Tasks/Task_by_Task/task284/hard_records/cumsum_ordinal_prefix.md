# task284 hard record: F16 CumSum ordinal/prefix route (2026-07-11 prompt) — DEAD

- **Live bar used for delta:** manifest live `2634`, sha `1721f6c7195abb2a3f50f977b903aa93f7c94a16fe39ed0e5212daa377ce6197`
  (`python tools/ng_current_bar.py --task 284 --json`).
- **Effective (reviewer-ready) best:** as of 2026-07-12, task-local staged `task284_f16coord_cost2537`, sha
  `556fd194324595f9d103007abcaaf28b02f5859cc465c190e9f15563d6741316` (F16-coord narrowing of the prior
  `task284_ceilfold_cost2593`, sha `eea9ac4636902baaf47995d0f8842d3a5b042ff267ee131f2451b2abbc104b04`).
  Per prompt "staged sidecar = effective best"; a win requires profile cost **strictly < 2537**. The 2537
  win came from a *different* lever (F16 scalar lane), not from CumSum — the CumSum route below loses to
  either bar by a wide margin.

## Attempted route
Prompt `Prompt_20260711T000000Z_6224_Task284_F16CumSumOrdinal.md` suggested a from-scratch rewrite:
opset-14 `CumSum` on F16 — mark run-starts/boundaries, accumulate ordinal IDs, then equality/rank logic to
place outputs; floor_est ~900, EV +18.39. Class STRUCTURAL.

## Why it is dead (priced, not asserted)
The task is **not** a run-length/ordinal problem. Visual + generator + Einsum trace all show it is a
**2-color bracket-drawing** problem: two singleton markers → arms + 5-tall spine + inward corner caps, each
marker painting its own side. The current 2593 graph solves it as a **free-output separable Einsum**
`lb,gc,gk,kr,ks->lcrs` over **K=6 geometry components** F16[6,30] per axis, with 3 color groups sharing the
6 components via `group_map[3,6]`.

**Scoring is fully additive** (`neurogolf_utils.calculate_memory` sums *every* intermediate tensor's bytes,
input/output excluded, refined to `max(static, profiled)`; `params` = Σ initializer/constant elements). So a
win must reduce *total* intermediate bytes, not peak-live memory.

CumSum can build a *closed interval* `[lo,hi]` as `CumSum(delta)` where `delta` has +1 at `lo`, −1 at `hi+1`.
But `lo`/`hi` are **data-dependent** (runtime coords), so `delta` must be placed with `Equal(idx, pos)`
index-planes. Built + measured both label lanes apples-to-apples (`prototypes/probe_cumsum_vs_band_2026-07-12.py`, exact
`calculate_memory` rule, final F16[6,30] counted as the Einsum operand it is; shared front-end coords injected
as graph inputs so only the label-construction cost is compared):

| label lane (per axis) | charged bytes | breakdown |
|---|---:|---|
| **U8-band (current 2593 family)** | **895** | F16[6,30] 360 + BOOL[6,30] concat 180 + 5×(U8 `d`[1,30] 30 + BOOL band[1,30] 30) 300 + valid 30 + scalars |
| **CumSum/prefix (prompt route)** | **1965** | F16[6,30] out 360 + step[5,30] 300 + eq_lo/eq_hi BOOL[5,30] 300 + eq_lo_f/eq_hi_f F16[5,30] 600 + delta[5,30] 300 + valid_f 60 |

**CumSum loses by +1070 B/axis (~2.2×), +2140 B over both axes.** Even a maximally-charitable variant
(imagine `delta` free) is `step[5,30]` 300 + valid_f 60 + concat 360 = 720, still not a structural win, and
`delta` cannot be free — placing +1/−1 at runtime positions requires the `Equal`/Scatter index-planes the
prompt's **KILL RULE** names ("BOOL/UINT8 CumSum or Range-heavy index matrices to make the labels → stop").
`ScatterElements` is no cheaper (also materializes `[5,30]`).

## Exact gate/probe run
- `prototypes/probe_cumsum_vs_band_2026-07-12.py` → `BAND = 895`, `CUMSUM = 1965` (charged bytes, per axis, static shape inference).
- Per-tensor fat map of the live 2593 graph (`prototypes/dump_charged_bytes.py`): memory 2485 + params 108 = **2593**;
  by dtype F16 820 / BOOL 781 / F32 544 / U8 340. Every dominant tensor traced to a forced role
  (F16[6,30]×2 = Einsum operands; BOOL[6,30]×2 = the one cheap concat-then-cast; F32[1,30]×2 valid counts =
  active-grid canvas, reduce output inherits F32 input dtype, un-narrowable without an 18 000 B input cast).

## Forced-floor argument (why 2593 is the wall for every formulation, not just this one)
1. **K=6 is minimal geometry.** Single-color bracket matrix is **rank-3** (spine-line ⊗ + caps ⊗ + arm ⊗,
   independent). Two colors cannot share a rank-1 term (they paint different colors) ⇒ 3×2 = **6**.
   `central_gap` is a distinct rank-1 rectangle (prior 1960-case proof), not a signed combo of the other 5.
2. **F16 is the cheapest legal Einsum operand dtype** (U8/I8 Einsum = ORT NOT_IMPLEMENTED; F32 doubles it).
   Components must span 30 (output width); narrowing to 24 forces a pre-pad F16[.,.,.,24] Einsum output
   = 14 400 B intermediate. ⇒ F16[6,30]×2 = 720 B forced.
3. **`valid` canvas forced**: empty active-grid cells need background channel = 1; `valid = ReduceSum(input
   over channels)` reconstructs the crop from one-hot padding (this is the 23×24-crop correctness fix).
   Reduce output = F32 (input dtype); F16 needs an added cast that eats the saving.
4. **2D routes lose by construction**: any materialized `[.,30,30]` F16 plane = 1 800 B, and a 2D summed-area
   fill needs ≥2 planes (delta + cumsum) = ≥3 600 B before color/coord machinery — already > 2593. This is the
   same #108 wall that killed the coordinate-predicate stencil (~7 500 B, `task284.md` 2026-06-22).
5. **Micro-levers priced to loss**: batched `[K,30]` bands = byte-neutral +20 (2613, prior W15); cast-before-
   concat = +180/axis; F16 coord lane = net loss (moments are F32 from the F32-input Einsum, so any F16 coord
   conversion adds ≥80 B of casts to save ≤72 B of scalars, and risks F16 rounding on the exhaustive suite).

## Verdict
**WALL — NO_CANDIDATE. 2593 is the hard floor.** The CumSum/ordinal/prefix card is a function-first mismatch
(cheap primitive selected first, then matched to the task); it does not reduce K, the component dtype, or the
component width, and its label lane is a measured +2140 B regression. No sub-2593 candidate staged.

## One remaining hypothesis (for the next batch, low-confidence)
Only revisit if someone finds a construction that supplies the 6 F16 geometry components **without** any
per-band `[1,30]` intermediate AND without an F32-locked reduction for the `valid` canvas — e.g. a single op
that emits an in-range indicator from `(idx, lo, span)` (no such ternary op exists in opset-≤21 as of this
record). Absent that, the 895 B/axis band lane is provably minimal for a comparison-based band.
