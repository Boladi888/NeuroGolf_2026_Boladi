# task053 — sub-14 WALL (dynamic-weight ConvTranspose floor)

**Date:** 2026-07-07 · worker `bo_ng_cld_worker1` (Claude lane)

## Live bar (at write time)
- **live_cost = 14** — authoritative source `manifest` (`ng_current_bar.py --task 53`),
  owner Vamshi, updated 2026-07-02T02:35Z.
- **live sha256 = `82363688d5fa95aaff1fcf43083ea8a7bc36034124fb263e690e4df186143664`**
  (`Team_Best_Onnx/task053.onnx`).
- Profiled: `profile_real_cost.py --task 53` → `static=14 profile=14` (memory 0 + params 14).
- NOTE: the local `task053.md` floor proof claiming "cost 30 is optimal" is **STALE/FALSE** —
  it predates Vamshi's 2026-07-02 ConvTranspose ship. The real floor is 14 (this record).

## What the champion actually is (NOT the old cost-30 single Gather)
A **dynamic-weight ConvTranspose**: the input grid is fed in as the convolution *weight*.
```
ConvTranspose(X=impulse[1,1,4,1], W=input[1,10,30,30], B=bias[10])
  impulse=[4,1,1,3]  (4 params)   bias=[-3,-2,-2,0,0,0,0,0,0,0]  (10 params)
  pads=[2,0,1,0] strides=[1,1] kernel=[30,30]   cost = 4+10 = 14
```
Verified closed form (byte-exact vs ORT): `out[co,oh,ow] = Σ_ih impulse[ih]·input[co,oh+pt−ih,ow] + bias[co]`, `pt=pad_top=2`.

## Attempted route that is DEAD
Beat 14 by shrinking the impulse (≤3 taps → cost ≤13) or dropping the bias (cost = tap-count).

## Exact gate that fails
`prototypes/search_sub14_convtranspose.py` — **exhaustive** over the *entire* task
distribution (color∈{1,2}, pixels in 6 cells r∈{0,1}, 2–3 pixels ⇒ **70 distinct grids**,
the whole truth set, not a sample) + 60 real pairs. Bias feasibility solved analytically
(a per-channel constant just slides the `>0` threshold; feasible iff `max_{off}O < min_{on}O`).
Result: **NO config with cost < 14 matches.** L=4 champion is the unique minimum (cost 14).

## Why 14 is forced (three coupled constraints)
1. **10 channels + shared row-mix T (30×30) + per-channel bias.** `out = einsum('oi,cij->coj',T,input)+bias`.
   T is identical across channels; only bias differs. Materialising T costs 900; the ConvTranspose
   Toeplitz-via-impulse expresses T in `#taps` params — that is the whole reason 14 < 30 is possible.
2. **Impulse ≥ 4 taps.** The stamp must simultaneously (a) source the *guaranteed-background*
   input row 2 into output row 0 ⇒ reach `pt ≥ 2`; and (b) carry the down-shift input-row0→output-row1
   ⇒ a tap at `ih = pt+1 ≤ L−1` ⇒ `pt ≤ L−2`. Together `2 ≤ pt ≤ L−2` ⇒ **L ≥ 4**. Exhaustively
   confirmed: every L≤3 impulse (any pt, any legal bias) misses ≥1 of the 70 grids.
3. **Bias mandatory, length 10.** The shift unavoidably leaks the all-background input row 2 into
   output row 3 (padding): pre-bias `out[ch0,row3] = d = 3 > 0` (measured). Padding must read ≤0,
   so `bias0 ≤ −3`, while background rows (pre-bias ≥4) must stay >0 — only a per-channel additive
   constant does this. ConvTranspose bias length = `C_out = 10` by op contract; a shorter bias is the
   §8h sub-declared-bias UB trap (visible-unstable). ⇒ bias = 10 params, non-negotiable.
   ⇒ **cost = 4 + 10 = 14.**

## Other families ruled out (strictly worse than 14)
- **Regular Conv, input-as-weight:** weight [1,10,30,30] ⇒ M=1 out-channel, not 10. Wrong shape. Dead.
- **MatMul/Gemm row-mix:** needs the full 30×30 T (900 params). Dead.
- **Multi-stage (ConvT→crop/Pad, Slice+Concat):** any counted intermediate is `[1,10,3,30]`=3600 B
  or `[1,10,30,30]`=huge memory. All ≫14 (already logged in task053.md attempts). Dead.
- **Single Gather / Slice:** 30-element row index (old cost-30). Dead — beaten by ConvT.
- **0-param leak-suppression (Clip/Pad/Transpose):** Clip(min=0) can't push the +3 leak negative;
  Pad zero-fill can't synthesise the ch0=1 background (the classic task053 translation trap). Dead.

## One remaining hypothesis (unproven, low probability)
An op outside the Conv/MatMul/Gather families that produces 10 channels with a **Toeplitz-cheap row
mix AND a sub-10-param per-channel correction** in one node. None found in the ONNX opset; the
per-channel constant on a 10-channel tensor is intrinsically ≥10 params (can't broadcast 3→10 without
a Slice/Concat that reintroduces intermediate memory). Treat as **WALL at 14** unless a novel op appears.
