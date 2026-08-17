# task289 hard_record — FreeOutputPadTailSweep WALL @ 324

**Batch:** 5608 (bo_ng_cld_worker1 / Claude lane), 2026-07-09
**Live bar when explored:** cost **324**, sha256 `ffbb8ecbbd332816e69ab20d1539bda1c778bc7372b01a37ba061d95766de62e`
(manifest, owner Sho, updated 2026-07-09T01:09:17Z — `python tools/ng_current_bar.py --task 289`).
Prompt named the bar as `None` at authoring; it is now 324 and fresh.

**Verdict:** WALL. No candidate strictly below 324 exists via the free-output/resize/einsum
families. The prompt's `floor_est ~129` (gap 2.51x, "HIGH confidence") is **not achievable** —
it assumed the charged `[1,10,4,4]` pre-Resize tensor is removable, which ORT's Resize
short-circuit forbids. Local staged bests (expand_reuse 529, kbasis5 614) are SUPERSEDED by
the team's live 324 and are not shippable improvements.

---

## The rule (reconfirmed)
`grid_enhance(size=3, enhance=k)`: 3x3 source, each cell -> `k x k` block, active output `3k x 3k`
on black, k = #distinct colours in {1..5}. Encoder (`ng_harness/encode.py`): within-grid one-hot
(black=ch0=1), **outside-grid = all-zero**. Hence exactly:
`output[n,r,c] = input[n, floor(r/k), floor(c/k)]` — verified byte-exact 268/268
(input's own zero padding rows>=3 supply the zero background for r>=3k).

**Grader is sign-based:** `session.run_raw` returns `(out > 0.0)` and compares with `np.array_equal`
(ng_harness/modes.py). Off-band must be `<= 0`; on-band `> 0`. F32 output + all-zero background
are both fine (why expand_reuse's all-zero background passes).

---

## DEAD ROUTE 1 — direct-input same-size Resize (~98 target). ORT short-circuit.
Build: `Resize(input[1,10,30,30], roi=[0,0,1/k,1/k], sizes=[..30,30], tf_crop_and_resize,
nearest, floor)`. Input free + output free => charged ≈ k-detection only (~98).
Prototype: `prototypes/build_task289_resize_roi/task289_resize_roi.onnx`.

- **Real gate: `ng_gate visible --task 289` = 0/268 FAIL.**
- Root cause (empirically isolated): **ORT returns `output == input` unchanged whenever
  output_H == input_H (30==30), ignoring roi/scales/mode entirely.** Tested `sizes` and `scales`
  paths, `tf_crop_and_resize`, `nearest_mode` floor/round — all give `out == input` on 268/268.
  `scales=1.0166` (output still 30) short-circuits; `scales=0.9834` yields output size 29.
  So the short-circuit is **shape-based (out==in spatial)** and cannot be defeated by scale
  tweaks at 30->30. This is the "ORT ignored the ROI" folklore in the old don't-do log — now
  mechanised. **Do not retry same-size dynamic-ROI Resize.**

## DEAD ROUTE 2 — small-input Resize at S=3 ([1,10,3,3] pre-Resize, 90B).
Would save 70B vs Sho's 4x4 (160B). Resize-tail test on hand-built 3x3 one-hot: **0/268 all k**.
A 3x3 input has no zero border row/col, so background (r>=3k) must come from `extrapolation_value`,
which ORT's nearest tf_crop_and_resize does not apply on this path. S=4 works (247/268 with a naive
roi; only k=5 fails on a float floor()-boundary at x_orig=3.0). Sho's `roi_num=9.8` (slightly above
`29/3=9.667`, giving x_orig ≈ 1.014·r/k) pushes the background boundary safely past 3.0 -> 268/268.
**Minimum working Resize input = [1,10,4,4] U8 = 160B, forced.**

## DEAD ROUTE 3 — cheaper free-output Einsum basis (target the 129 floor).
The free-output Einsum (expand_reuse, 529) has memory 69 but needs the k-dependent expansion
basis, which costs **either** a stored bank `expand[5,3,30]` = 450 params **or** a runtime-computed
`[3,30]` F32 = 360B (multiple [30,3] intermediates, dtype-invariant since params are element-counted
and any BOOL/U8 basis must Cast to F32 for the F32-input Einsum). Neither drops below ~450, so the
Einsum family floor ≈ 519 > 324.
- Bank zero-columns (cols 15..29 are structurally 0, since 3k<=15) can't be exploited: emitting the
  active `[1,10,15,15]` then Pad charges a 2250B intermediate; dense-init floor forbids sparse
  storage (traps §8e).
- Low-rank/CP factorization of the bank is dead: sign(`>0`) grading needs off-band `<= 0`, but an
  approximate bilinear basis contaminates off-band via cross-terms -> off-band positives
  (cp_rank11: visible 0/268). Direct pair-selector rank >= 41.

---

## Forced-component floor (why 324 is at/near global floor)
A free-output op producing `[1,10,30,30]` has irreducible charged inputs:
- **Resize** needs a spatially-small (<30, min 4x4 in ORT) **10-channel** U8 one-hot = `[1,10,4,4]`
  = **160B**, built via the label round-trip (Conv F32 `[1,1,3,3]`=36B + U8 quadratic reconstruction
  ~36B; a direct F32 slice is 360-640B). Plus k-detection+roi ~35B, params 57. Floor ≈ **316-324**.
- **Einsum** avoids the [1,10,H,W] tensor but pays the >=450 basis. Floor ≈ 519.
- **Label+Equal** (prompt's literal hint) needs a full-res `[1,1,30,30]` label = **900B** (>160) — worse.
- **Pad-tail** needs `[1,10,3k,3k]` up to 2250B (profiled at k=5 max) — worse.

Sho's 324 = memory 267 + params 57, fully decoded (Conv neg-pad crop -> quadratic QLinearConv
`logit[n]=1-(idx-n)^2` -> MaxPool/QLinearConv k -> `roi_end=9.8/k` -> Resize 4x4->30 @ scales 7.5).
Every intermediate is used; only a ~5-9B k-detect micro-trim (ReduceSum-1 vs the 10-param
`present_sum_w`) is even conceivable, and **ORT rejects ReduceSum on U8/INT32** (opset-13 axes-input
form: INVALID_GRAPH), so that trim is blocked too. EV of any surviving trim ≈ 0.005 pts.

## One remaining hypothesis (low confidence)
If a future ORT/opset exposes a **same-size Resize that honours a dynamic crop-ROI** (or a
DepthToSpace/ConvTranspose with a *dynamic* blocksize/stride), the direct-input Resize collapses the
whole graph to k-detection only (~98) and the 129 floor becomes reachable. Until then: WALL @ 324.
