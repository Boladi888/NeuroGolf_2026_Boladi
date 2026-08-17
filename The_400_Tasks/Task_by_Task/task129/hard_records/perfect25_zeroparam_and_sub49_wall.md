# task129 — hard record: perfect-25 (cost≤1) DEAD + sub-49 WALL

- **Live bar:** cost **49** (manifest, owner Vamshi), sha256 `7df7fc5e6df8c8e5ddb98cd1e10b092e28d63a85dd0e691c67b1d1f594dcab93`.
  Profiled: `static=49 profile=49 (memory 24 + params 25)`.
- Batch 53 worker `bo_ng_cld_worker3`, 2026-07-07.

## Rule (from data, 265/265 arc-gen)
3×3 input → output = 3×3 uniformly filled with the **dominant (mode)** input color,
lowest-index tiebreak, embedded at top-left of the [1,10,30,30] one-hot canvas (empty pad).
**Generator invariant: the dominant color always has count EXACTLY 3; no ties ever**
(max-count distribution = `{3: 265}`, strict-unique-max 265/265).

## Route 1 — perfect-25 / zero-param Identity·Transpose·flip·rotate family: DEAD
The output block color is `mode(9 cells)`, a count/argmax function of the input; input `4,4,8…`
≠ output `4,4,4`. No permutation-only op can produce it.
- `task129_identity_cost1_probe.onnx`: `[visible] 0/265 FAIL`. One visible miss = certain hidden 0.
- Do NOT re-run any cost≤1 direct-output / transpose / flip / rotate variant. Disproven by construction AND gate.

## Route 2 — zero-param `Equal(count, ReduceMax)` + input-stencil render: CORRECT but LOSES
`task129_equal_stencil_zeroparam.onnx` = `Einsum(bchw->bc)` → `ReduceMax` → `Equal` → `Cast` →
`Einsum(bc,bdhw->bchw)` (reuses input as occupancy stencil, 0 initializer params).
- `[visible] 265/265 PASS`; `static=94 profile=94 (memory 94 + params 0)`.
- Loses to 49: the two `[1,10]` F32 vectors (count 40 + fmask 40) cost 80 of the 94.
- **Lesson:** any route that materializes the [1,10] count/mask VECTOR pays ≥~80 memory and cannot
  beat 49. The incumbent wins by collapsing the 10-vector to a SCALAR code before rendering.

## Route 3 — moment-scalar-collapse (the incumbent architecture): at forced floor 49
Incumbent = 3 falling-factorial moments of the count vector via shared `channel_basis[2,10]`
(row0=ones, row1=`[-15,-12,…,9,16]`), `Mean` → scalar `target_code` = dominant color index
(because `count·(count−1)·(count−2)=6` iff count==3, else 0), then a rank-2 signed-factor
`Einsum` render that reuses `channel_basis` AND the input occupancy (`bdhw`) → output. No spatial mask.

**Forced-component floor argument (why 49 ≈ true floor):**
- The render forms a per-code-positioned channel indicator `v_c = a·1 + b·w_c` (renders `(w_c − t(code))`).
  This needs BOTH a **constant** row and a **linear** row over the 10 channels → `[2,10]` = **20 params forced**;
  a rank-1 (linear-only) basis gives a parabola centered at 0 with no per-code shift → cannot select the color.
- `factor_matrix[2,2]`=4 (the 2×2 bilinear form) + `one`[1]=1 (affine constant in `target_basis=[1,code]`) = **5 params forced**.
- Detection reuses the same `[2,10]`, so no extra detection params. Total params **25 forced**.
- Memory 24 = five tiny scalar/`[2]` intermediates (`m1, neg_m2, m3, target_code, target_basis`); already scalar-minimal.
  F16-recasting them is fragile (F32 input → F32 Einsum output; a Cast adds intermediates and risks visible-instability).

**Verdict: WALL at 49.** Perfect-25 impossible; vector routes cost ≥94; the moment-collapse is at its
forced 25-param + 24-scalar-memory minimum. No staged candidate (nothing beats 49; never stage a tie).

## One remaining hypothesis (low confidence, for a future batch)
A single fused output-Einsum that contracts the count into the scalar code AND renders in one node
*might* shave 1–2 bytes of scalar memory, but the code path needs the `Mean`/`Concat` affine step
(degree-3 combine) that Einsum alone can't express; unlikely to reach ≤48. Not worth a full rebuild.
