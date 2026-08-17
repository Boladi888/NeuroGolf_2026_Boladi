# task227 — hard record: perfect-25 rumor DEAD + sub-94 WALL

- Live bar at audit: **cost 94** (memory 64 + params 30), sha `b12b1571447134150ac7908ca5b6ad4a319afdc1a05c326c4d85c074bd3d7f01`
  (manifest, owner Vamshi, 2026-07-02). Source: `ng_current_bar.py --task 227`.
- Scoring model (confirmed by reading `tools/_score_v2.py`): `cost = memory + params`;
  `points = max(1, 25 - log(memory + params))`. **Perfect-25 ⇒ cost ≤ 1.**

## The actual rule (decoded from task JSON, verified on train ex1+ex2)
Input is **8×4** = two stacked 4×4 grids: top grid uses color **3**, bottom uses color **1**.
Output is **4×4**, color **2**. Output cell = 2 **iff the top-half AND bottom-half are both
empty (0)** at that cell — i.e. a **NOR** of the two halves, recolored 3/1→2. Nonlinear
(logical AND of two positions) + shape change 8×4→4×4.

## Route 1 — legal zero-param family (Identity / Transpose): DEAD
Only payload-free, output-contract-preserving ops on [1,10,30,30]→[1,10,30,30] are
Identity and the single shape-preserving Transpose (swap H↔W, perm=[0,1,3,2]); flips/rot
need a Slice/Gather initializer payload (price > 1, excluded by prompt).
- `zp_identity.onnx`     → `ng_gate visible --task 227` = **0/267 (0.00%) FAIL**
- `zp_transpose_hw.onnx` → `ng_gate visible --task 227` = **0/267 (0.00%) FAIL**
One visible miss = certain hidden 0 (traps §VISIBLE-FAIL). A zero-param geometric op cannot
express a nonlinear NOR with an 8×4→4×4 shape change. **Perfect-25 ≤1 is impossible.** Not a rumor to re-hunt.

## Route 2 — beat 94 with a cheaper NOR: WALL (forced-component floor = 94)
`cost = max_intermediate_bytes + total_param_count`. For any legal correct graph:
- First learnable op consumes [1,10,30,30]; combining input channel-0 (emptiness) of two
  rows 4 apart (the NOR) needs weight in-dim = 10 and kH = 2 taps ⇒ **≥ 20 source params**.
- Output contract [1,10,30,30] one-hot ⇒ final op must emit 10 channels; from the rank-1
  bottleneck that is a [10,1,1,1] weight ⇒ **≥ 10 render params**. Params ≥ **30**.
- A two-op factorization has exactly one intermediate; the smallest is the rank-1 result
  plane [1,1,4,4] F32 = **64 B** (16 output cells all required = spatially minimal; F32
  forced — Cast to a narrower dtype materialises a ≥9,000 B full-canvas intermediate that
  dwarfs the saving). ⇒ two-op floor = 64 + 30 = **94**.
- The one-op alternative deletes the intermediate (output not charged) but forces a
  [10,10,2,1] = **200**-param weight (built `oneop_direct.onnx` to confirm the 200-param
  shape). 200 > 94.
- Any rank-≥2 or ≥3-op design adds params and/or intermediates ⇒ strictly worse.

**Pareto floor = min(two-op 94, one-op 200) = 94.** The live graph already sits on it.
No candidate prices strictly below 94; nothing staged (ties never stage). Verdict: **WALL at 94.**

## One remaining (thin) hypothesis for a future batch
The only theoretical sub-94 would require the 4×4 result plane to be materialised in a
sub-F32 dtype *without* a full-canvas cast — e.g. an integer ConvTranspose kernel that ORT
runs natively at INT8/INT16 on an INT input. Input is F32 one-hot, so it needs a free
integer view of the canvas (none known). Treat as dead unless a zero-cost F32→int8 canvas
reinterpretation appears in the library.
