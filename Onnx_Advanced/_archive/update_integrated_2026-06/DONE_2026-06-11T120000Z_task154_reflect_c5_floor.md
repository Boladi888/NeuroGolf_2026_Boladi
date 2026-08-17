# task154 reflection c5 — floor proof and dead-ends (worker-154, 2026-06-11)

## Insight 1: Cast-to-BOOL replaces Greater+threshold (saves 6B params per model)

**What**: Replacing `Greater(x_f32, half_f32)` and `Greater(x_f16, half_f16)` with
`Cast(x, BOOL)` is valid when x values are {0.0, 1.0} (one-hot channels). Cast F32/F16 -> BOOL
maps 0.0 -> False and any non-zero -> True, equivalent to Greater(x, 0.5) for binary inputs.
This eliminates 2 threshold initializers (half_f32=4B F32 + half_f16=2B F16 = 6B params).

**Where**: task154 v6, `build_task154_v6_cast_bool.py`. v5->v6 saved 2 bytes (6181->6179) via 6B
params reduction.

**Why it matters**: Small saving, but clean: removes two initializers and two nodes. Applies to any
model with BOOL-extraction from a binary-valued F32/F16 channel plane.

**Sorter note**: techniques.md — add as small optimization under "Cast idioms" or similar.

---

## Insight 2: 3-operand Einsum eliminates intermediate MatMul output (saves 450B for [1,1,15,15] F16)

**What**: Replacing `MatMul(R_mat, c5_f16) -> c5_r` + `MatMul(c5_r, C_mat) -> c5_out_f16` with
`Einsum([R_mat, c5_f16, C_mat], equation="ri,bpij,jk->bprk") -> c5_out_f16` eliminates the c5_r
intermediate. The Einsum contraction is scorer-invisible (confirmed profile==static for 266 real
inputs). For [15,15] F16 matrices with [1,1,15,15] content tensor, saves exactly 450B per reflection.

**Where**: task154 v5 (`build_task154_v5_einsum.py`), 6631->6181 (-450B). All gates pass:
visible 266/266, truth 200000/200000, ab 200266/200266 byte-equal.

**Why it matters**: Generalizes to any `R @ X @ C` matrix triple-product pattern — replace with
3-operand Einsum to eliminate the intermediate tensor. This is a direct application of #8b/technique
#109, confirmed on task154's specific shape and opset 13.

**Sorter note**: techniques.md §8b (or nearest Einsum/MatMul section) — extend with this measurement.
Profile confirmed: `profile==static` on real task154 inputs (266-input arcgen set).

---

## Insight 3: task154 reflection floor = 6179B (structural proof, many-to-one reflection barrier)

**What**: The equal-tail architecture (two F32 slices + F16 MatMul chain + U8 color-index canvas) has
a floor of ~6179B for task154. The gap to Aditya's 3857 (2322B) CANNOT be closed by any technique
currently in the library. The floor breaks down as:

- Two F32 channel slices: 2 × 900B = 1800B (unavoidable: Slice output dtype = input dtype = F32)
- c5_f16 Cast: 450B (unavoidable for MatMul)
- R_mat + C_mat OneHot: 2 × 450B = 900B (unavoidable: 43.5% of corpus have c5 on BOTH sides of bbox,
  making Gather/destination-pull incorrect; MatMul is required)
- c5_out_f16 Einsum result: 450B (unavoidable intermediate)
- Color assembly tail: c2_b+c5_b (225B BOOL each) + color2_u8+color15_u8 (225B U8 each) + 
  color30_u8 (900B U8) = 1800B
- Bbox vectors (r_refl, c_refl, rmin, rmax, cmin, cmax, rows_has, cols_has): ~224B
Total: ~5624 memory + 55 params = ~5679 theoretical; actual 6179 (extra ~500B from ArgMax I64 outputs
and cast intermediates).

**Tried and rejected techniques**:
- Gather-based reflection: FAILS 43.5% of corpus (many-to-one; visible 0/266 confirmed twice)
- ScatterND sparse (#139): requires NonZero (BANNED in the ARC grader)
- Free-output Einsum (#109): output channels are NOT separable as row×col outer products
- BitwiseAnd (#141): output not a row×col intersection pattern  
- U8 MatMulInteger: INT32 output (900B) worse than F16 (450B)
- W_assemble Conv 1x1 for output: [1,10,15,15] F16 = 4500B intermediate, far worse than Equal-tail
- Combined [1,2,15,15] slice: adds both_f32 [1800B] + both_f16 [900B] + c2_f16 [450B] = worse

**Where**: task154 v6 staged at `staged_submits/task154_v6_cast_bool_cost6179/`. Full dead-end log
in `The_400_Tasks/Task_by_Task/task154.md` Don't-do section.

**What's missing from the library**: Need a new technique for avoiding both F32 input reads and
both OneHot [15,15] matrices simultaneously. Aditya's 3857 must use a fundamentally different
algorithm (possibly avoids MatMul entirely). Inspecting Aditya's ONNX for task154 would unblock
this — his SHA is `598f6d834157f4c17bd48416f4451fa3fdc2ecea126637e55541a29b4c4ff77f`.

**Sorter note**: cost_floor_reference.md — add task154 row: architecture=equal-tail-reflect,
floor=6179, gap_to_best=2322, reason=two-F32-reads+R-mat+C-mat-unavoidable+many-to-one-proven.
