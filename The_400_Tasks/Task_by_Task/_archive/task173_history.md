# task173 — archived history (moved 2026-06-24 by w08 per WORKER_PROTOCOL staging rule)

Superseded detail: pre-build novelty statement, full v1–v21 attempt-log table, and per-version cost analysis. Current best/floor proof lives in `Task_by_Task/task173.md`.

### Pre-build novelty statement (2026-05-20)

**What is materially new about my approach versus:**
- The standard polish passes (F16, onnxsim, safe_opt): The current canonical has F16-output trick applied (out25→Pad→output is F16). onnxsim+safe_opt produce 0 bytes. The DCE attempt (deferred Cast) is staged but not in canonical — it saves 33,750 by keeping the 3 `*_full → *_full_flat → *_full_t` Reshape+Transpose chains in BOOL through Cast-only-at-MatMul. My approach starts by applying DCE then attacks the per-branch 16-node MatMul-cooccurrence pipeline directly.
- spread_scan auto-detector hits: 0 bytes recoverable at the floor. Scan results: clip_mul_to_min/const_to_init/safe_opt/scalar_broadcast/slice_axis_collapse all "—" (no match); onnxsim/fp16 also 0 / ERR. The only positive scan suggestion is crop_to_size_advisory at -11000, but the graph is **already cropped** to 25x25 via Slice on input — false-positive case per traps.md §21 and crop_advisory_false_positive feedback.

**Hypothesis for where the 256k cost goes:**
Per `largest_intermediate.py` output the top 17 intermediates are 11,250 bytes each:
- `t173_input16` (F16 [1,10,30,30]) = 18,000 (input cast, pre-Slice)
- `x9` (F16 [1,9,25,25]) = 11,250 (the working tensor)
- `flat` (F16 [9,625]) = 11,250 (shared, reshape of x9)
- `flat_t` (F16 [625,9]) = 11,250 (shared, transpose of flat)
- Per branch × 3: `detected_sum [1,9,25,25] F16` + `full [1,9,25,25] F16` + `full_flat [9,625] F16` + `full_t [625,9] F16` = 4 × 11,250 = 45,000 per branch
- 3 branches × 45,000 = 135,000

That accounts for ~187k of 256k. The remaining ~70k is `out25` (12,500), BOOLs for `*_full_bool` (3×5,625=16,875), `out25_bool` (6,250), `out_color_bool` (5,625), small per-branch frame/center fill (3×~5k), Equal int32 cast (2,500), etc.

**Architecture changes targeted:**
1. **DCE** (proven, -33,750): keep `*_full_flat / *_full_t` in BOOL through Reshape+Transpose. Cast only at MatMul boundary.
2. **input→BOOL whole-input trick (rule 8)**: Cast input to BOOL once (9,000 vs 18,000 F16). Slice bg25 as BOOL directly, eliminating the trailing `Greater(bg25, half)` (since bg25_b is already BOOL). For x9, Cast back to F16 only for the Conv ops. Estimated -5k.
3. **Vector 4 algorithmic reformulation** (largest potential win, -~60k): replace the MatMul-based cooccurrence pipeline (full_flat × flat → cooccur [9,9] → Greater → Where(non_diag) → MatMul(color_values) → MatMul(flat_t, *)) with a direct Conv-based per-cell evaluation. Since `non_diag` is "1 off-diagonal" and the cooccurrence math reduces to "did any (corner color, center color) pair exist with corner!=center", we can compute per-cell `corner_idx = Conv(full_f16, [1..9])` (giving 0 or the unique color whose 4 corners match) and `center_idx = label` (already in the graph). Then `corners_frame_values = Where(And(any_full_bool, NotEqual(corner_idx, center_idx)), center_idx, 0)`. This eliminates `flat`, `flat_t`, and per-branch `full_flat`, `full_t`, `cooccur`, `pairs`, `frame_values_by_center`, `frame_values_flat`, etc. — replacing them with 2 Convs and a few small comparisons per branch.

**What does each of the 3 parallel branches compute, and what would a generic branch look like?**
Per branch with kernel K (K=2x2 corners / 1x2 h / 2x1 v, all dilation 2 in valid axes, group=9):
1. `full_bool[c, p] = (Conv(x9, K_c) == |K_cells|)` — at output position p, is color c the unique color filling all kernel positions?
2. `corner_idx[p] = Conv(full_f16, weights=[1,2,…,9])` — single color matching, 0 elsewhere (only one channel can be True per p since input is one-hot)
3. `any_full[p] = ReduceMax(full_f16, axis=1)` (or equivalently Greater(detected_sum_max, threshold) — actually detected_sum max over channels works)
4. `valid[p] = And(any_full > 0, corner_idx != center_idx)`
5. `frame_values[p] = Where(valid, center_idx, 0)`
6. `frame_fill[p] = Conv(frame_values, K, dilation=K_dil)` — spread center color to the corner positions (or to itself for center_fill)

A generic detector with stacked output channels could use a single 27-channel Conv if we accept zero-padded kernels (corners=[[1,1],[1,1]], h=[[1,1],[0,0]] but with dilation [2,2] not [1,2] — verifies shifts), but the dilations and pads are NOT compatible across the 3 branches (corners dil=[2,2] pads=[1,1,1,1], h dil=[1,2] pads=[0,1,0,1], v dil=[2,1] pads=[1,0,1,0]). So stack-into-1-Conv is blocked by dilation/pad differences. Keep 3 separate detection Convs but unify the downstream logic.

---

### Attempt log

| Date | Approach | Result | Cost | Notes |
|---|---|---|---|---|
| 2026-05-19 | deferred-Cast-past-Reshape+Transpose in 3 branches (DCE) | ✅ staged | 243,141 (−33,750 vs prior canonical 276,891) | worker-7; `task173_dce_t_E1079E23_on_5806_87`. NOT in current canonical (256,900). |
| 2026-05-20 | v1: DCE + input→BOOL + drop Greater(bg25, half) | 266/266 verify | 218,525 (−38,375 vs 256,900) | `build_task173_v1_dce_bool_input.py`. input_b BOOL replaces input16 F16 (9k vs 18k); bg25_b BOOL reused directly in final And; per-branch full_flat_b/full_t_b stay BOOL. |
| 2026-05-20 | v2: + opset 10→13 (Pad accepts BOOL; Equal accepts F16) | 266/266 verify | 203,525 (−15,000 vs v1) | `build_task173_v2_opset_bump.py`. Drops out25 F16 Cast before Pad (−12,500) and v_acc INT32 Cast (−2,500). Graph output dtype = BOOL. |
| 2026-05-20 | v3: + replace MatMul w/ Gemm transA/transB (drop flat_t + per-branch full_t) | 266/266 verify | 175,400 (−28,125 vs v2) | `build_task173_v3_gemm.py`. Eliminates shared flat_t F16 [625,9] (−11,250) and 3× full_t_b BOOL [625,9] (−16,875). Cast moves to full_flat_b → full_flat_f16 (single Cast per branch). |
| 2026-05-20 | v4: + runtime-Conv-weights for frame_fill (eliminates frame_values_flat + frame_values per branch) | 266/266 verify | 167,958 (−7,442 vs v3) | `build_task173_v4_runtime_conv.py`. Identity: `frame_fill[p] = sum_a by_center[a] * detected_sum[a, p]` ⇒ `Conv(detected_sum, by_center_4d_runtime, 1×1)`. Confirmed ORT supports Conv with runtime weights under ORT_DISABLE_ALL. |
| 2026-05-20 | v5: + sweep 3 unused spread_weight inits (orphaned by v4) | 266/266 verify | 167,950 (−8 params vs v4) | `build_task173_v5_cleanup.py`. |
| 2026-05-20 | v6: + Gemm transA=1 eliminates 3 *_pairs_t Transposes | 266/266 verify, 50/50 adv | **167,464 (−89,436 vs 256,900, −34.8%)** | `build_task173_v6_pairs_t.py`. **STAGED**: `task173_radical_t_3719fa42_on_5874_78/`. |
| 2026-05-24 | independent reference + task-shaped synth gate | 266/266 reference; 999/1000 construction synth | n/a | Raw-data numpy reference in `prototypes/task173_rule_reference.py`; task-shaped generator in `prototypes/task173_synthetic_gate.py`. Generator uses spaced complete templates plus center-only/frame-only instances; construction-vs-reference is 99.9%. |
| 2026-05-24 | v7: exact U8/QLinearConv + QLinearMatMul rewrite | 266/266 verify; 266/266 vs LIVE | 136,583 (−30,881 vs v6) | `build_task173_v7_u8_qlinear.py`. Replaces F16 detector/cooccurrence lanes with UINT8 `QLinearConv`/`QLinearMatMul`. This preserves v6 semantics and verifies exactly, but still materializes per-branch full-map transposes. |
| 2026-05-24 | v8: dynamic 1x1 QLinearConv projections | 266/266 verify; 266/266 vs LIVE; 1000/1000 ONNX-vs-reference synth | **116,012 (memory 115,705 + params 307)** | `build_task173_v8_dynamic_qconv.py`. Computes cooccurrence as frame-color rows × center-color columns, then applies `by_center`/`by_frame` with runtime-weight 1x1 `QLinearConv`s. Removes the per-branch spatial transposes used only for projection in v7. |
| 2026-05-24 | v9: exact compact label + `ScatterElements(reduction=max)` rewrite | 266/266 verify; 266/266 vs LIVE; 1000/1000 ONNX-vs-reference synth | 75,205 (memory 74,774 + params 431) | `build_task173_v9_scatter_label.py`. Collapses color channels to one label plane, learns C->F and F->C maps through row-map scatter, then gathers/spreads fills. Exact rule including `C != F`. |
| 2026-05-24 | v10: TopK2 pair-code probe | 266/266 visible; 1000/1000 synth | 127,249 | `build_task173_v10_topk2_label.py`. Correct but worse: `TopK`/pair-code path forces large INT32 maps. Abandoned. |
| 2026-05-24 | v11: lower-cost scatter-label, no explicit `C != F`, direct 30x30 final Equal | 266/266 verify; 266/266 vs LIVE; 1000/1000 ONNX-vs-reference synth | **59,855 (memory 59,424 + params 431)** | `build_task173_v11_scatter_label_nondiagless.py`. Uses INT64 label indices directly, shares input-nonzero mask, pads the 1-channel sentinel label plane before final `Equal(output)`, and drops the explicit diagonal guard. This is the first under-60k candidate, but hidden risk is higher if same-color frame/center decoys appear. |
| 2026-05-24 | v12: v11 + cropped sparse `QLinearConv` color-label extractor | 266/266 verify; 266/266 vs LIVE; 1000/1000 ONNX-vs-reference synth | **59,507 (memory 58,724 + params 783)** | `build_task173_v12_posconv_label_nondiagless.py`. Replaces `ArgMax`'s counted INT64 full-canvas/crop tensors with `Cast(input, UINT8) -> QLinearConv(6x6 positional label kernel) -> Cast(INT32 indices)`. Same hidden risk as v11, but lower official scorer cost. |
| 2026-05-24 | v13: sparse F32 Conv label extractor | 266/266 verify; 266/266 vs LIVE; 1000/1000 synth | 53,007 | `build_task173_v13_f32conv_label_nondiagless.py`. Removes the full-canvas `input_u8` cast by using one sparse F32 Conv over one-hot input, then casts the 25x25 label down. |
| 2026-05-24 | v14: compact row/column active-profile padding mask | 266/266 verify; 266/266 vs LIVE; 1000/1000 synth | 50,286 | `build_task173_v14_active_profile_nondiagless.py`. Replaces per-cell background slice with row-active and col-active reductions over the one-hot input. |
| 2026-05-24 | v15: dilated 2x2 label extractor | 266/266 verify; 266/266 vs LIVE; 1000/1000 synth | 49,966 | `build_task173_v15_dilated_label_nondiagless.py`. Same 25x25 label output as v14, but a 2x2 dilation-5 Conv replaces the 6x6 sparse kernel. |
| 2026-05-24 | v16-v18: vector slot-0 cleanup removes branch gates | 266/266 verify; 266/266 vs LIVE; 1000/1000 synth | 43,778 by v18 | `build_task173_v18_zero_pair_vectors_nondiagless.py`. Zeroes tiny `by_center[0]`/`by_frame[0]` vectors after scatter, allowing removal of full 25x25 `complete`, `frame_update`, `input_nonzero`, and `center_update` maps. |
| 2026-05-24 | v19: QLinearConv shift maps | 266/266 verify; 266/266 vs LIVE; 1000/1000 synth | 38,970 | `build_task173_v19_qshift_nondiagless.py`. Replaces eight `Slice -> Pad` label shifts with direct single-weight `QLinearConv` shifts. |
| 2026-05-24 | v20-v21: no explicit `frame_nz`; keepdims active profiles | 266/266 verify; 266/266 vs LIVE; 1000/1000 synth | **35,163 (memory 34,805 + params 358)** | `build_task173_v21_keepdims_active_nondiagless.py`. Slot-0 vector cleanup makes all-zero frame equality harmless, so branch masks no longer need `frame_nz`; v21 slices 4D row/column active profiles directly. Current lowest candidate. |

## Cost analysis

After v6 (cost 167,464), the structural floor is dominated by 9 mandatory F16 tensors at 11,250 bytes each (= 101,250 total = 60% of total cost):
- `x9` F16 [1,9,25,25] = 11,250 (shared, Conv input)
- `flat` F16 [9,625] = 11,250 (shared, used by 3 cooccur Gemms)
- 3 × `*_detected_sum` F16 [1,9,25,25] = 33,750 (Conv outputs, mandatory F16)
- 3 × `*_full_flat_f16` F16 [9,625] = 33,750 (Cast for Gemm cooccur)

Plus mandatory BOOLs (50k):
- `input_b` BOOL [1,10,30,30] = 9,000 (input→BOOL Cast result)
- `out25_bool` BOOL [1,10,25,25] = 6,250 (Concat output before Pad)
- `x9_b` BOOL [1,9,25,25] = 5,625 (Slice of input_b)
- 3 × `*_full_bool` BOOL [1,9,25,25] = 16,875 (Greater outputs)
- 3 × `*_full_flat_b` BOOL [9,625] = 16,875 (Reshape outputs)
- `out_color_bool` BOOL [1,9,25,25] = 5,625 (Equal output before Concat)

Plus 19 small F16 [1,1,25,25] or [625,1] tensors at 1,250 each = ~24k for the downstream lookup and spread logic.

The algorithm is "find templates with corners-color-F surrounding center-color-C, propagate to incomplete instances". MatMul-based cooccur (per branch: `Gemm(flat [9,625], full_flat_f16 [9,625], transB=1) → [9,9]`) is the byte-optimal cooccurrence calculation — alternative broadcasted-And gives [9,9,625] = 50,625 BOOL intermediate, 4-5× worse.

After v8 (cost 116,012), the graph is exact and mostly UINT8/BOOL:
- largest remaining tensors: `input_u8` [1,10,30,30] = 9,000; `out25_bool` [1,10,25,25] = 6,250; `x9_u8`, `flat_u8`, `flat_t_u8`, and per-branch `detected_sum/full_bool/full_u8/full_flat_u8` [9,25,25] or [9,625] = 5,625 each.
- runtime-weight 1x1 `QLinearConv` is accepted by ORT and the official scorer, and saves 20,571 bytes vs v7.
- onnx_opt_extended produced no cost change on v8.

After v11 (cost 59,855), remaining counted memory is dominated by:
- `label30_i64` [1,1,30,30] = 7,200 and `label25_i64` [1,1,25,25] = 5,000 from `ArgMax`/crop.
- 3 x `*_frame_full_i32` [1,1,25,25] = 7,500 for per-cell `Gather` indices.
- `bg25_f32` = 2,500 and the padded sentinel label plane `out_label_active30_u8` = 900.
- The former `out25_bool` [1,10,25,25] = 6,250 is gone: v11 pads the 1-channel label plane with sentinel `10`, then names the final broadcast `Equal(..., colors10)` as graph `output`, so the 10-channel bool tensor is not counted as an intermediate.

After v12 (cost 59,507), `ArgMax` is gone. The replacement label path costs `input_u8` [1,10,30,30] = 9,000 + `label25_i32` = 2,500 + `label25_u8` = 625, but deletes v11's `label30_i64` + `label25_i64` = 12,200. The sparse 6x6 label kernel increases params (783 total), but official score still drops by 348 bytes.

After v21 (cost 35,163), the current nondiagless floor is dominated by the unavoidable scalar-label index maps and per-branch frame index casts:
- `label25_f32`, `label25_i32`, and 3 x `*_frame_full_i32` are 5 x 2,500 = 12,500 bytes.
- Per-branch QLinearConv shifts removed the eight former shift slice tensors (4,704 bytes) while keeping the eight 625-byte shifted maps.
- Tiny vector cleanup trades 25x25 branch gates for 10-element `ScatterElements` repairs on `by_center` and `by_frame`.
- `ConstantOfShape` for the shared zero rowmap was probed and rejected: 35,167 vs v21's 35,163.

