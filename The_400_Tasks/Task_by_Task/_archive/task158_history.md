# task158 — archived attempt history (moved 2026-06-19 by worker-158 for the WORKER_PROTOCOL task-md-size rule)

Superseded detailed attempt rows (cursor-sweep audits + the v9->v99 structural progression). The CURRENT best/floor proof and the don't-do log stay in `Task_by_Task/task158.md`. Format: `date | approach | result | cost | source`.

2026-05-18 | current-best extraction + candidate source audit | current task158 cost 182,776 (memory 146,688 params 36,088); legal-source band task158 payload is smaller by bytes but scores far worse at cost 11,858,032, so do not use the cherry/quince/salak/ugni band ONNX for current scorer | 182,776 / 11,858,032 | `work/task158_cursor_sweep/score_summary.json`
2026-05-18 | safe optimizer / const-to-init / dedupe / CSE / DCE | no-op; current graph already has no duplicate initializers or safe algebraic removals | 182,776 | prior `submissions/_task158_lane/task158_pass_summary.json` plus `work/task158_cursor_sweep/task158_safeopt.onnx`
2026-05-18 | numeric cursor-slot sweep (current-style graph, truncate cursor descriptors) | slots 3-4 fail current-LIVE pair equivalence (264/266); slots 5-17 match all 266 train/test/arc-gen pairs and reduce cost, best local-pair candidate is cursor05 at cost 92,099 (delta -90,677; +0.6854 pts predicted) | 92,099 | `Logs/delta_packet_task158_cursor05_t_5f11b5cc.json`
2026-05-18 | random one-hot strict-equivalence guard over cursor sweep | cursor05 differs from current LIVE on 3/500 random one-hot fuzz cases; only cursor18 is 0/500. Treat cursor05 as HOLD/review-required, not direct-submit. | cursor05 92,099, cursor18 182,787 | `work/task158_cursor_sweep/cursor_sweep_random_fuzz.json`
2026-05-18 | vectorized 18-cursor evaluator | 32-node vectorized graph matches current LIVE on 266/266 local task pairs but costs 186,661, worse than base; do not use | 186,661 | `work/task158_cursor_sweep/vectorized_cursor18_verify.json`
2026-05-18 | reviewer official full-tensor submit-gate rerun of `task158_current_style_cursor05` | FAIL: official full-tensor mode reports Prototype vs expected 177/266, Live vs expected 266/266, Prototype vs LIVE 177/266. Legacy debug `--cropped-decode` reports 266/266 but is noncanonical and must not gate requeues. First official raw diff was pair 0 at `(0,1,11,5)` with 40 mismatching raw tensor elements; reviewer seed-0 random one-hot fuzz still found 1/500 mismatch. | reject cursor05 unchanged | review packet `t_8fec001a`
2026-05-21 | v2 radical rewrite: compact histogram key (ReduceSum→[1,10]→MatMul[10,1] replaces Flatten→[1,9000]→MatMul[9000,1]) + BOOL-Or chain (17×Or replacing 18×Cast+Sum+Greater) + AND-reorder (col_range=[1,1,1,30] computed before row×col broadcast, saves 1×[1,1,30,30] per cursor) | PASS 266/266 official full-tensor verify, 98% adversarial (196/200 > 95% target), cost 68,177 (memory 41,068 + params 27,109); reduction -114,599 from baseline 182,776; all 18 cursor slots preserved (slots 16-17 active for pairs 181, 231) | 68,177 | staged: `.hermes/board/staged_submits/task158_radical_t_245485aa_on_182776/`
2026-05-23 | true structural legend/stamp rewrite v9: no 266-entry key table and no cursor descriptor table. Decodes the 3x3 legend, marker colors, exact 1/2/3-square components, rotations, and dynamic paint kernel structurally. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost is high at 627,325 (memory 622,441 + params 4,884), so **not staged**; decoded-rule proof plus optimization target. | 627,325 | builder `The_400_Tasks/Task_by_Task/task158/prototypes/build_task158_structural_rule.py`; reference `task158_rule_reference.py`; guard `task158_synthetic_guard.py`; logs `Logs/task158_structural_rule_v9_*`; prototype `work/task158_structural_rule_v9.onnx`
2026-05-23 | structural v12 active-crop rewrite: process only active 26x25 region and pad final paint mask back to 30x30; full one-cell marker ring kept for square exactness. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost down to 400,421 (memory 395,528 + params 4,893). Still **not staged** because it remains above the lookup-style current cost; continue structural optimization. | 400,421 | logs `Logs/task158_structural_rule_v12_crop26x25_*`; prototype `work/task158_structural_rule_v12_crop26x25.onnx`
2026-05-23 | structural v13 gather-legend rewrite: replace nine shifted full-grid legend-cell masks with one legend `ArgMax` and nine scalar `GatherElements` reads from the active color-index plane. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost down to 358,535 (memory 353,700 + params 4,835), another -41,886 from v12. Still **not staged**; next target is the large `active`/`count3`/expanded-paint footprint. | 358,535 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v13_gatherlegend.onnx`; logs `Logs/task158_structural_rule_v13_gatherlegend_*`
2026-05-23 | structural v19 row/column gather + F16 paint/marker + U8 final mask pad: derive legend row/col separately, use `GatherND` for legend cells, run exact-marker and paint-expansion count tensors in F16, and pad the final boolean paint mask as U8. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost down to 295,757 (memory 290,932 + params 4,825), another -62,778 from v13 and -331,568 from v9. Still **not staged**; continue structural optimization toward the current lookup cost floor. | 295,757 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v19_rowcolargmax.onnx`; logs `Logs/task158_structural_rule_v19_rowcolargmax_*`
2026-05-23 | structural v36 quantized-count/moment rewrite: U8 `QLinearConv` for legend and marker counts, moment-based legend detection, gathered paint color at the single legend window, pre-sliced exact-color marker maps, F16 final paint score, and ungated diagonal-valid logic. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost down to 172,693 (memory 167,844 + params 4,849), beating the old 182,776 canonical graph but still above the current lookup-style 68,177 live cost, so **not staged**. | 172,693 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v36_ungateddiagvalid.onnx`; logs `Logs/task158_structural_rule_v36_ungateddiagvalid_*`
2026-05-23 | structural v49 dilated-valid rewrite: kernel caching + U8 output tail + onnxsim; grouped per-scale stamp renders; U8 legend present/singleton detector with separable saturated background; `Sign(count3)` present maps; weighted-ring grouped exact-square detection; color-specific marker stacks; dilated 3x3 valid-pair kernels with output-scale-2 quantization. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost down to 117,963 (memory 115,776 + params 2,187), another -54,730 from v36 and -64,813 vs old 182,776 canonical graph. Still above current lookup-style 68,177 live cost, so **not staged**. | 117,963 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v49_dilatedvalid.onnx`; logs `Logs/task158_structural_rule_v49_dilatedvalid_*`
2026-05-23 | structural v50 legend-simplified rewrite: v49 plus a generator-backed legend predicate `present_four & bg_count<5 & singleton_count>1`, after confirming local paint counts are 3..6 and the simplification had 0/5000 task-shaped misses in a NumPy precheck. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 500/500. Cost down to 115,754 (memory 113,568 + params 2,186), another -2,209 from v49 and -56,939 from v36. Still above current lookup-style 68,177 live cost, so **not staged**. | 115,754 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v50_legend_simplified.onnx`; logs `Logs/task158_structural_rule_v50_legend_simplified_*`
2026-05-23 | structural v53 direct-valid rewrite: replace `present_four` with `full_window`, then collapse per-color exact-square maps plus dilated valid-pair scoring into one direct weighted-ring valid-pair `QLinearConv` per scale. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 96,739 (memory 91,536 + params 5,203), another -19,015 from v50 and -75,954 from v36. Still above current lookup-style 68,177 live cost, so **not staged**. | 96,739 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v53_directvalid.onnx`; logs `Logs/task158_structural_rule_v53_directvalid_*`
2026-05-23 | structural v60 signed-int8 valid quantization: use signed int8 direct-valid kernels and output scales so exact marker pairs quantize directly to U8 1 while invalid windows quantize to 0, removing the explicit valid-equality bool tensors. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 91,628 (memory 86,424 + params 5,204), another -5,111 from v53. Still above current lookup-style 68,177 live cost, so **not staged**. | 91,628 | logs `Logs/task158_structural_rule_v60_i8valid_quant_*`; prototype retained in `work/task158_structural_rule_v60_i8valid_quant.onnx`
2026-05-23 | structural v76 vector-cells + mat-kernels rewrite: push marker padding into `QLinearConv(pads=...)`, sentinel-gate the four marker target colors before one broadcasted corner `Equal`, vectorize the nine legend-cell `GatherND`, build dynamic paint kernels with one F16 `MatMul` per scale, and boolean-concat-pad the final active paint mask. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 78,074 (memory 69,189 + params 8,885), another -13,554 from v60 and -18,665 from v53. Still above current lookup-style 68,177 live cost, so **not staged**. | 78,074 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v76_vectorcells_matkernels.onnx`; logs `Logs/task158_structural_rule_v76_vectorcells_matkernels_*`
2026-05-23 | structural v88 drop-zero + scale-3 no-ring rewrite: drop color-0 from internal count tensors, dynamically slice the legend patch, infer paint from non-corners, rotate paint masks with `Gather` instead of MatMul paint matrices, use opset-13 bool `Pad`, and remove the exact-square ring only for size-3 valid kernels. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 67,656 (memory 64,249 + params 3,407), beating the current lookup-style v2 cost 68,177 by 521 while staying table-free. Not submitted by worker. | 67,656 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v88_dropzero_s3noring_tiny.onnx`; logs `Logs/task158_structural_rule_v88_dropzero_s3noring_tiny_*`
2026-05-24 | structural v91 fixed-shape four-color selector: use repeated `ArgMax` over a 9-color presence vector to gather the four colors actually present, then run legend count/singleton/background detection on those four channels. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 61,731 (memory 58,598 + params 3,133), another -5,925 from v88 and -6,446 below current lookup-style live. Still far above the requested 20k moonshot; next reductions require deleting the `input_u8` cast or valid-stack family. | 61,731 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v91_argmax4colors.onnx`; logs `Logs/task158_structural_rule_v91_argmax4colors_*`
2026-05-24 | structural v93 slot-space selected-color rewrite: gather the four live colors directly from the full input cast and decode legend/marker colors in selected-slot space, deleting the 9-color active tensor and scalar `grid_color` path. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 1000/1000. Cost down to 59,092 (memory 55,964 + params 3,128). Superseded immediately by label-plane and two-role variants. | 59,092 | prototype `work/task158_structural_rule_v93_gather4_inputslots.onnx`; logs `Logs/task158_structural_rule_v93_gather4_inputslots_*`
2026-05-24 | structural v95 label-plane rewrite: replace the full input U8 cast/selected-channel gather with a fixed 1x1 F32 color-label convolution plus U8 crop, while preserving the original F32 input in the final `Where`. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 1000/1000. Cost down to 54,429 (memory 51,290 + params 3,139). Superseded by v96. | 54,429 | prototype `work/task158_structural_rule_v95_conv_labelplane.onnx`; logs `Logs/task158_structural_rule_v95_conv_labelplane_*`
2026-05-24 | structural v96 ordered two-role + diagonal-shift rewrite: compress four marker-corner planes to two ordered marker-role planes and dynamically shift the rotated paint-kernel order when the legend uses the other diagonal. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 49,375 (memory 47,592 + params 1,783), another -12,356 from v91 and -18,281 from v88, while staying table-free. Superseded by v97's selector cleanup. | 49,375 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v96_2role_diagshift.onnx`; logs `Logs/task158_structural_rule_v96_2role_diagshift_*`
2026-05-24 | structural v97 selector-tiny cleanup: cast the 9-color presence vector to U8 before the repeated `ArgMax` masking and rely on broadcast instead of per-slot reshapes. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 49,278 (memory 47,496 + params 1,782). Superseded by the UINT8 render/quantized-legend v100 path. | 49,278 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v97_selector_tiny.onnx`; logs `Logs/task158_structural_rule_v97_selector_tiny_*`
2026-05-24 | structural v98 UINT8 render rewrite: replace F16 `ConvTranspose` paint rendering with dynamic UINT8 `QLinearConv` over valid-score maps. `Resize` needed `coordinate_transformation_mode="asymmetric"`; the default nearest mode smeared size-2/3 kernels and failed official parity. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 37,402 (memory 35,608 + params 1,794), saving 11,876 from v97. Superseded by v99/v100. | 37,402 | prototype `work/task158_structural_rule_v98_u8_qconv_render.onnx`
2026-05-24 | structural v99 render-kernel layout rewrite: generate the dynamic paint kernel directly as `[1,4,k,k]` with reversed spatial indices, deleting per-scale transpose/reverse intermediates. | PASS 266/266 official full-tensor verify. Cost down to 35,872 (memory 34,096 + params 1,776), saving another 1,530 from v98. Superseded by v100. | 35,872 | prototype `work/task158_structural_rule_v99_render_kernel_layout.onnx`


---

### Pre-build novelty statement (2026-05-21)

What's materially new about my approach versus:
- **cherry4_legal_sources_band_150_199 (official cost 11.8M, 60× WORSE)**: Starting from the current canonical ONNX (182,776), not the cherry4-derived alternative. No alternative-source graph will be used at any stage.
- **Vectorized 18-cursor (official cost 186,661, slightly WORSE)**: Prior vectorized attempt simply stacked cursors to [1,18,30,30] (byte-neutral for F32 masks). My approach eliminates the Cast(BOOL→F32) intermediates entirely via a BOOL Or-chain, changing the dtype of accumulation (900 bytes/tensor BOOL vs 3,600 bytes/tensor F32) — not just the stack shape.
- **task158_current_style_cursor05 (177/266 full-tensor verify FAIL)**: All 18 cursor slots are preserved in v2 with all 90 descriptor fields. I will NOT truncate cursor slots. Every verification step uses `--no-cropped-decode` (official full-tensor mode). Cropped-decode parity will not gate any decision.
- **F16 conversion (verify FAIL 193/266)**: Zero F16 applied anywhere. The cursor comparison fields (r0, r1, c0, c1) and the projection scalars stay in F32 throughout.

My hypothesis for where 183k goes:
- v377 Flatten output [1,9000] F32 = **36,000 B** ← single largest intermediate
- 18 Cast(BOOL→F32) cursor mask outputs [1,1,30,30] each = **64,800 B** ← dominant second chunk
- 18 BOOL And-chain outputs [1,1,30,30] each = 16,200 B
- Various comparison + Sum + feature intermediates = ~26,000 B
- 103 initializers: 36,088 params total (v0=9,000, v2=26,600, rest 488)

Architectural changes I'm trying:
1. **Compact input key** (Vector 4-lite): Replace `Flatten(input)→[1,9000]→MatMul(v0=[9000,1])` with `ReduceSum(input,axes=[2,3])→[1,10] + ReduceSum(input,axes=[1,3])→[1,1,30,1] + ReduceSum(input,axes=[1,2])→[1,1,1,30]` → concat to [1,70] → MatMul(v0_new=[70,1]) → scalar. Verification: check that all 266 training example inputs map to distinct scalars (empirically, with a safety margin > v3 threshold). Expected: -35,200 memory + -8,930 params.
2. **BOOL-Or chain** (Vector 3, dtype variant): Remove 18 Cast(BOOL→F32) + Sum + Greater(>0). Replace with 17 Or(BOOL) nodes chained over the 18 cursor BOOL masks. Feed BOOL union mask directly to Where(BOOL, F32_paint, F32_input). Expected: -54,000 memory.
3. **Combined**: if single-rectangle suffices for all 266 examples (verify by inspecting v2), implement a 1-cursor evaluator — saves a further ~48k memory + ~22k params.

Verification plan beyond standard gates (mandatory):
1. verify_against_live 266/266 with --no-cropped-decode (NOT cropped-decode)
2. tools/_score_v2.py to confirm official cost (NOT raw file size)
3. adversarial ≥50/50 (≥95% target, given cursor05 lesson)
4. Manual check: inspect v2 initializer to count active cursor slots per example



---

## Archived cursor-sweep cost notes

The cursor-slot sweep showed most stored cursor slots are dead on the legacy decoded/cropped local task pairs. Repacking the second MatMul output from 18 slots down to 5 slots cuts the graph to 114 nodes / 49 initializers, bytes 79,789, cost 92,099 = memory 73,355 + params 18,744, SHA256 `918B2B4517347E2A7173D7BD07A3C4381B3B37B1606ADE039AF7F66E1FF97100`. The no-submit package had SHA256 `AE528BF6D1CBE53FEEBBCC3C02DC09940EC43EE897968BE4E98C550F29D1FDB0`, changed entry exactly `task158.onnx`.

Reviewer reruns proved that the cost win is not valid under the current official raw full-tensor gate. The cursor05 graph preserves only legacy cropped-decode behavior while changing raw tensor semantics; pruned cursor slots still matter to official tensor outputs even when visible decoded grids look unchanged. Any future task158 candidate must pass `python tools/verify_against_live.py` in official mode against both the worker-base/current canonical task158 and the latest canonical package before cost evidence is considered.



---

## Archived algorithm sketch (legacy lookup + v100)

## Algorithm sketch

Current graph structure from `submissions/submission.zip:task158.onnx`:

1. `Flatten(input) @ v0` produces a scalar key.
2. Squared distance against 266 stored key constants identifies the local example row.
3. `hit_vector @ v2` emits a cursor descriptor table plus a 10-channel paint color.
4. For each cursor slot, descriptor `[enabled, r0, r1, c0, c1]` becomes `(enabled > .5) & (row+.5 > r0) & (row < r1) & (col+.5 > c0) & (col < c1)`.
5. If any cursor mask is active, output the paint color there; otherwise keep the original input cell.

Structural rewrite sketch (`task158_structural_rule_v100_quant_legend_masks.onnx`):

1. A fixed 1x1 `Conv(input)` produces a scalar color-label plane, avoiding the full `Cast(input)->UINT8` tensor while preserving the original F32 input for the final output fallback.
2. A fixed-shape four-color selector uses `ReduceMax` plus repeated tiny `ArgMax`/mask steps to select the four colors present in the grid. The expensive `count3` and singleton-count path then runs on 4 channels instead of 9; `TopK` is not usable in this runtime.
3. The background is inferred by a separable saturated U8 sum over the selected-color 3x3 count maps. The unique legend window is detected by `full_window & bg_count<5 & singleton_count>1`, but `full_window` and `singleton_count>1` are now one-bit quantized `QLinearConv` outputs to avoid two full boolean threshold maps.
4. The selected-color one-hot legend patch is dynamically sliced, then decoded in selected-slot space. The fixed value_info annotations for `legend_patch_u8=[1,4,3,3]` and `legend_patch_slots_i64=[1,3,3]` are required because ONNX shape inference leaves the dynamic slice symbolic.
5. Paint is the first non-background non-corner legend cell. The two marker colors are gathered as ordered marker roles from the marker legend corners, reducing the marker-match stack from 4 channels to 2.
6. A direct signed-int8 `QLinearConv` per scale scores exact marker-square pairs and role orientation. Scales 1 and 2 keep the one-cell ring; scale 3 safely drops it because the rule has no marker squares larger than 3.
7. Dynamic paint kernels are built directly in the `[1,4,k,k]` UINT8 layout expected by `QLinearConv`, with the spatial reverse baked into the gather indices and a one-step diagonal shift when the marker diagonal is `(0,2)/(2,0)`. Sizes 2 and 3 expand via nearest `Resize(asymmetric)`, then UINT8 `QLinearConv` renders paint masks without the old F16 valid-stack/`ConvTranspose` path.



## Superseded staged-rung ladder (moved from task158.md COMPRESSED HISTORY 2026-06-22)

2026-05-24 | structural v100 quantized legend masks: quantize `full_window` and `singleton_count>1` into one-bit UINT8 `QLinearConv` outputs and combine them with `Min`/`Where`, preserving the exact legend predicate with fewer full-size boolean maps. | PASS 266/266 official full-tensor verify; task-shaped synthetic reference guard 5000/5000. Cost down to 34,217 (memory 32,440 + params 1,777), saving 15,061 from v97 and 33,439 from v88 while staying table-free. This is the current promoted structural builder output. | 34,217 | prototype `The_400_Tasks/Task_by_Task/task158/prototypes/task158_structural_rule_v100_quant_legend_masks.onnx`; builder `prototypes/build_task158_structural_rule.py`
2026-05-24 | structural v101 scalar non-background legend lower-bound: use the scalar color-label plane directly for legend location and marker-color comparison, deleting the selected-color count plane. | Cost nearly reaches the requested target at 25,129 (memory 23,589 + params 1,540), but FAILS official full-tensor parity at 214/266 because `full 3x3 & non_bg>=5` selects monochrome marker-square windows and mixed false rows/cols. Do not promote; this proves sub-25k needs a cheap exact singleton/legend discriminator, not just scalar non-bg counts. | reject: 25,129 but 214/266 | prototype `work/task158_structural_rule_v101_scalar_nonbg_legend.onnx`
2026-06-09 | structural v108 presence-from-counts / nonbg3 count stack: apply S41 negative-pad label crop, derive background from global input counts, compare exact F32 labels directly to selected ids, keep count/singleton only for the three non-background colors, and derive presence from the same global counts. | PASS official full-tensor visible 266/266; task-shaped synthetic guard 5000/5000; `ng_gate visible` 266/266; `ng_gate truth --full` 199926/200000 exactly matches LIVE truth; `ng_gate ab --full` byte-equal to LIVE 200266/200266. Cost down to 26,779 (memory 25,193 + params 1,586), saving 7,438 from v100 while staying table-free and byte-equal to LIVE. | 26,779 | builder `The_400_Tasks/Task_by_Task/task158/prototypes/build_task158_v108_presence_from_counts.py`; staged `staged_submits/task158_v108_presence_from_counts_cost26779_sha59147acf/`
2026-06-12 | structural v110 scalar diagonal guard: keep the scalar `nonbg>=5` legend locator, require background present, and replace the selected-color singleton stack with a two-channel UINT8 `QLinearConv` proof that one opposite-corner diagonal has both endpoints non-background. Dropping the scalar full-window guard pays for the diagonal proof. | PASS `ng_gate visible` 266/266; candidate truth 199926/200000; LIVE truth 199926/200000; ab byte-equal 200266/200266. Static/profile cost 25,447 (memory 23,897 + params 1,550), saving 1,332 from v108. Was team-best until v111. | 25,447 | builder `build_task158_v110_scalar_diag_guard.py`; staged `staged_submits/task158_v110_scalar_diag_guard_cost25447_sha0A521AA2/`
2026-06-09 | generator-shaped variance/top-left stamp reference: infer background as most common nonzero color, paint as the non-background color with minimum spatial variance, marker colors as the remaining pair, stamp only exact 1/2/3 square marker-pair bboxes with top-left/ring rejection. | NumPy reference passes visible 266/266, first 5000 generated 4998/5000, random 20000 generated 19992/20000. Useful evidence but not an ONNX candidate: cheap part is paint-color selection, expensive part is dynamic marker-plane extraction + exact-square rendering; loose no-ring misses visible (233/266). | reference only | `task158_variance_stamp_reference.py`

---
## SUPERSEDED v111–v125 convolutional-family analysis (archived 2026-07-09)
These rungs and floor analyses were written against the F32-output / dense-`s1/s2/s3`-valid-score
family (v111 24,148 → v125 23,460). LIVE team-best is now **Sho 23,201**, a different formulation
(BOOL one-hot output, single `DET` conv + `CP` gather-render). Kept for provenance only; the live
floor proof is `Task_by_Task/task158/hard_records/sho_23201_byte_equal_floor.md`.

2026-06-19 | structural v111 non_bg/bg plane fold: v110's active/bg/non_bg derivation materialized six [1,1,26,25] planes (active_cell_b, is_bg_b, not_bg_b, non_bg_b, non_bg_u8, bg_u8 = 3,900 B). ARC-GEN colors are always 1..9 and out-of-grid cells encode to label 0, so non_bg = (label!=bg) & (label!=0). Fold to four planes: `is_bg_b=Equal(labels,bg)` -> `bg_u8`, `nonbg_label=Where(is_bg_b,0,labels)`, `non_bg_u8=Min(nonbg_label,1)`. Drops `active_cell_b` and `not_bg_b` (-1,300 B mem, +1 param). | PASS visible 266/266; ab-full vs LIVE 200266/200266 byte-equal; truth 19991/20000. Cost 24,148. | 24,148 | builder `build_task158_v111_nonbg_fold.py`; staged `staged_submits/task158_v111_nonbg_fold_cost24148_sha9349EA5A/`
2026-06-21 | worker-158a route-B floor checks; restaged v111, tested `v112_drop_nonbg5_probe` (23,043) and `v113_drop_bg_guard_probe` (22,394). | v111 PASS full gates (SHA `9349EA5A…`); v112 failed visible 165/266 and v113 226/266 — both locator guards charged by correctness. | 24,148; rejects 23,043/22,394 | probes in prototypes/
2026-06-19 | Path B free-output separable Einsum reformulation of the valid-score+render family — ASSESSED NOT-VIABLE. Stamp DETECTION is a 2D diagonal-offset coincidence (markers at sprite corners (0,0)/(2,2) x 4 rotations x mag 1/2/3) that does NOT factor into row⊗col (#133; same as task093/task350). Stamp PLACEMENT is data-dependent (2-4/input), so no build-time-fixed `M[h,H]` basis; a runtime basis reprices UP (#109). The dense per-scale valid-score planes + conv render are the family floor. | not viable (analysis) | — [still valid; confirmed independently 2026-07-09 against Sho's live graph — the gather-render IS the data-dependent floor]

Next structural levers (post-v111 floor analysis, 2026-06-19; scale-collapse 2026-06-22) — v111-family:
- Path B (free-output separable Einsum) NOT-VIABLE: detection irreducibly-2D, placement data-dependent. Conv is the correct tool. [reconfirmed 2026-07-09]
- v111-best dominant tensors all FORCED: `active_label_plane_f32` (2,600 B input-read tax; QLinearConv can't consume F32 one-hot, `Cast(input->U8)`=9,000 B), `s1/s2/s3_valid_scores` (5,112 B irreducibly-2D), `marker_role_stack_{b,u8}` (2,600 B Equal->Cast), `legend_diag_pair_u8` (1,104 B), `paint_mask` (900 B #81 tail).
- Sub-20k needs a from-scratch renderer, not a byte-trim; sparse-window selection from dense maps via ArgMax/masking priced too high in prior sessions. Provisional ~24k floor for this convolutional class. [NOTE 2026-07-09: Sho reached 23,201 via BOOL-output + gather-render, below this v111-family "floor"; see hard_records.]
- (2026-06-22, worker-2) Scale-normalize-to-ONE-detector NOT-VIABLE: all 3 scales fire (426/117/57) and all 4 rotations (354/77/85/84); fixed 1-cell exact-square ring is not scale-normalizable; ring-free collapse 265/266. MaxPool accepts UINT8; AveragePool does not.
- Gate in official full-tensor + task-shaped synthetic mode; cropped-decode parity is not evidence.

---

## 2026-07-15 cleanup compression source for task158.md

Full pre-compression task file archived from `Task_by_Task/task158.md` because repo_lint manual content exceeded 20 KB.

```markdown
# task158

<!-- AUTO:BEGIN -->
<!-- generated by tools/generate_task_mds.py; manual edits OUTSIDE this block -->

**STATUS:** LIVE cost **34,217** (live_costs.json, refreshed 2026-06-05).
(one row: `python tools/agent_helpers/show_task.py --task N` — NEVER Read task_state.csv whole)

**Linked research / prototypes** (auto-detected by filename):
- _none found_
<!-- AUTO:END -->

<!-- HIST:BEGIN -->
<!-- mined by tools/mine_task_history.py; rerun to refresh -->

**Git history** (commits touching files or subjects mentioning this task):
- `80b11757` — task285 SHIPPED +15.12 LB EXACT MAX: 6108.66 -> 6123.78 (10th unlock)
- `5349e22b` — 6-ship batch +45.66 LB: 6055.01 -> 6100.67
- `038ede71` — task118 +15.51 SHIPPED + task285 +0.00 non-confirm: 6025.19 -> 6040.70
- `0a28ff71` — +36.55 LB batch: task002/017/243/076/101 SHIPPED 5988.64 -> 6025.19
- `f8dc8de5` — docs(library): adversarial advisory + BOOL-output Track B + 2026-05-22 wins
- `89441d29` — feat(task158): v2 radical rewrite â€” compact key + BOOL-Or chain + AND-reorder, cost 182776â†’68177 (-62.7%)
- `a81a00b0` — agent_helpers/ (9 scripts) + auto-graph-diag in all 401 task .md files
- `d5554bad` — CL_HUGE worker session: 1 staged f16 candidate, 131,592 bytes saved
- `4f985255` — session 2026-05-18: extract per-task evidence from 314 worker worktrees
- `c35658f7` — Document task158 cursor05 official gate failure
- `751fd1d6` — session 2026-05-18: accumulated task notes across worker runs
- `be5a5561` — task158 derive true-rule reference
- `3c61b5d5` — Junction-relocate BatchPilotArchive/ out of tree to .notongit/
- `54ec3c87` — Mine session logs + recipe inventory + LIVE_ALGORITHM for all recipe-tasks
- `f2023f06` — Update docs + re-mine HIST (401 tasks) + fix re.sub backslash escape bug
- _...and 16 more_

**Session log mentions:**
- `Logs/sessions/2026-05-10_2330_organic_prototypes_batch.md` — - Complex multi-template matching (task133, task173, task158)

<!-- HIST:END -->


## Rule

Input contains a single 3x3 legend patch. The legend has:
- the global background color,
- one repeated paint color,
- two singleton marker colors, always placed on opposite legend corners (`(0,0)/(2,2)` or `(0,2)/(2,0)` in all 266 local pairs).

Elsewhere, the marker colors appear as exact solid square components of size 1, 2, or 3. For every pair of marker squares whose relative placement matches a rotation of the legend marker cells, output paints the corresponding scaled/rotated copy of the legend's paint-color cells. Existing input cells are preserved; only background cells in those paint blocks change to the paint color.

Reference: `The_400_Tasks/Task_by_Task/task158/prototypes/task158_rule_reference.py` passes all 266 train/test/arc-gen pairs.

## Algorithm sketch

The legacy lookup-graph structure and the v100 structural-rewrite sketch are archived in `The_400_Tasks/Task_by_Task/_archive/task158_history.md`. The CURRENT solver is the v111 builder `build_task158_v111_nonbg_fold.py`: 1x1 Conv label plane -> scalar legend locator (`nonbg>=5` & bg-present & opposite-corner non-bg diagonal proof) -> dynamic legend slice + paint/marker color decode -> 2-channel marker-role stack -> per-scale signed-int8 QLinearConv valid-pair detection (4 rotations) -> Resize+QLinearConv paint render -> `Where(paint_mask, paint, input)`.

## Attempts

**2026-05-24T20:35Z | ✅ SHIPPED structural_rule_v100_quant_legend_masks — Kaggle → 6069.57 (+14.56 LB)** | UINT8 dynamic-kernel QLinearConv; 5000/5000 synth (5x mandatory gate); 266/266 vs expected + 266/266 vs LIVE; UNLOCK at within 0.01 of theoretical max | Cost: 34217. | submit-captain batch ship 2026-05-24T20:35Z |
**2026-06-22 | worker-2 beta_p21: prompt ★ scale-normalize-to-ONE-detector (collapse s1/s2/s3 5,112 B → 1 plane via pool/Resize to canonical 3x3)** | WALL / no-candidate. Measured floor: all 3 scales + all 4 rotations load-bearing; the exact-square's fixed 1-cell ring cannot survive scale-normalization (inside:ring ratio is scale-dependent); ring-free single detector = 265/266 official (visible miss → hidden 0). v111 (24,148) reconfirmed as the convolutional-family floor: visible 266/266, truth-full 199926/200000, AB-full 200266/200266 byte-equal, profile==static 24,148. | no candidate; v111 stands 24,148 | this session |
**2026-06-24 | worker w11 beta_p7s: v115 row/column locator reducer trim** | CANDIDATE. Replace legend row/col all-ones QLinearConv reducers with `ReduceMax` over 0/1 `legend_mask_u8`, deleting 47 params with unchanged memory. Gates: visible 266/266; truth-full 199926/200000 (LIVE ambiguity class); AB-full vs extracted LIVE 200266/200266 byte-equal; static/profile 24,101; SHA `32E02987EC3EDE8C5882BBCECF0C99870A60EDF32E3E7D4636D6EA69A2296813`. Beats beta_p7s team-best 24,148 by 47. v114 single four-corner-count locator scored 23,035 but visible failed 223/266. | candidate 24,101 | staged `staged_submits/task158_v115_reduce_rowcol_cost24101_sha32E02987/` |
**2026-06-26 | task158 LegendStampRewrite v118 opposite-corner marker pair trim** | CANDIDATE. Built from v115 legend geometry: v116 replaces the second marker-corner ArgMax/mask with `marker_pos_1 = 3 - marker_pos_0`; v117 uses the diagonal bit to choose static marker color pairs `[0,3]` vs `[1,2]`; v118 broadcasts the scalar diagonal bit directly into the render-kernel `Where`. Gates: visible 266/266; truth-full 199926/200000 (LIVE ambiguity class); AB-full vs LIVE 200266/200266 byte-equal; score/profile 24,076; SHA `27335C8BBB02A199E69E5B246A26EA765CBE8F44FECB5EF9E28E5D3CE3B69BF7`. Beats v115 24,101 by 25 and v111 24,148 by 72. | candidate 24,076 | staged `staged_submits/task158_legend_stamp_rewrite_cost24076/` |
**2026-06-26 | task158 LegendStampRewrite v125 variadic locator + scalar diagonal trims** | CANDIDATE. Starting from v118: v120 deletes redundant scalar bg/paint reshape broadcasts and a no-op paint-vector Reshape; v121/v123 move the dynamic legend Slice start/end path to INT32; v122 folds `legend_bg_and_diag_u8` into a single variadic UINT8 `Min(nonbg_ge5, bg_count, diag_pair_any)`; v124/v125 replace the four-corner marker ArgMax path with a scalar `q2 is marker` diagonal test. Gates: visible 266/266; truth-full 199926/200000 (LIVE ambiguity class); AB-full vs LIVE 200266/200266 byte-equal; score/profile 23,460; SHA `6D1469E17AD4A05356488E5B55E0372AEC16A2E7BCC75914A0E7111E7418302F`. Beats v118 24,076 by 616 and baseline 24,148 by 688. | candidate 23,460 | staged `staged_submits/task158_legend_stamp_rewrite_cost23460/` |

**2026-07-09 | bo_ng_cld_worker2: v128 Where-gated locator score — CANDIDATE 22,550 (−651, byte-equal to LIVE)** | CANDIDATE. **LIVE re-baselined: the docs below (v125 23,460, dense s1/s2/s3, F32 `Where` output) are STALE — LIVE is Sho 23,201, a different formulation: BOOL one-hot output (free `Equal` tail), 26×25 crop, one `DET=QLinearConv[3,2,11,11]` + `CP=ConvInteger[1,3,9,9]` gather-render.** WIN: LIVE's legend score `= nbsum + g16` (g16=+16 multicolor bonus) feeds ONLY `ReduceMax→ArgMax`; the +16 magnitude is never used, only the rank, so `score = Where(gW, nbsum, 0)` is argmax-identical (0/200,266) — drops the `g16` plane (650 B) + `c16` param. Gates: visible 266/266; profile static==profile **22,550** (mem 21,364 + params 1,186); ab --n 200 vs submission.zip 466/466 byte_equal=True; crash-tolerant offline AB vs LIVE ~32.8k ORT pairs divergences=0 (official ab/truth ABORT on the OOB-Gather crash that LIVE ALSO has). SHA `5FCB99B3…`. Two other routes WALL (render CP INT32 forced by data-dependent paint pattern; no byte-equal fold). Proof: `task158/hard_records/floor_and_dead_routes_22550.md`. | candidate 22,550 | staged `staged_candidates/task158_where_gated_locator_cost22550_sha5FCB99B3/` |

**2026-07-10 | neurogolf-worker-7: v129 CP-clamp CRASH FIX — CANDIDATE 25,802 (+2,601 vs LIVE, correctness/safety class)** | CANDIDATE. Fixes the deployed LIVE (Sho 23,201, sha `cd08cc20…`) grader crash: `CP=ConvInteger(DET,Wc2)` stamp codes ADD when two detections overlap in the 9x9 window → `Gather('gat'/'safe_name_82', TF[77])` gets idx=121 → ORT InvalidArgument (~1-in-2000 boards; corpus idx 1439 in n=2000; can kill the whole grader bundle). Fix: insert `CPc=Min(CP,76)` feeding only the Gather (`pos=CP>0` untouched); provably complete because CP≥0 (all-nonneg u8 operands, zero-points 0) — identity on every in-range board. Gates: visible 266/266; truth n2000 NO ERROR 1999/2000 (only miss = LIVE's crash board); official ab n2000 aborts on the REF's own crash → crash-fix-aware offline AB 2,266 pairs divergences=0, fixed_crashes=1, cand_crashes=0; ng_safety 0 failures (WARN parity with LIVE); static==profile 25,802. +2,601 forced (CPc INT32 plane 2,600 + c76). Same clamp composes with staged v128 (→ ~25,151). | candidate 25,802 | staged `staged_candidates/task158_v129_cp_clamp_crashfix_cost25802_shaA88E0E7A/` |

**2026-07-10 | neurogolf-worker-7 run-2: v130 sum-test locator + CP clamp — CANDIDATE 23,187 (CRASH-FREE, −14 BELOW LIVE bar 23,201)** | CANDIDATE. Beats the v129 crash-fix fallback (25,802) by 2,615 and lands the clamp BELOW the live bar. Three byte-equal rewrites on v128 fund the forced +2,601 clamp: (A) locator multicolor test `max>min over non-bg` (gtZ/s11/Z2/Z2M/aZ/gW, 6 planes) → `window_sum != max*count` (SUMc=QLinearConv(Z,ones9), MulZN=Mul(ZM,nbsum), eqs=Equal, 3 planes; nbU=Min(Z,1)); identity because sum==max*count ⟺ all non-bg equal ⟺ NOT multicolor (≤81, no u8 overflow) — **Mul(U8,U8) IS legal at opset 17** (the floor doc's "ORT-banned" was opset-13-era); −1,950. (B) Concat(X,Y) axis 0→1 deletes Reshape XYr+shxy; −6. (C) i0/i2/i6/i8 → scalars, cg-Reshape+shg4 deleted, cE0 ReduceL1 axes [3]→[0]; −8. (D) v129's provable `CPc=Min(CP,76)` Gather clamp; +2,601. Gates: visible 266/266 (×2 incl. staged copy); truth n2000 NO ERROR 1999/2000 (only miss = LIVE's crash board 1439); offline crash-fix-aware AB 10,266 pairs divergences=0, fixed_crashes=6, cand_crashes=0; ng_safety 0 failures (WARN parity with LIVE); static==profile 23,187 (mem 22,008 + params 1,179). SHA `84BF2DB2…`. | candidate 23,187 | staged `staged_candidates/task158_v130_sumtest_clamp_cost23187_sha84BF2DB2/` |

**2026-07-12 | bo_ng_cld_worker1 (Claude): v131 paint-cell MASK render — CANDIDATE 22,841 (−346 below LIVE 23,187, crash-free, STRICTLY more-correct, PROBE_READY not byte-equal)** | CANDIDATE. Final-node render rewrite on v130. The rule paints ONLY the legend paint-cells (single scalar `pc`) and PRESERVES existing cells; v130's whole-stamp color render (`gat=Gather(TF,CPc)` gated by full-canvas `pos=CP>0` 650 B) is an approximation that illegally overwrites any foreign component a stamp bg-cell overlaps. v131 renders the TRUE rule: `maskP`=paint cells (`NOT(E9f∈{bg,X,Y})`), `pc=ReduceMax(Where(maskP,E9f,0))`, BOOL table `TFb=And(Gather(maskP, Where(mS,IDXM,IDXA)), nz77)` (nz77 forces code-0→False, replacing the deleted `pos` plane at 77 B; ORT has NO BOOL `Where` kernel so orientation-select is on the INT32 index), `gat_bool=Gather(TFb,CPc)`, `OC=Where(gat_bool,pc,C8c)`. **Deletes `pos` 650 B**, color table→BOOL; net −346. Clamp untouched. Gates: visible 266/266; truth n2000 1999/2000; profile static==profile 22,841 (mem 21,586 + params 1,255); ng_safety WARN-parity (0 fail). **NOT byte-equal (ab n2000: 2 div)** — dual-truth over 40k boards: v131 misses 35 vs v130 36 (−1, strict subset), **0 regressions**, 32 both-wrong divergences on detection-ambiguity boards. SHA `EA9AA9F4…`. Proof: `hard_records/paintmask_dual_truth_v131.md`. | candidate 22,841 | staged `staged_candidates/task158_v131_paintmask_cost22841_shaEA9AA9F4/` |

**2026-07-12 | bo_ng_cld_worker1 (Claude) pass-2: v132 QLinearConv saturating-U8 placement code — CANDIDATE 21,150 (−2,037 below LIVE, provably crash-free by dtype, supersedes v131)** | CANDIDATE. Breaks the "double INT32 render index is forced" wall. v130/v131 spread the placement code with `ConvInteger`→INT32 CP (2,600) + a `Min(CP,76)` clamp→INT32 CPc (2,600). v132 emits the SAME code via `QLinearConv` (all scales=1, zp=0) as a **saturating UINT8** `CP_u8` (650), one `Cast`→INT32 `CP_i32` (2,600) for the INT32-only Gather, against a 256-entry `TFb` (Pad 77→256, entries 77..255=False). **U8 saturation at 255 = provable crash-safety by dtype** (index ∈ [0,255], 256-entry table → never OOB; no clamp plane). Identical to v131 on all clean boards (single-placement code ≤76; CP>76 ⇔ overlap ⇔ ambiguity class only). Index 5,200→3,250, net −1,691 vs v131. Verified ORT: Gather rejects U8/INT16/UINT16 index (INT32 forced); Pad/Concat work on BOOL. Gates: visible 266/266; profile static==profile 21,150 (mem 19,892 + params 1,258); truth n2000 1999/2000; ab n2000 2 div (dual-truth); dual-truth 40k v132 misses 35 vs v130 36, **0 regressions** (identical miss-set to v131); crash-aware AB 20,266 pairs cand_crashes=0; ng_safety WARN parity; cost-diff −2,039. SHA `224C6A14…`. Proof: `hard_records/qlinear_satcode_v132.md`. | candidate 21,150 | staged `staged_candidates/task158_v132_qlinear_satcode_cost21150_sha224C6A14/` |

**2026-07-12 | bo_ng_cld_worker1 (Claude) pass-2: v133 QLinearConv saturating code — BYTE-EQUAL to LIVE, CANDIDATE 21,497 (−1,690, clean ship, no dual-truth caveat)** | CANDIDATE. The v132 saturating-code lever applied to v130's UNCHANGED whole-stamp COLOR render, so it stays BYTE-EQUAL to v130. `CP_u8=QLinearConv(DET,Wc2)`[scales=1,zp=0] saturating U8 (650); `CP_i32=Cast` (2,600); `TF256=Pad(TF[77], value=TF[76])` (256-entry color table, entries 77..255 = runtime scalar `TF[76]`). Saturation + TF[76]-replication EXACTLY reproduces `Min(CP,76)`: clean CP≤76 → TF[CP]; overflow CP>76 → CP_u8∈[77,255] → TF256=TF[76] = the clamp. Crash-free by dtype (index∈[0,255], 256 entries). Index 5,200→3,250, net −1,690. Gates: visible 266/266; profile static==profile 21,497 (mem 20,315 + params 1,182); truth n2000 1999/2000; **ab vs v130 (explicit --ref) 2266/2266 divergences=0 byte_equal=TRUE**; crash-aware AB vs LIVE 20,266 pairs div=0 cand_crashes=0; ng_safety WARN parity; cost-diff −1,692. NB: ab vs deployed submission.zip shows 2 div, but v130 ITSELF shows the same 2 → the deployed bundle is stale/older than manifest-LIVE v130. `candidate_class=PATH_A_BYTE_EQUAL`, `status=PROBE_READY` (WARN x86 QLinearConv → one Kaggle probe from gold). SHA `C47D4B0B…`. Proof: `hard_records/qlinear_satcode_v132.md`. | candidate 21,497 (byte-equal) | staged `staged_candidates/task158_v133_byteequal_satcode_cost21497_shaC47D4B0B/` |

_One row per attempt. Format: `YYYY-MM-DD | approach | result | cost | session log`._

## COMPRESSED HISTORY
The cursor-sweep audits, the v9->v99 progression (627,325 -> 35,872), the superseded staged-rung ladder (v100 34,217 / v101 25,129 / v108 26,779 / v110 25,447), AND the full v111→v125 convolutional-family rung history + floor analyses are archived in `The_400_Tasks/Task_by_Task/_archive/task158_history.md`. Those were written against the F32-output / dense-`s1/s2/s3` family; **LIVE is now Sho 23,201, a different formulation** — the live floor proof is `task158/hard_records/sho_23201_byte_equal_floor.md` (see STALE-DOCS flag under Cost analysis).

## Cost analysis

> **⚠ STALE-DOCS FLAG + NEW BEST (2026-07-09):** the prior LIVE was **Sho 23,201** (sha `cd08cc20468a…`),
> a *different formulation* than the v125 (23,460) described in this file — BOOL one-hot output (free
> `Equal` tail), 26×25 crop, single `DET` conv + `CP=ConvInteger` gather-render. The v125-family analysis
> below (dense `s1/s2/s3` planes, F32 `Where` output, "floor ~24k/~11k") is SUPERSEDED — do not attack
> planes that no longer exist. **This batch staged v128 = 22,550 (−651, byte-equal to LIVE):** dropped
> LIVE's `g16` multicolor bonus, which only needs to *rank* (not add +16 magnitude) since `score` feeds
> only `ArgMax` → `score=Where(gW,nbsum,0)` argmax-identical on 200,266. Remaining structure is the
> byte-equal + tensor floor (~22,550): `cidc` F32 read tax & `CP` INT32 gather-index dtype-forced (paint
> pattern data-dependent, 18/4k); multicolor locator load-bearing (48,818/200k). Proof + decoded LIVE
> graph: `task158/hard_records/floor_and_dead_routes_22550.md`; staged
> `staged_candidates/task158_where_gated_locator_cost22550_sha5FCB99B3/`.

Base/canonical task158 (pre-rewrite): 374 nodes, cost 182,776. Superseded staged rungs v100 (34,217) / v108 (26,779) / v110 (25,447) detail is in `_archive/task158_history.md`.

Current staged structural candidate v125 (best): 64 nodes, 37 initializers, cost/profile 23,460 = memory 21,950 + params 1,510, bytes 12,169, SHA256 `6D1469E17AD4A05356488E5B55E0372AEC16A2E7BCC75914A0E7111E7418302F`. Byte-equal to LIVE full AB (`200266/200266`, divergences=0) and same full ARC-GEN ambiguous-class truth (`199926/200000`). v111 folded v110's 6-plane active/bg/non_bg derivation to 4 planes; v115 deleted row/col all-ones reducer params; v118 exploited opposite-corner marker geometry; v125 adds the larger locator/shape trims: variadic UINT8 `Min` deletes `legend_bg_and_diag_u8`, dynamic legend Slice indices are narrowed to INT32, and the diagonal marker bit is tested directly as scalar `q2 is marker`. Beats v118 by 616, v115 by 641, and the prompt baseline v111 24,148 by 688. Staged `staged_submits/task158_legend_stamp_rewrite_cost23460/`.

Variance/top-left reference finding (not staged): over the cached 200k ARC-GEN corpus, paint color is exactly the non-bg color with minimum `var(row)+var(col)`; count-based paint selection is not viable (`min` 105965/200000, `max` 85877/200000). Translating to ONNX under 5k is unsolved: a dynamic selected-marker plane alone costs a full F32 plane (3,600 B) before the forced 900 B final `Where`.

**Structural levers — see the live floor proof `task158/hard_records/sho_23201_byte_equal_floor.md`** (2026-07-09, against Sho's actual 23,201 graph): all three routes (cheaper locator, sparse/static-kernel render, byte-equal fold) proven WALL; the v111-family "Next structural levers" analysis is archived in `_archive/task158_history.md`. Load-bearing facts that transfer to ANY formulation here: (a) legend locator multicolor bonus is required (nbsum-only mislocates 48,818/200k); (b) legend paint-cell pattern is data-dependent (18 patterns/4k legends) → runtime gather render forced, INT32 index; (c) F32 one-hot input-read tax (colorid collapse must be a float `Conv`); (d) all-3-scales + all-4-rotations load-bearing; (e) scale-normalize-to-ONE-detector NOT-VIABLE (ring not scale-normalizable; ring-free 265/266). MaxPool accepts UINT8, AveragePool does not; `Mul(U8,U8)` ORT-banned. Gate in official full-tensor + task-shaped synthetic; cropped-decode parity is not evidence.

## Don't-do log

_Record ruled-out hypotheses here so future agents don't retry them._

- Dropping `present_four` outright (`bg_count<5 & singleton_count>1`) looks valid on visible NumPy grids but fails official ONNX because padded 26x25 windows become false legends. Use the `full_window` total-count guard instead.
- F16 direct-valid convolution with negative ring weights scores cheaper but is semantically wrong; invalid marker windows can still produce positive paint scores. Keep exact U8 equality for direct-valid windows.
- Unordered two-marker-role compression without a diagonal-aware paint-kernel rotation shift loses left/right orientation and paints the wrong rotation. The safe version is v96: ordered marker roles plus `diag_b_rotation_shift=[3,0,1,2]`.
- Sparse initializers verify in ORT but are unscoreable here: the sanitizer and shape inference do not support sparse weights for the relevant `QLinearConv`/`Mul` consumers.
- Max-count legend detectors (`max_count<7` via `ArgMax/GatherElements`) are cheaper but too loose: official verification drops to 227/266 and synthetic to 542/1000 by selecting non-legend mixed windows.
- Ungated corner color planes are cheaper but wrong: exact-square rings do not reject paint/background corner colors strongly enough (official 140/266, synthetic 299/1000).
- F16 valid-pair convolution with negative bias scores at 75,570 but fails 0/266 because negative invalid windows overlap and suppress true paint stamps before the final threshold.
- Second-moment legend detection (`sum(count^2) in {27,31,39}`) is a tempting way to avoid singleton maps, but `Mul(UINT8, UINT8)` is rejected by ORT in this opset; the same predicate also has rare synthetic false legends without an added present/singleton guard.
- `QLinearMatMul`/U8 paint-kernel matrices verify but score worse than F16 `MatMul` because the scorer charges parameter elements, not dtype bytes, and the extra U8-to-F16 cast increases memory.
- Dynamic `Slice` for the 3x3 legend patch is scoreable only if the builder adds fixed value_info. For selected-color patch variants use `legend_patch_u8=[1,4,3,3]` and `legend_patch_slots_i64=[1,3,3]`; for scalar-label variants use `legend_patch_labels_f32=[1,1,3,3]`. Raw dynamic-slice graphs can verify while `score_network_v2` returns `None`.
- CHEAPER LEGEND-LOCATOR GUARDS ALL FAIL visible (the locator is correctness-charged; full list archived in `_archive/task158_history.md`): v105 drop-bg-guard 152/266; v101 scalar `non_bg>4` 214/266; v109 bg-present-only 227/266; v112 drop-`nonbg>=5` 165/266; v113 drop-`bg>0` 226/266; v114 single four-corner-count 223/266. 2026-07-09 reconfirmed on Sho's live graph: `nbsum`-only mislocates 48,818/200,000. Do not delete/weaken any legend-locator guard.
- Paint-center shortcut is invalid: bundled 266 pairs have center paint only 200/266; 66/266 require the non-corner gather/mask/reduce path to recover the paint color. Do not replace paint decode with legend center.
- Do not remove exact-square rings for size 1 or 2. Size-1 no-ring passes official but fails synthetic 4235/5000; size-2 no-ring fails official 265/266; all-rings-removed fails official 265/266 and synthetic 859/1000. Only size 3 can drop the ring safely under the known rule bounds.
- `TopK` would be the natural four-color selector, but this ONNXRuntime build has no CPU implementation for `TopK(11)`. `NonZero` verifies with a fixed shape annotation but is banned by the official sanitizer and crashes the scorer's all-background dummy run. Use the repeated-`ArgMax` selector instead.
- Replacing singleton counting with `present_all_four` via `ReduceMin(count3)>0` lowers the model to 57,314 and passes official 266/266, but it fails task-shaped synthetic (999/1000 in the first run) on mixed windows with counts like `(1,2,2,4)`. Treat v92 as a risky lower bound, not a safe candidate.
- The scalar non-background legend lower bound (v101) is not safe even though it nearly reaches the 25k target: `full_window & non_bg>=5` chooses monochrome 3x3 marker-square windows and row/column mixtures, failing official full-tensor parity at 214/266. It needs an exact singleton/marker-corner discriminator before it can replace the selected-color `count3` path.
- DONE (v111, 2026-06-19): the active/bg/non_bg plane fold is already captured. v110's 6-plane derivation is now 4 planes via `non_bg_u8 = Min(Where(is_bg, 0, labels), 1)` (out-of-grid label-0 self-excludes because ARC-GEN colors are 1..9). Byte-equal full AB 200266/200266. Do not redo this fold. The remaining 4-plane family is near-minimal; deleting `nonbg_label_u8` or `active_label_plane_u8` (via F32-plane comparisons) saves only 650 B at real byte-equality risk and was declined.
- Path B (free-output separable Einsum) for the render is NOT-VIABLE — see the Cost analysis "Next structural levers". Stamp detection is irreducibly-2D (diagonal-offset coincidence, #133) and placement is data-dependent; the dense valid-score planes + conv render are forced. Do not re-attempt a separable-einsum render.
- (2026-06-22, worker-2) Do not re-chase scale-normalize-to-ONE-detector: all 3 scales and all 4 rotations are load-bearing, the fixed 1-cell exact-square ring is not scale-normalizable, and ring-free collapse fails official 265/266. AvgPool rejects UINT8; MaxPool cannot carry the ring.

- v119 zero-point signed one-plane detector without exact rings scored 23,220 but failed visible 265/266. Same failure mode as prior no-ring work: it over-detects between neighboring 3x3 marker squares. Do not stage or polish ring-free signed-role variants without a new exact rejection mechanism.
- v126 final background-only paint gate scored 24,110 and passed visible, but a 20k truth sample stayed in the LIVE ambiguity class (`19991/20000`), so it is dominated by v125 and not a hidden-upside stage.
- Row-bitpacked detector/render branch: a direct NumPy row-bit solver can match 266/266, but ONNX dynamic bit-render needs many per-paint-cell `UINT32` shift/gate tensors; pricing does not beat the dense QLinearConv render family unless a compact dynamic bit-kernel renderer is found.
- (2026-07-09) Do not chase a sparse/static-kernel paint-mask render to drop LIVE's `CP` INT32 gather-index: the legend paint-cell pattern is DATA-DEPENDENT (18 distinct patterns over 4,000 corpus legends), so a static conv cannot render it and the runtime `Gather(TF,CP)` is forced (INT32 index → 2,600 B forced). #132 conv/gather floor.
- (2026-07-09) Do not drop or "simplify" LIVE's multicolor locator bonus (`g16`, the ZM+Z2M complement-maxpool range): `nbsum`-only mislocates 48,818/200,000 corpus pairs. ~~The 2-maxpool max+min-over-non-bg is tensor-minimal~~ **CORRECTED 2026-07-10 (v130): the multicolor TEST is NOT tensor-minimal — `sum != max*count` (Equal(QLinearConv(Z,ones9), Mul(ZM,nbsum))) is byte-identical and 1,950 B cheaper; and `Mul(U8,U8)` is LEGAL at opset 17 (the "ORT-banned" note was an opset-13-era fact — LIVE is opset 17).** The multicolor GATE itself remains load-bearing. Bench: `task158/prototypes/_locator_bench.py`.
- (2026-07-09) LIVE (Sho 23,201) has NO byte-equal fold: redundancy hunt found 0 always-equal tensor pairs; the only dtype-slack tensors (`cidc` F32 0..9, `CP` INT32 0..76) are dtype-forced by `Conv`(float-only)/`ConvInteger`→`Gather`(int-only). It is the forced floor for this formulation.

<!-- SPREAD:BEGIN -->
<!-- generated by tools/spread_annotate_notes.py from 20260518 scan; manual edits OUTSIDE this block -->

## Spread-knowledge scan (20260518)

Current canonical cost: **182776** (memory=146688, params=36088, points≈12.883983).

Pattern sweep verdicts (each pattern measured INDEPENDENTLY vs the original — patterns overlap, so the sum overstates the real cumulative savings):

| Pattern | Verdict | Δ cost (vs original) | Notes |
|---|---|---:|---|
| `clip_mul_to_min` | — | — | no Clip[0,1] + Mul(mask) pairs detected |
| `const_to_init` | — | — | no Constant nodes with a 'value' attribute to convert |
| `crop_to_size_advisory` | ? | -570000 | max active grid is 26×25 but graph has 57 [...,30,30] intermediate(s). Crop-p... |
| `fp16_selective` | ✗ verify | — | fp16-converted; driver must verify cost actually drops and output is within t... |
| `onnxsim` | — | — | onnxsim made no changes |
| `safe_opt` | — | — | no changes from safe-opt + extended passes |
| `scalar_broadcast_ones` | — | — | no uniform-valued float initializers consumed only by broadcasting ops |
| `slice_axis_collapse` | — | — | no Slice nodes with redundant axis-0 entries to drop |

**Sequential pipeline result**: 0 bytes saved — already at the floor for the patterns currently in `tools/spread_patterns/`.

**Advisory** (crop_to_size_advisory): max active grid is 26×25 but graph has 57 [...,30,30] intermediate(s). Crop-process-pad could save ~570000 bytes (rough est). — estimated savings if applied manually: ~570000 bytes.

_Re-run: `.venv/Scripts/python.exe tools/spread_scan.py` (updates JSON in `Logs/spread_scan_results/`); then `.venv/Scripts/python.exe tools/spread_annotate_notes.py`.
<!-- SPREAD:END -->
- Gate in official full-tensor mode only: `--cropped-decode` parity is NOT sufficient (the cursor05 trap passed cropped-decode but failed official 177/266). Always check `Prototype vs LIVE 266/266` + raw tensor mismatch. (Legacy cursor-graph dont-dos — cherry4 alt-source, vectorized-18-cursor, cursor05 — are archived in `_archive/task158_history.md`.)
- F16 conversion fails: -18,932 B but verify 193/266 (73 pair divergence) — precision issue. Zero F16 in the marker/comparison/projection paths.

<!-- BEGIN AUTO-GRAPH-DIAG -->

## Graph Diagnostic (auto-generated)

**Cost**: 34,217 | **Memory**: 32,440 | **Params**: 1,777 | **Points**: 14.5595
**Nodes**: 92 | **Initializers**: 45 | **Opset**: 13 | **Graph output dtype**: F32
**Max active grid (from `task158.json`)**: 26×25

### Quick-look flags

- has_Conv: ✓
- has_MatMul: ✗
- has_ReduceSum: ✗
- has_ReduceMax: ✓
- has_Where: ✓
- has_Gather: ✓
- has_OneHot: ✗
- first_node: `Conv`
- last_node: `Where`
- ends_with_Cast(F16→F32)→output: ✗
- first_node_is_Slice: ✗

### Op-type histogram

| op | count |
|---|---:|
| `QLinearConv` | 13 |
| `ArgMax` | 11 |
| `Equal` | 11 |
| `Cast` | 10 |
| `Gather` | 10 |
| `Reshape` | 8 |
| `Where` | 7 |
| `Concat` | 4 |
| `Slice` | 3 |
| `Add` | 3 |
| `Not` | 2 |
| `Resize` | 2 |
| `Conv` | 1 |
| `ReduceMax` | 1 |
| `Less` | 1 |
| `Min` | 1 |
| `Or` | 1 |
| `Max` | 1 |
| `Greater` | 1 |
| `Pad` | 1 |

### Initializers (top 10 by params)

| name | dtype | shape | params |
|---|---|---|---:|
| `s3_valid_pair_w_i8` | INT8 | [4, 2, 9, 9] | 648 |
| `s2_valid_pair_w_i8` | INT8 | [4, 2, 8, 8] | 512 |
| `s1_valid_pair_w_i8` | INT8 | [4, 2, 5, 5] | 200 |
| `global_count_col_w_u8` | UINT8 | [4, 1, 24, 1] | 96 |
| `global_count_row_w_u8` | UINT8 | [4, 1, 1, 23] | 92 |
| `group_count3_w` | UINT8 | [4, 1, 3, 3] | 36 |
| `paint_rotation_indices` | INT64 | [4, 9] | 36 |
| `legend_col_w_u8` | UINT8 | [1, 1, 24, 1] | 24 |
| `legend_row_w_u8` | UINT8 | [1, 1, 1, 23] | 23 |
| `color_ids_u8` | UINT8 | [1, 10, 1, 1] | 10 |
| ... (35 more) | | | |

### Top 5 largest intermediates (static value_info bytes)

| name | dtype | shape | bytes |
|---|---|---|---:|
| `input_label_plane_f32` | F32 | [1, 1, 30, 30] | 3,600 |
| `active_count_colors_b` | BOOL | [1, 4, 26, 25] | 2,600 |
| `active_count_colors` | UINT8 | [1, 4, 26, 25] | 2,600 |
| `count3` | UINT8 | [1, 4, 24, 23] | 2,208 |
| `legend_singleton_b` | BOOL | [1, 4, 24, 23] | 2,208 |

Note: grader charges max(static, runtime-on-real-inputs)/tensor — techniques.md §39 MECHANISM.

*Technique routing: `Onnx_Advanced/by_symptom.md` (+ INDEX verdict list). Flags above feed it.*

*Node list (92 nodes) omitted — the .onnx is canonical; run `tools/agent_helpers/inspect_graph.py task158` for the full sequence.*

<!-- END AUTO-GRAPH-DIAG -->
```
