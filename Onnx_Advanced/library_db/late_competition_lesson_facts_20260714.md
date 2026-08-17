# Late Competition Lesson Facts - 2026-07-14

Reviewed fact blocks from the root `.update` backlog, the three pending
`Onnx_Advanced/.update` drops, and the current `Team_Best_Onnx` snapshot. These
facts are intentionally compact so `lib_query.py search|facts` can route workers
without loading the full packet bodies.

~~~library-fact
id: neurogolf.late20260714.team_best_authority
fact_type: source_authority
subject: Current team-best authority for late competition batches
claim: As of 2026-07-14T00:05:59Z, use K-NeuroGolf-OnnxlyFans Team_Submission/Backend/manifest.csv plus Team_Best_Onnx as the current cost/ONNX authority; local task_state and old task docs can be materially stale.
status: current
confidence: live_manifest
mutable: true
source_anchor: source removed from live tree - Onnx_Advanced/team_best_snapshot_20260714.md (git history)
~~~

~~~library-fact
id: neurogolf.late20260714.loose_submission_zip_not_authority
fact_type: source_authority
subject: Loose submission.zip differed from Team_Best_Onnx
claim: On inspection the loose Backend/submission.zip differed from Team_Best_Onnx on task243.onnx and had a sha not matching current_leaderboard_state's accepted snapshot; do not use the loose zip as canonical without byte-identical reconciliation.
status: current
confidence: local_audit
task_id: task243
mutable: true
source_anchor: source removed from live tree - Onnx_Advanced/team_best_snapshot_20260714.md (git history)
~~~

~~~library-fact
id: neurogolf.late20260714.task196.binary_ring_min_tail
task_id: task196
fact_type: current_team_best_idiom
subject: Binary K=2 ring detector plus Min tail beat the prior task196 detector
claim: The confirmed public task196 win moved 4482->3167 by replacing the larger detector with a binary K=2 ring detector and a size-agnostic Min(paint,color-1) tail; profile equaled static and the Kaggle row landed as expected.
status: current
confidence: confirmed_public
op: QLinearConv
source_anchor: external sibling repo - K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/_shipped/task196.json
~~~

~~~library-fact
id: neurogolf.late20260714.task286.signed_wall_remap
task_id: task286
fact_type: technique_micro
subject: Signed wall-value remap collapses open-mask planes
claim: Remapping wall/open values so Greater(v,0) directly means open collapsed five BOOL 25x25 planes to three on task286, yielding a byte-equal 18672->17422 exact rewrite; the remaining task is still non-local maze reachability, not a cheap QLinearConv floor.
status: current
confidence: confirmed_public
source_anchor: external sibling repo - K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/_shipped/task286.json
~~~

~~~library-fact
id: neurogolf.late20260714.multicolor.sumtest
task_id: task158
fact_type: technique_micro
subject: Window multicolor test via sum != max*count
claim: To test whether a small-valued nonzero window has at least two distinct values, compare window_sum against window_max*window_count; when max*area<=255 this replaces complement-min/MaxPool pairs and saved 1950 B on task158.
status: current
confidence: byte_equal
op: QLinearConv;MaxPool;Mul;Equal
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-10T232000Z_multicolor_sumtest_beats_complement_maxpool_pair.md
~~~

~~~library-fact
id: neurogolf.late20260714.convinteger.x_zero_point_scalar
task_id: task219
fact_type: ort_runtime_caveat
subject: ConvInteger input zero point is scalar-only in pinned ORT
claim: Pinned ORT rejects ConvInteger x_zero_point as a real per-input-channel 1-D tensor even when ONNX checker passes; w_zero_point per-output-channel is fine, so K-state decoders may need a forced Add(+const) before BitwiseAnd to move background off origin.
status: current
confidence: runtime_repro
op: ConvInteger
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-12T015336Z_convinteger_scalar_zeropoint_only.md
~~~

~~~library-fact
id: neurogolf.late20260714.single_axis_flood_reach
task_id: task243
fact_type: trap
subject: Single-axis alternating flood under-reaches winding gated paths
claim: A single-axis VHVH flood with N alternating rounds only guarantees path-distance N/2; to cover a K-round combined 4-connected reference it needs N>=2K and a gated-comb adversarial test, because visible/truth/public probes missed the max45 hidden-fail.
status: current
confidence: hidden_fail_attribution
op: Einsum
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-12T223700Z_gated_flood_reach_safety.md
~~~

~~~library-fact
id: neurogolf.late20260714.task029.conv_inner_transport
task_id: task029
fact_type: technique_micro
subject: Conv inner transport fuses channel-collapse and fixed-window crop
claim: A single Conv can read a fixed inner window and collapse one-hot channels to a color-index plane without materializing the full F32/U8 canvas; dilating a single-tap Conv is the legal low-param substitute for rejected sparse-init tricks.
status: current
confidence: byte_equal
op: Conv
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-08T065936Z_conv_inner_crop_transport_dilated_lowparam.md
~~~

~~~library-fact
id: neurogolf.late20260714.live_decode_before_floor
task_id: task158
fact_type: validation_workflow
subject: Re-decode live ONNX before trusting mature task floor docs
claim: On mature tasks, inspect the actual current Team_Best ONNX before following task.md floor prose; task158's documented family was superseded, and the real win came from auditing the newer live graph.
status: current
confidence: byte_equal
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-09T115500Z_relive_baseline_and_offline_pregate.md
~~~

~~~library-fact
id: neurogolf.late20260714.offline_full_corpus_pregate
task_id: task158
fact_type: validation_workflow
subject: Offline full-corpus pre-gate for ambiguous-class candidates
claim: For ambiguous byte-equal-to-LIVE classes, instrument the pivotal LIVE intermediate, validate a numpy replica against ORT, and bench over the full cached corpus before burning heavy gate or Kaggle quota.
status: current
confidence: byte_equal
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-09T115500Z_relive_baseline_and_offline_pregate.md
~~~

~~~library-fact
id: neurogolf.late20260714.rank_only_bonus_strip
task_id: task158
fact_type: technique_micro
subject: Strip additive bonuses that only affect rank separation
claim: If a plane feeds only ArgMax/comparison, replace additive or scale bonuses that merely separate classes with a rank-preserving gate such as Where(mask,value,0); task158 dropped a bonus plane for 651 B byte-equal.
status: current
confidence: byte_equal
op: Where;ArgMax
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-09T115500Z_relive_baseline_and_offline_pregate.md
~~~

~~~library-fact
id: neurogolf.late20260714.freeoutput_gather_index_floor
task_id: task249
fact_type: cost_floor
subject: Row-independent permutation floor is often the int32 Gather index
claim: If the live graph already emits final output by Gather(input,int32_idx), a free-output Equal/label-map rewrite regresses; the residual forced cost is the INT32 index, not a hidden one-hot canvas.
status: current
confidence: floor_confirmation
op: Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-09T184440Z_gather_index_beats_free_output_for_permute_tile.md
~~~

~~~library-fact
id: neurogolf.late20260714.freeoutput_padtail_selfdefeating
task_id: task254
fact_type: trap
subject: FreeOutputPadTail is self-defeating when live output is already free
claim: A pad-tail or color-index collapse prompt is false when the live graph already renders output directly by free-output Einsum/Gather; shrinking then padding adds a charged prepad instead of deleting a canvas.
status: current
confidence: floor_confirmation
op: Einsum;Gather;Pad
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-09T191447Z_freeoutput_padtail_selfdefeating_when_best_already_freeoutput.md
~~~

~~~library-fact
id: neurogolf.late20260714.resize_same_size_short_circuit
task_id: task289
fact_type: ort_runtime_caveat
subject: Same-size Resize short-circuits and cannot crop direct input
claim: ORT Resize short-circuits when output spatial size equals input spatial size, so a direct-input 30->30 Resize does not implement crop/ROI semantics; for integer upsample, small-input Resize can beat free-output Einsum only when the pre-Resize tensor is cheap.
status: current
confidence: floor_confirmation
op: Resize
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-09T201927Z_ort_samesize_resize_shortcircuit_and_resize_beats_freeoutput.md
~~~

~~~library-fact
id: neurogolf.late20260714.argmax_indexplane_int64_trap
task_id: task064
fact_type: trap
subject: Spatial ArgMax index plane is INT64 and usually a cost trap
claim: Color-index collapse via a spatial ArgMax plane is usually false economy because ArgMax output is INT64, making a full index plane 4608-7200 B; inspect whether the live graph already collapsed color reasoning to scalar moments first.
status: current
confidence: floor_confirmation
op: ArgMax
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-10T202012Z_argmax_indexplane_is_int64_costtrap_coloridx_collapse_falsefloor.md
~~~

~~~library-fact
id: neurogolf.late20260714.param_deletion_not_total_cost
task_id: task133
fact_type: trap
subject: Deleting a param bank is not a floor when memory dominates
claim: A prompt that says delete the hash/table params can be stale if live params are already tiny; for variable-scale 2D template-stamp tasks the render/intermediate memory floor dominates and floor_est near output-tail cost is false.
status: current
confidence: floor_confirmation
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-10T220523Z_paramdeletion_not_totalcost_variablescale_stamp_render_floor.md
~~~

~~~library-fact
id: neurogolf.late20260714.floor_est_live_over_gap_tell
task_id: task208
fact_type: prompt_triage
subject: floor_est equal to live_cost/gap is not evidence
claim: If a prompt's floor_est is just live_cost divided by the quoted class gap, treat it as a mechanical EV estimate with no forced-component proof; for one-hot output, the broadcast output-tail mask alone costs H*W bytes.
status: current
confidence: floor_confirmation
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T041239Z_task208_coloridx_falsefloor_second_confirmation.md
~~~

~~~library-fact
id: neurogolf.late20260714.unpromoted_staged_win_reconcile
task_id: task160
fact_type: workflow_guardrail
subject: Reconcile task-local staged wins against current team bar at batch close
claim: At batch close, compare every task-local staged_candidates cost against the current manifest/team bar; a fully gated candidate below live with no Staged creator sidecar is an unpromoted win that must go to Submit Captain.
status: current
confidence: artifact
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T183412Z_task160_freeoutput_padtail_thirdconfirm_and_staged_win_never_promoted.md
~~~

~~~library-fact
id: neurogolf.late20260714.topk_second_place
task_id: task005
fact_type: technique_micro
subject: TopK collapses ArgMax-penalize-ArgMax second-place selectors
claim: Replace ArgMax -> mask/penalty -> ArgMax-again second-place selectors with TopK(k=2) plus split when ties/eviction match; task005 saved 125 B byte-equal and broke a detection wall.
status: current
confidence: byte_equal
op: TopK;ArgMax
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T203244Z_topk_collapses_argmax_penalty_reargmax_and_pertensor_audit_breaks_detection_wall.md
~~~

~~~library-fact
id: neurogolf.late20260714.moment_variance_normalize_by_count
task_id: task054
fact_type: trap
subject: Unnormalized moment variance numerator favors high-count classes
claim: Per-channel spread detectors using counts*m2-m1^2 rank by n^2*variance and can mis-rank toward high-count classes; normalize by count before ArgMax when the rule needs spatial spread rather than count.
status: current
confidence: correctness_root_cause
op: Einsum;ArgMax;Div
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T212329Z_moment_detector_centercolor_unnormalized_variance_misranks_by_count.md
~~~

~~~library-fact
id: neurogolf.late20260714.dihedral_coord_int8_narrow
task_id: task076
fact_type: technique_micro
subject: Dihedral coordinate algebra often narrows from F16 to INT8
claim: Relative-marker and dihedral coordinate stacks often hold only small integers, so F16 coordinate algebra can be changed to INT8 and INT64 Div/Mod indices to INT32 after a range audit; task076 saved 509 B byte-equal.
status: current
confidence: byte_equal
op: Cast;Div;Mod;Clip;Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T212440Z_f16_dihedral_coord_algebra_narrows_int8_int32_byteequal.md
~~~

~~~library-fact
id: neurogolf.late20260714.korientation_matchmap_floor
task_id: task191
fact_type: cost_floor
subject: K-orientation template-match map is a forced floor
claim: For D4/K-orientation template tasks, reduced detection empirically fails because each orientation footprint is distinct; exact detection needs the K-channel match map, while box shape can still come from the local matched-footprint bbox.
status: current
confidence: floor_confirmation
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T221500Z_korientation_matchmap_forced_floor_and_localbbox_boxshape.md
~~~

~~~library-fact
id: neurogolf.late20260714.reduce_sum_uint8_blocked
task_id: task191
fact_type: ort_runtime_caveat
subject: ReduceSum UINT8 is rejected
claim: ReduceSum rejects UINT8 in the pinned ORT, blocking ones-vector MatMul -> ReduceSum parameter levers for U8 projections unless the cast tax is still profitable.
status: current
confidence: runtime_repro
op: ReduceSum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T221500Z_korientation_matchmap_forced_floor_and_localbbox_boxshape.md
~~~

~~~library-fact
id: neurogolf.late20260714.value_info_orphan_phantom_cost
fact_type: scorer_trap
subject: Orphan value_info names can be phantom memory cost
claim: calculate_memory charges value_info names, so node-removal surgery can leave orphaned value_info that inflates cost; after surgery clear or rebuild value_info, except preserve concrete value_info when dynamic Slice needs it for scoring.
status: current
confidence: local_audit
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T221500Z_valueinfo_orphan_phantom_cost_equalwidth_sharedmask_unpack_and_param_lever_on_memoryfloor.md
~~~

~~~library-fact
id: neurogolf.late20260714.equal_width_bit_unpack
fact_type: technique_micro
subject: Equal-width bit-unpack lanes can share one pow mask
claim: When unpacking packed lanes, equal-width splits such as 7/7/7 can reuse one pow mask at the same memory cost, while uneven 8/8/5 lanes need more distinct masks; representation-floor tasks can still have param levers.
status: current
confidence: byte_equal
op: BitShift;BitwiseAnd
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T221500Z_valueinfo_orphan_phantom_cost_equalwidth_sharedmask_unpack_and_param_lever_on_memoryfloor.md
~~~

~~~library-fact
id: neurogolf.late20260714.freeoutput_gt0_integer_lock
task_id: task117
fact_type: trap
subject: Free-output >0 decoders are exact-at-zero and lock integer weights
claim: Graphs decoded by output>0 are exact at zero, so non-integer refits/rank reductions can make off-cells positive even with tiny raw diffs; free-output vote tensors are locked to exact integer restructures.
status: current
confidence: visible_fail_root_cause
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-11T225553Z_freeoutput_gt0_decode_integer_weight_lock_and_reflection_basis_floor.md
~~~

~~~library-fact
id: neurogolf.late20260714.sparse_param_lever_dead
task_id: task326
fact_type: scorer_trap
subject: Sparse initializers are not a legal param-golf route
claim: Sparse initializers may look cheap in calculate_params but scorer full_check/strict shape inference rejects sparse-fed op inputs or charges dense memory after value_info repair; dense initializer shape/factor compression remains the legal route.
status: current
confidence: runtime_repro
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T004531Z_sparse_param_lever_dead_and_dynamicslice_valueinfo_preserve.md
~~~

~~~library-fact
id: neurogolf.late20260714.dynamic_slice_preserve_value_info
fact_type: scorer_trap
subject: Dynamic Slice with constant output shape needs concrete value_info
claim: After graph surgery, clear stale value_info generally, but a data-dependent Slice whose output size is invariant may need concrete value_info preserved or rebuilt so score_v2 can price it instead of returning None.
status: current
confidence: local_audit
op: Slice
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T004531Z_sparse_param_lever_dead_and_dynamicslice_valueinfo_preserve.md
~~~

~~~library-fact
id: neurogolf.late20260714.monotonic_priority_argmax_fold
task_id: task209
fact_type: technique_micro
subject: Monotonic priority table can fold into ArgMax select_last_index
claim: A monotonic lookup table feeding Where(mask,table,0)->ArgMax can often be deleted by using ArgMax(mask,select_last_index=1 or 0), preserving tie priority without a positional table.
status: current
confidence: byte_equal
op: ArgMax;Where
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T014613Z_task209_monotonic_priority_table_folds_to_argmax_select_last_index.md
~~~

~~~library-fact
id: neurogolf.late20260714.maxpool_qlinearconv_1d_morphology
fact_type: technique_micro
subject: Cheap 1-D morphology via wide MaxPool and QLinearConv erosion
claim: A wide-kernel MaxPool computes prefix/suffix OR over a contiguous 1-D band, and QLinearConv with y_scale=k can implement run-threshold erosion without Pad+Slice+And chains.
status: current
confidence: byte_equal
op: MaxPool;QLinearConv
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T063259Z_maxpool_prefixsuffix_or_and_qlinearconv_erode_cheap_1d_morphology.md
~~~

~~~library-fact
id: neurogolf.late20260714.gatherelements_offset_fusion
fact_type: technique_micro
subject: Unsqueeze small operand before broadcast-add for GatherElements offsets
claim: For GatherElements probe lanes, unsqueeze the small Kx1 operand before broadcast-add so the large KxP broadcast result is not materialized, then zero-pad the gathered lane to delete Mod/Less/Where OOB guards.
status: current
confidence: byte_equal
op: GatherElements;Unsqueeze;Pad
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T071108Z_offset_fusion_and_zeropad_oob_probe_delete_mod_where.md
~~~

~~~library-fact
id: neurogolf.late20260714.hierarchical_translation_frame
fact_type: technique_micro
subject: Hierarchical framing can beat flat translation-union render floors
claim: For data-dependent translation/tiling unions, build early copies in a small frame and use Pad-translation as a fourth render family; it can beat Conv/Gather/coordinate render floors for diagonal spreads.
status: current
confidence: byte_equal
op: Pad
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T072453Z_hier_frame_translation_union_and_pad_family_beats_3family_floor.md
~~~

~~~library-fact
id: neurogolf.late20260714.einsum_droppable_operand
task_id: task387
fact_type: technique_micro
subject: All-ones and squared-coordinate Einsum operands can be implicit
claim: If an Einsum operand is all ones or a squared-coordinate weight that can be expressed by repeated use of an existing coordinate operand, delete the initializer and let the contraction sum or re-square implicitly.
status: current
confidence: byte_equal
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T074247Z_einsum_droppable_operand_folds_allones_and_squared_coord.md
~~~

~~~library-fact
id: neurogolf.late20260714.value_cse_constructed_triggers
task_id: task002
fact_type: validation_workflow
subject: Value-level CSE needs constructed triggers on repair artifacts
claim: A byte-equal floor audit needs both static CSE and dynamic value-level equality checks; visible-repair artifacts can make masks constant-zero on random inputs, so test remask candidates with constructed trigger inputs before pruning.
status: current
confidence: floor_confirmation
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T165519Z_value_cse_hunt_and_repair_coverage_artifact_trap_bytefloor_audit.md
~~~

~~~library-fact
id: neurogolf.late20260714.flat_gather_neighborhood
task_id: task080
fact_type: technique_micro
subject: Data-dependent KxK neighborhood read should be flat Gather
claim: For a data-dependent KxK read around a computed cell, use flat Gather(Reshape(cell,[HW]), center_flat+static_offsets) rather than GatherND 2D indices or shift-bank Conv-scatter; task080 saved 640 B byte-equal.
status: current
confidence: byte_equal
op: Gather;GatherND
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T181448Z_flatgather_beats_gathernd_and_shiftbank_for_datadependent_KxK_read.md
~~~

~~~library-fact
id: neurogolf.late20260714.overprovisioned_stamp_noop
task_id: task133
fact_type: technique_micro
subject: Last stamp can be a permanent no-op when base canvas carries one sprite
claim: In multi-stamp render graphs, expose per-instance fires across the generator range; if the base canvas already carries one instance, the final stamp slot may be permanent no-op and removable with its TopK/Gather selector width.
status: current
confidence: byte_equal
op: Resize;TopK;Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T190208Z_overprovisioned_stamp_is_noop_when_template_carried_by_base_canvas.md
~~~

~~~library-fact
id: neurogolf.late20260714.int8_ranking_where_rewrite
task_id: task096
fact_type: technique_micro
subject: INT32 ranking chains can narrow to INT8 with Where arithmetic rewrite
claim: Small-magnitude INT32 ranking/code chains can narrow to INT8 after a full-corpus range audit; because Where lacks INT8 data support, rewrite Where(cond,a,b) as b + cond*(a-b) when safe.
status: current
confidence: byte_equal
op: Cast;Where;Mul;Add;Sub
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T192605Z_int8_narrow_ranking_arithmetic_byteequal_where_rewrite.md
~~~

~~~library-fact
id: neurogolf.late20260714.einsum_52letter_wall
task_id: task138
fact_type: technique_micro
subject: Free-output Einsum walls can be the 52-letter budget
claim: Before declaring a free-output Einsum floored, count distinct equation letters; if it is at the 52-letter limit, squeeze size-1 operand dims or pre-contract small axes to free letters and fold precomputed slices into the fused kernel.
status: current
confidence: byte_equal
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T194650Z_einsum_52letter_budget_is_the_wall_squeeze_dummy_dim_and_fold_shapeslice_into_kernel.md
~~~

~~~library-fact
id: neurogolf.late20260714.distance_domain_render
task_id: task096
fact_type: technique_micro
subject: D4-symmetric renders can use distance-domain table plus reflection gathers
claim: For rings/frames/shells whose output is a pure function of distance from center, render a small distance-domain table and reflect it through two 1-D gathers instead of full-size position planes; task096 saved 361 B on top of INT8 ranking.
status: current
confidence: byte_equal
op: Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T195359Z_distance_domain_render_table_reflect_gather_symmetric_output.md
~~~

~~~library-fact
id: neurogolf.late20260714.freeoutput_einsum_f32_input_lock
task_id: task198
fact_type: ort_runtime_caveat
subject: Free-output Einsum with raw F32 input is F32 dtype-locked
claim: If a free-output Einsum takes the raw F32 graph input as an operand, ORT rejects mixed F16/F32 operands, so narrowing only the other operands is a phantom floor estimate; this killed the top task198 lever.
status: current
confidence: runtime_repro
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_2026-07-12T205725Z_freeoutput_einsum_f32input_locks_dtype_ort_rejects_mixed.md
~~~

~~~library-fact
id: neurogolf.late20260714.compare_vector_color_fold
task_id: task014
fact_type: technique_micro
subject: Fold color map into final Equal compare vector
claim: A color-map lookup feeding only the final Equal compare can often be folded into the compare vector, deleting a redundant crop/copy; static graph cost diff can miss dynamic crop tensor changes, so profile real cost.
status: current
confidence: byte_equal
op: Equal;Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task014_compare_vector_crop_fold.md
~~~

~~~library-fact
id: neurogolf.late20260714.select_then_threshold_union
task_id: task042
fact_type: technique_micro
subject: Mutually exclusive gated unions prefer select then threshold
claim: For mutually exclusive branch unions, select the active branch first and threshold once instead of thresholding each branch then ORing; task042 saved 200 B byte-equal.
status: current
confidence: byte_equal
op: Where;Or
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task042_select_tail_gated_union_combine.md
~~~

~~~library-fact
id: neurogolf.late20260714.dynamic_weight_convtranspose_shift
task_id: task053
fact_type: technique_micro
subject: Dynamic-weight ConvTranspose can beat Gather for grid shifts
claim: For some grid-shift tasks, using the input as a dynamic ConvTranspose weight can implement the shift cheaper than Gather index routes; task053's cost-14 live floor silently beat an older cost-30 proof.
status: current
confidence: floor_confirmation
op: ConvTranspose;Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task053_dynamic_weight_convtranspose_shift.md
~~~

~~~library-fact
id: neurogolf.late20260714.pad_float_only_coupling
task_id: task056
fact_type: ort_runtime_caveat
subject: Free attribute Pad is coupled to float-only old Pad forms
claim: On old Pad/opset forms, the cheap/free attribute-pad route is float-only; switching a one-hot pad tail to BOOL/UINT8 can require newer input-form Pad and lose the expected Cast deletion.
status: current
confidence: floor_confirmation
op: Pad
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task056_pad_float_only_coupling.md
~~~

~~~library-fact
id: neurogolf.late20260714.int8_detect_domain_render_reuse
task_id: task090
fact_type: technique_micro
subject: Split bit domain from INT8 arithmetic domain and reuse live render tail
claim: In bit-pack detection tasks, keep bit ops in UINT32 but move arithmetic/selection into INT8 when values fit, then reuse the live render tail instead of rebuilding the whole graph; task090 saved 287 B.
status: current
confidence: dual_truth_shippable
op: BitShift;BitwiseAnd;ArgMax;Where
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task090_int8_detect_domain_reuse_render.md
~~~

~~~library-fact
id: neurogolf.late20260714.computed_qlinearconv_weight
task_id: task118
fact_type: technique_micro
subject: Computed Where-selected QLinearConv weight is scoreable in a real graph
claim: A Where-selected tiny QLinearConv kernel can replace two data-selected Gather shifts plus a Where blend; toy graphs can score None due to dummy I/O mismatch, so test computed weights inside the real full-I/O graph.
status: current
confidence: byte_equal
op: QLinearConv;Where;Gather
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task118_computed_convweight_scoreable_kernelselect.md
~~~

~~~library-fact
id: neurogolf.late20260714.mode_fill_scalar_collapse
task_id: task129
fact_type: cost_floor
subject: Mode-fill floor collapses to scalar dominant color
claim: For mode-fill with generator-known count invariants, detect the dominant color as one scalar first and render late; vector materialization is dominated by scalar-collapse plus free-output render.
status: current
confidence: floor_confirmation
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task129_mode_fill_scalar_collapse_floor.md
~~~

~~~library-fact
id: neurogolf.late20260714.reversesequence_dead
task_id: task155
fact_type: ort_runtime_caveat
subject: ReverseSequence is validator-illegal and axis-locked
claim: ReverseSequence is not a viable row/col reversal shortcut: the scorer forbids Sequence ops and ORT restricts time_axis/batch_axis to 0 or 1, so moving spatial axes would materialize a huge transpose/reshape.
status: current
confidence: runtime_repro
op: ReverseSequence
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task155_reversesequence_illegal_axis_locked.md
~~~

~~~library-fact
id: neurogolf.late20260714.freeoutput_cp_floor_tools
task_id: task163
fact_type: cost_floor
subject: Free-output CP floors need rank/sign-change proof tools
claim: Proving a single-Einsum free-output CP graph is floored requires forced-component tools such as channel-rank, sign-change, decode-region don't-care checks, and factor reuse audits, not a tried-hard wall.
status: current
confidence: floor_confirmation
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task163_freeoutput_cp_forced_floor_proofs_2026-07-09.md
~~~

~~~library-fact
id: neurogolf.late20260714.maxroipool_upscale_floor
task_id: task223
fact_type: cost_floor
subject: Integer upscale into fixed canvas can be one MaxRoiPool
claim: True nearest-neighbor pixel upscale is not a perfect-25 direct-output task; task223's forced floor is one MaxRoiPool with a 5-element ROI initializer.
status: current
confidence: floor_confirmation
op: MaxRoiPool
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task223_upscale_into_canvas_maxroipool_floor.md
~~~

~~~library-fact
id: neurogolf.late20260714.column_position_rank2_floor
task_id: task262
fact_type: cost_floor
subject: Column-position color remap floor is single rank-2 Einsum
claim: For column-position to color remap, a single free-output rank-2 Einsum can be the floor; slicing to shrink params loses if it creates a charged intermediate.
status: current
confidence: floor_confirmation
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task262_col_position_remap_single_einsum_floor.md
~~~

~~~library-fact
id: neurogolf.late20260714.freeoutput_recolor_threshold_floor
task_id: task344
fact_type: cost_floor
subject: Free-output recolor floor uses positivity decode and identity base
claim: For free-output recolor/relabel graphs, verify decode is output>0 rather than argmax; threshold decode plus the 3-slot identity base can be a real floor when the live graph already has no charged prepad.
status: current
confidence: floor_confirmation
op: Einsum
source_anchor: .update/_archive/update_integrated_2026-07/DONE_task344_freeoutput_recolor_threshold_base_floor.md
~~~
