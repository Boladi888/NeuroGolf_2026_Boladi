# NeuroGolf Lesson Facts

Librarian-reviewed fact blocks for compact lesson packets that are useful as searchable memory but too granular for a new `techniques.md` section. Markdown docs remain canonical; the generated SQLite DB indexes these blocks for `lib_query.py facts` and `search`.

~~~library-fact
id: neurogolf.pack.20260703.webprompt_params_element_count
fact_type: prompt_contract
subject: Browser WebPrompt cost model must say params are element counts
claim: The self-contained browser WebPrompt template now defines initializer params as dtype-blind element counts and computes params as sum(prod(init.dims)); changing an initializer from F32 to F16 is not a param win, only shape/factor compression reduces params.
status: current
confidence: artifact
source_anchor: The_400_Tasks/Task Prompts/_TEMPLATE_Prompt_TaskNNN_WebPrompt.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.conv_integer_ban_not_literal
fact_type: ort_runtime_caveat
subject: External pack ConvInteger hard-ban is not local doctrine
claim: The Creator Aditya pack lists ConvInteger among hard bans for ready external candidates, but local Onnx_Advanced docs have validated/useful integer-kernel routes; treat ConvInteger and related integer kernels as probe-until-locally-gated, not as a blanket library ban.
status: current
confidence: artifact
op: ConvInteger;MatMulInteger;QLinearConv
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/01_technical_hacks/FAILURE_BANS_AND_PUBLIC_ZERO_NOTES.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.blocker_taxonomy
fact_type: workflow_router
subject: External pack blocker taxonomy for quick candidate triage
claim: Classify stalled candidates by their dominant blocker before building: full-grid mask floor, 10-channel intermediate floor, candidate enumeration floor, sparse update table floor, dynamic/profile-invalid floor, param floor, or tie-break/hidden-semantics floor; choose the route that deletes that tensor family first.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_BLOCKERS_INDEX.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.candidate_enumeration_delete_first
fact_type: workflow_router
subject: Candidate-enumeration floors need choose-first render-late rewrites
claim: For candidate-enumeration floors, generic pruning, CSE, and dtype cleanup are secondary; the primary move is to select the winning candidate from scalar or tiny descriptors first, then render one output late, as shown by task209 and branch-pruning lessons.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_BLOCKERS_INDEX.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.dynamic_profile_invalid_rebuild_static
fact_type: validation_workflow
subject: Dynamic/profile-invalid candidates must be rebuilt static before ready
claim: If a candidate uses symbolic dims, dynamic crop renderers, sparse constants shape inference cannot price, control-flow wrappers, or otherwise returns performance-unmeasurable/invalid, treat it as a probe and rebuild as a static-shape measurable output path before candidate_ready.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_BLOCKERS_INDEX.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task296.final_no_bias_dense_conv
task_id: task296
fact_type: technique_micro
subject: No-bias dense final Conv can reach near-19 when it deletes crop/fold tensors
claim: The Aditya pack lists task296 at score 19.00 via a no-bias dense final Conv; the reusable lesson is to spend arithmetic in the final output node when that removes counted crop/fold tensors and keeps params small.
status: current
confidence: artifact
op: Conv
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task303.final_polynomial_einsum
task_id: task303
fact_type: technique_micro
subject: Final polynomial Einsum can collapse a full rule into one output contraction
claim: The Aditya pack lists task303 at score 19.57/cost228 via a final polynomial Einsum; use this as evidence that polynomial/color contractions are attractive when they render directly to output and avoid materialized rule surfaces.
status: current
confidence: artifact
op: Einsum
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task231.row0_threshold_final_gather
task_id: task231
fact_type: technique_micro
subject: Row-0 threshold plus INT32 final Gather can encode width selectors cheaply
claim: The Aditya pack lists task231 at score 19.52/cost239 with row-0 thresholding plus INT32 final Gather; width and selector inference can be kept as tiny descriptors before a final gather instead of full masks.
status: current
confidence: artifact
op: Gather;Greater
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task126.center_column_scatternd
task_id: task126
fact_type: technique_micro
subject: Center-column descriptors plus additive ScatterND beat broad mask construction
claim: The Aditya pack lists task126 at score 18.57/cost620 via center-column Conv plus additive ScatterND; sparse update tasks should carry compact support and update cells directly instead of building broad yellow/object masks.
status: current
confidence: artifact
op: Conv;ScatterND
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task097.final_3x3_conv_param_blocker
task_id: task097
fact_type: technique_boundary
subject: Final 3x3 Conv isolation can delete mask tails but leaves a param blocker
claim: The Aditya pack lists task097 at score 18.18/cost910 via final 3x3 Conv isolation; deleting cast/mask/tail tensors works, but remaining params become the next blocker.
status: current
confidence: artifact
op: Conv
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task057.descriptor_crop_color_counts
task_id: task057
fact_type: technique_micro
subject: Scalar coordinate decoder plus channel-0 crop deletes 10-channel crop
claim: The Aditya pack lists task057 at score 18.62/cost588 by using scalar coordinate decoding, channel-0 crop, and color counts; if only nonzero support is needed, avoid carrying the full 10-channel crop.
status: current
confidence: artifact
op: Gather;ReduceSum
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task065.scalar_moments_modulo_final_einsum
task_id: task065
fact_type: technique_micro
subject: Scalar moments modulo folding and final coefficient Einsum fit odd-one-out rules
claim: The Aditya pack lists task065 at score 18.46/cost687 using scalar moments, modulo folding, and a final coefficient Einsum; odd-one-out style rules can often be descriptor-first rather than grid-first.
status: current
confidence: artifact
op: Einsum;Mod;ReduceSum
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task203.color_count_rank_vectors
task_id: task203
fact_type: technique_micro
subject: Compact color-count rank vectors replace full complement grids
claim: The Aditya pack lists task203 at score 19.39 via compact color-count rank vectors; express complement/rank relations as small vectors before rendering rather than full grids.
status: current
confidence: artifact
op: ReduceSum;ArgMax
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task393.avgpool_topk_descriptor_score20
task_id: task393
fact_type: technique_micro
subject: AveragePool TopK descriptors can cross score 20
claim: The Aditya pack lists task393 at score 20.01/cost146 via AveragePool/TopK descriptor plus final channel select; compact fixed-window descriptors can cross the score-20 threshold when they avoid full render surfaces.
status: current
confidence: artifact
op: AveragePool;TopK;Gather
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.generic_sweep_polish_not_structure
fact_type: technique_boundary
subject: Generic Einsum and pruning sweeps are polish after rule structure
claim: The Aditya pack's generic sweep ledger shows Transpose/MatMul->Einsum, Mul/ReduceSum->Einsum, and dead graph pruning can save meaningful cost on tasks such as 285, 361, 366, and 233, but they rarely solve high-score blockers without a structural rule rewrite.
status: current
confidence: artifact
op: Einsum;MatMul;ReduceSum;Transpose
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/02_task_ledgers/TASK_WIN_LEDGER.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.external_lane_ready_probe_discard
fact_type: validation_workflow
subject: External task-pack artifacts must be classified ready probe or discard
claim: A no-repo/web-lane candidate is ready only when the one-task zip contains exactly taskNNN.onnx at root, ORT CPU loads and runs it, original plus ARC-GEN validation passes, measured memory/params/cost improve, and no banned constructs are present; otherwise classify as probe or discard with failure counts and first-failure evidence.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/03_validation_and_packaging/VALIDATION_AND_SUBMISSION_GATES.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.external_report_contract
fact_type: validation_workflow
subject: Web-lane reports need measurable cost validation and dominant tensor family
claim: External optimizer reports should include task id, candidate name, baseline and candidate memory/params/cost/points, validation counts, zip contents, dominant removed tensor family, plain-English rule summary, ops changed, known risks, and a ready/probe/discard decision; do not accept a downloadable ONNX that lacks this evidence.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/04_prompts_and_task_packs/PROMPT_SNIPPETS.md
~~~

~~~library-fact
id: neurogolf.pack.20260703.chatgpt_lane_success_gate
fact_type: validation_workflow
subject: ChatGPT web lanes can work but only after exact gates and dedupe
claim: The Aditya pack records real ChatGPT-lane public successes on task049 and task335, but later waves repeated the same fingerprints; keep clean small graphs from web lanes, revalidate locally before merge, and do not resubmit duplicate hashes or candidates without public/local improvement.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.pack.20260703.chatgpt_public_zero_family
fact_type: validation_trap
subject: Plausible ChatGPT visual rules repeatedly public-zero without exact validation
claim: ChatGPT waves public-zeroed or errored repeatedly on tasks 013, 023, 076, 137, 170, 208, 224, and 285 when artifacts lacked exact original plus generated validation, first-failure analysis, ORT kernel proof, or generator semantics; future prompts must require executable validation output before candidate_ready.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/01_technical_hacks/FAILURE_BANS_AND_PUBLIC_ZERO_NOTES.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task239.dynamic_crop_public_zero
task_id: task239
fact_type: validation_trap
subject: Correctness-only dynamic TopK Slice Pad crop scored public zero
claim: A task239 dynamic TopK->Slice->Pad crop candidate passed original plus 1000 local correctness examples but local performance could not be measured and isolated Kaggle ref53594523 scored 0.00; rebuild dynamic crop renderers as static-shape final renderers before marking candidate_ready.
status: current
confidence: artifact
op: TopK;Slice;Pad
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task232.sparse_einsum_kaggle_error
task_id: task232
fact_type: validation_trap
subject: Sparse Einsum theoretical score-20 route errored on Kaggle
claim: A task232 sparse-Einsum route passed 1254/1254 local correctness and looked like cost146/score20.016, but local scoring could not infer the sparse constant rank and isolated Kaggle ref53600008 returned ERROR; rebuild as a non-sparse measurable factorization.
status: current
confidence: artifact
op: Einsum;SparseTensorProto
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task209.descriptor_crop_render_breakthrough
task_id: task209
fact_type: technique_micro
subject: Descriptor crop render beats candidate enumeration on task209
claim: The task209 breakthrough was collapsing candidate enumeration into descriptor -> crop -> render and then deleting 20x20/16x20 mask and class-map tensor families; generic CSE, fp16 cleanup, and semantic guesses are lower-value noise while broad render/vote surfaces remain.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task209.inspect_only_geometry_direction
task_id: task209
fact_type: safety_boundary
subject: Private compact-geometry reference is clean-room direction only
claim: The Aditya pack records a private task209 reference at cost9487 using compact geometry/descriptor logic instead of broad render-vote surfaces; use it only as high-level clean-room direction and never copy, derive constants from, or submit the private model.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task215.maxpool_formula_dead
task_id: task215
fact_type: validation_trap
subject: Public row-max shorthand did not become a valid zero-cost MaxPool graph
claim: Direct final MaxPool variants for the public task215 row-max shorthand scored attractive zero counted cost but failed original/default validation; task215 needs true phase/channel factorization rather than a mechanical pooling rewrite.
status: current
confidence: artifact
op: MaxPool
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task215.inspect_only_phase_compression
task_id: task215
fact_type: safety_boundary
subject: Private phase-compression reference is high-level direction only
claim: A private task215 reference shows phase/sign/channel factorizations can be smaller than the accepted one-node phase-Einsum family, but it is inspect-only; independently search smaller phase maps and never copy constants or exact contractions from the private file.
status: current
confidence: artifact
op: Einsum
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task023.public_zero_repeat_guard
task_id: task023
fact_type: validation_trap
subject: task023 prior public-zero routes need exact validation and no repeats
claim: The Aditya pack records repeated task023 public-zero outcomes; even when a static boundary-prior score table lowered TopK from 18 to 14, prior public-zero families must not repeat without exact original plus generated validation and package checks.
status: current
confidence: artifact
op: TopK
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task076.crop15_tiebreak_blocked
task_id: task076
fact_type: validation_trap
subject: task076 crop15 upside blocked by dihedral tie-breaks
claim: The task076 crop15 reduction had real cost upside but failed per-anchor dihedral and tie-break validation; future candidates need explicit anchor/dihedral ambiguity checks and first-failure analysis before producing another ONNX.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task133.margin_padding_unsafe
task_id: task133
fact_type: validation_trap
subject: task133 static margin padding covered too few defaults
claim: Static margin padding was unsafe on task133 because margin-1 covered only 135/255 defaults; later ChatGPT output scored nonzero but below goal, so future routes should preserve behavior-valid pieces while improving validation reporting and cost reasoning.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task285.einsum_sweep_partial_only
task_id: task285
fact_type: technique_boundary
subject: Transpose-MatMul to Einsum is useful cleanup not a high-score route
claim: Generic Einsum sweeps on task285 confirmed transpose+MatMul absorption as the largest generic gain, saving about 18k cost, but the result remained far from 20+ and later ChatGPT waves public-zeroed; use this as polish only after the rule structure is correct.
status: current
confidence: artifact
op: Einsum;MatMul;Transpose
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip::Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703/05_source_docs/learning/task_lessons.csv
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task387.sparse_initializer_shapeinfer_fail
task_id: task387
fact_type: scoring_trap
subject: Sparse initializers fail strict shape inference before scoring
claim: A SparseTensorProto initializer can load in raw ORT, but the scorer's check_model(full_check) plus infer_shapes(strict_mode) can leave shape-sensitive consumers such as Einsum, MatMul, Gemm, or Reshape seeing the sparse-fed input as rank-0/unknown; adding value_info repairs shape inference but recharges the dense memory, so the sparse param discount is unreachable.
status: current
confidence: lesson
op: SparseTensorProto;Einsum;MatMul;Gemm;Reshape
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-03T075500Z_sparse_init_shapeinfer_failmode.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task387.dense_initializer_cheapest_constant_home
task_id: task387
fact_type: scoring_internals
subject: Dense initializer is the cheapest home for a constant
claim: A dense initializer costs N params by element count, while moving the same F16 constant into a materialized graph value costs 2N memory bytes; task387 no-P measured params -528 but memory +1056, so do not turn constants into runtime tensors unless the tensor is deleted or lands on graph output.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-03T075500Z_sparse_init_shapeinfer_failmode.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task251.fixed_unroll_full_corpus_tail
task_id: task251
fact_type: floor_trap
subject: Fixed-unroll round count is pinned by the full-corpus tail
claim: For hardcoded flood/BFS/morphology unrolls, the safe round count equals the maximum convergence round over the full cached truth corpus; task251 needed R10 because only 2 of 200000 grids reached round 10, while 20k/30k samples under-counted and R9 failed 199998/200000.
status: current
confidence: lesson
op: BitShift;BitwiseOr
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-03T075558Z_unroll_round_floor_full_corpus.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task251.v_then_h_shift_dilation_is_8_connected
task_id: task251
fact_type: op_cost_pattern
subject: Vertical-then-horizontal bit-shift dilation is 8-connected
claim: In a packed flood graph, vertical = reach|up|down followed by vertical|(vertical<<1)|(vertical>>1) reaches diagonals in the same round, so it is an 8-connected dilation and can converge faster than a 4-connected reference BFS; model the actual op sequence when predicting K.
status: current
confidence: lesson
op: BitShift;BitwiseOr
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-03T075558Z_unroll_round_floor_full_corpus.md
~~~

~~~library-fact
id: neurogolf.artifact.20260703.creator_aditya_knowledge_pack
fact_type: artifact_index
subject: Creator Aditya shareable NeuroGolf knowledge pack is archived
claim: The 2026-07-03 Creator Aditya NeuroGolf knowledge bundle is a self-contained shareable zip with technical hacks, task ledgers, validation workflow, prompt/task-pack instructions, copied source docs, selected tools, and handcrafted builders; it is an archived artifact, not an open Research/Todo card.
status: current
confidence: artifact
source_anchor: Onnx_Advanced/Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip
~~~

~~~library-fact
id: neurogolf.research.20260703.o7.euler_2x2_local_reduction
fact_type: op_cost_pattern
subject: Euler characteristic is a zero-border 2x2 local reduction
claim: A binary 30x30 mask can compute exact Euler characteristic by zero-padding to 32x32, encoding each 2x2 window with Conv weights [[8,4],[2,1]], counting code classes, and applying 4*chi_fg4_bg8=n1-n3+2*nd or 4*chi_fg8_bg4=n1-n3-2*nd; same foreground/background connectivity is not locally computable.
status: current
confidence: research
op: Pad;Conv;Equal;Cast;ReduceSum
source_anchor: Onnx_Advanced/Research/Done/_archive/Research_o7_topology_integral_geometry_detectors.md
~~~

~~~library-fact
id: neurogolf.research.20260703.o7.single_component_holes_boundary
fact_type: technique_boundary
subject: Hole tests are local only after component count is known
claim: Since holes = components - chi, a known single connected component has holes = 1 - chi and has_hole = (1 - chi) > 0, but unrestricted component count, same-component tests, reachability, and largest-component aggregation are non-local and still require flood or labeling style routes.
status: current
confidence: research
op: Conv;ReduceSum
source_anchor: Onnx_Advanced/Research/Done/_archive/Research_o7_topology_integral_geometry_detectors.md
~~~

~~~library-fact
id: neurogolf.research.20260703.o7.perimeter_corner_local_reductions
fact_type: op_cost_pattern
subject: Perimeter and corner counts ride the same 2x2 code map
claim: The same zero-bordered 2x2 code-count map gives exact perimeter P=n(Q1)+n(Q2)+2*n(QD)+n(Q3), convex corners C+=n(Q1), and concave corners C-=n(Q3); use INT32 or F32 rather than F16 for area-derived intermediate formulas that can exceed 2048 before comparison.
status: current
confidence: research
op: Conv;Equal;ReduceSum
source_anchor: Onnx_Advanced/Research/Done/_archive/Research_o7_topology_integral_geometry_detectors.md
~~~

~~~library-fact
id: neurogolf.research.20260703.o8.f16_narrowing_safety_gate
fact_type: validation_trap
subject: Cast-before-selector narrowing needs absmax finite and dataflow proofs
claim: Moving Cast(F32->F16) before Gather or Slice narrows values that the old graph may never have selected; approve it only after proving absmax <= 65504 for values reaching arithmetic or a documented compare-only +inf carrier, dumping decode-relevant tensors for no new NaN/undocumented infinities, and excluding invalid patterns such as 0*inf or inf+(-inf).
status: current
confidence: research
op: Cast;Gather;Slice
source_anchor: Onnx_Advanced/Research/Done/_archive/Research_o8_f16_inf_nan_dtype_narrowing.md
~~~

~~~library-fact
id: neurogolf.research.20260703.o8.inf_nan_positive_decode
fact_type: validation_trap
subject: Positive decode tolerates +inf but not nan
claim: Under output > 0 style decode, +inf behaves as true while NaN behaves as false in ONNX-reference/IEEE comparison behavior, so truth percentage can hide an F16 narrowing bug unless +inf/-inf/NaN counts are inspected separately.
status: current
confidence: research
op: Cast;Greater
source_anchor: Onnx_Advanced/Research/Done/_archive/Research_o8_f16_inf_nan_dtype_narrowing.md
~~~

~~~library-fact
id: neurogolf.research.20260703.o8.f16_einsum_runtime_probe
fact_type: ort_runtime_caveat
subject: F16 Einsum support must be target-probed
claim: ONNX allows float16 Einsum, and a 2026-07-03 local mini-probe loaded and ran F16 Einsum on onnx 1.21.0 / ORT 1.24.4 CPU, but ORT source/kernel listings can still omit F16; before relying on an F16 Einsum path, build and run the exact target graph on the exact runtime, and treat contractions as high-risk consumers for narrowed non-finite values.
status: current
confidence: research
op: Einsum;Cast
source_anchor: Onnx_Advanced/Research/Done/_archive/Research_o8_f16_inf_nan_dtype_narrowing.md
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task010.polynomial_logits_rank
task_id: task010
fact_type: technique_micro
subject: Signed polynomial logits can replace exact one-hot selectors
claim: For threshold-decoded one-hot outputs, small rank routes can sometimes emit signed polynomial logits instead of materializing an exact one-hot selector.
status: current
confidence: lesson
technique_section: 124
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task010
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task064.padded_count_span_dirty
task_id: task064
fact_type: validation_trap
subject: Count-derived span masks are dirty on padding-perturbed LIVE
claim: Thresholding row or column counts summed over a padded extent is structurally non-byte-equal on padding-perturbed LIVE; task064 count-span routes had 341 and 163 A/B divergences, while the robust span needs the explicit 48-param coordinate index.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T174656Z_task064_countspan_dirty_dualtruth.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task138.free_output_einsum_retracts_canvas_floor
task_id: task138
fact_type: floor_trap
subject: Free-output rank-1-basis Einsum retracts the crop-canvas floor
claim: A canvas-floor proof for task138 was representation-specific; the current cheaper family reconstructs the full one-hot output as a free-output separable Einsum over shared rank-5 spatial bases and channel-modulated terms, with no per-cell crop plane.
status: current
confidence: lesson
op: Einsum
technique_section: 119
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033708Z_task138_freeoutput_inf_f16.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task138.f16_inf_nan_gate
task_id: task138
fact_type: validation_trap
subject: Inf-carrying F16 free-output graphs need nan scans
claim: In a free-output graph whose decode is output greater than zero, F16 inf can be tolerated while nan silently decodes false in ignored cells; structural edits such as cast-before-gather must be checked for nan and inf on real inputs, not only truth percentage.
status: current
confidence: lesson
op: Cast;Gather;Einsum
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033708Z_task138_freeoutput_inf_f16.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task138.cast_before_gather_absmax_gate
task_id: task138
fact_type: op_cost_pattern
subject: Cast-before-Gather narrowing requires abs-max proof
claim: Cast-before-Gather can delete a gather-result tensor only when the gathered data never overflows the target dtype; measure corpus abs-max first, because overflowing F16 chains can create inf times zero nan despite passing small truth gates.
status: current
confidence: lesson
op: Cast;Gather
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033708Z_task138_freeoutput_inf_f16.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task243.single_axis_einsum_flood
task_id: task243
fact_type: technique_micro
subject: Single-axis Einsum flood collapses two state tensors into one
claim: For walk-counting Einsum floods, the per-step operator sets the letter budget and therefore the number of charged state tensors; a single-axis one-letter round can fit many more rounds in one Einsum than a combined three-letter step, deleting one F32 state tensor.
status: current
confidence: lesson
op: Einsum
technique_section: 37
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033745Z_einsum_singleaxis_flood_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task243.dual_truth_superset_flood
task_id: task243
fact_type: validation_trap
subject: Single-axis flood can be more-correct-than-LIVE despite A/B divergence
claim: On ambiguous-class flood tasks, a cheaper single-axis flood that diverges from LIVE can still be shippable if same-corpus attribution shows candidate misses are a subset of LIVE misses; task243 had zero regressions on 8000 sampled grids.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033745Z_einsum_singleaxis_flood_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task286.column_packed_flood
task_id: task286
fact_type: technique_micro
subject: Column-packed bit-parallel flood beats grid geodesic planes
claim: For maze reachability, packing each column into a UINT32 scalar and using hoisted segmented run masks plus horizontal sweeps moves flood state from full grid tensors to O(W) scalars; task286's floor moved to 21558 and the old 23210 geodesic-plane floor is retracted.
status: current
confidence: lesson
op: BitShift;BitwiseAnd;BitwiseOr
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033820Z_bitparallel_column_flood_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task286.full_log_vertical_fill_loses
task_id: task286
fact_type: floor_trap
subject: Full-log vertical fill does not lower packed flood pass count enough
claim: Deepening packed vertical fill shifts from {1,2,4} to {1,2,4,8,16} raises per-pass cost more than it reduces passes, because horizontal direction reversals bound convergence; do not retry this sub-lever for task286-like mazes.
status: current
confidence: lesson
op: BitShift
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033820Z_bitparallel_column_flood_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.scoring.output_tensor_exempt
fact_type: scoring_internals
subject: Land expensive assembly on graph output
claim: The scorer skips tensors named input and output, so the final graph output tensor can be large without a memory charge; arrange the last broadcast, scatter, or unpack to write output directly rather than creating an equivalent predecessor.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T033820Z_bitparallel_column_flood_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.recapture.probe_snapshot_sha_recovery
fact_type: workflow
subject: Recover cleaned candidates from Submit Captain probe snapshots
claim: If a Kaggle-probed ONNX was cleaned from staged submits, search consumed Submit Captain update JSON for the cost and SHA, extract taskNNN.onnx from the referenced _probe/_snapshots submission.zip, SHA-match it, then rerun gates live.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T034145Z_recapture_from_probe_snapshots.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task017.pathb_bar_truth_check
task_id: task017
fact_type: validation_trap
subject: Verify the banked bar's own truth on Path-B recaptures
claim: For Path-B recaptures, run truth on the current banked/team-best graph as well as the candidate; a cheaper recapture can also be more correct than a banked Path-B bar, as task017 showed.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T034145Z_recapture_from_probe_snapshots.md
~~~

~~~library-fact
id: neurogolf.lesson.20260702.task002.byte_equal_floor_audit
task_id: task002
fact_type: floor_trap
subject: WALL verdicts require byte-equal compression audit
claim: Before declaring a structural-LIVE graph minimal, audit exact-valued duplicate initializers and removable remask or idempotent-clip nodes whose invariant is re-established downstream; task002 recovered 132 bytes byte-equal and Kaggle-confirmed.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-02T034200Z_floor_audit_dedupe_initializers.md
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task011.finite_verifier_scope
task_id: task011
fact_type: validation_gate
subject: Finite verifier scope for slow full-corpus gates
claim: When full corpus gates are impractically slow, a reviewer-scoped finite verifier proves a localized algebraic rewrite only if it compares against the prior full-gated formulation and covers generator-valid plus stress classes.
status: current
confidence: lesson
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task011
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task015.active_grid_shift_basis
task_id: task015
fact_type: technique_micro
subject: Active-grid shift basis beats full-canvas shift bases
claim: For fixed active grids, replace full-canvas shift bases with active selector S[30,H] plus small in-grid basis B[2,H,H].
status: current
confidence: lesson
technique_section: 118
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task015
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task027.scatternd_false_bank_shrink
task_id: task027
fact_type: technique_micro
subject: Shrink invariant-false ScatterND current-read banks
claim: For sparse ScatterND patch graphs, audit target cells whose current value is invariant false; shrink only that current-read bank and pad a false bool vector back to the full patch length.
status: current
confidence: lesson
op: ScatterND
technique_section: 139
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task027
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task032.tail_activity_size
task_id: task032
fact_type: technique_micro
subject: Tail activity flags can recover small fixed-range square size
claim: For small fixed-range square grids, compute size from tail activity flags as base plus active last slots, then shift the row ramp by the base to delete full active-count reductions.
status: current
confidence: lesson
technique_section: 144
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task032
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task059.wide_contract_then_cast
task_id: task059
fact_type: dtype_micro
subject: Contract before narrowing casts when the reduced result is smaller
claim: When a cast narrows a vector before a contraction, try contracting in the wider dtype and casting only the reduced result.
status: current
confidence: lesson
op: Cast
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task059
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task065.argmax_f16_onehot_selector
task_id: task065
fact_type: technique_micro
subject: ArgMax plus F16 OneHot can beat ReduceMax Equal Cast
claim: For unique max-channel selectors, ArgMax plus F16 OneHot can beat ReduceMax plus Equal plus Cast by a few bytes even after OneHot params.
status: current
confidence: lesson
op: ArgMax;OneHot
technique_section: 147
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task065
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task079.where_u8_tail
task_id: task079
fact_type: ort_op_dtype
subject: Where UINT8 tails can preserve channel-0 background
claim: For one-hot output tails, Where(BOOL, UINT8, UINT8) can replace F16 index reconstruction when channel-0 background must be preserved; Where(BOOL, BOOL, BOOL) remains unsupported.
status: current
confidence: lesson
op: Where
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task079
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task106.shared_einsum_selector_init
task_id: task106
fact_type: technique_micro
subject: Share square anchored tile basis selector initializers
claim: Square anchored tile basis-Einsums can reuse the same [30,k] selector initializer for row and column operands; params count the initializer once even if the Einsum consumes it twice.
status: current
confidence: lesson
op: Einsum
technique_section: 132
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task106
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task225.compact_role_helpers
task_id: task225
fact_type: technique_micro
subject: Compact rowrole colrole helpers make quad scatter practical
claim: For quad scatter color operands, materializing compact rowrole and colrole helpers can turn an impractically slow one-terminal role Einsum into a full-gated practical ship while keeping full ARC-GEN profile equal to static cost.
status: current
confidence: lesson
technique_section: 109
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task225
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task283.signed_qlc_binary_carrier
task_id: task283
fact_type: qlinearconv_dtype
subject: Quantize binary F32 crops directly to signed QLinearConv carriers
claim: For binary F32 crops feeding signed QLinearConv classifiers, QuantizeLinear can map {0,1} directly to a signed carrier and delete Cast plus Add plus Add; with scale 0.5, doubled QLinearConv weights let that same scale serve all QLinearConv scales.
status: current
confidence: lesson
op: QuantizeLinear;QLinearConv
technique_section: 137
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task283
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task289.expansion_bank_source_reuse
task_id: task289
fact_type: technique_micro
subject: Reuse source selector slices already present in an expansion bank
claim: If an expansion bank already contains a source selector slice, reuse it through a one-hot basis vector instead of carrying a separate source matrix.
status: current
confidence: lesson
technique_section: 138
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task289
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task327.pad_axes_param_shrink
task_id: task327
fact_type: opset_param_golf
subject: Opset-18 Pad axes input shrinks spatial-only pad vectors
claim: For Pad ops in opset 18, the axes input can shrink pad parameter vectors when only spatial axes are padded.
status: current
confidence: lesson
op: Pad
technique_section: 59
source_anchor: source not committed - Codex librarian lesson 20260626T010445Z task327
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task047.generator_bound_validation_trap
task_id: task047
fact_type: validation_trap
subject: Generator bounds do not automatically cover fixed validation examples
claim: Source-read generator bounds can shrink reconstruction masks, but fixed validation examples may violate random bounds and must be priced explicitly.
status: current
confidence: lesson
technique_section: 133
source_anchor: source not committed - Codex librarian lesson 20260626T010446Z task047
~~~

~~~library-fact
id: neurogolf.lesson.20260626.task217.reducemax_attribute_axes
task_id: task217
fact_type: opset_param_golf
subject: Static ReduceMax reductions can use attribute axes by lowering opset
claim: For static ReduceMax reductions, lowering opset to attribute axes can remove the axes initializer without changing graph semantics.
status: current
confidence: lesson
op: ReduceMax
technique_section: 142
source_anchor: source not committed - Codex librarian lesson 20260626T015713Z task217
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task340.u8_diff_reducemin_membership
task_id: task340
fact_type: technique_micro
subject: U8 Sub plus ReduceMin gives one-plane per-axis scalar membership
claim: For a per-row or per-column test that asks whether value V appears, reduce UINT8 Sub(source,V) with ReduceMin==0 instead of Equal plus Where plus ReduceMax; this spends one full U8 plane instead of two, requires opset 14 U8 Sub, and needs V disjoint from filler values.
status: current
confidence: lesson
op: Sub;ReduceMin;Equal;Where
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T152747Z_u8_diff_reducemin_membership_plane.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task379.u8_underflow_interval_membership
task_id: task379
fact_type: technique_micro
subject: U8 underflow interval membership folds Ge plus Le plus And
claim: For bounded interval masks low<=r<=high, use LessOrEqual(Sub(r,low), Sub(high,low)) in UINT8 to reject both below-low underflow and above-high values, replacing GreaterOrEqual plus LessOrEqual plus And with two planes; requires opset 14 U8 Sub and a safe empty-interval sentinel.
status: current
confidence: lesson
op: Sub;LessOrEqual;GreaterOrEqual;And
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T163000Z_u8_underflow_interval_membership.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task338.sentinel_value_reuse
task_id: task338
fact_type: opset_param_golf
subject: Reuse an existing constant as a Pad sentinel value
claim: In sentinel-Pad and out-of-region masking graphs, the sentinel value is often unconstrained; retarget the Div or threshold band so an already-paid constant can also be the Pad fill and delete the dedicated sentinel initializer.
status: current
confidence: lesson
op: Pad;Div;Equal
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T120500Z_task338_sentinel_value_reuse_param_fold.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task002.visible_repairs_can_be_net_negative
task_id: task002
fact_type: validation_trap
subject: Memorized visible repair seeds can be net-negative on broad truth
claim: On structural-LIVE tasks with memorized visible-repair seeds, measure the repair-free model on the broad truth corpus; task002 repairs preserved visible officials but cost bytes and slightly reduced broad truth, so official repair coverage can be overfit.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T130000Z_task002_repair_net_negative_truth.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task002.ab_includes_corpus
task_id: task002
fact_type: validation_gate
subject: mode_ab byte-equality includes the corpus after officials
claim: Do not treat mode_ab as officials-only byte-equality; it streams official pairs first and then corpus tensors, so a candidate that matches the visible set but diverges off-officials fails the ab gate.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T130000Z_task002_repair_net_negative_truth.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task138.cast_collapsed_canvas_to_u8
task_id: task138
fact_type: dtype_micro
subject: Cast the collapsed color canvas to U8 before crop and gather
claim: For one-hot F32 input color re-emission tasks, collapse channels once with Conv, then Cast the small [1,1,H,W] canvas to U8 and keep crop, gather, pad, compare, and masks in U8/BOOL; paying one F32 plane can beat F32 downstream planes.
status: current
confidence: lesson
op: Conv;Cast;Gather;Pad;Equal
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T120000Z_u8canvas_floor_and_undersize_gradergap.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task138.crop_undersize_grader_gap
task_id: task138
fact_type: validation_trap
subject: Crop or gather undersize can pass dummy scoring and crash real inputs
claim: A statically priced graph can pass score_v2 dummy input but crash on real inputs when crop or gather bounds are undersized; run profile_real_cost, and if LIVE itself crashes then byte-vs-LIVE gates are invalid and truth is the useful gate.
status: current
confidence: lesson
op: Gather
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T120000Z_u8canvas_floor_and_undersize_gradergap.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task191.qlinearconv_runtime_template
task_id: task191
fact_type: qlinearconv_dtype
subject: Runtime-extracted templates can feed INT8 QLinearConv
claim: QLinearConv is not limited to constant kernels; runtime-extracted integer templates can be gathered, transposed, sliced, oriented, and fed to INT8 QLinearConv so an [1,N,H,W] match map is UINT8 instead of F16.
status: current
confidence: lesson
op: QLinearConv;Gather;Slice;Transpose
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T061309Z_task191_qlinearconv_runtime_template_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task191.f16_conv_forced_false_floor
task_id: task191
fact_type: floor_trap
subject: F16 Conv forced floor proofs miss INT8 QLinearConv
claim: Re-audit template-match floors that claim Conv forces an irreducible F16 match map; if the data and template are integer-valued, INT8 QLinearConv can halve the match map and remove F16 paint paths.
status: current
confidence: lesson
op: Conv;QLinearConv;ConvTranspose
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T061309Z_task191_qlinearconv_runtime_template_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task191.runtime_kernel_arithmetic_before_concat
task_id: task191
fact_type: technique_micro
subject: Move runtime-kernel arithmetic onto small patches before orientation Concat
claim: When oriented runtime kernels apply linear permutations, do per-element plus/minus arithmetic on small [1,1,k,k] base patches before orientation Concat; the final weights stay bit-identical, though savings are partial because small per-orientation ops and multiple final layouts can remain forced.
status: current
confidence: lesson
op: Concat;QLinearConv;Sub;Transpose
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T061309Z_task191_qlinearconv_runtime_template_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task204.sparse_initializer_dead_under_sanitize
task_id: task204
fact_type: scoring_trap
subject: SparseTensorProto initializers break the official sanitizer
claim: Sparse initializers look cheaper because params count nnz, but sanitize_model renames graph inputs and initializers without renaming sparse_initializer values, leaving dangling node references and causing ORT load failure, so sparse_initializer is unstageable under the local gate.
status: current
confidence: lesson
op: SparseTensorProto
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T170000Z_sparse_initializer_breaks_sanitize.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task204.u8_pipeline_zero_sum
task_id: task204
fact_type: scoring_internals
subject: All-U8 pipelines make param-memory reshuffles zero-sum
claim: In an all-U8 pipeline, one intermediate byte equals one initializer element in scorer cost, so moving work between conv weights and U8 outputs is zero-sum and extra planes are strictly additive; real wins delete load-bearing tensors or shrink forced F32 intermediates.
status: current
confidence: lesson
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T170000Z_sparse_initializer_breaks_sanitize.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task219.where_uint16_unsupported
task_id: task219
fact_type: ort_op_dtype
subject: Where has no UINT16 kernel in the pinned ORT build
claim: Do not fold Cast(bool->uint16) plus Mul(uint16_mask, selector) into Where for UINT16 masks; Where(BOOL, UINT16, UINT16) is not implemented, while uint16 Mul works.
status: current
confidence: lesson
op: Where;Cast;Mul
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T210000Z_where_uint16_unsupported_select_cast_fold.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task182.profile_hash_unsafe_for_containment
task_id: task182
fact_type: validation_trap
subject: 1D profile hashes are unsafe for contained-object disambiguation
claim: When the rule must identify an object inside a frame or container and ignore decoys, 1D row or column profile hashes can miss the 2D containment relation; use a 2D frame-plus-shape hash even if it costs more.
status: current
confidence: lesson
op: Conv
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T193000Z_task182_2d_frame_containment_safe_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task182.convolve_on_free_input
task_id: task182
fact_type: scoring_internals
subject: Convolve on free one-hot input instead of materializing channel planes
claim: Because NeuroGolf cost treats input and output tensors as free but charges intermediates by bytes and params by element count, shape-match-render tasks should convolve directly on the free one-hot input rather than slice or project channels into materialized 30x30 F32 planes.
status: current
confidence: lesson
op: Conv;Slice;Cast;QLinearConv
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T193000Z_task182_2d_frame_containment_safe_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260630.task182.sliding_hash_kernel_optimum
task_id: task182
fact_type: cost_floor
subject: Sliding hash kernel size has an interior cost optimum
claim: For a k by k C-channel hash on 30x30 free input reduced to a scalar, check C*k*k + dtype_bytes*(31-k)*(31-k); smaller kernels are not always cheaper because output-plane size rises as k shrinks.
status: current
confidence: lesson
op: Conv
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-06/DONE_2026-06-30T193000Z_task182_2d_frame_containment_safe_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260703.task352.single_fused_conv_floor
task_id: task352
fact_type: cost_floor
subject: Single fused Conv to output can be the forced floor
claim: If a graph already writes the full computation with one Conv named output, rank factoring, sparse-channel decomposition, or channel-drop rewrites can only win if they shrink the single weight tensor itself; params count elements not nonzeros, and any second node births at least one charged 30x30 intermediate (900 B at BOOL/U8), often exceeding the whole single-Conv budget.
status: current
confidence: lesson
op: Conv
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-03T235900Z_fused_conv_vs_decompose_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260707.task024.per_channel_decode_not_argmax
task_id: task024
fact_type: validation_trap
subject: NeuroGolf output decode is per-channel threshold, not argmax
claim: The harness decodes output with per-channel out > 0, so a one-hot output is correct only when the true color channel is positive and all other channels are non-positive; argmax-perfect logits can still gate 0/266. Fit or test the actual thresholded one-hot condition.
status: current
confidence: lesson
op: Einsum
technique_section: 119
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-07T140000Z_einsum_perchannel_decode_rank_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260707.task024.argmax_rank_floor_shortcut
task_id: task024
fact_type: cost_floor
subject: Argmax fit can falsify a free-output Einsum rank
claim: For free-output one-hot Einsum rank searches, per-channel threshold exactness is a subset of argmax exactness; if the best held-out argmax fit at rank k is below 100%, rank k cannot pass the real per-channel truth gate, so skip the harder fit and full gate at that rank.
status: current
confidence: lesson
op: Einsum
technique_section: 119
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-07T140000Z_einsum_perchannel_decode_rank_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260708.task091.subbyte_dtype_no_scored_savings
task_id: task091
fact_type: scoring_internals
subject: UINT4 and INT4 save neither memory nor params in the scorer
claim: UINT4 and INT4 activations have numpy itemsize 1 in the scorer, the same as UINT8/BOOL, and initializer params are dtype-blind element counts; sub-byte dtypes buy file bytes only, not scored memory or params.
status: current
confidence: lesson
op: UINT4;INT4
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-08T114409Z_subbyte_dtype_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260708.task091.full_canvas_floor_calibration
task_id: task091
fact_type: cost_floor
subject: Full-canvas output plane remains 900 B even with sub-byte probes
claim: A bare Conv->Cast->Slice->Pad->Equal skeleton for task091 measured 2258, with forced 15x15 F32 combine source (900 B) and 30x30 U8 pre-Equal output plane (900 B); expanding early to a 10-channel BOOL crop then free Pad measured worse at 3317 because the Pad predecessor is charged.
status: current
confidence: lesson
op: Conv;Cast;Slice;Pad;Equal
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-08T114409Z_subbyte_dtype_floor.md
~~~

~~~library-fact
id: neurogolf.lesson.20260709.task310.dynamic_output_shape_dead
task_id: task310
fact_type: validation_trap
subject: Dynamic-shape graph output is grader-dead even when scorer-cheap
claim: Baked value_info can rescue a fixed-shape dynamic Slice intermediate, but a Slice used directly as a smaller graph output fails the official np.array_equal contract against the padded [1,10,30,30] benchmark output; task310 direct Slice scored 469 but visible-gated 0, while Slice->Pad kept the output shape and passed at 3133.
status: current
confidence: lesson
op: Slice;Pad
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-09T170000Z_dynamic_slice_output_grader_dead.md
~~~

~~~library-fact
id: neurogolf.lesson.20260709.task365.stubbed_skeleton_floor_probe
task_id: task365
fact_type: floor_discipline
subject: Stubbed skeletons cheaply falsify directive floor claims
claim: To test a claimed floor, build the claim family's real input read and output path while replacing disputed expensive quantities with free initializers; if that generous skeleton already prices above the claimed floor, no implementation in that family can reach it. Task365's fully stubbed scalar-selector skeleton priced 1324 against a claimed 900 floor.
status: current
confidence: lesson
op: Einsum;Slice;Pad;QLinearConv
technique_section: 134
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-09T190000Z_skeleton_probe_directive_floor_audit.md
~~~

~~~library-fact
id: neurogolf.lesson.20260709.task365.directive_floor_source_check
task_id: task365
fact_type: process_trap
subject: Directive floor claims can be stale or self-referential
claim: Before building from a directive floor, check the task doc floor proof, provider consistency output, team manifest, and whether the cited technique is already in the current team-best; task365's directive reused #156 from the same task and cited a stale 3260 team_best while the manifest had 2431.
status: current
confidence: lesson
technique_section: 156
source_anchor: Onnx_Advanced/.update/_archive/update_integrated_2026-07/DONE_2026-07-09T190000Z_skeleton_probe_directive_floor_audit.md
~~~

~~~library-fact
id: neurogolf.workflow.alias.smallest_onnx_min_cost_search
fact_type: workflow_alias
subject: Search aliases for smallest ONNX and byte-floor work
claim: Agents asking for smallest ONNX, min-cost, tiny architecture, byte floor, score floor, or cost golf should start with lib_query.py status and context taskNNN, then search techniques_catalog, cost_floor_reference, tiny_architectures, by_symptom, and task facts for the target task before opening large docs.
status: current
confidence: documented
source_anchor: Onnx_Advanced/WORKER_PROTOCOL.md
source_anchor_note: also Onnx_Advanced/INDEX.md
~~~

~~~library-fact
id: neurogolf.workflow.alias.gate_and_cost_tooling
fact_type: workflow_alias
subject: Search aliases for ng_gate, score_v2, profile_real_cost, and arcgen
claim: Tooling aliases: ng_gate visible/truth/ab checks correctness; tools/_score_v2.py estimates static score; tools/agent_helpers/profile_real_cost.py checks real-input profile cost; --arcgen 266 means use the full ARC-GEN corpus/profile set when validating or profiling a candidate.
status: current
confidence: documented
source_anchor: Onnx_Advanced/WORKER_PROTOCOL.md
~~~

~~~library-fact
id: neurogolf.workflow.alias.bobrain_provider_direct_read
fact_type: workflow_alias
subject: Use the direct NeuroGolf provider for BoBrain context checks
claim: For BoBrain-backed NeuroGolf context, prefer the direct provider CLI commands neurogolf_provider.cli health, context taskNNN, consistency taskNNN, team-score taskNNN, search QUERY, and ort-op OP; these read live source-policy, team manifest, task routes, facts, and library DB without relying on stale task_state cost fields.
status: current
confidence: documented
source_anchor: external local path - Q:/AIz/BoBrain/Providers/NeuroGolf_Provider/README.md
~~~
