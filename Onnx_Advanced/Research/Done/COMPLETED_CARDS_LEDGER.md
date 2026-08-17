# Completed research-card ledger (moved from `Todo/README.md` 2026-06-10 — don't re-research these)

> Card bodies: `cards/` (this folder). Full deliverable bodies: extracted into the root docs per the
> filing protocol, originals archived to `_archive/Research_<name>.md` (rule 7). Earlier-series bodies
> were already in `_archive/` prefixed `Research_*` / `TodoDone_*`.

## O-series (2026-06-20, algorithm/assignment/scatter follow-up)

> Delivered, card bodies retired to `cards/`, and **extracted into the root docs 2026-06-20**. Targets:
> techniques.md (#152-#154, #94, #8c/#8d/#34), by_symptom.md routes, WORKER_PROTOCOL digest,
> ort_reality.md scatter/Attention facts, scoring_internals.md ConstantOfShape caveat, and
> op_cost_patterns.md scatter shelf.

| Card | Outcome |
|---|---|
| ~~O7~~ | Topology / integral-geometry detectors — **DONE + EXTRACTED 2026-07-03**: `_archive/Research_o7_topology_integral_geometry_detectors.md` (verdict: exact Euler characteristic, single-component hole tests, perimeter, and corner counts are local `2x2` reductions under a zero border and a foreground/background dual-connectivity convention; unrestricted connected-component counting/reachability stays non-local) -> `op_cost_patterns.md`, `ort_reality.md`, `library_db`. |
| ~~O8~~ | F16 `inf`/`nan` dtype narrowing — **DONE + EXTRACTED 2026-07-03**: `_archive/Research_o8_f16_inf_nan_dtype_narrowing.md` (verdict: `Cast` before `Gather`/`Slice` is conditionally safe only after abs-max, finite-output/non-finite dump, and no-invalid-op dataflow checks; `+inf > 0` can be an intentional carrier, but `nan > 0` decodes false; F16 `Einsum` remains an exact-runtime probe target because source tables and local probe notes conflict) -> `op_cost_patterns.md`, `ort_reality.md`, `library_db`. |
| ~~O1~~ | General algorithm mining — **DONE + EXTRACTED 2026-06-20**: `_archive/Research_o1_general_algorithm_mining_static_grid_cost_reducers.md` (verdict: integral-histogram/SAT cache is the best new direction for repeated rectangle color/rank/stat queries; exact tiny assignment and Kronecker/TT factor mining are follow-up probes; generic morphology/distance transforms mostly do not lower graph cost) -> techniques #152/#153. |
| ~~O2~~ | Scatter reduction modes audit — **DONE + EXTRACTED 2026-06-20**: `_archive/Research_o2_scatter_reduction_modes_deep_audit.md` (verdict: `ScatterElements` reductions are real/deterministic on ORT CPU; spec-clean `min`/`max` need opset 18; duplicate-target `ScatterND` reductions are unsafe because ORT parallelizes shared-output writes) -> techniques #8c/#8d/#34, ort_reality, op_cost_patterns. |
| ~~O3~~ | Attention-23 corpus probe — **DONE + EXTRACTED 2026-06-20**: `_archive/Research_o3_attention23_candidate_corpus_probe.md` + artifact archive (verdict: `Attention-23` loads/runs/scores as a fused zero-memory output, but 7,162 scanned ONNX artifacts contain zero Softmax/attention chains; parked future trigger only) -> techniques #94 + ort_reality. |
| ~~O4~~ | ConstantOfShape scorer confirmation — **DONE + EXTRACTED 2026-06-20**: `_archive/Research_o4_constantofshape_scorer_confirmation.md` + artifact archive (verdict: local full-shape final-output A/B confirms dense Constant=9000 params vs ConstantOfShape=4 params/0 memory; no task-valid scalar-filled output target, no Kaggle confirmation) -> scoring_internals + techniques #94. |
| ~~O5~~ | Latent correspondence / assignment algorithms — **DONE + EXTRACTED 2026-06-20**: `_archive/Research_o5_latent_correspondence_assignment_algorithms.md` (verdict: current slot/rank routing dominates unless a true one-to-one objective is load-bearing; subset-state LAP DP is the best exact static probe, Sinkhorn+reject is approximate) -> techniques #153. |
| ~~O6~~ | Boolean exact synthesis to packed bitwise — **DONE + EXTRACTED 2026-06-20**: `_archive/Research_o6_boolean_exact_synthesis_to_bitwise.md` (verdict: add a narrow row-packed uint32 XAG lane for XOR/equality/symmetry-rich binary rules; keep K4 Conv-compile for monotone thresholds/counting) -> techniques #154 + by_symptom. |

## L-series (2026-06-07, born from the einsum-final breakthrough)

| Card | Outcome |
|---|---|
| ~~L1~~ | Exact CP/rank decomposition — **Done 2026-06-07**: `_archive/Research_l1_exact_cp_decomposition.md` (verdict: unrestricted-real rank is ∃R-hard, skip; the ONE regime = bounded-dyadic MIQCP + dictionary + symmetry breaks; flattening ranks = instant lower-bound kill test; permutation banks CANNOT OR-compress) → techniques #87 |
| ~~L2~~ | ORT Einsum internals — **Done 2026-06-07**: `_archive/Research_l2_ort_einsum_internals.md` (in-kernel temporaries = temp-space, SCORER-INVISIBLE; operand order IS the left-deep path — kill big non-output labels early; no operand cap; split only for ~10x; diagonal/permute slow paths; #18654 edge) → techniques #89 |

## R-series (REPO-ACCESS, 2026-06-08)

| Card | Outcome |
|---|---|
| ~~R1~~ | Big-fish formulation-recon — **DONE 2026-06-08**: `Logs/research/R1_bigfish_*` (66 rows, 40 decoded + 26 compaction-skip; top-EV 342/371/362/183/019/289/046/343/029; 11 label-misnomers flagged). |
| ~~R2~~ | Selector-bank rank reduction — **DONE 2026-06-08**: `Logs/research/R2_selector_bank_rank.md`. task312 residual REAL — exact rank-18 {0,+1,-1} CP shipped 600->570; ranks 10-17 OPEN (needs M2 solver stack). task304 DEAD (no cheaper factorization). task357 N/A (packed-lookup already cheaper than any bank). |
| ~~R3~~ | Peephole rewriter — **DONE 2026-06-08 (v1)**: `tools/peephole_rewrite.py` built (--task/--sweep/--check); swept 219 workable, found+staged 5 add_chain_to_sum cuts (185/190/205/208/308, all byte-equal + profile-gap-0). v1 ships ONE safe rewrite; future v2 adds free-output/dtype/negpad/groups rules -> re-run as the library grows. `Logs/research/R3_peephole_sweep.md`. |
| ~~R4~~ | task312 deep rank harvest — **DONE 2026-06-08**: `Logs/research/R4_task312_deep_rank.md` + runner `tools/scripts/task312_rank_cpsat.py`. CP-SAT proved rank<=17 infeasible over {0,1}/{0,1,2}; signed 17->13 OPEN-timeout, no witness -> rank-18 cost 570 stays best, NO gain. |
| ~~R5~~ | Aditya idiom harvest — **DONE 2026-06-08**: → techniques #111–#127 (cross-creator calibration + 16 idioms). |
| ~~R6~~ | ONNX op candidate matrix — **DONE + EXTRACTED 2026-06-19**: `Logs/research/R6_onnx_op_candidate_matrix.md` + `.csv` + 9 tiny probes (body stays in Logs/research/). Verdict: no broad new surface; `LpPool(p=1)` only task-specific new candidate (scorer-verified memory=0/params=0, nonnegative-only), `SplitToSequence`/`ReverseSequence` scorer-rejected ("not permitted"), opset26 not loadable on ORT 1.24.4, `ConstantOfShape` value-attribute accounting needs Kaggle confirmation before doctrine → techniques #75/#94, ort_reality, scoring_internals. |

## N-series (2026-06-19, ONNX/ORT candidate follow-up)

> Delivered, card bodies retired to `cards/`, and **extracted into the root docs 2026-06-19** (full bodies archived to `_archive/Research_n*.md` per rule 7). Targets: ort_reality.md (substring-"Sequence" ban incl. `ReverseSequence`, opset-25 load ceiling, generator+spatial CPU dtype gaps), techniques.md #94 (opset 23-26 audit) / #75 (LpPool verify + ReverseSequence ban) / #44, scoring_internals.md (ConstantOfShape final-output accounting), zero_param_generators.md (EyeLike/Trilu dtype fixes), core.md (ban examples).

| Card | Outcome |
|---|---|
| ~~N1~~ | Opset 23-26 delta audit — **DONE + EXTRACTED 2026-06-19**: `_archive/Research_n1_opset23_26_delta_audit.md` (verdict: mostly barren; only `Attention-23/24` worth a narrow [SPECULATIVE] probe when replacing a large `MatMul -> Softmax -> MatMul` chain; `TensorScatter` lacks ORT 1.24.4 CPU kernel; opset26 `BitCast`/`CumProd` rejected before kernel lookup) → techniques #94 + ort_reality opset ceiling. |
| ~~N2~~ | Sequence/container ops probe — **DONE + EXTRACTED 2026-06-19**: `_archive/Research_n2_sequence_container_ops_probe.md` (verdict: no `probe-now`; whole sequence/optional/string/sparse family stays declined — R6 confirmed the validator's substring-"Sequence" ban rejects even the "only-if-needed" `SplitToSequence`/`SequenceAt` path) → ort_reality banned-ops. |
| ~~N3~~ | Shape/constant generators cost audit — **DONE + EXTRACTED 2026-06-19**: `_archive/Research_n3_shape_constant_generators_cost_audit.md` (verdict: full-size generated intermediates lose; wins are final-output generators, small broadcast carriers like `[30]` iotas / `[30,30]` masks, structural ops that avoid materialized masks; EyeLike/Trilu/OneHot CPU dtype gaps) → zero_param_generators + ort_reality dtype matrix. |
| ~~N4~~ | Obscure spatial ops audit — **DONE + EXTRACTED 2026-06-19**: `_archive/Research_n4_obscure_spatial_ops_cost_audit.md` (verdict: `DepthToSpace` best obscure u8 pixel-shuffle/final-render probe [already #80/#44]; `CenterCropPad` function-backed validator-risk; `GridSample` loses unless it deletes a larger generated-coordinate chain) → techniques #75/#44 + ort_reality. |

## M-series (2026-06-08, gap-targeted post-batch-16)

| Card | Outcome |
|---|---|
| ~~M1~~ | Opset 19-22 audit — **DONE 2026-06-08**: `_archive/Research_m1_opset19_22_op_audit.md` (verdict: mostly barren; 3 levers = ConstantOfShape param-shrink / bool-Reduce@20 / Pad-wrap@19; traps = bf16-not-honored-on-CPU, no-kernel DeformConv/Bernoulli/GroupNorm, AffineGrid loses) → techniques #94 + ort_reality. |
| ~~M2~~ | Exact-rank tooling recipe — **DONE 2026-06-08**: `_archive/Research_m2_exact_rank_tooling_recipe.md` (verdict: OR-Tools CP-SAT, runnable script included; S={0,±1}+table-products+symmetry-breaks+2^24 guards) → techniques #87. UNBLOCKED the 312 ranks-16-13 harvest (R4). |
| ~~M3~~ | Min-cost integer linear map — **DONE 2026-06-08**: `_archive/Research_m3_min_cost_integer_linear_map.md` (7-step decision procedure: boundary -> zero-param -> selector -> conv-sharing -> block -> low-rank -> sparse -> dense; structure->op table + kill-numbers) → techniques #104. Pairs with R1. |

## K-series (2026-06-06/07, five for five decision-grade)

| Card | Outcome |
|---|---|
| ~~K1~~ | `_archive/Research_k1_superoptimization_survey.md` — SKIP all superoptimizer ports; sanctioned fallback = tiny ONNX-native peephole over the exact-rule shortlist → techniques #66 |
| ~~K2~~ | `_archive/Research_k2_ort_exact_semantics.md` — v1.24.4 source-read; TopK sorted=1 mandatory, Where-bool no kernel, integer Div/Mod/Pow native-C++ semantics, pin Resize attrs → ort_reality.md K2 section + traps §K2 ACT-NOW PAIR |
| ~~K3~~ | `_archive/Research_k3_unexploited_op_audit.md` — 72 residual ops; finds: Sum/RoiAlign/LpPool-p1/Hardmax; CenterCropPad CONFIRMED kernel-less; NMS validator-fatal → techniques #75 |
| ~~K4~~ | `_archive/Research_k4_compression_native_minimal_nets.md` — Nothing beats hand construction; ONE transfer = offline ESPRESSO/ABC logic-synthesis pass → 1×1-Conv compile → techniques #76 |
| ~~K5~~ | `_archive/Research_k5_accumulation_exactness.md` — Load-bearing F16-Conv claim REFUTED at source (NO native F16 Conv kernel; cast-insertion→F32); REAL hazard = native F16 Gemm stepwise rounding; QLinearConv ≤255 confirmed → techniques #77 |

## Earlier series (all delivered + extracted; bodies in `_archive/` prefixed `Research_*` / `TodoDone_*`)

- **A1/A2/A3, C1, E1** — deliverables promoted to live `Onnx_Advanced/` root docs: `symmetry_and_affine.md`, `arc_rule_taxonomy.md`, `synth_oracle_strategies.md`, `cellular_automata.md`, `mlp_rule_extraction.md`.
- **F1/F2** — extractions in techniques.md; full bodies `_archive/radius_encoded_weighted_cover.md`, `_archive/row_signature_rank_reduction.md` (retired root stubs in `_archive/stubs_2026-06-10/`).
- **F3/F4** — extractions in techniques.md; full bodies `_archive/multistage_key_first_compositional.md`, `_archive/direct_detection_multi_qlinearconv.md`. Caveat both: worked examples were researcher inferences from public ARC hints, not project-verified specs.
- **G1** (ORT profiler vs dynamic shapes → techniques §39-UPGRADE), **G2** (log-round reconstruction: doubling provably NON-exact → techniques §37-ADDENDUM), **G3** (n-ary fan-in catalog → techniques §40), **H1** (spatial data-movement → #44), **H2** (bit-packing law → #45), **J1** (row-packed u32 morphology → #47), **J2** (Einsum equation catalog → #46).
