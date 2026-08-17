<!-- GENERATED FILE - DO NOT EDIT BY HAND.
     Regenerate after ANY techniques.md edit:
         python Onnx_Advanced/.update\tools\build_techniques_catalog.py
     source-sha256:2ce7a89e21433043 sections:165 generated:2026-07-15T17:20:41Z -->

# techniques_catalog.md — the complete §-index of `techniques.md` (GENERATED)

**Use this file FIRST for discovery**: grep/scan here (~15 KB, every technique), then open
`techniques.md` at the §-number (`grep "^# 89\." techniques.md`). Titles are verbatim from the
source — title-grep works in both files. `was` = old number before a renumber (breadcrumb map).
Tags are coarse keyword auto-tags, a browse aid only. Routing by graph-symptom: `by_symptom.md`.

| § | technique (title verbatim) | tags | was |
|---|---|---|---|
| #0 | Strategic principle (2026-05-23 update): RADICAL ALGORITHM REWRITE BEATS TRICK STACKING | - |  |
| #1 | Bool-first family (12 rules) | bool |  |
| #2 | Crop-to-active-grid + dimension-tied initializer rewrites | pad-crop shape-move |  |
| #3 | F16-output / BOOL-output trick (drop final Cast) | bool dtype |  |
| #4 | F16 placement rules | dtype |  |
| #4b | Detector→paint→`input+delta`→`Pad` ⇒ output `Where` (drop the 10-channel intermediates) | pad-crop where detection |  |
| #4c | De-Morgan dedupe: `a·(1-b)·(1-c)·(1-d)` ⇒ one Conv + `Greater(==0)` + `Where` | conv where |  |
| #4h | CumSum→ArgMax run-length (drops the F32 CumSum track; opset-12; 2026-06-01, task216) | topk-argmax reduce |  |
| #5 | Equal-fold (algebraic identity) | algebra |  |
| #6 | Add-chain → Sum | reduce |  |
| #7 | Shape-only patterns: the cost floor | floor-proof pricing |  |
| #8 | Whole-grid summary via single Conv | conv shape-move reduce |  |
| #8b | Collapse one-hot → single-channel color-index canvas BEFORE gather/mirror (kill the ×10) | scatter-gather onehot |  |
| #8c | Count-per-value / dominant-color / mode via `ScatterElements(reduction='add')` (task222 v2, 2026-06-01) | scatter-gather reduce |  |
| #8d | Per-category min/max of a positional quantity via `ScatterElements(reduction='min'/'max')` (task018 v7, 2026-06-01) | scatter-gather |  |
| #8e | Per-class k-cell-motif histogram via a multi-input Einsum (scorer-invisible) — the scatter is NOT a floor (task222 v9, 2026-06-08, GRADER-VALIDATED byte-equal-LIVE 200266 + profile==static) | einsum scatter-gather floor-proof free-cost |  |
| #9 | Active-cell / non-bg detection (correct version) | detection |  |
| #10 | nonzero_shortcut: `Mul(input, gate) → ReduceSum` collapses | reduce |  |
| #11 | Static lookup table → dynamic compute (rank-aware) | rank-separable lut |  |
| #12 | Projection-cache (skip the wide intermediate) | - |  |
| #13 | Slice-CSE (common-subexpression elimination) | pad-crop algebra |  |
| #14 | Slice-chain-to-multi-axis | pad-crop |  |
| #15 | Separable mask collapse | rank-separable detection |  |
| #16 | Dead-Relu after pad-zero MaxPool | pad-crop algebra |  |
| #17 | opt_safe blind spots — 5 algebraic-identity patterns to manually scan for | algebra |  |
| #18 | Const-node → initializer | - |  |
| #19 | Conv with runtime weights | conv |  |
| #20 | Gemm `transA` / `transB` eliminate redundant Transpose intermediates | shape-move |  |
| #21 | Drop Conv padding when downstream threshold excludes overflow positions | conv pad-crop |  |
| #22 | Compact-mask projection for perpendicular thickening | detection |  |
| #23 | Color-index scalar representation (instead of one-hot channel expansion) | onehot |  |
| #24 | Merged Conv for "W-wide isolated run" detection | conv detection |  |
| #25 | Active-subset projection before pair-matrix enumeration | - |  |
| #26 | Compact histogram key replaces full flattened MatMul hash | - |  |
| #27 | BOOL-Or chain replaces Cast→Sum→Greater accumulation | bool dtype reduce |  |
| #28 | AND-reorder for row_range × col_range cursor checks | - |  |
| #29 | NOOP sentinel position for ScatterND when ALL positions are valid | scatter-gather |  |
| #30 | Memorized-lookup discriminator must stay F32 when hash sums exceed F16 exact range | dtype lut reduce |  |
| #31 | Direct channel slices vs staged `[1, N, H, W]` mask tensor | pad-crop detection |  |
| #32 | Cell(0,0) base detection via ArgMax (replaces N diagonal Slice + variadic Max) | pad-crop topk-argmax detection |  |
| #33 | Negative-MaxPool trick for MinPool / connected-component labeling (octaviograu 6042.85 technique, opset 11+) | - |  |
| #34 | ScatterND-add histogram for size-counting (opset 17+, octaviograu 6042.85 technique) | scatter-gather reduce |  |
| #35 | Conv weighted-sum color decode + asymmetric-kernel free crop (task368 v9, shipped +2.08, 2026-06-04) | conv pad-crop reduce |  |
| #36 | Cumulative ≥k masks + deepest-rank-outermost Where nesting (task368 v9) | rank-separable where detection |  |
| #37 | Corpus-measured iteration floors for flood/CC chains (task325 + task279, shipped 2026-06-04) | flood-morph floor-proof |  |
| #38 | Marker-sentinel rewrite + sparse ScatterND canvas updates (task122 v7, shipped 2026-06-04) | scatter-gather |  |
| #39 | GatherND(batch_dims=1) sparse reads — read ONLY the cells the rule needs (task195, shipped +3.23, 2026-06-04) | scatter-gather |  |
| #40 | Plane-breaker catalog — what actually beats the 2-input conservation law (2026-06-04, from `_archive/Research_nary_fanin_catalog.md`) | - |  |
| #41 | 2D imos corner marks + QLinearConv U8 prefix-sum — rectangle-interior fill at ~1/3 the parity cost (task338 v7, 2026-06-04) | qlinearconv dtype reduce pricing |  |
| #42 | Free-input Einsum marginals: 0-memory channel reductions (task119 v5, 2026-06-04) | einsum free-cost reduce |  |
| #43 | Negative-bias cliff Conv — threshold detectors without Equal chains (task364, 2026-06-04; pointer section added 2026-06-05) | conv detection |  |
| #44 | Spatial data-movement quick table (2026-06-04, from `_archive/Research_spatial_data_movement_catalog.md`) | lut |  |
| #45 | The packing law — "plane-neutral" is FALSE (2026-06-04, from `_archive/Research_bit_packing_calculus.md`) | packing |  |
| #46 | Einsum repeated-subscript DIAGONAL READS — adopt immediately (2026-06-04, from `_archive/Research_einsum_equation_catalog.md`) | einsum periodicity reduce |  |
| #47 | Row-packed uint32 flood — VERDICT: adopt for 4-NEIGHBOR floods with K ≥ 12 (2026-06-04, from `_archive/Research_row_packed_uint32_morphology.md`) | flood-morph packing |  |
| #48 | Output-overlay `Where(cond, const_onehot, input)` + line-mask conditions (task119 v3, 2026-06-04) | onehot where detection | #41 |
| #49 | Strip-extracted runfill + ScatterND row-writes — K-bounded sparse lines (task054 v5, 2026-06-04) | scatter-gather | #41 |
| #50 | Crossing-parity interior fill (point-in-polygon, NO BFS) + F16 triangular-MatMul prefix-sum (task338 v5, 2026-06-04) | dtype reduce | #40 |
| #51 | Merged-Conv (pair-Conv + Equal → single Conv + Equal) | conv | #23 |
| #52 | Separable-depth rewrite for nested-frame / "onion" tasks (task377, 2026-06-01: 63,682 → 9,887, 6.4×) | rank-separable |  |
| #53 | Fold "colored mask" + "active mask" into ONE weighted-conv input read (task192, 2026-06-01: 56,531 → 18,166, 3.1×) | conv detection algebra |  |
| #54 | Runtime variable-template extraction → Conv-with-runtime-weights (task191, 2026-06-01: 63,060 → 32,841, 1.92×) | conv detection |  |
| #55 | task013 trio: aspect-branch + clamped Mod/Gather periodic fill + broadcast-And assembly (promoted from `.update/` 2026-06-05; shipped byte-equal, 58,000 → 2,774) | scatter-gather periodicity |  |
| #56 | Hash-MLP architectural fingerprint — "is this MLP a hash table?" (promoted from `mlp_rule_extraction.md` 2026-06-05) | lut |  |
| #57 | Boolean-safe reduction scaffold (promoted from `symmetry_and_affine.md` 2026-06-05) | bool algebra |  |
| #58 | Sub-presence period detection (ReduceSum − ch0_presence) | periodicity detection reduce | #33 |
| #59 | Opset-18 Pad axes input + shared init with ReduceMax | pad-crop reduce | #34 |
| #62 | Staircase-code rank-1 assembly — separable outputs WITHOUT planes (task055, 2026-06-06; 31,628 → 2,386 byte-equal, 13.3×) | rank-separable | #58 |
| #62b | BITAND bit-detector upgrade — halves #62's assembly again (task055 v8-v11, 2026-06-06; 2,386 → 1,174, ab-byte-equal-full) | packing detection | #58b |
| #63 | Einsum × OneHot selected-channel marginals — skip the all-color count map (task079, 2026-06-06; 24,409 → 1,382 shipped) | einsum onehot reduce | #59 |
| #64 | Cast-BOOL occupancy → ArgMax bbox extents — the cheapest data-dependent bounding box (task228, 2026-06-06; 23,286 → 2,087 shipped byte-equal) | bool dtype topk-argmax |  |
| #65 | Assembly-choice laws — Equal-vs-Where and the mild-dynamic 600B threshold (task166 floor campaign, 2026-06-06; 21,648 → 1,900 shipped) | floor-proof where |  |
| #66 | Superoptimizer verdict + the exact-rewrite shortlist (research card K1, 2026-06-06) | - |  |
| #67 | FREE-OUTPUT catalog — the output node is never charged; make the expensive op BE the output (2026-06-07 wave) | free-cost |  |
| #68 | Floor proofs are FORMULATION-RELATIVE — enumerate final-op families FIRST (task256, 2026-06-07) | floor-proof |  |
| #69 | Sign-arithmetic kit — channels need the right SIGN, not {0,1} (task256/078, 2026-06-07) | - |  |
| #70 | Einsum place-value / dominance packing — many scalars per ONE free-input reduction (078/256/247, 2026-06-07) | einsum packing free-cost reduce |  |
| #71 | Moment-correlation mono-window detection — variance-zero identity (task143 → task206 family, 2026-06-07) | detection reduce algebra |  |
| #72 | QLinearConv rounding-boundary gadgets — logic ops from quantization rounding (task077, 2026-06-07) | qlinearconv |  |
| #73 | Locate-then-detect + separable ones-banks + marginal-profile architecture (task205, 2026-06-07) | rank-separable detection reduce pricing |  |
| #74 | 3-operand free-input Einsum VERIFIED + INT8/U8 vector lanes at opset 14 (task148, 2026-06-07) | einsum dtype free-cost reduce |  |
| #75 | K3 residual-op audit — five real finds + three confirmed dead (2026-06-07, `_archive/Research_k3_unexploited_op_audit.md`) | algebra |  |
| #76 | K4 verdict — logic-synthesis offline pass is the ONE transfer; nothing else beats hand construction (2026-06-07, `_archive/Research_k4_compression_native_minimal_nets.md`) | - |  |
| #77 | F16 EXECUTION MODEL on the grader CPU — K5 source-read + ship-empirics reconciled (2026-06-07, `_archive/Research_k5_accumulation_exactness.md`) | dtype |  |
| #78 | FREE-OUTPUT catalog v2 — four MORE output-node families (wave 2, 2026-06-07; extends #67) | free-cost |  |
| #79 | Small-structure reconstruction kit — Equal(iota, sum-code) + the parameter-LUT crossover law (345/357, 2026-06-07) | lut reduce |  |
| #80 | DepthToSpace(CRD) block-stamp interleave — fractal layouts in ONE op (task304, 2026-06-07; extends #44) | shape-move |  |
| #81 | PAD-FLOOR CROSSOVER LAW — price the Equal-tail BEFORE building it (sweep wave-3, worker-7, 2026-06-07; extends #67/#78) | pad-crop floor-proof pricing |  |
| #82 | Singleton-object plane rebuild kit — einsum scalar reads + Equal-ramp planes (353/047/389, 2026-06-07; composes #42/#46/#79) | einsum reduce |  |
| #83 | NIBBLE-PACKED transport plane — 2 cells/byte halves any color-index plane (task209 v6, 2026-06-07; pair with #81 before converting) | packing |  |
| #84 | Placer-Conv structural toolkit — IG one-hot-sum identity, groups-halving, dilated quadrant reads (task180, 2026-06-07; params-only class 3rd member: 180@210 joins 215@110, 252@280) | conv flood-morph onehot reduce |  |
| #85 | Free-output QLinearConv toolkit — int32-bias binarizer, in-conv morphology, the bias-leak law, crop-instead-of-mask (task265, 2026-06-07; extends #72/#74/#78) | qlinearconv conv pad-crop flood-morph |  |
| #86 | Occupancy-pack + max-bound existence — the aggregation IS the detector (task346, 2026-06-07; extends #69/#70/#71) | packing detection |  |
| #87 | Selector-bank rank minimization — the L1 method (research deliverable, 2026-06-07; serves #68-class rank residuals like 312's +0.51) | rank-separable |  |
| #88 | u8 flood kit — row-byte packed flood (#47 crossover FLIPS at width<=8) + Min-masked MaxPool + seed/priority tricks (task048, 2026-06-07; 8,831 -> 699) | dtype flood-morph packing detection |  |
| #89 | Einsum-final design rules — ORT 1.24.4 source-read (L2 deliverable, 2026-06-07; the einsum-final class #68's runtime contract) | einsum reduce |  |
| #90 | No-Mul colour decode — scaled-conv weights + raw-channel u8 Max merge (task020, 2026-06-07) | conv dtype |  |
| #91 | Einsum-final vs Pad-last crossover for free-output spatial remaps (task142 519→456, 2026-06-07) | einsum pad-crop free-cost reduce |  |
| #92 | Marginal-split orientation detector — 2 depthwise marginal-diff Convs beat the 40-ch 2D conv (task253 23974→2434, 2026-06-08) | conv detection reduce |  |
| #93 | Weighted-marginal Einsum + mod-k — a position-invariant feature from ZERO spatial planes (task253, 2026-06-08; extends #42/#82/#89) | einsum reduce |  |
| #94 | Post-opset-18 levers + bf16 trap (M1 opset 19-22 + N1 opset 23-26 audits, 2026-06-08/06-19; `_archive/Research_m1_opset19_22_op_audit.md`, `_archive/Research_n1_opset23_26_delta_audit.md`) | - |  |
| #95 | DISJOINT-SOURCE MERGE WITHOUT A `Mul`: scaled-conv weights + `ch_ids` remap (task020 v2, 2026-06-07, GRADER-CONFIRMED 0% gap) | conv |  |
| #96 | FUSED LINEAR-COLLAPSE + NEGPAD-CROP DETECTION CONV: one conv from the raw one-hot to the detection signatures (task264, 2026-06-08, grader-confirmed byte-equal-LIVE 200265/200265) | conv pad-crop onehot detection |  |
| #97 | uint8 SIGNATURE DETECTION + (K+1)-BIN ScatterElements ROUTING: route N detected objects to K slots without a `[K,H,W]` one-hot (task264, 2026-06-08, grader-confirmed byte-equal-LIVE 200265/200265) | dtype scatter-gather onehot detection |  |
| #98 | DILATED-KERNEL SPATIAL-CROP + CHANNEL-REDUCE in ONE conv (task046, 2026-06-08, GRADER-VALIDATED 2026-06-08 via A/B primary-scored; gate-clean 200267 + 200k truth) | conv pad-crop flood-morph reduce |  |
| #99 | TopK STREAM-COMPACTION: remove gaps / left-pack occupied columns WITHOUT cumsum+OneHot+matmul (task046, 2026-06-08, GRADER-VALIDATED 2026-06-08 via A/B primary-scored; gate-clean 200267 + 200k) | topk-argmax packing onehot reduce |  |
| #100 | FOLD per-window CLASS + VALIDITY into ONE uint8 conv channel (task264 v5, 2026-06-08, grader byte-equal-LIVE 200k) | conv dtype algebra |  |
| #101 | TopK(k=N) for EXACT-COUNT object extraction → collapse the scatter index (task264 v7, 2026-06-08, grader byte-equal-LIVE 200k, profile==static) | scatter-gather topk-argmax reduce pricing |  |
| #102 | eps-FRACTIONAL FOLD: pack a discrete CLASS + a quantized PAYLOAD into ONE float channel so TopK/sort carries both (task264 v9-v11, 2026-06-08, grader byte-equal-LIVE 200k, profile==static==2979) | qlinearconv topk-argmax packing pricing |  |
| #103 | MaxPool sprite-preserving DOWNSAMPLE before TopK (task264 v12, 2026-06-08, byte-equal-LIVE 200k, profile==static==2605) | topk-argmax pricing |  |
| #104 | Min-cost ONNX realization of a fixed integer linear map y=Wx — the 7-step decision procedure (M3, 2026-06-08; `_archive/Research_m3_min_cost_integer_linear_map.md`) | pricing |  |
| #105 | "STRICTLY BETWEEN TWO MARKERS" span-fill via 4 directional U8 MaxPool scans (task350, 2026-06-08, reviewer structural-floor, byte-equal-LIVE 200267 + truth 200000) | dtype floor-proof |  |
| #106 | FIXED-CORNER MASK kills a DYNAMIC OneHot selector when the index is generator-constant (task183, 2026-06-08, reviewer structural-floor, byte-equal-LIVE 200265) | rank-separable floor-proof onehot detection |  |
| #107 | CAUCHY-SCHWARZ EINSUM "filled-rect / all-rows-equal" test — per-channel shape detection with NO [C,H] tensor (task234, 2026-06-08, gate-5 profile==static, 266/266 + 10000 fresh) | einsum detection reduce pricing |  |
| #108 | COLLAPSE N nested/shell/ring/frame mask planes into ONE per-cell distance predicate (task096, 2026-06-08, reviewer floor-verified, byte-equal-LIVE 200266) | floor-proof detection |  |
| #109 | FREE-OUTPUT multi-term separable Einsum — build the WHOLE one-hot output with NO canvas/Equal-tail (task234, 2026-06-08, gate-5 profile==static==2808, 266/266 + 10000 fresh) ] | einsum rank-separable free-cost onehot |  |
| #110 | BIT-PACK + PREFIX/SUFFIX LUT-GATHER for ray-fill / axis-aligned propagation (task064, 2026-06-08, reviewer floor-verified, byte-equal-LIVE 200267) | scatter-gather packing floor-proof lut |  |
| #111 | CROSS-CREATOR CALIBRATION — Aditya wins by COMPUTING rules at zero/low-param where we spend learned weights (Team_Best_Onnx analysis, 2026-06-08) | where |  |
| #112 | BIT-PACK (value / position<<k) then MaxPool to SELECT-BY-POSITION, mask to extract value (Aditya task009, R5 2026-06-08) | packing detection |  |
| #112b | BITSHIFT as cheap integer ×2^k / ÷2^k (Aditya task012/263) | packing |  |
| #113 | GatherND STENCIL-READ + binary place-value PATTERN-ENCODE + ScatterND reduction-WRITE (Aditya task027) | scatter-gather packing detection |  |
| #114 | CumSum over run-START markers = ORDINAL / rank labeling of bands (Aditya task184) | rank-separable reduce |  |
| #115 | VARIANCE-IDENTITY spatial EXTENT from einsum moments: D = sqrt(2·Q − S²) (Aditya task012) | einsum reduce algebra |  |
| #116 | OneHot(color-index grid, depth=10) as the DIRECT one-hot output (Aditya task007) — crossover vs #81 Equal-tail | onehot shape-move |  |
| #117 | RESIZE with a DATA-DEPENDENT ROI to scale/place a constant template (Aditya task021) | detection |  |
| #118 | BASIS-MATRIX axis remap / upscale / placement as an EINSUM OPERAND (Aditya task034/068, einsum-catalog 2026-06-08) | einsum reduce |  |
| #119 | DEEP separable output — push the free-output Einsum to 6–30 operands (Aditya task009/084/068; extends #109) | einsum rank-separable free-cost reduce |  |
| #120 | QLinearConv y_scale != 1 = conv + BUCKET / THRESHOLD in one op (Aditya task278/349/396; extends #85) | qlinearconv conv |  |
| #121 | GridSample for COMPUTED spatial remap / warp (Aditya task351) | shape-move |  |
| #122 | MatMulInteger for cheap INTEGER projection / reduction (Aditya task297/355) | - |  |
| #123 | Cheap PREDICATE + COORDINATE idioms (recurring across his nets) | - |  |
| #124 | RANK / SORT small-N by pairwise-comparison count (Aditya task010; R5 deep-dive batch 1) | rank-separable reduce |  |
| #125 | Cheap boolean & encoding micro-idioms — the connective tissue of his rules (Aditya, R5 batch 2) | bool |  |
| #126 | ITERATIVE MaxPool FLOOD-FILL — 2D region growth + enclosure/hole detection (Aditya task251) | flood-morph detection |  |
| #127 | GEOMETRIC / coordinate micro-idioms (Aditya task375/329/373) | - |  |
| #128 | SHAPE-PROPERTY via a MOMENT IDENTITY — derive area/size algebraically, skip the bbox (Boladi task100) | reduce algebra |  |
| #129 | GatherND SUB-SHAPE REPLICATION / grid-tiling — the WRITE side of #113 (Boladi task033) | scatter-gather shape-move |  |
| #130 | ANALYTIC / TRIG-FORMULA output: memory-0 closed-form + `>0`-logit decode (Boladi task007, LB-probed 2026-06-09) | - |  |
| #131 | FORCED FLOOR for paint-mask-preserve tasks — decompose before you target a number (Boladi task102) | floor-proof detection |  |
| #132 | STAMP-TILING via a rank-K separable-einsum basis + free-output (Boladi task327; the einsum twin of #129) | einsum rank-separable free-cost shape-move |  |
| #133 | CROP to the generator's HARD content bound + check separability first (Boladi task093) | pad-crop |  |
| #134 | FLOOR DISCIPLINE — a forced-component PROOF vs a no-lever CLAIM (Boladi task365 vs task102/task350) | floor-proof |  |
| #135 | Base-16 LOG-POSITION scalar decode — compact position encoding (Boladi task300) | packing |  |
| #136 | DERIVE, don't RE-COMPUTE — audit every Conv against the labels you already have (Boladi task022) | conv |  |
| #137 | Narrow a THRESHOLD-ONLY carrier to U8 via QLinearConv — a dtype golfing lever (Boladi task192) | qlinearconv dtype |  |
| #138 | Scorer-INVISIBLE expansion temporaries — free block-upsampling (Boladi task289) | free-cost |  |
| #139 | ScatterND TINY-PATCH avoids the 900 B Where-mask + Einsum-midpoint (not NonZero) for dynamic indices (Boladi task371) | einsum scatter-gather where detection |  |
| #140 | SCALAR ROW-MOMENT dissolves a presence plane (task188) | reduce |  |
| #141 | Axis-bitset intersection as FREE OUTPUT (task047) | free-cost |  |
| #142 | Single-color one-hot dissolve: mask × color vector (task217) | onehot detection |  |
| #143 | Size-conditional index remap + index-map cost law (task298/task382) | pricing |  |
| #144 | Scalar dimension recovery from a full one-hot count (task067) | onehot reduce |  |
| #145 | Sparse scattered F32 reads: flat GatherND beats separable Slice/Gather (task033) | pad-crop scatter-gather rank-separable |  |
| #146 | Dynamic Slice value_info rescue + Slice-vs-GatherND crop law (task036) | pad-crop scatter-gather |  |
| #147 | TopK-marker shrink stack: F16 OneHot, I32 indices, F16 counts (task391) | dtype topk-argmax onehot reduce |  |
| #148 | Reversed-Slice ArgMax for the LAST (or Nth-from-end) active element (task288, 2026-06-12) | pad-crop topk-argmax |  |
| #149 | Concat-prepend beats Slice+Pad for a zero-edge shift (task199, 2026-06-12) | pad-crop |  |
| #150 | One triangular operand serves BOTH directions via subscript swap + Where(f,x,x) NOP removal (task270, 2026-06-12) | where |  |
| #151 | ch0-background Slice replaces a ch0-penalty mask; the shared-[1] init lever (task291, 2026-06-12) | pad-crop detection |  |
| #152 | Integral-histogram / summed-area cache for repeated rectangle color queries (O1, 2026-06-19) | periodicity reduce |  |
| #153 | Small-N exact correspondence via subset-state DP; use matchers only when a key is not enough (O1/O5, 2026-06-19) | - |  |
| #154 | Row-packed XAG/bitwise lane for XOR/equality/symmetry binary rules (O6, 2026-06-19) | packing |  |
| #155 | Broadcast-offset index assembly + tiny fixed-offset ScatterND masks (task250, 2026-06-19) | scatter-gather detection |  |
| #156 | Tiny-K scalar selector: chained `Greater`/`Where` beats `Concat`->`ArgMax`->`Gather` (task365, 2026-06-21) | scatter-gather topk-argmax rank-separable where |  |
| #157 | Bit-unpack U8-lane narrowing: EXIT expand in U8, not packed-word dtype (task077, 2026-06-24) | dtype packing |  |

*Named sub-blocks (e.g. `§39 MECHANISM CONFIRMED`, `F16-CONV VERDICT`, `§39 GAP-ledger`) are not*
*top-level §s; grep them directly in `techniques.md`. Doctrine/traps live in `traps.md`, not here.*
