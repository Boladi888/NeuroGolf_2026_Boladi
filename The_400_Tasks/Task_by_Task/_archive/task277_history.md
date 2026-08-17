# task277 — archived history

Superseded detail moved out of `task277.md` on 2026-07-02 to satisfy the
`task-md-size` lint (WORKER_PROTOCOL staging rule). The CURRENT best rung
(`task277_xor_qlc_count_cost3481`, byte-equal to the accepted team-best) keeps
its full prose in `task277.md`; everything below is retained for provenance.

## Current gate — full structural-vs-LIVE interpretation (2026-05-23 Prompt v4)

The 11-task bulk cherry13b submission completed 2026-05-23T19:45Z. Replacing
task277 LIVE with the cost-0 Identity/all-zero stub produced **0.00 public LB
delta**, so the current LIVE contributes 0 public score. This does **not** prove
latent task277 weight is zero; it means the current LIVE/NN-distilled family is
wrong on hidden. Per the task066 wallpath precedent, the remaining path is a
true structural object-dedup rewrite.

Raw `verify_against_live` is contradictory for this task: structural candidates
that match expected full tensors 266/266 still score 0/266 vs LIVE because LIVE
emits signed raw logits/padding and is visibly wrong against expected. Treat
`prototype_vs_expected`, task-shaped synthetic-reference gates, and the v4 public
diagnostic as the active decision signals; do not treat BOOL-vs-signed-LIVE
divergence as evidence against structural candidates.

Submit-captain-only staged structural probes (historical):
- `.hermes/board/staged_submits/task277_bbox_area_min_edges_t_8c0ccd6c_on_5964_63_cost16748/` — cost 16,748, expected 266/266, roles synthetic 724/1000, catalog synthetic 565/1000.
- `.hermes/board/staged_submits/task277_shape_catalog_dup_safeungated_t_7e08e567_on_5964_63_cost33795/` — cost 33,795, expected 266/266, roles synthetic 995/1000, catalog synthetic 314/1000.

## Superseded Attempts (full prose)

- 2026-06-20 | **Slice+mask10h-elimination param shave (beats 4049)** | **CANDIDATE staged** at `staged_submits/task277_conv_slice_cost3844/` (SHA `74E00463CA881B0273B9E4192C4A812F788CFFCC5C8F1ED7BB07E6DF9F5D788B`). Built on the 4049 u8_maxpool prototype. Two changes: (A) eliminated `mask10h` (F16 Cast of mask10, 200B) by casting mask10 directly to BOOL for `mask_bool` and doing `ReduceSum(mask10_f32) -> count_total_f32` + `Cast(F16) -> count_total` for the Sub path; (B) shrank Slice init arrays from [4]-element to [3]-element (axes=[1,2,3] skipping batch dim, -3 params); also removed `axes_all` (4 params) via empty-axes ReduceSum, `zero` (1 param), `false_bool` (1 param via optional Pad input). Net: memory 3921→3725 (-196B: eliminated mask10h 200B + small scalar changes), params 128→119 (-9). Cost **4049→3844 (-205)**. Gates: visible `266/266`; truth `--full 199211/200000 (99.606%)` — identical to team best; AB `--full` vs u8_maxpool (team best) `200266/200266 divergences=0 **byte_equal=True**`; `score_v2=(0,3725,119)`; profile `--arcgen 266` static=profile=3844. Expected delta vs team best `ln(4049/3844)=+0.052`. Floor not claimed; remaining charges are mask10 F32 (400B, Slice reads free input but output is 400B F32 - can't go lower), mask_c1/mask_c2 F16 planes (200B each, ReduceSum(BOOL) rejected forcing F16 floor), rgb3 BOOL 300B, U8/BOOL propagation planes. | 3,844 | Worker-277 2026-06-20 |
- 2026-06-12 | **F16 -> U8 label propagation (beats 5250)** | **CANDIDATE staged** at `staged_submits/task277_U8_MaxPool_cost4049/` (SHA `270DC183575227B7E93068347E0EB6B6B76E109F4462A562B1DD47F69A9275ED`). Rework per reviewer-277. Built on the 5250 MaxPool-pads candidate. Converted the 11 F16 `[1,1,10,10]` propagation/working planes (`nlabels_0..5`, `pooled_0..4`) to U8 by remapping the label scheme: `neg_idx` F16 (0..-99 = negated position index) -> `lbl_u8` (`100 - pos` in {1..100}); sentinel `neg_big`=-999 -> `bg_u8`=0 in the five propagation Where fills; ReduceMin background fill (`nlabels_fg0`, node 15) F16 `zero` -> `hi_u8`=255 so bg can't win the min over positive labels. MaxPool-max on positive labels picks largest `100-pos` = smallest position index = SAME component-representative propagation. ReduceMax/ReduceMin/Equal run on U8 at opset 14. **Count path (ReduceSum nodes 24-26) untouched** — it already casts BOOL->F16 (`mask_c1/mask_c2`) and reads `mask10h` F16, so no `ReduceSum(U8)` introduced (`ReduceSum(U8)` is rejected by ORT). Cost **5250 -> 4049 (-1201)** (memory 5123->3921; the 11 planes 200B->100B = -1100, plus neg_idx 200B->lbl_u8 100B = -100). Gates: visible `266/266`; truth `--full 199211/200000 (99.606%)` first_fail_idx **1259** — identical count AND first_fail to team-best; AB `--full` vs team-best 6698 `200266/200266 divergences=0 **byte_equal=True**`; ORT A/B vs the 5250 F16 over 5000 random grids `0 mismatches`; `score_v2=(0,3921,128)`; profile `--arcgen 266` static=profile=4049. Expected delta vs team-best `ln(6698/4049)=+0.503`, vs fallback `ln(5250/4049)=+0.260`. | 4,049 | rework-worker-277 task277 |
- 2026-06-12 | **MaxPool-pads fold (beats 6698)** | **CANDIDATE staged** at `staged_submits/task277_beat6698_cost5250/`. Built on the 6698 BOOL-tail team-best. Folded the five F16 label-propagation `Pad` nodes into their `MaxPool` via the `pads=[1,1,1,1]` MaxPool attribute — ORT MaxPool intrinsic padding ignores OOB cells for the max (equivalent to pad-with-`neg_big`), so the propagation is byte-identical. Removed five `[1,1,12,12]` F16 pads (288 B each = 1440 B) + unused `pads_hw` init (8 params). Cost **6698 -> 5250 (-1448)**. Gates: visible `266/266`; AB vs team-best `200266/200266` divergences=0 **byte_equal=True**; truth `199211/200000 (99.606%)`; `score_v2=(0,5123,127)`; profile `--arcgen 266` static=profile=5250. SHA `9FA010DC6C0208DAD75641F12294F3D2756818D8033DA67731E7FAB7F0DA4D36`. Expected delta vs 6698 `ln(6698/5250)=+0.244`. | 5,250 | Worker worker-277 task277 |
- 2026-06-11 | team-best BOOL-tail compression | **CANDIDATE staged** at `staged_submits/task277_team_bool_tail_cost6698/`. Source-of-truth baseline is sibling manifest task277 by Aditya, cost 7,598, not the stale home 16,748 route. Converted the team-best final F16 output tail to BOOL: removed three final Casts, kept decoded output byte-equal to the sibling team-best over full AB `200266/200266`, visible `266/266`, `score_v2=(0,6563,135)`, profile `6698`. SHA `A98589EB091441CE8074050E90BCAE8658D3B2BB24B1EF062F9D0DBC146DC64D`. Expected delta vs team-best `ln(7598/6698) ~= +0.126`. 4-/3-/2-round propagation variants rejected on visible (`246/266`, `205/266`, `84/266`). | 6,698 | Worker worker-5 task277 |
- **2026-05-23T20:50Z | ✅ SHIPPED bbox_area_min_edges structural rewrite — Kaggle 5968.46 → 5983.73 (+15.27 LB)** | Cost 46,539 → 16,748 (-29,791). points 0.0 → 15.27 (LIVE was 0/266 vs expected per bulk cherry13b, contributing 0 LB; structural rewrite generalized to hidden at full theoretical max). visible 266/266 vs expected; synth role-shaped 724/1000 (72.4%); polyomino 23.9%. **Second confirmation of task066 wallpath solve-from-scratch unlock pattern.** Package SHA `BA747D13D847EB525D95AAFC5A863272A52AF07B62DDED0E21792BC1A72D8D92`. Builder: `prototypes/task277_bbox_area_dup_min_edges.onnx` with `--visible-pruned --edge-pruned --min-edges` flags. | 16,748 | submit-captain manual ship 2026-05-23T20:50Z |
- 2026-05-23 | 8-hidden distilled sign MLP + BOOL output tail | Cost 4,201, threshold-decodes 266/266 vs expected, fails raw `verify_against_live` (BOOL vs signed logits). Synthetic 59/1000. Not staged. Builder: `prototypes/build_task277_mlp8_bool.py`. | 4,201 | Codex Task277 lower-cost push |
- 2026-05-23 | optimized 8-hidden distilled sign MLP + BOOL output tail | Cost 3,900, threshold-decodes 266/266; adversarial-against-LIVE 127/250 (advisory). BOOL vs LIVE signed. Artifact: `prototypes/task277_mlp8_bool_opt.onnx`. | 3,900 | Codex Task277 under-5k push |
- 2026-05-23 | 12-hidden visible-fit distilled sign MLP + BOOL output tail | Cost 4,736, decodes 266/266; adversarial 132/250. Artifact: `prototypes/task277_visible_mlp_H12_seed6.onnx`. | 4,736 | Codex Task277 under-5k push |
- 2026-05-23 | 36-hidden shift/flipped distilled sign MLP + BOOL output tail | Cost 9,752, decodes 266/266 and 6,670/6,670 shifted/flipped; adversarial 130/250. Builder: `prototypes/build_task277_mlp_bool_from_npz.py`. | 9,752 | Codex Task277 lower-cost push |
- 2026-05-23 | bbox+area duplicate detector with exact 2x3/area4 collision override | Decodes 266/266 but costs 53,500 (profiler-visible Conv/edge branches). Rejected. Builder: `prototypes/build_task277_bbox_area_dup.py`. | 53,500 | Codex Task277 lower-cost push |
- 2026-05-23 | raw-live scalar polish of older prototype MLP | Scalars replace broadcast inits. Raw live 266/266, cost 46,749 -> 46,551, but current submission already lower at 46,539. Artifact: `prototypes/task277_live_scalar_polish.onnx`. | 46,551 | Codex Task277 lower-cost push |
- 2026-05-23 | Conv/ConvTranspose rewrite of pruned H7 visible MLP | Pruned H8→H7, rebuilt as `Conv(10x10)->Relu->ConvTranspose(10x10)`. Cost 3,358, decodes 266/266; raw live 0/266 (BOOL one-hot vs signed). Artifact: `prototypes/task277_mlp_H7_pruned_convtranspose.onnx`. | 3,358 | Codex Task277 under-3k push |
- 2026-05-23 | H5 pruned/fine-tuned ConvTranspose under-3k attempt | Best under-3k artifact 2,948 but only 265/266 (foreground false positive at pair 258 / cell 57). Not exact, not staged. Artifact: `prototypes/task277_h5hard_convtranspose.onnx`. | 2,948 | Codex Task277 under-3k push |
- 2026-05-23 | H5 plus one exact visible hard-case unit | Cost 3,153, decodes 266/266. Trained/visible-fit — **forbidden for further work** under Prompt v3. Artifact: `prototypes/task277_h5hard_plus_exactunit.onnx`. | 3,153 | Codex Task277 ONNXAdv under-5k push |
- 2026-05-23T17:42Z | **SUBMITTED Kaggle calibration**: 2,948 candidate submitted as single-task wildcard on 5949.01. **Result: ZERO delta.** Prompt v3 later marked this **INCONCLUSIVE** (visible-fit NN can share LIVE's wrong hidden behavior). | 2,948 → no LB delta | NN/CAE/MLP family closed by policy |
- 2026-05-23T19:27Z | **STAGED cherry13b Identity diagnostic** | Cost 0, `check_network=True`, visible 0/266 by design, package SHA `26C4F51B5CC99178947610589597E2667E4F816C061EC01E7724E0F410898F95`. | 0 | `.hermes/board/staged_submits/task277_cherry13b_identity_.../manifest.json` |
- 2026-05-23 | structural bbox+area duplicate detector, UINT8-Max combiner | Cost **44,100**, decodes 266/266; raw live 0/266. Hidden-risk stress found false cases. Artifact: `prototypes/task277_bbox_area_dup.onnx`. | 44,100 | Codex Task277 rule-rewrite push |
- 2026-05-23 | visible-pruned bbox+area detector, pruned edges + gated-hit spreading | Cost **17,992**, decodes 266/266; raw live 0/266. Higher hidden risk (visible-dead detectors removed). Builder flags `--visible-pruned --edge-pruned`. | 17,992 | Codex Task277 rule-rewrite cost push |
- 2026-05-23 | cheaper combo-exact structural variant | Cost **17,630**, decodes 266/266. Polyomino stress 168/500. Builder flags `--visible-pruned --edge-pruned --combo-exact`. | 17,630 | Codex Task277 rule-rewrite cost push |
- 2026-05-23T19:45Z | **Bulk cherry13b public diagnostic complete** | 0.00 LB delta. NN/MLP/CAE/NPZ cost compression closed; continue structural rewrite only. | 0.00 LB diagnostic | Prompt v4 |
- 2026-05-23 | exact pairwise shape-code structural prototype | Expected 266/266, roles 793/1000, catalog 828/1000, polyomino 489/1000, cost **330,026**. Artifact: `prototypes/task277_shape_code_pairwise_observed_f16.onnx`. | 330,026 | Codex Task277 v4 structural rewrite |
- 2026-05-23 | min-edge bbox+area structural probe | Expected 266/266, cost **16,748**. Roles 724/1000, catalog 565/1000, polyomino 239/1000 (LIVE 7/1000 on roles). Staged. | 16,748 | `prototypes/task277_bbox_area_dup_min_edges.onnx` |
- 2026-05-23 | grouped exact observed-shape catalog | Full catalog expected 266/266, catalog 999/1000, cost **75,041**. Dup-only 266/266, catalog 643/1000, cost **47,808**. Safe-ungated dup-only 266/266, roles 995/1000, catalog 314/1000, cost **33,795** (staged). | 33,795-75,041 | Codex Task277 v4 structural rewrite |
- 2026-05-23 | hybrid exact-mixed prototype | Exact detectors only for colliding bbox/area combos. Expected 266/266, roles 833/1000, catalog 516/1000, cost **32,194**. Not staged (weaker on role-shaped than safe-ungated). | 32,194 | Codex Task277 v4 structural rewrite |

## Superseded Cost analysis blocks

2026-06-11 team-best compression:
- Sibling team-best is Aditya's 7,598-cost component-label/count graph, not the local stale 16,748 bbox route. It uses five F16 Pad+MaxPool propagation rounds, then component-size comparisons, then an F16 output tail.
- BOOL-tail rewrite saves exactly 900 cost: final charged tensors change from three `[1,1,10,10]` F16 Casts plus `[1,3,10,10]` F16 Concat (1,200 B) to one `[1,3,10,10]` BOOL Concat (300 B). Profile equals static at 6,698.
- Propagation shortening is not available on this route: 4 rounds would cost 6,010 but fails visible `246/266`; 3 rounds `205/266`; 2 rounds `84/266`. Remaining dominant charges are `mask10` F32 400 B, five `[1,1,12,12]` F16 pads at 288 B each, and the F16 label planes.

2026-05-23 under-3k frontier:
- Generic F16 BOOL-tail MLP costs by hidden size: H4 2,849; H5 3,054; H6 3,259; H7 3,464; H8 3,669. H4/H5 random foreground-only training did not fit (best H4 368 wrong; best H5 240 wrong).
- Conv/ConvTranspose form preserves the MLP math but avoids the flat-score reshape path. Measured exact H7: 3,358. Measured H5 near-miss: 2,948.
- H5 can be pushed to 1-2 foreground-cell errors but did not reach 0 in the sweeps tried. Fixed-feature column refits show the last column is not linearly separable with the current H5 activations; further progress likely needs a better H5 hidden representation, not just output-bias tuning.

2026-05-23 structural rewrite frontier:
- `task277_bbox_area_dup.onnx` now costs 44,100 after the UINT8-Max combiner rewrite. Savings came from removing 47 OR-tree intermediates and one `rect_bool` tensor per branch.
- The graph detects duplicate `(bbox height, bbox width, area)` groups and exact-handles the visible 2x3/area4 collision. This is not yet the full normalized-shape rule.
- Visible-only structural pruning pushed cost to 17,992; cheapest visible variant is 17,614 after `onnxsim` (`task277_bbox_area_dup_pruned_edges_combo_sim.onnx`).
- ONNXAdv under-5k pass: the existing H7 visible-fit ConvTranspose graph was already near-floor at 3,358. The lower exact visible artifact is the H5+exact-unit graph at 3,153. Prompt v3 forbids continuing this family.
- Prompt v4 update: cherry13b bulk returned 0.00, so cost-only NN/MLP compression is dead. Structural frontiers: (a) bbox/area min-edge at 16,748; (b) exact dup-shape safe-ungated catalog at 33,795; (c) full observed-shape catalog at 75,041; (d) pairwise shape-code at 330,026.
- `prototypes/task277_synthetic_gate.py` is the repeatable gate (`--mode roles|catalog|polyomino`).

### 2026-05-20 - CL_HUGE worker (f16 attempted, no winning candidate)
- Tried: f16 conversion (robust + direct + opset bump), spread_scan (all 21 patterns)
- Result: NO WIN
- Reason: Has Flatten+MatMul (KNN-style, crop blocker); f16 not productive.
- Future workers: try crop-to-active-grid (small graphs <30 max), Equal-fold manual, full rebuild from scratch
