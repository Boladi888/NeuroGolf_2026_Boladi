# task284 — archived history (superseded provenance)

Moved from `Task_by_Task/task284.md` 2026-06-21 (WORKER_PROTOCOL md-size rule). These are the long-superseded 2026-05-18/05-22 high-cost lanes (dynamic ScatterND ~3.7k-9.6k = None/0 unscoreable, v2/v5 f16 ~139k-254k, safeopt). All dominated by the current **2593** band-Einsum floor. Kept only as provenance.

## Superseded attempts (2026-05-18 / 2026-05-22)

2026-05-22 | `task284_scatter_dynamic`: radical whole-tail replacement. Kept sparse crop Conv + U8 marker presence, then replaced the 23x24 mask factory with dynamic `Range` segment lists and two `ScatterND`s: first fill the active rectangle with zero over a sentinel crop canvas, then scatter only the colored bracket/arm segments clipped to active height/width. Final scalar U8 Pad + channel `Equal` emits BOOL [1,10,30,30]. **SHIPPED to Kaggle 2026-05-22T22:10Z as 3-stack component on 5900.63 base: 0.00 LB CONTRIBUTION** (the 3-stack landed 5900.63 → 5903.78 = +3.15 LB, exactly accounted for by task382 +2.17 + task077 +0.98; the predicted task284 +2.75 evaporated). **NOT a regression** (LB went up, not down), but the cost reduction didn't translate to LB. Interpretation: scatter_dynamic produces visible+contract-synthetic-correct output (266/266 + 569/569) but either (a) task284 has zero or partial public weight on this Kaggle slice (like task366/324/133/182/018 family), or (b) silent partial failure on hidden inputs offsets the cost gain perfectly. cost **20,895 -> 7,569**; verify 266/266; synthetic-LIVE 569/569; generic adv 19/150 off-contract. | 7,569 | shipped as 3-stack component for 0.00 LB |
2026-05-22 | `task284_scatter_gather_f32_clamped_dynamiccanvas`: aggressive dynamic-canvas variant. Builds the active output canvas at runtime with `ConstantOfShape(active_h, active_w)`, scatters only colored segments once, and pads dynamically back to 30x30. This removes the fixed 23x24 sentinel initializer, the active-base ScatterND, the fixed crop canvas intermediate, and the fixed full-color Pad intermediate from static accounting. This is the least structural and most dynamic tail replacement in the family; treat it as experimental despite strong rectangular-contract parity. | PASS: static local cost **6,227 -> 3,722** (`memory=3,609`, non-scalar params `113`); verify_against_live 266/266; targeted aligned two-marker LIVE 6529/6529 fixed 23x24 and 600/600 random rectangular cases; generic adversarial-against-live 130/150 advisory with candidate stable and 11 live-runtime failures on off-contract random masks. Polish no-op/score-failed due dynamic-shape scorer behavior; no Kaggle submit; no `submission.zip` edit. | 3,722 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_gather_f32_clamped_dynamiccanvas.onnx`, `Logs/task284_scatter_gather_f32_clamped_dynamiccanvas_verify_against_live.txt`, `Logs/task284_scatter_gather_f32_clamped_dynamiccanvas_adversarial_against_live.txt`, `Logs/task284_scatter_gather_f32_clamped_dynamiccanvas_targeted_two_marker_live.txt` |
2026-05-22 | `task284_scatter_gather_f32_clamped_sharedweighted_activemax`: compressed active-rectangle sizing by using the rectangular one-hot input invariant: `ReduceMax(active_cols_full_f32)` gives active height and `ReduceMax(active_rows_full_f32)` gives active width, eliminating the active-row/col Slice -> Greater -> Cast -> ReduceSum chains and six slice initializers. Includes the shared weighted-profile endpoint reuse below. Still a tail replacement; keep the 0.00-LB scatter warning attached. | PASS: static local cost **6,652 -> 6,227** (`memory=5,557`, non-scalar params `670`); verify_against_live 266/266; targeted aligned two-marker LIVE 6529/6529 fixed 23x24 and 600/600 random rectangular cases; generic adversarial-against-live 130/150 advisory with candidate stable and 11 live-runtime failures on off-contract random masks. Polish no-op/score-failed due dynamic-shape scorer behavior; no Kaggle submit; no `submission.zip` edit. | 6,227 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_gather_f32_clamped_sharedweighted_activemax.onnx`, `Logs/task284_scatter_gather_f32_clamped_sharedweighted_activemax_verify_against_live.txt`, `Logs/task284_scatter_gather_f32_clamped_sharedweighted_activemax_adversarial_against_live.txt`, `Logs/task284_scatter_gather_f32_clamped_sharedweighted_activemax_targeted_two_marker_live.txt` |
2026-05-22 | `task284_scatter_gather_f32_clamped_sharedweighted`: micro-compressed the best clamped F32-profile scatter candidate by reusing the already-built `row_weighted` / `col_weighted` compact profiles for endpoint `ReduceMax` instead of rebuilding duplicate weighted profiles. Also pre-clamped the four endpoint color samples once with shared active-grid highs. Still a tail replacement; keep the 0.00-LB scatter warning attached. | PASS: static local cost **6,652 -> 6,558** (`memory=5,882`, non-scalar params `676`); verify_against_live 266/266; generic adversarial-against-live 141/150 with 0 runtime failures. Polish no-op/score-failed due dynamic-shape scorer behavior; no Kaggle submit; no `submission.zip` edit. | 6,558 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_gather_f32_clamped_sharedweighted.onnx`, `Logs/task284_scatter_gather_f32_clamped_sharedweighted_verify_against_live.txt`, `Logs/task284_scatter_gather_f32_clamped_sharedweighted_adversarial_against_live.txt` |
2026-05-22 | `task284_scatter_gather_f32_clamped_profilemax`: follow-up under-10k dynamic scatter variant. Kept the dynamic segment canvas, clamped all endpoint color `GatherND` indices to the active grid so generic off-contract inputs no longer crash, pushed `ReduceMax(color_crop_f32)` before the compact presence compare, and sampled the four endpoint colors directly from the F32 Conv output so the full 23x24 `UINT8` color-index sheet is gone. This is still a tail replacement, so the earlier 0.00-LB scatter warning applies; do not submit without explicit reviewer/owner acceptance. | PASS: cost **7,569 -> 6,652**; verify_against_live 266/266; generic adversarial-against-live 141/150 with 0 runtime failures; targeted aligned two-marker synthetic-LIVE 1332/1332 on 23x24 backgrounds and 600/600 random variable-background cases. Polish no-op; no Kaggle submit; no `submission.zip` edit. | 6,652 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_gather_f32_clamped_profilemax.onnx`, `Logs/task284_scatter_gather_f32_clamped_profilemax_adversarial_against_live.txt`, `Logs/task284_scatter_gather_f32_clamped_profilemax_targeted_two_marker_live.txt`, `Logs/task284_scatter_gather_f32_clamped_profilemax_random_two_marker_live.txt` |
2026-05-22 | `task284_scatter_gather_clamped_profilemax`: intermediate under-10k dynamic scatter variant. Kept the dynamic segment canvas, but clamped endpoint color `GatherND` indices and pushed `ReduceMax(color_idx_u8)` before `Min(...,1)` to avoid the full 23x24 nonbg U8 sheet. Superseded by the F32-scalar gather variant above. | PASS: cost **7,569 -> 7,064**; verify_against_live 266/266; generic adversarial-against-live 141/150 with 0 runtime failures; targeted aligned two-marker synthetic-LIVE 1332/1332 and random variable-background 600/600. No submit/zip edit. | 7,064 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_gather_clamped_profilemax.onnx` |
2026-05-22 | `task284_scatter_momentcolor`: safer color-recovery dynamic scatter branch. Replaced endpoint `GatherND` color sampling with v2-style row/col color moments, which avoids OOB failures on broad generic adversarial grids while staying below 10k. Kept as evidence/fallback because clamped gather now matches its advisory safety at lower cost. | PASS: cost **7,569 -> 9,649**; verify_against_live 266/266; generic adversarial-against-live 141/150 with 0 runtime failures; targeted aligned two-marker synthetic-LIVE 1332/1332 and random variable-background 600/600. No Kaggle submit; no `submission.zip` edit. | 9,649 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_momentcolor.onnx`, `Logs/task284_scatter_momentcolor_adversarial_against_live.txt` |
2026-05-22 | `task284_u8tail_gather_f32profile` / `task284_u8tail_moment_profilemax2_polished`: continued structural v2-family compression after the scatter 0.00-LB evidence. The gather-f32 profile keeps F32 color index compact-profile reductions and U8 final tail; the moment profilemax branch keeps safer moment color recovery. | PASS: gather-f32 cost **20,895 -> 19,763**, verify_against_live 266/266, but generic adversarial 19/150 due GatherND OOB; moment-profilemax polished cost **22,759**, verify_against_live 266/266, adversarial 141/150. These remain structural-ish comparison points; neither was submitted and `submission.zip` was untouched. | 19,763 / 22,759 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_u8tail_gather_f32profile.onnx`, `The_400_Tasks/Task_by_Task/task284/prototypes/task284_u8tail_moment_profilemax2_polished.onnx` |
2026-05-22 | `task284_u8tail_u8coloridx_minprofile_equalmask_gather`: continued structural v2 compression. Kept sparse crop Conv, then replaced moment color recovery with `GatherND` marker sampling, moved the tail to `UINT8` color ids + BOOL full-output Equal, used U8 `Min(color_idx,1)` profiles for marker presence, nested the final color-image `Where`s, and collapsed exact row/col masks from `Less/Not/Greater/Not/And` to `Equal`. | PASS: cost 34,965 -> **20,895**; verify_against_live 266/266; balanced contract synthetic-LIVE 790/790. Generic adversarial-against-live 19/150 is advisory/off-contract: that generator creates many colored cells and can drive GatherND indices out of range, while Task284's stated rule is exactly two aligned singleton markers. No Kaggle submit; no `submission.zip` edit. | 20,895 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_u8tail_u8coloridx_minprofile_equalmask_gather.onnx`, `Logs/task284_u8tail_u8coloridx_equalmask_synthetic_live_report.json`, `Logs/task284_u8tail_u8coloridx_equalmask_adversarial_against_live.txt`
2026-05-22 | `task284_sparsecropconv_moment_equal_bool_pad13`: structural v2 compression only. Replaced one-hot color weighting with dilated sparse crop Conv, moved output to F16 Equal + BOOL Pad-13, compacted row/col center and coordinate recovery, pushed orientation gates into compact masks, and recovered marker colors from row/col color moments. | PASS: cost 118,850 -> **34,965**; verify_against_live 266/266; contract synthetic-LIVE 790/790 (odd/even row+col, distance-2, edges, random); generic adversarial-against-live 148/150 (98.7%, advisory). No Kaggle submit; no `submission.zip` edit. | 34,965 | `The_400_Tasks/Task_by_Task/task284/prototypes/task284_sparsecropconv_moment_equal_bool_pad13.onnx`, `Logs/task284_sparsecropconv_moment_synthetic_live_report.json`, `Logs/task284_sparsecropconv_moment_adversarial_against_live.txt`
2026-05-18 | Rebuilt v2 live-even-parity ONNX from failed v1 hypothesis after finding the hidden/live even-distance branch. | Correctness strong (266/266 expected+LIVE; 5000/5000 broad same-axis LIVE match), but current LIVE task284 is already cheaper (196,021) than v2 (254,402), so this is **no-submit / no-improvement evidence only**. | 254,402 | `Logs/delta_packet_task284_v2_live_even_parity_t_aa8e199f.json`
2026-05-18 | Tried generic optimizer variants on current LIVE (`onnx_opt_with_fold`, `onnx_opt_extended`, `onnxsim`, `onnxoptimizer`). | `onnx_opt_with_fold` and `onnxsim` tied `onnx_opt_safe` at 195,898; `onnx_opt_extended` and selected `onnxoptimizer` passes did not beat current cost. Kept the smaller/safer `onnx_opt_safe` output as the candidate. | best 195,898 | session stdout in kanban run
2026-05-18 | Applied `tools/onnx_opt_safe.py` to the current LIVE task284 ONNX: initializer dedupe + CSE only, preserving broad LIVE behavior. | Candidate `task284_current_live_safeopt` improves current LIVE by 123 cost (196,021 → 195,898); package changes only `task284.onnx`, preserves entry order, and verifies 266/266 expected+LIVE plus 5000/5000 broad same-axis LIVE match. | 195,898 | `Logs/delta_packet_task284_current_live_safeopt_t_aa8e199f.json`
2026-05-18 | Reviewer packet for `task284_current_live_safeopt`: exact and clean vs the worker base, but returned because the gain is only +0.0006276807 expected points (196,021 → 195,898) and is economically dominated by the previously reviewed exact `task284_v3_onehot_pruned_opset10_t_36152161` lane at 157,225 cost (+0.2205438959), which should be repaired/isolation-tested instead. | returned / not queued | 195,898 | `.hermes/board/review_artifacts/task284_current_live_safeopt_t_aa8e199f_review.json`
2026-05-18 | worker-8 (t_d7182876 context) diagnosed v3 bug: `prune_to_output` removed `t284v2_active=ReduceSum(input)` gate, so OneHot(0) for outside-canvas cells → ch0=1 (wrong under official zero-padding). Three fix variants (v4 ReduceMax+Greater+Cast+Mul, v4b ReduceSum+Mul, v4c ReduceSum+Greater+Where) all cost more than live: 201325/196825/197726 respectively. Root cause: OneHot output must be an intermediate for any gate, adding 36000B overhead. | v4b 196825 (+804 vs live) — not usable | `The_400_Tasks/Task_by_Task/task284/prototypes/`
2026-05-18 | worker-8 `task284_v5_f16_t_d7182876_20260518T181232Z_on_5757_42` | Full float16 rewrite of v4b: Cast(input→f16), all initializers→f16, all bool→float Casts upgraded to f16, ReduceSum outside-canvas gate. 266/266 verify_against_live; adversarial 84/84 (100%); cost 139,013 vs live 196,021 (delta -57,008, expected +0.3437 pts). Staged for review at `.hermes/board/staged_submits/task284_v5_f16_t_d7182876_20260518T181232Z_on_5757_42/`. Review-required on card t_d7182876 (card currently todo waiting on parent t_225b209c). | 139,013 | staged
2026-05-18 | Reviewer packet for `task284_v5_f16_t_d7182876_20260518T181232Z_on_5757_42` | Approved for submit_queue promotion after independent reruns against the worker-original valid `task033_where_t_568dd001` package: official convert_to_numpy parity 266/266 vs expected and LIVE, preserved 401-entry packaging with only `task284.onnx` changed there, and local cost drops `196,021 -> 139,013` (`+0.3436543424`). Current-best later advanced on task216 only, so submit-captain should merge just `task284.onnx` onto the latest canonical package before any Kaggle attempt. | ready | 139,013 | `.hermes/board/review_artifacts/task284_v5_f16_t_d7182876_20260518T181232Z_on_5757_42_review.json`


## v3 generator-exact rewrite + 196k-era cost analysis (2026-05-29 / 05-22, superseded by 2593)

### 2026-05-29 | `task284_bracket_v3` — generator-exact correct rewrite (fixes LIVE's hidden bug)

**ARC-GEN-first methodology.** Read `external/ARC-GEN/tasks/task_b7249182.py` (UUID resolved
via task_list line 284). Confirmed distance is ALWAYS odd `2*half-1` → the "even-distance
branch" is dead code. Built generator-exact numpy ref (`prototypes/decode_probe.py:solve_gen`),
then a structural ONNX that draws the single bracket topology with coordinate-grid masks on a
SOUND 24×24 window (single-channel crop only; never crops the 10-channel input), orientation
gate, BOOL per-channel `Equal` one-hot tail, `active`-gated ch0.

Decoded rule per parity bucket: **odd only** (no even bucket exists). Discriminators: **50
exhaustive** param combos where mine==generator and LIVE≠generator (all `half=10, width=24,
xpose=1`; 12 saved to `prototypes/discriminators/even_bucket_discriminators.json`). Numpy ref:
266/266 visible + 1960/1960 exhaustive. ONNX: **266/266 vs expected**, **3000/3000 fresh
generator**, **1960/1960 EXHAUSTIVE** (full param space), **266/266 vs LIVE on visible**
(matches LIVE everywhere visible → no regression). LIVE on the same exhaustive set: 1910/1960
(fails the 50 tall-transposed corners — the 23-row crop truncates row 23).

cost **118,850 → 58,202** (`memory=58,112, params=90`, points 13.31→14.03). Builders:
`build_task284_bracket_v3.py` (best/ship), `_unified` (64,047), `_correct` (67,677, v1). Output
dtype BOOL (decoder thresholds `>0`). **Asymmetric ship**: LIVE forfeits this task's hidden
points (stub-probe ~0 LB); the correct rewrite recovers them. **No Kaggle submit; captain
stages + ships.** Cost note: ~58k is the clean *static* floor (fixed overhead = 10ch→1ch Conv
+ active reduction ~11.8k, 10-channel one-hot output ~5.8k, geometry masks). Sub-30k would need
a dynamic ScatterND canvas (prior `scatter_*` lane reached 3.7k) but that path hit scorer
dynamic-shape failures and was only ever verified against LIVE, not the generator — deferred as
correctness-first. | 58,202 | `prototypes/task284_bracket_v3.onnx`, `prototypes/verify_bracket.py`, `prototypes/exhaustive_test.py`, `prototypes/decode_probe.py` |

### Pre-build novelty statement (2026-05-22)
v1_midpoint_brackets Kaggle-NEGATIVE understood: the even-distance branch is the hidden-sensitive part, so I am not changing the midpoint/parity/bracket construction; any candidate must preserve the v2/live-even graph outputs byte-for-byte and only compress equivalent subgraphs.
v2_live_even_parity status: the current canonical is the cropped/F16 v2-family payload at 118,850 cost, already cheaper than the older v5 F16 staged lane and already carrying the crop-to-active-grid improvement.
Polish status: F16 output already applied, opset 10, first-node crop and final Pad already applied.

My approach: structural compression of v2, not algorithm replacement. First target is replacing the one-hot channel weighting `Slice -> Cast -> Mul(channel_values) -> ReduceSum` and active-cell gate with narrower equivalent color-index/active projections while leaving nodes 6-197 midpoint logic intact.
- Rule 12 candidates identified: `Greater(color_index, half) -> Cast -> ReduceSum(nonbg)` row/col count chains are not direct ReduceMax swaps because counts and marker masks are reused; same-row/same-col already reduce before compare via `ReduceMax(row_count/col_count) -> Greater`; remaining full-grid comparisons mostly feed mask construction and cannot be blindly swapped without changing the v2 branch.
- Rule 11 INT8 opset bump candidates: no valid `BOOL -> Cast -> ReduceMax/ArgMax` chain dominates this graph; the available casts feed `ReduceSum`, and current ORT notes reject UINT8/INT8/BOOL for ReduceSum, so opset bump is secondary unless a new ReduceMax-only subchain is introduced.

Synthetic-LIVE plan: 50+ inputs covering odd and even same-row/same-column marker distances, distance-2 no-emission cases, near-edge markers inside the 23x24 active window, color pairs across 1-9, and padding/outside-window perturbations for advisory reporting.

## COMPRESSED HISTORY

The 2026-05-18/05-22 superseded high-cost lanes (dynamic ScatterND None/0 unscoreable, v2/v5 f16, safeopt) moved to `Task_by_Task/_archive/task284_history.md`. Current floor = **2593** band-Einsum.

## Cost analysis

Current LIVE task284 scores `(memory=195,936, params=85, cost=196,021)`. The safeopt candidate scores `(memory=195,816, params=82, cost=195,898)`, a 123-cost reduction from removing duplicate initializers and four common-subexpression comparison nodes (`safeopt_stats={dedupe:3, cse:4}`). Reviewer re-checks confirmed the current canonical package still carries the same task284 payload as the worker base (SHA `5C5674CA...`), so the safeopt lane is a clean single-entry delta but is economically dominated by the previously reviewed exact `task284_v3_onehot_pruned_opset10_t_36152161` line at cost `157,225` (`+0.2205438959`). The v2 parity rewrite is useful rule evidence but not a current-live cost improvement.

v5 (float16) cost breakdown: memory 138,934 vs live 195,936 = -57,002. The savings come from halving all float intermediates to f16 (each [1,10,30,30] tensor goes from 36000B float32 to 18000B float16). The OneHot output [1,10,30,30] at 18000B and the ReduceSum mask [1,1,30,30] at 1800B, total 19800B overhead for the gate, fully within the float16 saving of 57,002B. No v4b integer routing (v4b cost was 196,825 = +804 vs live because the gate added 39,600B overhead; v5 saves 57,008B vs live by switching to f16).

---
## Superseded LIVE-era attempts (archived 2026-07-02 by bo_ng_cld_worker3, moved out of task284.md for the 20 KB task-md-size lint)

### 2026-05-30 | `task284_bracket_v4..v10` — cost reduction on the v3 parent (zero-downside)

After v3 landed +14.03 LB at ratio 1.000, drove cost down with the **byte-equal-vs-v3 gate**:
since v3 == generator on all 1960 exhaustive param combos, any candidate that is 1960/1960
byte-equal to the generator is provably zero-downside. Every version below holds 1960/1960.

| ver | cost | technique |
|---|---|---|
| v3 | 58,202 | correct rewrite (parent, SHIPPED) |
| v4 | 52,202 | canvas+broadcast: single `Equal([1,1,30,30] U8 canvas, [1,10,1,1] ids)` -> `output` (skipped); padding via sentinel 10 |
| v5 | 45,291 | color image via nested `Where` (disjoint fig sides) + `Where(active,·,10)` sentinel |
| v6 | 39,390 | shifted Conv weight `[1..10]`: `active=(cidx_s>0)` falls out free (drops the separate ReduceSum + its crop) |
| v7 | 32,671 | centers/arm-ends from 1-D marker line-counts (`ReduceSum` perp-axis); `cR=max`, `cL=col_sum-cR` (no full coord products) |
| v8 | 31,323 | `col_center==midc+0.5` (odd dist) => marker-split mask IS the color-split mask; compute once |
| v9 | 27,017 | colors via marker-color line-profiles (only `mc=cidx·nonbg_f` full); `sec=total-pri` |
| **v10** | **25,855** | `out_idx=Where(fig_p,pri,Where(fig,sec,0))` drops `fig_s` + `sec_side` |

cost **118,850(LIVE) -> 58,202(v3) -> 25,855(v10)**; points 13.31 -> 14.03 -> **14.84** (+0.81 over v3).
All single-channel intermediates; output BOOL via broadcast `Equal`. Builder:
`build_task284_bracket_v3.py --tail broadcast`. | 25,855 | `prototypes/task284_bracket_v10.onnx` (+ v4..v9), `prototypes/exhaustive_test.py`
(Superseded by the 2593 band-Einsum ceilfold floor.)

### 2026-05-30 | dynamic ScatterND lane — CORRECT but NOT locally scoreable (DON'T ship)
Adapted the prior `scatter_dynamic` to the sound crop (`build_task284_scatter_correct.py`: Conv
`dilation=[6,6]` -> [1,1,24,24] crop instead of [7,6]->23x24; clamp active dims to crop so the
scorer's dummy input can't OOB `ScatterND`). Result: **1960/1960 byte-equal vs generator** and
runs without crashing on any real input. BUT `score_network_v2` returns **None**: the
`Range`/`Expand` segment construction leaves **153 tensors with dynamic dims** (`unk__*`), which
`calculate_memory` rejects. A *scoreable* (static fixed-length) scatter needs per-cell masking
~= v10's cost (no win). **Decision: dynamic lane is a dead end for shipping.** | None | `prototypes/build_task284_scatter_correct.py`

### 2026-05-20 | PR_HUGE worker (LIVE-era, 196k/5835 canonical)
- Canonical already exposes graph output as float16 — F16 trick already applied; onnxconverter_common rejects re-conversion.
- **crop_to_active_grid** (delta=-20,040 bytes): crop input to 23x24 max active grid; shrink row/col grids; prepend Slice + append Pad (opset-10). Verify 266/266; adversarial 92.2% (worst `add_isolated_cell` 29.5% — off-window cells missed by crop). Superseded by the separable-Einsum family.

---

## 2026-07-15 cleanup compression source for task284.md

Full pre-compression task file archived from `Task_by_Task/task284.md` because repo_lint manual content exceeded 20 KB.

```markdown
# task284

<!-- AUTO:BEGIN -->
<!-- generated by tools/generate_task_mds.py; manual edits OUTSIDE this block -->

**STATUS:** LIVE cost **25,855** (live_costs.json, refreshed 2026-06-05).
(one row: `python tools/agent_helpers/show_task.py --task N` — NEVER Read task_state.csv whole)

**Linked research / prototypes** (auto-detected by filename):
- _none found_
<!-- AUTO:END -->

<!-- HIST:BEGIN -->
<!-- mined by tools/mine_task_history.py; rerun to refresh -->

**Git history** (commits touching files or subjects mentioning this task):
- `1c760ab2` — work: per-task .md + prototype builders/harnesses + prompt archive (session)
- `4e77d6ef` — restructure: migrate Research/ content to natural homes + rename to Research_Archive/
- `0a28ff71` — +36.55 LB batch: task002/017/243/076/101 SHIPPED 5988.64 -> 6025.19
- `ced387ee` — feat(stack): task008+task051+task157 3-stack +5.05 LB â†’ 5908.83 new high
- `3d86f169` — feat(stack): task382+task284+task077 3-stack +3.15 LB â†’ 5903.78 new high
- `0cc50fd6` — feat(task018+task089): +1.97 LB stack, task018 confirmed zero-weight (5/5 family)
- `367c884c` — docs(library): integrate .hermes/cln/ lessons + preserve worker prototypes
- `6a2b99ee` — feat(task175): v12 +2.98 LB shipped (5895.68 -> 5898.66) + library lessons
- `859624fe` — chore: session artifacts â€” task284 reports + task077 alt prototype + task382 v4
- `37a20356` — feat(tasks): task069/task383/task358 +6.16 LB wins + prompts refresh + worker prototypes
- `f8dc8de5` — docs(library): adversarial advisory + BOOL-output Track B + 2026-05-22 wins
- `a81a00b0` — agent_helpers/ (9 scripts) + auto-graph-diag in all 401 task .md files
- `06a9b208` — PR_HUGE worker: 2 additional candidates (task284 crop, task080 dropcast)
- `1911db7d` — OB_HUGE worker: final wrap-up notes for 8 unstaged tasks
- `7418571b` — worker-7 GROUP-O session: task238 v1 restage + crop docs + batch build scripts
- _...and 27 more_

**Session log mentions:**
- `Logs/sessions/2026-05-15_1451_neurogolf-orchestrator-closeout.md` — - `codex-68ea`: clean branch bundle/patch for task071/task085/task284 work.

<!-- HIST:END -->


## ⛔ CORRECTED DIAGNOSIS (2026-05-29) — the "even-distance branch" is a PHANTOM

The ARC-GEN ground-truth generator (`external/ARC-GEN/tasks/task_b7249182.py`) was
read directly. It proves:

- **Marker distance is ALWAYS ODD**: `distance = 2*half - 1` with `half = randint(4,10)`,
  so `distance ∈ {7,9,11,13,15,17,19}`. **There is no even-distance case, ever.** Every
  "even-distance hypothesis" in prior prompts/notes describes dead code that never runs.
- **The rule decode was already CORRECT.** Both `prototypes/task284_numpy_reference.py`
  (midpoint-floor formula) and a fresh generator-exact reimpl score **266/266 visible +
  3000/3000 fresh generator + 1960/1960 EXHAUSTIVE** (every valid param combo). The rule
  was never the problem.
- **LIVE's real bug is the 23×24 crop.** Exhaustive sweep: LIVE fails **50/1960**, *all*
  `half=10 (width=24), xpose=1` — transposed inputs with **24 active rows**. LIVE's
  `t284v2_row_grid` only spans 0..22 and its first-node Slice crops to 23 rows, so it
  literally cannot represent row 23. On the 266 visible pairs LIVE is 266/266 (no tall
  case present), which is why `verify_against_live` never caught it — the classic
  **synth-independence trap** (lessons.md 2026-05-27). A hidden suite that contains ≥1
  tall-transposed instance makes LIVE score 0 on the task → matches the stub-probe's ~0 LB.
- **Fix = draw the (single, unambiguous) shape on a SOUND frame** (≤24 in each axis, or
  full 30×30) — the route that led to the current 2593 separable-Einsum floor.

## Rule (the ONE topology — no parity branch)

Two non-zero singleton markers (distinct colors, both 1-9) are aligned on one row or one
column at distance `2*half-1`. Each marker grows an **arm** toward the midpoint, then a
**5-tall spine**, then **1-cell corner caps** pointing inward at the top & bottom rows.
Each marker's color paints its own side of the figure. (Same-column = the transpose of
same-row.) Concretely, same-row markers `(r, cL)` and `(r, cR)`, `mid = floor((cL+cR)/2)`:
- spine cols `{mid-1, mid+2}` over rows `[r-2, r+2]`;
- cap cols `{mid, mid+1}` at rows `{r-2, r+2}`;
- arm at row `r`, cols `[cL, mid-1] ∪ [mid+2, cR]` (2-cell center gap);
- color: col ≤ mid → left marker's color, else right marker's color.

<details><summary>superseded pre-2026-05-29 "rule" text (kept for history; the even-distance half is wrong)</summary>

Two non-zero singleton markers are always aligned on one row or one column. Current LIVE behavior is a midpoint-cursor family:
- odd marker distances draw two mirrored bracket halves around the midpoint, preserving each marker color;
- even marker distances do **not** draw the perpendicular bracket caps; they emit only same-axis arms ending at `mid - 2` / starting at `mid + 2` (distance 2 emits no colored cells).

</details>

## Algorithm sketch

For same-row markers `(r, c0, color_a)` and `(r, c1, color_b)` with `c0 < c1`:
- `mid = (c0 + c1) / 2`.
- If `c1 - c0` is odd: draw color_a over the left bracket (`row r`, `c0..floor(mid)-1`, vertical side at `floor(mid)-1`, top/bottom caps at `r±2`) and color_b over the mirrored right bracket.
- If `c1 - c0` is even: draw only color_a over `row r, c0..mid-2` and color_b over `row r, mid+2..c1`.
The same transform is transposed for same-column markers.

## Attempts

_One row per attempt. Format: `YYYY-MM-DD | approach | result | cost | session log`._

### 2026-07-12 | `task284_f16coord` — CURRENT BEST 2537 (beats 2593 by 56), `bo_ng_cld_worker2`
While pricing the prompt's CumSum route (dead, below), found an **unexplored micro-lever the prior
"<1000 impossible" floor proof did not cover**: the 2593 graph runs its entire front-end coordinate lane
in **F32** (~32 scalar `[1,1]` wall/where/block/gap tensors @ 4 B). The marker coords are exact small
integers and the only fractional op is `/2` (result integer or half-integer, **exact in F16** for values
≤ 2048). Ran the wall/Where/block/gap arithmetic in **F16** (2 B/scalar), casting the 4 coords in at the
boundary. Kept F32 (forced): the two moment Einsums (F32 input) + `top_r/bottom_r/left_c/right_c` (used by
the color one-hot `Equal` against F32 moments) + the `valid`-count reductions. **Byte-identical**: all 10
band endpoints are integers after the `Where` selectors (the half-integer same-row walls are selected away),
so `Cast(F16_endpoint, U8)` matches the F32 path exactly.

`ng_graph_cost_diff`: **NET −56** (32× `dtype_narrow` F32→F16, −3 F32 consts +3 F16 consts +4 coord casts;
memory −56 reconciles, params +0). Static **2537** = profile **2537**.

Gates: exhaustive **1960/1960** byte-equal vs generator; visible **266/266**; truth **--n 2000 2000/2000**;
ab **--n 2000 2266/2266 byte_equal=True div=0**; `ng_safety` **PASS 0/0**; profile==static.
SHA256 `556fd194324595f9d103007abcaaf28b02f5859cc465c190e9f15563d6741316`.
candidate_class **PATH_A_BYTE_EQUAL** (byte-equal output to the 2593 current-best via mutual exhaustive
equivalence). Staged: `staged_candidates/task284_f16coord_cost2537/`. Builder:
`prototypes/build_task284_f16coord.py`. `x86_required:false`, `kaggle_probe_required:false`, **LOCAL_GOLD**.
| 2537 | −56 | WIN |

**Pass-2 floor re-confirmation (2026-07-12, forced second pass):** re-challenged 2537 with the full
library + BoBrain lever set (constant-operand #119, ConvTranspose stamp #132, CumSum-ordinal #114,
channel-selector collapse #142). Every genuinely-different formulation is documented-dead or derives to a
loss (CumSum +2140 measured, batched-[6,30] +40, 2D/MatMulInteger ≥+1000, predicate ~+5000, U8/I8-Einsum
no-kernel). Full forced-component byte accounting + rank argument: `hard_records/floor_2537_forced_components.md`.
**2537 is the tight floor; WALL below it.**

### 2026-07-12 | F16 CumSum ordinal/prefix (prompt Task284 route) — DEAD, NO-CANDIDATE, `bo_ng_cld_worker2`
`2026-07-11` prompt suggested rebuilding the band-label components via `CumSum` (mark starts,
accumulate ordinals, place by rank; floor_est ~900). **Built + priced the CumSum label lane vs the
current U8-band lane apples-to-apples** (`prototypes/probe_cumsum_vs_band_2026-07-12.py`, exact `calculate_memory` rule,
final [6,30] component counted as the Einsum operand it is; shared front-end coords injected as inputs):
- **U8-band (current 2593 family): 895 B / axis.**
- **CumSum/prefix: 1965 B / axis (+1070, ~2.2×).**
Root cause = the prompt's own KILL RULE: placing +1/−1 deltas at *data-dependent* positions needs two
`Equal(idx, pos)` index-planes `BOOL[5,30]` + F16 casts + `delta[5,30]` + `CumSum step[5,30]` — the
"Range-heavy index matrices to make the labels" the card says to stop on. No cheaper delta primitive
exists (ScatterElements also materializes `[5,30]`). The task is a 2-color **bracket-drawing** problem,
not a run/ordinal problem — the CumSum card is a function-first mismatch. Confirmed **K=6 separable is the
minimal geometry** (single-color bracket = rank-3 {spine⊗caps⊗arm}; ×2 colors = 6; `valid` canvas forced
to reconstruct the active-grid crop from one-hot padding). **Floor stands at 2593.** Hard record:
`hard_records/cumsum_ordinal_prefix.md`. | NO_CANDIDATE | cost > 2593 (CumSum label lane +2140 B total) |

### 2026-06-22 | coordinate-predicate stencil (#108/#130) — FLOOR CONFIRMED, NO-CANDIDATE, worker W17

**The ONE untested break from prompt folder-20 (CoordPredicateBreak) was priced in memory before building.**

The bracket figure has three sub-shapes per color side: ARM (row-indicator ⊗ col-band), SPINE (row-band ⊗ col-indicator), and CAP (2-point row-set ⊗ col-indicator). A per-cell offset predicate `(dr=0 AND 0≤dc_L≤half-2) OR (dc_L=half-2 AND 1≤dr≤2) OR (dc_L=half-1 AND dr=2)` could express color A in closed form.

**Why the predicate route LOSES to 2593:**
The predicate starts as 1D arrays: `dr[1,1,24,1]` (48 B) and `dc_L[1,1,1,24]` (48 B). As soon as any ARM/SPINE/CAP sub-predicate ANDs a row-type mask with a col-type mask, the result BROADCASTS to `[1,1,24,24]` = **576 B** per plane. The surviving-plane taxonomy for both color sides:

| Plane | Shape | dtype | Bytes |
|---|---|---|---|
| ARM_A = `(dr==0) AND (dc_L_in_range)` | [1,1,24,24] | BOOL | 576 |
| SPINE_A = `(dc_L==arm_end) AND (1≤dr≤2)` | [1,1,24,24] | BOOL | 576 |
| CAP_A = `(dc_L==arm_end+1) AND (dr==2)` | [1,1,24,24] | BOOL | 576 |
| COLOR_A = `ARM_A OR SPINE_A` | [1,1,24,24] | BOOL | 576 |
| COLOR_A_full = `COLOR_A OR CAP_A` | [1,1,24,24] | BOOL | 576 |
| ×5 symmetric for color B | [1,1,24,24] | BOOL | 2880 |
| output merge (Where ×2 + add) | [1,1,24,24] | U8 | 1728 |

**Total 2D plane bytes ≈ 7,488 B** (vs 2,085 B for band-Einsum components in 2593). Even at minimum, 10 forced [1,1,24,24] planes = 5,760 B, far above 2485.

**The predicate form that avoids 2D planes IS the free-output separable Einsum** — exactly what 2593 already implements (K=6 rank-1 row⊗col terms). The coordinate-predicate stencil does not reduce K or the component dtype; it only reorganizes the same 6 terms via AND/OR instead of Einsum, forcing 2D intermediates as a penalty. Per #108 WALL: "materializing AND(row-pred, col-pred) costs one full-size plane PER op." **2593 is the hard floor.** No candidate staged. | NO_CANDIDATE | cost > 2593 (predicate route ≈ 7,500+ B memory floor) |

**2026-06-24 worker-08 reconciliation:** `Agenters/Orchestrator/runs/beta_p7t_codex_cli/team_best.csv`, `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`, and `NG_OnnxlyFans_TaskStatus.xlsx` still list task284 at 2634 / sha `1721f6...`, but staged `task284_ceilfold_cost2593` is present, unincorporated, and current (sha `eea9ac...`). Re-ran the effective floor artifact: score `memory=2485, params=108, cost=2593`; profile `--arcgen 266` = 2593; visible `266/266`; exhaustive generator `1960/1960`; truth `200000/200000`; AB `200266/200266` byte-equal. Coordinate-predicate break remains no-candidate for the plane-tax reason above.

**2026-07-02 recapture re-verify (bo_ng_cld_worker3):** 41_special prompt assumed the 2593 ONNX was cleaned — it is intact/staged (`staged_candidates/task284_ceilfold_cost2593/`, sha `eea9ac...`). Live bar `2634`/sha `1721f6...` → 2593 wins **−41 (+0.0157 pts)**. Gates GREEN: visible `266/266`; exhaustive `1960/1960`; **AB `--full` `200266/200266` byte_equal, div=0**; profile `--arcgen 266` = 2593. Upstream prototypes had been cleaned; rebuilt via `build_task284_gapmerge_u8band.py` + `fold_task284_u8band_ceil.py` → **byte-identical** re-derivation (sha `eea9ac...`), a determinism proof. Verdict = **VERIFIED, no re-stage needed**; sub-2593 WALL stands (floor proof above).

### 2026-06-21 | `task284_batchband` — direct [K,30] build, NO-CANDIDATE (2613 > 2593), worker W15

FromScratch crush attempt per prompt (target <1000). Re-measured: LIVE row desynced 25,855; staged
**2593** is the true bar (re-scored, memory=2485). Goal: escape the K=6 band-Einsum (#119) floor.

Tried (all priced/tested, none beats 2593):
- **Direct [K,30] band build** (`build_task284_batchband.py`): replace the 10 individual `[1,30]`
  bands + Concat with one `Sub(idx_u8[1,30], lo_vec[5,1]) -> U8[5,30]` + `Less(.,span_vec[5,1]) ->
  BOOL[5,30]` per axis. **Byte-NEUTRAL on bands** (5×[1,30] == 1×[5,30] = 150 B; scorer charges by
  element count), and the `lo_vec/span_vec[5,1]` add +20 B -> **2613 (loss, +20)**. Confirms band
  data is fundamentally `K×30` elements regardless of slicing.
- **K=5 fold (drop central_gap into signed group_map):** INFEASIBLE. `central_gap_r × central_gap_c`
  is a distinct rank-1 rectangle (the gap is over-subtracted/re-added between the two block terms);
  it is NOT a signed combination of the other 5 row⊗col products. K=6 is the separable-TERM floor
  for this exact figure (arm + spine/caps-rect + gap-correction per side).
- **Avoid the F16 component cast:** REFUTED by ORT probe — `Einsum` has NO U8/I8 kernel
  (NOT_IMPLEMENTED); F16 (2 B) is the cheapest legal Einsum dtype, I32 is 4 B. F16[6,30]×2 = 720 B
  is the hard component floor.
- **Cast-each-then-F16-concat (skip BOOL[6,30]):** WORSE (180 BOOL bands + 360 F16 bands + 360 F16
  concat = 900 vs 720 for single-cast-of-BOOL-concat). The current single-cast is optimal.
- **F32 extent counts -> U8/BOOL:** REFUTED. The extent reduce reads the FREE F32 input and emits
  `[1,30]` F32 = 120 B/axis; any sub-F32 reduce first needs `Cast(input)->U8 [1,10,30,30] = 9000 B`.
  240 B is forced. Output dtype of ReduceSum/Max == input dtype (F32).
- **U8 color canvas / 2D-plane / broadcast-Or routes (prompt routes 1-3):** any `[.,.,30,30]` plane
  is >=900 B charged; a union of ~3 rectangles/side needs multiple OR-combines (>=900 B each) or the
  Einsum's free rank-1 summation. Loses to the free-output Einsum (which materializes NO 2D plane).

**Charged-tensor census of 2593:** component/band bytes = **2084** (F16 720 + BOOL concat 360 + U8
sub 360 + individual bands), scalar/coord front-end = **401** (coords + [1,10] moments, near-minimal).
**FLOOR CONFIRMED ~2485 memory / 2593 cost** for the guaranteed-correct static separable-Einsum
family. <1000 is structurally impossible: F16 components (720) + forced F32 extents (240) + minimal
U8/BOOL builds + scalar coords already exceed 1000, and escaping Einsum forces >=900 B 2D planes that
cost MORE. No candidate staged (a tie/loss adds 0 points). Loss prototype `.onnx` deleted; builder
kept as a documented dead-end. | 2613 | `prototypes/build_task284_batchband.py` |

### 2026-06-20 | `task284_gapmerge_ceilfold` — 2593 (SUPERSEDED 2026-07-12 by 2537 `task284_f16coord`; floor proof below still governs the K=6 core), worker-task284

Visual/generator pass: output size equals input size; two distinct singleton marker colors paint
their own side of the bracket; same-column cases are exactly same-row cases transposed.
ARC-GEN falsifications: (1) no even-distance branch exists because distance is always
`2*half-1`; (2) fixed 23-row/24-col crop formulations are unsound because `half=10,xpose=1`
creates 24-active-row transposed cases.

Micro-fold: starting from `task284_gapmerge_u8band`, rewired every U8-band lower endpoint
`Cast(lo + 0.999, U8)` to `Cast(lo, U8)`. Exhaustive arithmetic over the full 1960 generator
parameter space proved all 10 lower endpoints are integers, so ceil(lo)=lo. This removes ten
scalar F32 Add intermediates (40 B) and the now-unused `c0_999` initializer (1 param):
static `memory=2485`, `params=108`, `cost=2593`.

Gates: exhaustive **1960/1960**; visible **266/266**; truth **200000/200000**; AB full
**200266/200266 byte-equal**; `profile_real_cost --arcgen 266` **2593 == static**. SHA256
`eea9ac4636902baaf47995d0f8842d3a5b042ff267ee131f2451b2abbc104b04`.
Staged: `staged_submits/task284_ceilfold_cost2593/task284.onnx`. No new floor claim; the
2634 forced-component proof still explains the dominant tensors, with the consumed ceil lever
removed. Remaining named marginal lever: share the U8 ramp.

### 2026-06-12 | `task284_gapmerge_u8band` — CURRENT BEST 2634 (beats Aditya 2818 by 184), worker-284

**Source of truth correction:** team-best is **Aditya's `task284_gapmerge` at 2818**
(`K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task284.onnx`, sha
`928afdcd…`), NOT the local v10 (25,855). Aditya's graph is a **deep free-output
separable Einsum** (#119): 2 moment Einsums → scalar coords/walls → **6 row + 6 col
1-D band components** `[valid, firstline(armL), firstblock(spine+capsL rect), secondline(armR),
secondblock(rect R), central_gap(shared subtract)]` assembled by
`Einsum('lb,gc,gk,kr,ks->lcrs', bg_vec, channel_groups[3,10], group_map[3,6], row_comp[6,30],
col_comp[6,30])` (F16, free output). The single graph serves both orientations via
`same_row`-gated Where selectors on the scalar bounds. score_v2 = (0, 2745, 73) = 2818,
profile==static, 1960/1960.

**My win (#148 U8-wraparound band primitive):** Aditya builds each closed band
`lo≤idx≤hi` as `And(GreaterOrEqual(idx,lo), LessOrEqual(idx,hi))` = **3 charged BOOL[1,30]
(90 B)/band**. I replace it with `Less(Sub(idx_u8, ceil(lo)), span)` = **2 charged tensors
(U8[1,30] + BOOL[1,30] = 60 B)/band**: the unsigned Sub wraps `idx<lo` to ≥256 (≥span) so the
lower bound falls out free. For integer idx the closed band equals the half-open U8 interval
`[ceil(lo), floor(hi)]`; `floor(hi)=Cast(hi,U8)` (truncation), `ceil(lo)=Cast(lo+0.999,U8)`.
**Verified byte-exact vs the F32 ge/le band on all 1960 cases × 10 bands** before building.
Saves 30 B × 10 bands = 300 B; net memory 2745→2525 after +U8-ramp/scalar overhead; params
73→109 (idx_u8 ramp 30 + axes/scalars). Opset 14 (U8 Sub). **Net 2818 → 2634 (+184).**

Gates: exhaustive **1960/1960**; visible **266/266**; **truth --full 200000/200000 (100%)**;
**ab --full byte-equal 200266/200266** (divergences=0, == submission.zip which carries Aditya's
2818); score_v2 **2634**; profile_real_cost --arcgen 266 **2634 == static**. sha
`1721f6c7195abb2a3f50f977b903aa93f7c94a16fe39ed0e5212daa377ce6197`.
Staged: `staged_submits/task284_bracketfold_cost2634/task284.onnx`. | 2634 |
`prototypes/build_task284_gapmerge_u8band.py`, `prototypes/build_task284_gapmerge_opt.py` (replica), `prototypes/aditya_ref/task284_aditya_2818.onnx`

**Floor analysis (≈2600 forced, sub-2400 NOT reachable static).** Dominant charged tensors:
row/col_components **F16[6,30]×2 = 720 B** — F16 is the cheapest ORT-Einsum-legal dtype (INT8/INT16
have NO kernel → score None; INT32 is 4 B), so 720 B is the component floor. BOOL[6,30] concat
×2 = 360 B (needed before the F16 Cast). valid_row/col_count **F32[1,30]×2 = 240 B**: the active-grid
extent (height=rand(7,10), width=2·half+rand(1,4)) is NOT recoverable from the markers, so a
genuine `ReduceSum(input)` (F32, 120 B/axis) is forced. Band machinery now at its 60 B/band U8
floor (Sub+Less). **K=6 is the minimum component count** (arm-merge blocked by 2 distinct colors;
gap-merge already saves one vs spine+cap-as-2-terms). The dynamic Range/ScatterND lane that reached
3.7 k is BANNED (leaves dim_param → calculate_memory returns None → 0). **2400 is unreachable in a
guaranteed-correct static graph of this family.** Remaining marginal levers (≤~40 B): fuse per-band
ceil Add for the 4 always-integer-lo bands; share the U8 ramp.

(Superseded LIVE-era attempts — `task284_bracket_v4..v10` (25,855 zero-downside ladder), the
dynamic-ScatterND non-scoreable lane, v3 58k rewrite prose, and the 196k-era cost analysis —
moved to `_archive/task284_history.md`. All dominated by the 2593 band-Einsum ceilfold floor.)


## Don't-do log

**(2026-06-21) Do not "batch the bands" expecting a saving.** Building the `[K,30]` component matrix
as one `Sub`/`Less` (broadcasting `lo_vec[5,1]` against `idx_u8[1,30]`) instead of 10 separate `[1,30]`
bands + Concat is **byte-NEUTRAL** — the scorer charges element count, so `5×[1,30] == 1×[5,30] = 150 B`.
The `lo_vec/span_vec[5,1]` even add +20 B (net 2613, a loss). The 2593 individual-band form is optimal.
**(2026-06-21) Do not try to drop below K=6 separable terms.** `central_gap` is a distinct rank-1
rectangle, not a signed combination of the other 5 row⊗col products. **(2026-06-21) Do not try U8/I8
Einsum** — ORT has NO kernel (NOT_IMPLEMENTED); F16 (2 B) is the cheapest legal Einsum component dtype.
**(2026-06-21) <1000 is structurally impossible for this static family**: F16 components 720 + forced
F32 extents 240 + U8/BOOL builds + scalar coords > 1000; escaping Einsum forces >=900 B 2D planes that
cost MORE. **2593 is the confirmed floor.**
**(2026-06-22) Do not try the coordinate-predicate stencil (#108) as a cheaper alternative to the K=6 Einsum.** AND-ing any row-predicate `[1,1,24,1]` with any col-predicate `[1,1,1,24]` forces a `[1,1,24,24]` BOOL plane = 576 B. The bracket figure needs ≥10 such AND/OR intermediates (5 per color side) = ≥5,760 B in 2D planes alone. The only predicate form that avoids 2D materialization is the free-output separable Einsum — which IS the current 2593 solution. The K=6 decomposition (ARM⊗col-band, spine-band⊗col-ind, cap-2pts⊗col-ind — 3 terms per color side = 6 total) is both the #108 collapse AND the current Einsum. No new structure exists here.
**(2026-05-29) Do not chase an "even-distance branch": it does not exist.** The generator only
emits odd distances (`2*half-1`). Any time you find yourself hypothesizing even-distance rules
(the trap-escape descriptor test), STOP — you are defending a phantom. The rule is one fixed
topology; the only real bug is LIVE's 23×24 crop truncating 24-row transposed inputs.
**(2026-05-29) Do not "verify against LIVE" and call a candidate correct.** LIVE is 266/266 on
visible AND matches every visible pair, yet fails 50/1960 of the true param space. Gate on the
ARC-GEN generator (`solve_gen` + `exhaustive_test.py`), never on LIVE or self-synth.
**(2026-05-29) The shipping candidate is `task284_bracket_v3.onnx` (cost 58,202, 1960/1960
exhaustive).** It matches LIVE on all visible (no regression) and fixes the tall-transposed
hidden failures. v1 `_correct` (67,677) and `_unified` (64,047) are equivalent-correct fallbacks.

Do not resubmit or promote `task284_v1_midpoint_brackets`: it was public-negative despite visible+odd-distance LIVE match because broad even-distance LIVE behavior differs. *(2026-05-29 note: the public-negative was NOT an even-distance issue — even distances never occur. v1_midpoint_brackets most likely diverged from the true odd-shape on a boundary/tall case the way LIVE's crop does; the generator-exact `bracket_v3` supersedes it.)*
Do not promote `task284_v2_live_even_parity` against current-best: it matches LIVE semantics but is costlier than the current LIVE payload.
Do not queue `task284_current_live_safeopt` unchanged: it is exact, but the expected gain is only `+0.0006276807` and the stronger exact v5 f16 lane (139,013 cost) is already staged.
Do not queue `task284_v3_onehot_pruned_opset10_t_36152161` unchanged: the pruned v3 tail dropped the outside-canvas gate and is superseded by `task284_v5_f16_t_d7182876_20260518T181232Z_on_5757_42`, which restores that gate while cutting cost further to `139,013`.
Do not try v4b (float32 ReduceSum gate) without float16: minimum cost overhead is 39,600B (36000B OneHot intermediate + 3600B ReduceSum mask) which pushes past live cost of 196,021.
Do not skip the outside-canvas ReduceSum gate: without it, OneHot(index=0)=ch0=1 for zero-padded outside-canvas cells, failing 0/266 under official convert_to_numpy encoding.

<!-- SPREAD:BEGIN -->
<!-- generated by tools/spread_annotate_notes.py from 20260518 scan; manual edits OUTSIDE this block -->

## Spread-knowledge scan (20260518)

(LIVE-era 196021-cost scan — fully superseded by the 2593 separable-Einsum family; archival
detail in `_archive/task284_history.md`. Re-run to regenerate: `tools/spread_scan.py` then
`tools/spread_annotate_notes.py`.)
<!-- SPREAD:END -->

(2026-05-20 PR_HUGE LIVE-era notes — F16-already-applied + `crop_to_active_grid` −20,040 B, adversarial 92.2% — moved to `_archive/task284_history.md`.)

<!-- BEGIN AUTO-GRAPH-DIAG -->

## Graph Diagnostic (auto-generated)

**Cost**: 25,855 | **Memory**: 25,762 | **Params**: 93 | **Points**: 14.8397
**Nodes**: 131 | **Initializers**: 19 | **Opset**: 13 | **Graph output dtype**: BOOL
**Max active grid (from `task284.json`)**: 23×24

### Quick-look flags

- has_Conv: ✓
- has_MatMul: ✗
- has_ReduceSum: ✓
- has_ReduceMax: ✓
- has_Where: ✓
- has_Gather: ✗
- has_OneHot: ✗
- first_node: `Conv`
- last_node: `Equal`
- ends_with_Cast(F16→F32)→output: ✗
- first_node_is_Slice: ✗

### Op-type histogram

| op | count |
|---|---:|
| `Sub` | 18 |
| `And` | 17 |
| `Equal` | 15 |
| `Or` | 14 |
| `Add` | 13 |
| `Mul` | 11 |
| `ReduceSum` | 10 |
| `Greater` | 9 |
| `Less` | 8 |
| `Cast` | 6 |
| `ReduceMax` | 3 |
| `Where` | 3 |
| `Conv` | 1 |
| `Slice` | 1 |
| `Not` | 1 |
| `Pad` | 1 |

### Initializers (top 10 by params)

| name | dtype | shape | params |
|---|---|---|---:|
| `rowgrid` | F16 | [1, 1, 24, 1] | 24 |
| `colgrid` | F16 | [1, 1, 1, 24] | 24 |
| `channel_w` | F32 | [1, 10, 1, 1] | 10 |
| `chan_ids` | UINT8 | [1, 10, 1, 1] | 10 |
| `cv_pads` | INT64 | [8] | 8 |
| `cs` | INT64 | [2] | 2 |
| `ce` | INT64 | [2] | 2 |
| `ax_2_3` | INT64 | [2] | 2 |
| `p5_f32` | F32 | [] | 1 |
| `c0_5` | F16 | [] | 1 |
| ... (9 more) | | | |

*Technique routing: `Onnx_Advanced/by_symptom.md` (+ INDEX verdict list). Flags above feed it.*

*Node list (131 nodes) omitted — the .onnx is canonical; run `tools/agent_helpers/inspect_graph.py task284` for the full sequence.*

<!-- END AUTO-GRAPH-DIAG -->
```
