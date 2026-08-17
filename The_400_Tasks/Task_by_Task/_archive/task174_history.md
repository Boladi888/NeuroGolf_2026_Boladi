# task174 — archived history (moved 2026-06-22 by Worker-8 to keep task174.md under the size cap)

Superseded attempt logs and the pre-2026-06 cost/don't-do prose. Current best + active floor proof + don't-do log live in `task174.md`.

_2026-05-19 status below is superseded by the 2026-05-23 staged rewrite._

### Top-3 TopK direct two-pair LR rewrite (2026-05-23) — **SHIPPED Kaggle +2.50 LB CLEAN**

Follow-up to the 6,023 direct-LR candidate. Replaced the repeated present-color `ArgMax`/mask-out ladder with a single allowed `TopK(score, k=3)` over non-background color presence. The LR selector and winner crop are otherwise the same direct two-pair rule as the 6,023 variant.

- Candidate: `task174_top3_topk_direct_lr.onnx`
- Builder: `task174/prototypes/build_task174_top3_topk_direct_lr.py`
- Staged: `.hermes/board/staged_submits/task174_top3_topk_direct_lr_t_99895f41_on_70716_cost5763/`
- Cost: **70,716 -> 5,763** (`memory=5,679`, `params=84`), delta **-64,953 bytes** / +2.5072 raw points.
- Verification: **266/266 proto-vs-LIVE** and **266/266 proto-vs-expected** against repo `submissions/submission.zip`.
- Advisory adversarial: generic random-fill suite **3802/6650 (57.2%)**, same as the 6,023 direct-LR model under the same 5-color off-contract rerun. This is below the older task-specific perturbation log because the generic suite often violates the observed exactly-3-color generator invariant.
- Risk note: currently the cheapest verified Task174 candidate. Same semantic risk as 6,023, plus reliance on `TopK` support; local scorer accepts it.
- Under-3k search note: true under-3k was only reachable by dropping the selector, which failed visible verification. `maxcolor_crop` cost **2,952** but verified **91/266**; `mincolor_crop` cost **2,952** but verified **99/266**. A solid-rectangle LR output variant cost **5,192** and verified **259/266**. The exact current architecture already spends 1,200 bytes on three candidate F32 masks, 400 bytes on the winner mask, and 900 bytes on the padded label plane before LR pair checks, bbox/crop shifts, and index plumbing, so sub-3k exact looks blocked without a fundamentally different output/selection primitive.
- **2026-05-23 SHIPPED Kaggle** as single-task wildcard on 5927.56 → **5930.06 (+2.50 LB clean)**. The 57.2% generic-5-color random-fill adversarial was advisory only — hidden test slice respects the exactly-3-color generator (266/266 visible). Confirms TopK presence scores + LR symmetry + crop winner is a safe algorithm rewrite when generator invariant is observed across all visible+arc-gen.

### Top-3 select-LR / crop-winner rewrite (2026-05-23) — STAGED SAFER VARIANT

Heartbeat follow-up improved the staged top-3 rewrite by separating winner selection from final crop. The graph now finds the three present non-background colors, tests LR symmetry with column-only comparisons for each candidate, then dynamically slices/crops only the winning color once for output.

- Candidate: `task174_top3_select_lr.onnx`
- Builder: `task174/prototypes/build_task174_top3_select_lr.py`
- Staged: `.hermes/board/staged_submits/task174_top3_select_lr_t_f4249c5e_on_70716_cost7421/`
- Cost: **70,716 → 7,421** (`memory=7,311`, `params=110`), delta **-63,295 bytes** / +2.2544 raw points.
- Verification: **266/266 proto-vs-LIVE** and **266/266 proto-vs-expected** against repo `submissions/submission.zip`.
- Advisory adversarial: **6313/6650 (94.9%)**, same as the previous top-3 candidate. Color swap/permutation/flip are 100%; failures are mostly off-contract perturbations that add/remove colors.
- Risk note: still relies on the observed generator invariant that all 266 pairs have exactly three non-background colors.
- Supersedes: `.hermes/board/staged_submits/task174_top3_dynamic_slice_lr_t_781b1688_on_70716_cost8717/`.

### Top-3 direct two-pair LR rewrite (2026-05-23) — STAGED ASSUMPTION VARIANT

At user request, staged a more generator-assumption-heavy variant alongside the 7,421 candidate. It keeps the same top-3 present-color assumption, but replaces the width-specific LR selector with direct outer/inner mirrored column-pair checks and drops the explicit `width <= 5` guard because no observed nonwinner passes the direct two-pair test.

- Candidate: `task174_top3_direct_lr.onnx`
- Builder: `task174/prototypes/build_task174_top3_direct_lr.py`
- Staged: `.hermes/board/staged_submits/task174_top3_direct2_lr_t_112496de_on_70716_cost6023/`
- Cost: **70,716 → 6,023** (`memory=5,925`, `params=98`), delta **-64,693 bytes** / +2.4631 raw points.
- Verification: **266/266 proto-vs-LIVE** and **266/266 proto-vs-expected** against repo `submissions/submission.zip`.
- Advisory adversarial: **6455/6650 (97.1%)**. Worst remains `add_isolated_cell` at 77.6%; color swap/permutation/flip are 100%.
- Risk note: cheaper and still rule-shaped, but more assumption-heavy than 7,421 because only two mirrored column pairs are tested. Keep both staged.

### Top-3 dynamic-slice LR rewrite (2026-05-23) — SUPERSEDED

Built a radical from-scratch graph that first finds the three present non-background colors, dynamically slices only those three 10×10 color masks, runs the bbox-crop LR-symmetry test on those candidates, then emits a UINT8 label plane with BOOL one-hot output.

- Candidate: `task174_top3_lr.onnx`
- Builder: `task174/prototypes/build_task174_top3_lr.py`
- Staged: `.hermes/board/staged_submits/task174_top3_dynamic_slice_lr_t_781b1688_on_70716_cost8717/`
- Cost: **70,716 → 8,717** (`memory=8,607`, `params=110`), delta **-61,999 bytes** / +2.0934 raw points.
- Verification: **266/266 proto-vs-LIVE** and **266/266 proto-vs-expected** against both default live zip and repo `submissions/submission.zip`.
- Advisory adversarial: **6313/6650 (94.9%)**. Color swap/permutation/flip are 100%; failures are mostly off-contract perturbations that add/remove colors (`add_isolated_cell`, `empty_quadrant`).
- Risk note: relies on the observed generator invariant that all 266 pairs have exactly three non-background colors. A robust 9-color version also verified but costs **21,224**.
- Rejected squeeze: replacing crop/bbox stacks with `Where(bool, bool, bool)` would save ~400B but ORT does not implement that op form; kept the valid stack+Gather selection.

### From-scratch rebuild (2026-05-19)

Built a clean re-implementation matching the live algorithm exactly (opset 10, 829 nodes). Verified **266/266** against live model. Cost = 74,587 (memory=74,515, params=72) vs baseline 70,971 — **+3,616 bytes regression**. Not staged. The extra cost comes from one extra Concat intermediate ([1,9,10,10] = 3,600 bytes) and two unused nodes (winner_idx ArgMax/Squeeze). Even with those removed (~826 nodes), estimated cost ≈ baseline with no net savings. From-scratch is a dead end for this task.

## Cost analysis

- Current best (2026-06-12): 4,786 (`memory=4,705`, `params=81`) via presence-slice param-dissolve of the polished boolprof model; real-input profile matches static over all 266 official inputs.
- The 2026-06-12 change saves 8 param elements by replacing 9-element `nonbg_channels_i32` Gather with 1-element `ten_i64`+`sel1_i64` Slice for the TopK score channel selection.
- The 2026-06-10 candidate deletes the fourth 400B F32 winner slice by reusing candidate BOOL masks, halves the six LR mismatch cast planes to F16, and halves most scalar/vector index plumbing from INT64 to INT32.
- Baseline: 70971 (memory=70899, params=72)
- params=72 elements (tiny, mostly small int64/float32 scalars). No compress opportunity.
- memory=70899 dominated by 826-node parallel computation on 9 color channels.
- 9×88-node pipelines each create many [1,1,10,10] and [1,1,19,19] intermediates (Pad adds 9 cells to last 2 dims: [1,1,10,10]→[1,1,19,19]).
- `crop_to_size_advisory` claims -32000B savings but is a FALSE POSITIVE: model already slices to 10×10 on node 0; the one [...,30,30] "intermediate" is the graph output, which the profiler correctly excludes from peak memory.

## Don't-do log

- **F32 slice dissolve blocked (2026-06-12)**: The three 400B F32 dynamic slices (`k0/k1/k2_mask`) cannot be dissolved. Any replacement (Gather on cast-to-U8 input, sequential ReduceMax split, GatherND) either materializes a larger intermediate or adds 100B+ BOOL casts per pair column. The F32 input-read tax is structural for dynamically-chosen channels from an F32 input.
- **mask_stack Where-based replacement marginal (2026-06-12)**: Replacing `mask_stack BOOL [3,1,10,10]` (300B) + `win_mask_b` (100B) with two chained `Where(U8, U8)` operations saves 300B (mask_stack) but requires `tmp_mask_u8` intermediate (100B) and ORT rejects `Where(BOOL, BOOL, BOOL)` — so needs explicit Cast or U8 masks throughout. Net at best 200B, but the LR pair Xor requires BOOL, making U8 refactor add 120B+ pair casts. Not worth pursuing.
- **sub-3k selector drops (2026-06-10)**: `maxcolor_crop` and `mincolor_crop` are cheap (2,952) but fail visible badly (91/266 and 99/266). They are not candidates.
- **solid-rectangle LR output (2026-06-10)**: costs 5,192 but fails visible 259/266; do not treat it as a near-miss candidate without a real repair.
- **naive mask U8 cast (2026-06-10)**: casting the F32 Slice masks to U8/BOOL downstream does not remove the charged F32 Slice tensor. Savings came from using BOOL reducers/reuse, not from adding a post-Slice U8 carrier.
- **crop_to_size_advisory (2026-05-19)**: Advisory misleads — the only [30,30] tensor is the graph output. Model already crops to 10×10 on node 0. Pre-crop would add overhead.
- **fp16 conversion (2026-05-19)**: Not attempted — model structure is opset 10 with 127 Cast(bool→float32) ops and tiny initializers. fp16 would need verification and risk type mismatches similar to task377. Params=72 elements (no byte savings from dtype change since score counts elements).
- **slice_axis_collapse**: Saves 3 bytes. Too small to stage (risk-return ratio too poor).
- **from-scratch rebuild (2026-05-19)**: Built 829-node opset-10 model matching live algorithm exactly; verified 266/266. Cost=74,587 vs baseline 70,971 = +3,616 regression. Extra [1,9,10,10] Concat intermediate costs 3,600 bytes. Even with that fixed, model would only tie baseline — no benefit.

<!-- SPREAD:BEGIN -->
<!-- generated by tools/spread_annotate_notes.py from 20260518 scan; manual edits OUTSIDE this block -->

## Spread-knowledge scan (20260518)

Current canonical cost: **70971** (memory=70899, params=72, points≈13.829973).

Pattern sweep verdicts (each pattern measured INDEPENDENTLY vs the original — patterns overlap, so the sum overstates the real cumulative savings):

| Pattern | Verdict | Δ cost (vs original) | Notes |
|---|---|---:|---|
| `clip_mul_to_min` | — | — | no Clip[0,1] + Mul(mask) pairs detected |
| `const_to_init` | — | — | no Constant nodes with a 'value' attribute to convert |
| `crop_to_size_advisory` | ? | -32000 | max active grid is 10×10 but graph has 1 [...,30,30] intermediate(s). Crop-pr... |
| `fp16_selective` | ✗ verify | — | fp16-converted; driver must verify cost actually drops and output is within t... |
| `onnxsim` | — | — | onnxsim made no changes |
| `safe_opt` | — | — | no changes from safe-opt + extended passes |
| `scalar_broadcast_ones` | — | — | no uniform-valued float initializers consumed only by broadcasting ops |
| `slice_axis_collapse` | ✓ SAVES | -3 | collapsed axis-0 no-op from 1 Slice node(s) |

**Free LB available (sequential pipeline truth)**: running `tools/polish_candidate.py` on the current ONNX applies `slice_axis_collapse` and saves **3 bytes** (byte-exact verified on 3 inputs). Note: the per-pattern Δ above was measured against the ORIGINAL, so the per-pattern numbers do not sum to this — patterns subsume each other when applied sequentially.

**Advisory** (crop_to_size_advisory): max active grid is 10×10 but graph has 1 [...,30,30] intermediate(s). Crop-process-pad could save ~32000 bytes (rough est). — estimated savings if applied manually: ~32000 bytes.

_Re-run: `.venv/Scripts/python.exe tools/spread_scan.py` (updates JSON in `Logs/spread_scan_results/`); then `.venv/Scripts/python.exe tools/spread_annotate_notes.py`.
<!-- SPREAD:END -->

### 2026-05-20 � PR_HUGE worker (f16 attempted, no winning candidate)
- Tried: f16 conversion (robust onnxconverter_common + direct manual converter, both with and without opset bump)
- Result: NO WIN � either f16 conversion produced larger cost (Cast wrapper overhead > savings), broke output type, or failed ORT load
- Future workers: try crop-to-active-grid (some have <30 max grid), bool_andfold, or full rebuild from scratch with smaller algorithm

