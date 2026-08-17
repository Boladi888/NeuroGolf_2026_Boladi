# task324 — archived history (moved 2026-06-19 by worker-324 per WORKER_PROTOCOL staging rule)

Full prose for superseded rungs (compress_v3 18,545 / compress_v1 23,491 / v5b SHIPPED 66,092) and the legacy Algorithm sketch. The one-line attempt rows remain in task324.md. Current best + diagonal-floor proof live in task324.md.

## Algorithm sketch

**Structural rewrite (2026-05-24, shipped for submit-captain):**

1. `ReduceSum(input, axes=[0,2,3])` → color histogram [10]. `ArgMax` → bg0 (most frequent), suppress + `ArgMax` again → bg1 (second most).
2. Gather bg0/bg1 channels from input: `bg0_ch [1,1,30,30]`, `bg1_ch [1,1,30,30]`.
3. **3-way majority vote** for which background class each cell belongs to:
   - Vote 1: 4-neighbor diff-conv of `(bg0_ch - bg1_ch)` via cardinal kernel Conv; ≥ 0 → bg0.
   - Vote 2: Plaid formula — strict-bg0 rows (have bg0, no bg1) OR (not strict-bg1 AND col-bg0-dominant).
   - Vote 3: Row majority — `ReduceSum(bg0_ch, axis=3) > ReduceSum(bg1_ch, axis=3)`.
   - 2-of-3 majority via `Or(And, And, And)`.
4. Marker detection: cells not in bg0_ch and not in bg1_ch.
5. **Diagonal spread** via precomputed sentinel tables D1_MEM [59,30] and D2_MEM [59,30] (sentinel=900): for each of the 59 possible NW-SE diagonals (k = r−c) and 59 NE-SW diagonals (s = r+c), the table lists the flat indices of cells on that diagonal. `Gather → ReduceMax` propagates marker presence to all diagonal members.
6. Color selection: marker cells on bg0-class majority get mc_bg0, bg1-class get mc_bg1. Colors are derived via `ArgMax` of marker+majority mask summed over the flat dimension.
7. **eye_10 Gather** (not OneHot) converts scalar color index to one-hot [10] vector (avoids OneHot intermediate op).
8. `Where(bg1_on_diag, mc_bg1_4d, input)` then `Where(bg0_on_diag, mc_bg0_4d, prev)` → output. Input is all-zeros for padding cells, so they naturally produce all-zero output.

**Local/private (legacy, no-submit):** GatherND 7-cell F16 hash, sparse packed stamps, sorted 25-tier prefix tables. Cost 44,417.

## Attempts

### 2026-05-31 | compress_v3 — EXACT corner-based rewrite → cost 66,092 → 18,545, MORE correct than v5b (RECOMMENDED, captain ships)

**Outcome**: a structural rewrite that DOMINATES v5b — **cost 18,545** (memory 16,968 + params 1,577) AND **exact** (100% vs ARC-GEN ground truth, vs v5b's ~98.5%). Score 13.90 → **15.17 pts**; predicted **+1.27 LB** (`ln(66092/18545)`). Builder `task324/prototypes/build_task324_compress_v3.py`. **NOT submitted — captain ships.**

**The breakthrough — a novel exact base-vs-stripe disambiguation** (found after the byte-equal v1; replaces v5b's whole 3-way vote): the task reduces to ONE global decision — which of {bg0, bg1} is the BASE color B (the other is stripe S). Generator fact: bg1 stripes are full rows `brows` / full cols `bcols`, both starting at `randint(2,7) ≥ 2` → **rows 0-1 and cols 0-1 are never stripes → the top-left 2×2 corner is ALWAYS base.** Markers (≤3, non-bg) can't fill all 4 corner cells, so **B = the bg color present in the 2×2 corner**. Then `on_base(r,c) = (countB_row[r]>0) AND (countB_col[c]>0)` is EXACT; recolor: diagonal cell colored B → mc_base, colored S → mc_stripe.

**Why it beats v5b**: v5b's 3-way majority vote (4nb-Conv + plaid + row, 2-of-3) only APPROXIMATES the bg-class and mis-buckets markers on ~1.5% of configs. The corner pins the base exactly, so v3 is exact. Drops the entire vote machinery (4nb Conv, bg1_ch, plaid sub-vote, row vote, 2-of-3) — ~5k cheaper than byte-equal v1 — and adds only the tiny corner detection.

**Failed precursors (informative)**: naive `(bg0_row>0)AND(bg0_col>0)` assuming bg0=base → **88/266** (bg0=most-common is the STRIPE color when stripes dense); v5b's plaid sub-vote alone → **250/266**. The corner disambiguation is what makes the simple formula exact.

**Gates**: `verify_against_live` **266/266 vs expected AND vs LIVE**. v3-vs-ground-truth: numpy 266/266 + 600/600; **ONNX 266/266 + 1500/1500 = 100.0%** vs **v5b 1477/1500 (98.47%)**; v3 diverges from v5b ONLY on the 23 cases v5b gets WRONG.

**Path-A vs Path-B**: v3 is NOT byte-equal to v5b (it is MORE correct) → a structural rewrite, not a byte-equal compression. Strictly safer than v5b (exact on the hidden ground-truth distribution) and cheaper. **compress_v1 (23,491, byte-equal) is the conservative fallback** for guaranteed identical-to-v5b behavior.

### 2026-05-31 | compress_v1 — single-channel index canvas (Path A byte-equal vs v5b) → cost 66,092 → 23,491 (CANDIDATE, captain ships)

**Outcome**: byte-equal Path-A compression of the shipped v5b. **cost 66,092 → 23,491** (memory 62,912→21,912, params 3,180→1,579). Score 13.90 → **14.94 pts**; predicted **+1.03 LB** (`ln(66092/23491)=+1.034`). Builder `task324/prototypes/build_task324_compress_v1.py` → `task324_compress_v1.onnx`. **NOT submitted — captain ships.**

**The lever**: v5b carried the whole pipeline in 10-channel F16. FOUR `[1,10,20,20]` tensors (`inp_f32` 16k + `inp`/`temp_out`/`cropped_out` 8k each = **40,000 B = 64% of v5b memory**). compress_v1 collapses to a **single-channel color-index canvas** and only reconstructs the one-hot at the very end.

**Method (exact v5b-logic mirror, only the representation changes)**:
1. **Canvas = color+1** via a 1×1 `Conv` (`weight[0,c,0,0]=c+1`): collapses the one-hot to `[1,1,30,30]` F32 (3,600 B) — cheaper than `ArgMax` INT64 (7,200). `color+1` makes padding (0) naturally distinct from real color-0, so **no separate active mask is needed for the bg comparisons**. Cast→uint8 @30 (900) → Slice→uint8 `[1,1,20,20]` (400) → one F16 copy (`canvas_h`, 800) only for the mc `ReduceMax`.
2. **bg detection identical to v5b**: `ReduceSum(input)` histogram (reads `input`, free) → `ArgMax` → suppress → `ArgMax`. bg0_idx/bg1_idx byte-identical.
3. **bg masks via `Equal(canvas_u8, bg_idx+1)`** == v5b's `Gather(one_hot, idx)>0` exactly (padding→False either way). The F16 0/1 maps the votes need = `Cast(Equal, F16)`.
4. **3-way majority vote, marker detection, mc — identical ops/values to v5b** (required: v5b's vote genuinely misclassifies markers on ~2% of out-of-distribution configs; byte-equality demands reproducing that, incl. the **mc=0 → color-0** degenerate, see below).
5. **ONE X-kernel** `[1,1,39,39]` (1.0 on both diagonals) replaces v5b's two diag kernels (3,042→**1,521 params**). Conv is linear & non-negative ⇒ `(Conv(m,d1)>0 OR Conv(m,d2)>0) == (Conv(m,d1+d2)>0)` — proven and A/B-verified byte-equal.
6. **Scalar `Where` recolor** on the uint8 canvas (no `eye_10`, no `[1,10,1,1]` one-hots); mc via `Where(marker,canvas,0)→ReduceMax` (skips the Cast+Mul). 4-neighbor vote uses `Greater(diff_conv, -0.5)` (== v5b's `Not(Less(.,0))` since diff_conv ∈ exact ints {-4..4}), dropping one bool tensor.
7. **Equal-broadcast tail**: `Pad(out_u8)→[1,1,30,30]` then `Equal(., [1..10])` → **bool** `output` (bool passes `np.array_equal` vs F16 live; `output` is excluded from memory regardless of dtype).

**Critical trap caught (mc=0 clamp)**: in +1 space, a no-marker class gives `ReduceMax=0`; v5b's `Gather(eye_10, 0)` paints **color 0** there (not blank). Without a `Max(.,1)` clamp, compress_v1 painted blank → **37/2000 arcgen + 844/2000 adversarial divergences**. The vote misclassifies markers often enough that this fires on *valid* inputs, not just degenerate ones. Fix: `mc_p1 = Max(ReduceMax(...), 1)` → no-marker→value 1→color 0, matching v5b. (Generator legality `len(seen)==2` guarantees a marker per class on truly-valid inputs, but the vote can still bucket both into one class.)

**Gates**: `verify_against_live --tid 324 --live-zip submissions/submission.zip` → **266/266 vs LIVE and vs expected**. Direct A/B vs v5b (`ab_gate_v1_vs_v5b.py`): **0 divergences on 2000 ARC-GEN valid + 2000 adversarial striped + 1000 fully-random**. Synth gate **823/1500 (54.9%) — identical to v5b**. v1-vs-ground-truth 97.95% = v5b's own rate (inherited byte-for-byte). uint8 op support probed: `Equal/Greater/Where/Pad/Cast` OK; `Mul/ReduceMax/Max/Sub` reject uint8 (→ mc & 4nb stay F16).

**Why ~23.5k not the prompt's ~3,000 floor**: the prompt described task324 as "same-position recolor" — it is actually a **two-bg plaid + diagonal-spread** (ARC `d07ae81c`, grids up to 20×20, not ≤19×12). The diagonal X-kernel (1,521 params), the forced F32 conv-collapse (3,600 B — input is F32; ArgMax INT64 would be 7,200), and the full 3-way vote machinery (~9.5k of required bool masks, byte-locked to v5b) dominate the structural floor (~22.3k). The 10-channel→1-channel collapse is the real, achievable win (62,912→21,912 memory). Pushed to ~1.05× floor; further gains need a different parent (a cleaner non-vote algorithm would break byte-equality).

### 🎯 2026-05-25T17:22Z | SHIPPED v5b structural (no-Einsum) → Kaggle +13.90 LB UNLOCK

**Outcome**: 6136.41 → 6150.31 = **+13.90 LB exact match for v5b cost-points** (25 - log_e(66,092) = 13.901). 11th solve-from-scratch unlock this session.

**Why this worked**: same pattern as task133 — canonical task324 was a 22-node lookup table at cost 160,289 (cherry13b zero-weight confirmed, scored ~0 on hidden). v5b is a structural rewrite at cost 66,092 — fully generalizes hidden, contributes full cost-points from a previous-zero baseline.

**Pre-ship gates**:
- 266/266 visible ✓
- v4-vs-v5b intra-version A/B: **0/1600 on (1000 row-striped + 100 col-equal-count + 500 col-unequal-count) adversarial inputs** ✓
- canonical-vs-v5b A/B: 995/1000 differ (EXPECTED — different algorithms; rule from traps.md §27d covers this case)

**Critical methodology contribution: caught v5 Einsum bug before ship**:
- v5 (with Einsum mc-detection): cost 60,528, looked attractive
- v4-vs-v5 A/B: 75/100 col-equal-count diverge, 179/500 col-unequal-count diverge → BLOCKED
- Root cause: Einsum(bchw,bxhw->bc) does count-ArgMax; ReduceMax(color_index * mask) does index-ArgMax — different tiebreak semantics. See traps.md §27b.
- v5b kept the Slice-then-Cast + plaid + majority simplifications but REVERTED the Einsum: cost 66,092 (5.5k more than v5, but ship-safe)
- Net session arc: v3 78,853 → v4 68,532 → v5b 66,092 → SHIPPED. v5 60,528 staged as research-only.

**Methodology takeaway**: for cherry13b zero-weight canonicals, intra-version A/B (v(N-1) vs v(N)) with adversarial column-striped inputs is the critical gate. The standard canonical-zip A/B does NOT apply (different algorithms). The 1600-input adversarial set (row + col-equal + col-unequal) caught a bug that 267/267 visible verify would never have seen.


---

## Superseded Attempts rows (moved from task324.md 2026-06-22 worker-3, size-budget compression)

Full prose for the pre-`u8mc_v1` rungs. The live task324.md keeps these as one-line `version | cost | verdict` rows; this archive holds the detail.

- 2026-05-22 | `cherry13b` Identity diagnostic on 5874.78 base | Kaggle stayed 5874.78 (delta +0.00) => zero public weight; STOP public-LB radical rewrite | cost 0 | staged `.hermes/board/.../task324_cherry13b_identity_on_5874_78/`
- 2026-05-22 | local/private sparse-stamp rewrite (GatherND 7-cell F16 hash, sorted 25-tier prefix, scalar clamp) | 266/266 vs expected+LIVE; no-submit (zero public weight) | 44,417 (mem 33,038 + params 11,379) | builder build_task324_sorted_tiered_lookup.py
- 2026-05-24 | structural rewrite (30×30): ArgMax bg, 3-way majority vote, D1_MEM/D2_MEM [59,30] sentinel tables, eye_10 Gather one-hot, input-as-Where-false | 266/266; synth 823/1500 | 164,514 (mem 159,042 + params 5,472) | build_task324_structural_diagonal.py
- 2026-05-24 | crop-to-20×20 v1 (Slice input, process 20×20, Pad back) | 266/266 | 107,141 | superseded by v2
- 2026-05-24 | crop-to-20×20 v2 (Reshape-before-Cast, dedupe) | 266/266; synth 54.9%; staged | 101,937 (mem 99,432 + params 2,505) | build_task324_structural_diagonal_crop20.py
- 2026-05-25 | structural v3 (two [1,1,39,39] f16 Conv diag kernels replace D1/D2 tables; f16 mc chain) | 266/266 | 78,853 (mem 75,672 + params 3,181) | build_task324_v3_convdiag_f16mc.py
- 2026-05-25 | v4 (full F16 pipeline, Cast-input-first, F16 output) | 266/266 | 68,532 (mem 65,352 + params 3,180) | build_task324_v4_f16pipeline.py
- 2026-05-25 | v5 (Einsum mc) FAILED A/B GATE — Einsum=count-ArgMax vs V4 max-index diverge under misclassification (179/500 col-unequal); BLOCKED | 60,528 | build_task324_v5_einsum_mc.py
- 2026-05-25 | v5b (Slice-then-Cast + plaid/majority simplifications) SHIPPED +13.90 LB; 0/1600 A/B vs V4 | 66,092 (mem 62,912 + params 3,180) | build_task324_v5b_no_einsum.py
- 2026-05-31 | compress_v1 (single-channel U8 index canvas, ONE [1,1,39,39] X-kernel, scalar Where recolor, Max(.,1) clamp) byte-equal vs v5b (0 div on 5000) | 23,491 (mem 21,912 + params 1,579) | build_task324_compress_v1.py
- 2026-05-31 | compress_v3 (EXACT corner-base rule, drops 3-way vote; on_base=(countB_row>0)AND(countB_col>0)) 266/266 + 100% truth (v5b 98.47%) | 18,545 (mem 16,968 + params 1,577) | build_task324_compress_v3.py
- 2026-06-11 | dilated_bridge_v4 (#98 dilated crop-conv K[1,10,2,2] dil=[10,10] → canvas20_f32 1600B directly; eliminates canvas30 3600+900) byte-equal | 15,667 (mem 14,068 + params 1,599) | build_task324_dilated_bridge_v4.py
