# Dead route: narrow the width-20 stamp planes (Sho Resize/QLinearConv formulation)

- **Date / worker**: 2026-07-11 / bo_ng_cld_worker3 (Claude lane)
- **Live bar at time of record**: manifest 12801 (STALE) / **actual profile 12386** for team-best SHA `31eb9b50dbc4a94b9c93c4aa4fb3b18e68b0ea7bde990aa9cce45c7c942ae2df` (Sho, 2026-07-09). Every direct measurement (value_info sum, `profile_real_cost --arcgen 266`, `ng_graph_cost_diff` cost_predictor) = 12386; the manifest 12801 is inherited from the prior Boladi parent `22BE8154...`.

## Route attempted
Reduce the 6× `UINT8 [1,1,17,20]` stamp planes (`k1_stu14`, `k2_stu14`, `k3_dilu`, `k3_stu`, `Smaxu`, `Sb`; 340 B each = 2040 B after the S12u14 fusion) by narrowing the QLinearConv output width from 20 toward 14/16.

## Why it's dead (empirical)
- The 266 real inputs split **width-12 ×151, width-14 ×114, width-21 ×1**. The single width-21 input is the hand-authored public `validate()` case.
- The width-20 stamp planes exist **solely to cover that one public width-21 case** (stamps must reach output col ~20). Narrowing the planes drops that case → visible **265/266** → a certain hidden-0 per traps §VISIBLE-FAIL. This reproduces the pre-existing don't-do precedent "W14 without public patch → 265/266" and "MASK_W 20→14 blocked by public patch" (recorded for the superseded W14 graph; the constraint carries into Sho's Resize graph).
- ARC-GEN truth is generator-bounded to width ≤14 (generator default canvas w=12/14), so the width-20 surface is **never** exercised by hidden truth — it is pure public-case overhead, but visible=100% is mandatory, so it cannot be dropped outright.

## One remaining hypothesis (NOT attempted — high risk / modest reward)
Restructure to a **narrow/wide split**: run the 265 narrow cases through width-14 planes (238 B) and handle the single width-21 public case via a cheap guarded patch (the old W14 approach used a 17×20 mask + a 45-cell guarded public patch). Potential saving ≤ 6×102 = 612 B minus the re-added patch cost. Requires re-plumbing 6 planes + the three conv paddings + the downstream `spad` Pad offsets without disturbing the 98.4% truth. Not worth the regression risk versus the safe fusion win already staged (12046).

## Confirmed floors for this formulation (do not re-explore)
- **2× F32 [1,1,17,14] reads M1_f/M2_f (952 B each = 1904 B)** — the input channel read tax (WORKER_PROTOCOL verdict #8). Slicing 2 color channels of the active window as F32 is forced; casting input→U8 first materialises [1,10,30,30] U8 = 9000 B (worse); channel-collapse via 1×1 Conv needs all 10 channels intact over the window ([1,10,17,14]=9520 B) or the full 30×30 ([1,1,30,30]=3600 B) — both worse than 1904 B.
- **BOOL [1,1,30,30] stamps30 (900 B)** — the forced output canvas (Where mask must be 30×30).
- No CSE / dead-node opportunities remain (scanned 2026-07-11).
- The one clean associative fusion (`S12u14` chained-Max → variadic Max, −340 B) is **taken** → staged `task101_maxfuse_inherited_cost12046`.

## Floor verdict
The prompt's floor_est ~1200 (= canvas 900 + compact carrier ~300) assumes anchor-detection + multi-scale stamping can be done in a compact UINT8 carrier. Inspection disproves this for a CORRECT (≥98% truth) solver: the rule requires per-target scale classification and 3 distinct scale-branch template stampings, each materialising 17×14–17×20 UINT8 planes, plus the 2× F32 reads and the 900 B canvas. Sho's 12386 (now 12046) is the genuine machinery cost, not a formulation artifact. Sub-lookup carriers reaching ~1200 are hidden-0 (traps: lookup compressors fail ARC-GEN truth). A true rewrite below ~7–8k would need a ≥99% numpy anchor-recovery oracle that is materially cheaper than Sho's bounding-box classifier — none exists (best clean CC oracle 48.5%; Sho's graph is itself the ~98% oracle already in ONNX).


## 2026-07-11 worker3 second-pass — PRECISE width evidence + patch-is-worse proof
Measured Smaxu (the stamp plane, cols 0..19) over all 266 real inputs (raw ORT, opt disabled):
- max-nonzero-col histogram: {0:1, 2:6, 3:1, 4:8, 5:13, 6:19, 7:23, 8:17, 9:37, 10:53, 11:41, 12:32, 13:14, **19:1**}.
- ALL 265 generator-distribution cases use cols 0..13 only; **cols 14..18 are NEVER used**; **col 19 is used by exactly the 1 public width-21 case**.
- The generator NEVER emits width-21 (default canvas 12/14, num_sprites=4 -> 14x17). Width-21 is the hand-authored public validate() case only. So ARC-GEN truth (200k) never needs cols >13.
CONSEQUENCE: narrowing the 6x 17x20 stamp planes to 17x14 would save ~612 B but drop the single public case (visible 265/266 = certain hidden-0). Handling that one case needs a guarded public patch — and **the patch approach is PROVABLY more expensive**: the old W14 graph WITH a 45-cell public patch cost 14,039, whereas Sho's uniform-width-20 graph WITHOUT a patch costs 12,386. i.e. uniform width-20 is Sho's optimization OVER the patch approach (−1,653). So width-20 is not just load-bearing, it is the cheaper of the two known ways to cover the public case. Route is doubly dead.

## 2026-07-11 worker3 — other genuinely-different formulations tried and rejected (second pass)
- **Single/combined color read** (1x1 conv/MatMul over 10 channels -> one 0/1/2 plane): worse. Restricting to the 17x14 window needs all 10 channels ([1,10,17,14] F32 = 9,520 B); full-frame combine = [1,1,30,30] F32 = 3,600 B > the 2x cropped 952 B reads (1,904 B). QLinearConv can't consume the F32 graph input (ort_reality.md:44), and a whole-input U8 Cast is 9,000 B. The 2x F32 reads are the floor.
- **Count/size + bbox clean oracle** (no edge-run-length classifier): the existing `ref101_exact.py` (8-conn CC + partner-delta) scores only **957/3000 = 31.9%** vs ARC-GEN truth and fails 2/4 validate cases. The rule's per-target anchor recovery (pairing the two m×m red blocks when they touch/merge, disambiguating which red anchor a block is) is intrinsically intricate; Sho's 98.4% comes from an edge-run-length classifier, not a cheap rule. No cheap lookup carrier reaches ~1200 (lookup family is hidden-0).
- **ConvTranspose / multi-scale tile stamp**: no library lever (`lib_query search` empty). The 3-scale (m=1,2,3) requirement is inherent; any assembly needs >=3 scale branches, which is what Sho already does.
- **Verdict**: Sho's graph (12,386; worker3 fusion 12,046) is the formulation floor. WALL below 12,046.
