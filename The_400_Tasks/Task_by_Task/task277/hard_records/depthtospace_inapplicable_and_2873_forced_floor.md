# task277 — DepthToSpace prompt is misapplied; live 2873 is the structural floor (WALL)

**live bar:** 2873 bytes, sha `8ef03725c8af8913224ba70d6a1b4df9120a2815f290f341b39de0365d053863`
(ng_current_bar manifest, owner Sho, 2026-07-09; profile static=profile=2873, visible 266/266). No lower staged sidecar in `Staged_Boladi/`.
Written by `bo_ng_cld_worker3` (Claude lane), batch of 2026-07-12.

## 1. The prompt's DepthToSpace / block-stamp angle is inapplicable to this task
The Prompt_..._Task277_DepthToSpaceBlockStamp card (floor_est ~1200, "stamp/sprite/checker/copy")
was auto-selected op-first then matched to the task — it does **not** match the actual rule.
task277's rule (verified 266/266, `prototypes/task277_numpy_reference.py`): 4-connected color-8
components on a 10×10 grid, grouped by **translation-normalized shape**; unique-shape component →
color 2, duplicated-shape → color 1. This is **connected-component shape-deduplication**, whose
output is a data-dependent per-component recolor — there is no small channel-packed motif and no
tiled expansion of a fixed template. DepthToSpace expands a fixed packed motif to a canvas; it has
no role here. The prompt's own kill rule ("packed motif as large as the expanded canvas → stop")
fires immediately. **Do not build DepthToSpace/tile/stamp variants for task277.** floor_est ~1200
is an artifact of the mismatched card; the real floor is ~2820 (below).

## 2. Connected-component labeling is REQUIRED (no local/detector shortcut)
Refutes the "single fused detector" remaining-hypothesis in the sibling records
`fused_detector_skeleton.md` and `no_extra_carrier_width_bbox.md`: a purely local/convolutional
discriminator provably cannot separate unique-cells from matched-cells. Test (per-cell KxK
foreground-neighborhood pattern → {matched, unique} label, over all 266):
- K=3: 82 conflicting patterns; K=5: 246; **K=7: 111 conflicting patterns**
(same local window appears on both a unique cell and a matched cell). Since the matched pair are
translated copies, unique-vs-matched is a **global** component property (area/shape identity), not a
local one. ⇒ a component-labeling flood is mandatory; flood-free bbox/width/detector routes are dead.

## 3. Sho's 2873 graph = the structural floor; every charged tensor is forced
Graph (35 nodes): Slice ch8 → Cast U8 → 5-round 3×3 masked-MaxPool flood (max-rank label) →
top-2 component extract (ReduceMax/Equal/Where) → 2× separable U8 QLinearConv area count →
Less/Equal decision → default+2-overwrite coloring → Equal one-hot [1,3,10,10] → Pad [1,10,30,30].
Charged memory 2730 + params 143 = 2873. Forced-component proof:
- **Input is F32** (harness `session.py` feeds `x.astype(np.float32)`), so ch8 Slice = F32 [1,1,10,10]
  = **400 B forced** (only non-1-byte charged tensor); Cast→U8 = 100.
- **Flood = 1100 forced.** 5 rounds needed: 4 rounds fails visible for every seed tried
  (raster 245/266, center 251-253, random-search best 264/266) — max component 8-conn radius = 4 but
  no fixed 10×10 seed centers all radius-4 components. Larger kernels bridge (min inter-comp gap = 1;
  5×5→105/266, 7×7→37/266). Re-mask (×mask) each round is required (else background bleed bridges
  components). 6 Mul + 5 MaxPool × 100 B = 1100.
- **Area discrimination needs ≥2 extractions + 2 counts.** total+1-count can't tell A from B when the
  top component is a matched. Counting must be U8 QLinearConv — ORT rejects ReduceSum on U8/I8
  (only I32, which is 4× the bytes). m1/sup/m2 (300) + 2×(u8 cast 100 + col 10 + count 1) (222).
- **Coloring minimal at 3 tensors (300).** Sho's default(lv3)+overwrite-c1+overwrite-c2 avoids ever
  materializing the 3rd component (`rest = mask−m1−m2` costs 200-300 extra). Any odd_mask+single-Where
  rebuild priced ≥500. Data property exploited: the unique component is ALWAYS the smallest (264/264).
- **Output one-hot [1,3,10,10] = 300 forced.** Grader compares the full [1,10,30,30] tensor exactly
  (`(out>0)==Y`); ch0(bg)/ch1/ch2 all required; Pad to 10ch/30×30 is free.
Sum ≈ 400+100+1100+522+300(+6 scalars)+300 = 2728-2730 mem + 143 params = 2873.

## 4. Verdict: WALL. No candidate staged.
2873 is the floor for the only feasible formulation (CC-label + area-discriminate). All independent
formulations rejected with quantitative evidence: DepthToSpace (inapplicable), local/detector
(K=7 111 conflicts), reduced-round flood (visible-fail ≤264/266), larger-kernel flood (bridging),
per-cell area-map / autocorrelation (priced > Sho). No sub-2873 candidate exists; do not re-explore.

**one remaining hypothesis** (thin): a labeling primitive cheaper than a 5-round masked-MaxPool flood
that still yields per-component identity in <1100 B, OR an output assembly that emits the 3 color
planes in <300 B. Neither is known in the ORT op set; both would need to beat 1-byte-per-cell tensors
that are already dtype-minimal. Absent that, task277 is banked at 2873.

## 5. Pass-2 (2026-07-12, forced second pass) — library/BoBrain mining CONFIRMS the floor
Mined `techniques.md` levers for cheaper CC-labeling / counting / assembly. Every relevant lever
confirms Sho's structure is the endorsed cheapest, and 4 more genuinely-different formulations died:
- **§34 cost-falsification (task364):** on small canvases the **U8 dominance flood STRICTLY DOMINATES**
  §33 (MinPool forces F16 = 1760 B/step) and §34 (ScatterND histogram = unreliable duplicate-target +
  INT64 machinery). Sho already uses the U8 dominance flood ⇒ labeling is at the library floor.
- **Counting:** §95/§90 confirm `Add/Mul/ReduceSum` have NO U8 kernel at opset13 (Sho @opset14 gets U8
  Mul, but ReduceSum still needs I32 = 4×). Sho's separable U8 **QLinearConv** count is the cheapest
  U8 summation; §34 ScatterND size-count is worse (unreliable + INT64). No shave.
- **Axis-separable 1D labeling (§114 CumSum run-labeling):** DEAD — components are not row/col
  separable (row-disjoint 61/266, col-disjoint 22/266, **neither 185/266**); genuine 2D flood required.
- **Bit-packed / byte-lane flood (§88/§243, task048 8831→699):** DEAD for this task — packing works for
  *binary reachability* floods (bit c = col c, shift-based). task277 needs *multi-valued* per-component
  labels (to compare areas/widths), which cannot be bit-packed; §88's own note "ROUND COUNT = corpus BFS
  diameter, k not tunable; F32 encoder bridge is dtype-locked" matches our 5-round + 400 B F32 findings.
- **Packed extraction (Conv-pack the F32 bridge to [1,1,10,1]):** DEAD — unpacking 10-bit rows back to a
  usable [1,1,10,10] needs I32 BitShift/And = 400 B, MORE than Sho's Slice(400)+Cast(100)=500.
- **§89 Einsum in-kernel-temporary hiding:** N/A — applies to separable multilinear masks; task277's
  flood (iterated MaxPool) and data-dependent Where-coloring are not contractions.
Conclusion unchanged and reinforced: **2873 is the structural floor; WALL.**

## 6. Reviewer sign-off (bo_ng_cld_reviewer1, Claude lane, 2026-07-12) — DROP, WALL upheld
Independently re-verified (did NOT trust worker prose):
- **Bar re-read** `ng_current_bar --task 277`: live 2873, Sho, sha `8ef03725…d053863`, authoritative=manifest.
- **Re-profiled the live model** myself: `profile_real_cost --arcgen 266` → static=profile=**2873** (mem 2730 + params 143). **Visible 266/266 PASS** on the live model — the floor is a valid, hidden-scoring model (points 17.0369).
- **Graph decomposition verified tensor-by-tensor** on `Team_Best_Onnx/task277.onnx` (35 nodes, opset14): flood `labels_0..5`+`pooled_0..4` = 11×100 = **1100**; F32 `mask`(400)+`mask_u8`(100)=**500**; extract+count (`labels_sup1`, 4 comp masks, 2 col-counts)=**522**; coloring (`id_c3default`/`id_c1`/`id10`)=**300**; output `small_output` BOOL[1,3,10,10]=**300**. Worker's breakdown is accurate (rare — usually the "floor" mis-identifies the dominant tensor; not here).
- **Card premise falsified** via `task277_numpy_reference.py`: rule is 4-conn CC translation-normalized-shape dedup, NOT a tile/stamp. `floor_est ~1200` / `EV +14.87` are op-first-keyword misclassification artifacts; DepthToSpace has no role.
- **Two skeptical levers I tried, both dead:** (a) #67 output→single value-canvas: pad-then-Equal charges `[1,1,30,30]` U8 = **900 B > 300**; at 10×10 (>9.5×9.5) the 3-channel one-hot IS the #81 floor. (b) 4-round flood (−200 B): formulation-relative; worker's search best 264/266 (visible-FAIL). Even if closed it's ~0.2 pt AND the 5th round is the hidden-margin — reference flags CC as **hidden-bad family**, so a barely-passing 4-round variant is a negative-EV probe when Sho already banks a hidden-VALID floor.
**Verdict: DROP** — confirmed at structural floor; no strict-gain candidate exists or is narrowly namable; card EV rests on a false premise. Do not re-assign task277 as a fresh worker slot.
