# task118 hard record — structural floor + convshift zero-pad dead route

- **Live bar:** 13,260 (manifest, authoritative; `ng_current_bar.py --task 118`), live sha `2522855037193826ccb446a1674f6b0a74359f228238aa0c607b4df911befa62`, owner Boladi.
- **This session (2026-07-08, bo_ng_cld_worker3):** staged a byte-equal **12,621** (Δ -639). Below are the routes proven DEAD so the next batch does not re-explore.

## DEAD: convshift zero-pad tie-break (cost 11,885, do NOT ship)
- Route: fold the two tie-break shift Gathers (`chosen_right_count`+`chosen_down_count`, -675 each) + their INT64 shift-index inits into ONE radius-selected `QLinearConv` shift kernel. Proto `prototypes/task118_structural_convshift.onnx`.
- **Failed gate:** `ab --full` vs LIVE = **78 divergences / 200,267, byte_equal=False**. The conv zero-pads while LIVE's Gather edge-clamps, so it selects spurious cross-centers at the last row (r2) / last col (r3).
- Not rescuable via §DUAL-TRUTH: truth is a TIE (cand 16256/20000 == LIVE 16256/20000 = 81.28%), so LIVE_misses − cand_misses = 0 ≠ 78 → lateral, not more-correct-than-LIVE. Owner wants LIVE-equivalence → dead.
- **Fix that shipped:** edge-Pad(bottom+right +1) before the shift conv (`pads=[1,1,0,0]`) replicates the clamp → byte-equal, but the +728 B pad drops the net win from -1,375 to **-639**. Staged at `staged_candidates/task118_crosscoveru8qconvcompress_cost12621/`.
- Useful sub-result (NOT dead): **a computed (Where-output) QLinearConv weight IS scoreable inside a full task graph** (the scorer only rejects it in an isolated toy graph because its dummy inference is [1,10,30,30] F32). Reuse this fold elsewhere.

## WALL: BoBrain floor_est ~3,300 is unreachable for the exact LIVE-equivalent rule
Real irreducible floor ~12,600. Each clause verified in numpy over all 267 officials this session:
1. Crop tight 25x27 — color-2 evidence + color-8 paint both span rows 0-24, cols 0-26 (all 4 edges); `ab` byte-equal on 200k confirms. Cannot shrink.
2. F32 color-2 slice **2,700** forced (cast-first=9,000, argmax=7,200 are worse) — already ~82% of the 3,300 est before any compute.
3. Radius chooser needs BOTH count maps: r3-always=257/267, r2-always=227/267, elementwise-max=257/267.
4. Local-max needed: lm-only=264/267.
5. Directional tie-break needed on BOTH radii: right-only=265 (fails 62,115), down-only=266 (fails 163). uint8 ramp-score alt overflows (27*11+26=323>255); INT32=4x bytes=worse. Irreducible.
6. Final 30x30 paint pad 900 forced (Where mask must broadcast to [1,10,30,30]).

## One remaining hypothesis (unproven, low confidence)
The only untried lever is a **byte-equal cut inside the peak+select+cover block** (currently lm, neighbor, strict, is_peak, selected, wsel, cover, paint_crop). No fusion found this session keeps 267/267 AND cuts a map without a computed op that the scorer rejects. A sub-12,000 byte-equal result would need a genuinely new scoreable primitive (e.g. a fused directional-argmax) — not seen in the op set. A sub-3,300 result requires abandoning LIVE-equivalence (the lookup/Cherry13b family), which is off-limits.
