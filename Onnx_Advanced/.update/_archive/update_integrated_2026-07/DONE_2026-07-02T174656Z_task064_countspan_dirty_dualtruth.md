# task064 — count-derived in-box span is inherently DIRTY on padding-perturbed corpora (DUAL-TRUTH dead-end)

## What I discovered
For task064 (rectangle dot-ray renderer, #110 dual-pack bit-pack ray-fill family), the in-box row/col span mask can be built two ways:

1. **Coordinate index** (the byte-equal floor, v20/9743): `row_in_box = (row_idx >= box_top) AND (row_idx <= box_bottom)`, using an INT64 coordinate-index initializer (`row_idx [1,24,1]` + `col_idx [1,1,24]` = 48 params) compared against the robust ArgMax bbox scalars.
2. **Count-derived span** (cheaper by 48 params, but DIRTY): derive the span from the box-color occupancy counts (`box_row_counts`/`box_col_counts`, already computed by the renderer's Einsum), deleting `row_idx`/`col_idx`.

**Both count-derived variants are DIRTY vs the LIVE 200k adversarial corpus and neither is byte-equal:**
- v14/v19 `box_*_counts > 0` (occupancy): **341/200267 A/B divergences**, cost 9707.
- v21 `box_*_counts >= ReduceMax(box_*_counts)` (only full-width rows): **163/200267 divergences**, cost 9715.

Root cause: the box counts sum box-color over ALL 30 padded columns/rows, so LIVE's legacy-encoded padding perturbations (sparse box-color cells in the 24→30 padding border) leak into the counts and paint rows/cols OUTSIDE the rectangle. `>0` includes any leaked row (overpaint); `>=max` fixes overpaint but a leak into an IN-box row inflates the max above the uniform width W and EXCLUDES legitimate rows (underpaint). No count threshold reaches byte-equal.

**DUAL-TRUTH attribution proves it's churn, not more-correct-than-LIVE:**
- Shippability identity (traps §DUAL-TRUTH): `ab_divs == LIVE_misses − cand_misses`.
- Measured: LIVE truth 199522/200000 (misses 478); v19 & v21 truth both 199525 (misses 475). `LIVE_misses − cand_misses = 3`.
- ab_divs = 341 (v19) / 163 (v21), both ≫ 3 → NOT a strict superset of LIVE-correct. Churn: ~half the divergences the candidate is WRONG where LIVE is RIGHT.

**Takeaway: a span/mask derived by thresholding counts that are summed over a padded extent is structurally non-byte-equal on a padding-perturbed LIVE corpus. The robust span REQUIRES an explicit coordinate index (48 params here). The robust-max threshold is a plausible-looking fix that only halves the divergences — do not mistake fewer-divergences for clean.**

## Where I encountered it
- Task: task064 (Color_and_Logical, rectangle dot-ray).
- Builders: `The_400_Tasks/Task_by_Task/task064/prototypes/build_task064_v19_rank2_pad_default_axes.py` (9707 count-span `>0`), `build_task064_v21_robust_countspan.py` (9715 `>=ReduceMax`, NEW this session), `build_task064_v20_clean.py` (9743 byte-equal coordinate-index floor).
- Staged: `staged_candidates/task064_rectdotray_cost9707/` (PATH-B RISK_SINGLE_PROBE only, honest manifest) and `staged_candidates/task064_rank2pad_paramfloor_cost9743/` (byte-equal gold, beats team-best 9751 by -8).

## Why it matters
- Saves a future worker from re-attempting the 48-param recovery via count thresholds (three variants now falsified: `>0`, `>=max`, and by extension any single-threshold-on-padded-count).
- Calibrates the recapture claim: the prior "9707 byte-bound Kaggle-confirmed" is a Path-B RISK — it is NOT byte-equal to the LIVE adversarial corpus. Whether it scores clean on Kaggle depends on whether Kaggle's real hidden inputs carry the legacy padding perturbations; that requires a captain submit-probe, not a bank.
- Reusable discriminant: when a count/reduction feeds a mask on a task whose LIVE encoding has padding perturbations, gate ab-full and run the DUAL-TRUTH identity before trusting a "≥99% truth" Path-B candidate.

## What's missing from the library
`cost_floor_reference.md:258` records task064 as "39,510 → 14,706 → 9,751 v12 renderer floor". It should be updated: the byte-equal floor is now **9743 (v20)**, and the count-span 9707/9715 routes are explicitly Path-B dirty (not the floor). The DUAL-TRUTH churn discriminant for padded-count-derived masks is a general trap not currently called out under the padding-perturbation entries.

## Sorter note: cost_floor_reference.md — update task064 entry: byte-equal floor 9743 (v20, mem 8858 + params 885); count-span 9707/9715 are Path-B dirty (341/163 LIVE divergences), not the floor. Robust in-box span requires the 48-param coordinate index.

## Sorter note: traps.md §DUAL-TRUTH (or §padding-perturbation) — add dead-end: a mask thresholded from counts summed over a PADDED extent is structurally non-byte-equal on a padding-perturbed LIVE corpus; the robust-max threshold only halves divergences (task064: `>0`=341 → `>=max`=163, neither byte-equal). Gate ab-full + DUAL-TRUTH identity before trusting a ≥99%-truth Path-B candidate.
