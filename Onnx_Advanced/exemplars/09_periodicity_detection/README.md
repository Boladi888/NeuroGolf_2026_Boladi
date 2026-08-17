# 09_periodicity_detection — worked case studies

A new category established by task110's radical-rewrite session (2026-05-20). Periodic-fill algorithms with per-direction scoring + iterative fill.

## Files

- `task110_periodicity.md` — The canonical periodicity-detection rebuild. -94,303 bytes, +12.73 LB. Includes the 9-phase algorithm template and the calibration anomaly (predicted +0.18 LB, actual +12.73 LB — cross-task cascade unlock).

## When applicable

Periodic-fill tasks:
- Input has repeating pattern along H, W, or both.
- Some cells are missing (sentinel value, typically bg/0).
- Output fills the missing cells using the inferred period.

## Critical lessons from task110

1. **Per-direction scoring** — joint V+H ArgMax fails on asymmetric-period pairs. Score V and H independently.
2. **`Where(bool, bool, bool)` workaround** — use And/Or/Not chains.
3. **Padding via fi=10 trick** — set out-of-range OneHot index to produce all-zero rows.
4. **Cost floor for 7-candidate × 2-direction periodicity** ≈ 210k bytes minimum.
5. **Calibration anomaly** — predicted gain way under-shot actual. Cross-task memory cascade unlock suspected.
