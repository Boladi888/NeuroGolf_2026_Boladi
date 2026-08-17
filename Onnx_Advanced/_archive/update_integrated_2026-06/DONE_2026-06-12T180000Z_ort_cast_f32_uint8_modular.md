# ORT Cast(F32→UINT8) is MODULAR, not saturating — breaks ArgMax fg detection

**Source:** task281 nibble audit, 2026-06-12, worker-281.

## Finding

ORT's `Cast(float32 → uint8)` kernel uses **modular arithmetic** (C-style truncation of int cast, i.e., `(uint8_t)(int)x` which wraps), NOT saturation clamping. Verified on ORT 1.18+:

- `Cast(1027.0 → UINT8)` = `1027 mod 256` = **3** (NOT 255)
- `Cast(256.0 → UINT8)` = **0** (NOT 255)
- `Cast(11.0 → UINT8)` = **11** ✓

This breaks any scheme where you rely on "large F32 → UINT8 max" for ArgMax-based first/last fg detection. A combo value of 1027 (from 4 fg pixels × 255 combo_mask) wraps to 3, which is LESS than bg-only rows (11), causing ArgMax to find the wrong position.

## Implication

When using `Cast(F32→UINT8)` as an ArgMax input for fg/bg detection:
- Design combo_mask so that fg rows' combo values mod 256 are ALWAYS greater than all possible bg-only combo values mod 256. This is possible but fragile and grid-size-dependent.
- Safer: keep the `Greater(fg_thresh) → BOOL → Cast(UINT8)` chain. ArgMax over {0,1} UINT8 correctly finds first/last 1.

## Cross-ref

- `ort_reality.md` (to add under Cast dtype table or Cast quirks section)
- task281 don't-do log (already updated)

**Suggested §-slot:** New entry under Cast quirks in `ort_reality.md`, after the INT8/INT16 notes.
