# task157 — LIVE Sho model crash fixed by index clamps; residual ~3.8% wrong answers remain (2026-07-10)

**Bar**: LIVE Sho `task157.onnx` cost **6237**, sha `ba3c4625ec637934c21e6ed0a15ef238cab454444718810e11e012041edabcfd` (manifest, 2026-07-09).

## What was fixed (staged: `staged_candidates/task157_sho_clampfix_crashfree_cost6512/`, sha `42410ae2…`, cost 6512, +275)

LIVE crashes on 13/2000 truth boards (InvalidArgument kills the whole bundle):
- `Gather single5_norm` idx up to 6 into 5 rows — `single5_rows = top_width_5(0..4) + [0,1,2]`.
- `ScatterElements ch1_active_flat_3` idx=90 into 90 — flat place index `anchor*15 + start + offsets(≤48)`;
  the obj3/obj4 `Where` col-guard validates the **wrong object's columns** when `case_argmax` selects a
  cross-object start (obj3 guard checks `top_cols_3` while the selected start may be `top_start_idx_4`);
  obj1/obj2 have **no guard at all** (statically reach 92).
- Latent: `obj4_selected_anchor` stack contains `top_width_5` (≤4) and `bottom_start_idx_i32_1` (TopK idx ≤14),
  feeding an axis-size-3 Gather (`obj4_code_matrix`) and `obj4_row_base*15`.

Fix = six int32 `Min` clamps (upper bound only; all values statically ≥ −3 and negative wrap is legal, so
Min is provably identity on every board LIVE runs): `single5_rows≤4`, `obj4_selected_anchor≤2`,
`obj{1..4}_place_indices_flat≤89`. Verified: visible 265/265; truth n=2000 **no error** 1912/2000;
ab byte-equal 2252/2252 on all LIVE-runnable boards (custom runner skipping LIVE-crash boards,
`prototypes/ab_skip_live_crash_2026-07-10.py`); every other data-dependent index site statically bound-proven
safe (see `prototypes/build_task157_clampfix_v1.py` docstring).

## Residual (NOT fixed, deliberately)

- **75/1987 LIVE-runnable boards wrong on both LIVE and candidate** (~3.8%; inherited byte-equal).
- The 13 formerly-crashing boards now run but all 13 give wrong answers (clamped indices are arbitrary,
  not semantic) → candidate truth 95.6%.

**Remaining hypothesis**: a large share of the 75 wrong boards is the SAME cross-object col-guard mismatch —
the `Where` valid mask for obj3/obj4 scatter placement should be computed from the case-**selected** start
(`Less(selected_start + [0..3], 15)`), and obj1/obj2 should get the equivalent guard. Fixing it changes
behavior on non-crashing boards (loses byte-equal vs LIVE → Path-B / DUAL-TRUTH attribution needed:
run truth on both, check `ab_divs == LIVE_misses − cand_misses`). Estimated cost ≈ +24 B/site (Add[4]+Less[4]+
Unsqueeze) minus possibly retiring the [16] clamps. Not attempted this session to keep the crash fix
provably LIVE-equivalent.

**Don't-do**: do not clamp `obj*_place_base` instead of the flat [16] indices — base can legally reach 42–44
on non-crashing boards when col 3 is invalid (guard routes those lanes to 0), so a base clamp is NOT identity.
Do not add lower-bound (Max/Clip-min-0) clamps — obj1/obj4 `col_direct_start` legally goes to −3 (negative
wrap is defined ONNX behavior) and a Max would silently change LIVE-runnable boards.
