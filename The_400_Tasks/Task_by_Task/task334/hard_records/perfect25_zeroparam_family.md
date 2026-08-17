# task334 — DEAD ROUTE: perfect-25 zero-param family (Identity/Transpose/flip/rotate)

- **Date:** 2026-07-07 (bo_ng_cld_worker2, Claude lane)
- **Live bar at audit:** cost **64** (manifest, owner Vamshi, sha `e3c6c9605a7a6e0c05269847bc58e99ee2ff5edd02f3d6837ff8ccb85ad01588`, updated 2026-07-02T02:35Z). `ng_current_bar.py --task 334` authoritative_source=manifest.
- **Prompt hypothesis audited:** "perfect-25 ZeroopsweepCeilingAudit, floor ~1, legal zero-param identity/transpose family."

## Verdict: DISPROVEN. The zero-param family is impossible here.

**Why (structural):** task334 is a *color-conditional, shape-changing glyph lookup*, not a whole-grid symmetry.
- ARC shapes: input **5×5**, output **3×3** (different shape) → no identity/transpose/flip/rotate can map 5×5→3×3.
- The 5×5 grid contains a single color `c ∈ {1,2,3}`; the 3×3 output is one of **3 fixed glyphs selected by `c`**, drawn in color 5:
  - c=1 → `+` (010/111/010), c=2 → `T` (111/010/010), c=3 → `L` (001/001/111).
- Output depends on input *values* (which color), so a constant is also wrong (3 distinct outputs).

## Exact failed gate (fast probe, cheapest-first)
Built in the official `[1,10,30,30]` F32→INT32 encoding (`prototypes/build_onenode_sweep.py`):
- `cand_identity.onnx`  (Cast only, cost≤1) — `ng_gate visible --task 334` → **0/271 (0.00%) FAIL**
- `cand_transpose_hw.onnx` (Transpose perm [0,1,3,2] + Cast, cost≤1) — **0/271 (0.00%) FAIL**

Both are certain hidden-0 (§VISIBLE-FAIL AXIOM). Kill rule fired: no candidate visible-correct at cost≤1. No truth/ab/battery run (forbidden on visible-zero).

## Forced-floor argument (why floor ≫ 1, and ≈ 64)
The transform forces three components in the 10-channel encoding:
1. **Color detect** — distinguish c∈{1,2,3}: a channel-spanning projection over the 10 input channels (current `code_w`[1,10,1,1] = 10 params). Length-10 forced by input channel dim (a Slice to fewer channels adds its own starts/ends/axes/steps + a sliced intermediate — net wash/loss).
2. **Glyph synthesis** — 3 distinct 3×3 patterns keyed on c. Current best already does this from just 2 predicates (a=c≥2, b=c≥3) + derived booleans, with **no stored 3×3 bank**. A stored bank ([3,1,3,3]=27 params or [3,10,3,3]=270) is strictly worse.
3. **Channel route** — place glyph into output channel 5 of 10 (current `render_w`[10,1,1,1] UINT8 = 10 params via ConvInteger). Length-10 forced by output channel dim; Concat/Pad-zeros alternatives cost 81/8+ params → worse.

Two length-10 projections (~20 params) + ~7 scalar consts + ~37 B minimal intermediate memory ≈ the current **64**. The live graph (13 nodes, `task334_scalar_signed_u8u8_final_convint`, profile=static=64) already sits at this floor.

## One remaining hypothesis (unexplored, low-EV)
Merge/shorten the two length-10 projections into a single combined weight, or synthesize the glyph with pure arithmetic (fewer materialized Concat rows) to shave a handful of intermediate bytes. Predicted gain <5 B and high correctness risk; not worth a rebuild that would likely tie. **Never stage a tie.**

**Bottom line:** WALL. EV +4.16 was predicated on a floor ~1 that does not exist; real floor ≈ 64 and the current best achieves it.
