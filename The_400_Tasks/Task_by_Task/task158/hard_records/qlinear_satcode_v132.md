# task158 v132 — QLinearConv saturating-U8 placement code KILLS the INT32 clamp — WIN 21,150 (−2,037) (2026-07-12, bo_ng_cld_worker1 pass 2)

**Bar:** LIVE v130 = 23,187 (manifest, sha `84bf2db2…`).
**Candidate:** v132 = 21,150 (profile==static), sha `224C6A14…`. **−2,037 vs bar, −1,691 vs v131.**

## The lever — break the "double INT32 render index is forced" wall
v130/v131 spread the per-cell placement code with **ConvInteger → INT32 CP** `[1,1,26,25]` = 2,600, then
make the OOB-Gather crash-safe with a provable **`Min(CP,76)` clamp → a SECOND INT32 plane CPc** = 2,600.
That 5,200 double-INT32 was documented as the forced render floor. It is NOT.

**QLinearConv with all scales=1 and all zero_points=0 computes `saturate_u8(Σ DET·Wc2)`** — bit-identical
to ConvInteger's integer sum whenever the sum ≤ 255, emitted as a **UINT8** plane `CP_u8` = **650**. Then a
single `Cast → INT32` `CP_i32` = 2,600 feeds the (INT32-only) Gather, against a **256-entry** paint table
(`TFb` padded 77→256, entries 77..255 = False).

- Index cost **5,200 → 3,250** (`CP_u8` 650 + `CP_i32` 2,600), table +256. Net **−1,691** vs v131.
- **Crash-safety is now PROVABLE BY DTYPE (stronger than the clamp):** UINT8 saturates at 255, so the
  Cast→INT32 index is always in `[0,255]`, and the table has exactly 256 entries → the Gather can never
  OOB. No clamp op. Confirmed `cand_crashes=0` over 20,266 boards incl. overflow.
- **Identical on every clean board:** a single placement's code = ≤4(rot)·19(max Wc2) = 76; a cell only
  exceeds 76 when two placements overlap in the 9×9 spread window = the overlap/ambiguity class only
  (CP>76 ⇔ overlap ⇔ LIVE's own truth-miss/crash class). So `CP_u8 == CP ≤ 76` on all clean boards →
  Gather hits identical entries → v132 == v131 == LIVE. On overflow boards v132 saturates to no-paint
  (dual-truth, same ambiguity class).

## Gate ladder (all at --n 2000 or the stated N)
- visible **266/266**; profile **static==profile 21,150** (mem 19,892 + params 1,258).
- truth n2000 **1999/2000**; ab n2000 match 2264/2266 (**2 div, dual-truth, not byte-equal**).
- dual-truth 40k vs official Y: v130 misses **36**, v132 misses **35** (−1, strict subset), **0 regressions**
  (identical miss-set to v131 — the render-index swap changed nothing on correctness, only cost).
- crash-aware offline AB 20,266 pairs: **cand_crashes=0**. safety WARN parity with LIVE.
- ng_graph_cost_diff: **NET −2,039** (memory −2,118, params +79), reconciled.

## v133 — the SAME lever kept BYTE-EQUAL (21,497, clean ship)
The saturating-code trick is independent of the paint-mask-vs-whole-stamp choice. Applied to v130's
UNCHANGED whole-stamp COLOR render with a 256-entry COLOR table `TF256 = Pad(TF[77], value=TF[76])`
(the runtime scalar `TF[76]` fills entries 77..255), saturation + replication EXACTLY reproduces
`Min(CP,76)` on overflow → **byte-equal to v130 on every board** (`ab --ref v130`: 2266/2266, div=0,
byte_equal=True; crash-aware AB vs LIVE 20,266 pairs div=0, cand_crashes=0). Cost 21,497 (−1,690),
crash-free by dtype, `candidate_class=PATH_A_BYTE_EQUAL`, no dual-truth caveat — the safe/standard-ship
recommendation. v132 (21,150) is 347 cheaper by ALSO taking the v131 paint-mask render (dual-truth).
Staged `staged_candidates/task158_v133_byteequal_satcode_cost21497_shaC47D4B0B/`.

## ⚠ Deployed submission.zip is STALE vs manifest-LIVE v130 (gate-reading trap)
`ng_gate ab --task 158` compares to the DEPLOYED `submission.zip`, which currently lags the manifest:
v130 ITSELF shows `divergences=2, byte_equal=False (first_div=1705)` vs submission.zip. So a candidate
that is truly byte-equal to v130 still reports 2 divergences on the default `ab`. To test byte-equality
to the actual LIVE, pass `--ref <v130.onnx>` explicitly (v133 → 0 divergences). Do NOT read the default
`ab`'s 2 divergences as a candidate defect on this task.

## Transferable ORT facts (verified this session, correct the floor docs)
- **ORT `Gather` index must be INT32/INT64** — UINT8, INT16, UINT16 all raise `INVALID_GRAPH` (tested).
  So a data-dependent per-cell Gather always costs a full-canvas INT32 index (650 cells → 2,600 B). What
  is NOT forced is a *second* INT32 plane to bound it: emit the code as saturating UINT8 (½ the width) and
  let a 256-entry table absorb the saturated range.
- **QLinearConv(scale=1, zp=0) == saturating integer ConvInteger** but at UINT8 output (650 vs 2,600).
  Use it anywhere a bounded integer conv result feeds a fixed-size table — the U8 clamp is free (dtype),
  replacing an explicit `Min`/`Clip` INT32 plane.
- **ORT `Pad` and `Concat` both work on BOOL** — build an N-entry BOOL lookup table cheaply.

## Remaining floor (below 21,150) — WALL for my routes
Every remaining block re-priced and forced:
- `cidc` F32 read 2,600 + `C8c` U8 650: input is F32 one-hot; Conv is the only cheap channel-reducer and
  is float-only (QuantizeLinear to U8 = 9,000). Forced.
- Render index `CP_u8` 650 + `CP_i32` 2,600: Gather INT32-only (tested), QLinearConv U8 is the minimum
  bounded producer. Forced (this session's optimization).
- 2 marker masks 2,600: 2 data-dependent roles, `Equal`→BOOL + `Cast`→U8 for the quantized DET conv.
- `DET` 1,656: 3 scales load-bearing (scale-normalize-to-one DEAD, 265/266).
- Locator 5,850 (9-plane multicolor+count): minimal; count- and variance-based paint/legend selection
  proven non-viable (compressed history); max-min / sum-of-squares multicolor variants price equal-or-worse.
- render mask+fallback+pad 1,300+900; table+extraction ~700; params 1,258.
No further INT32/wide-dtype narrowing exists (INT16 Gather rejected); no redundant plane (prior
`_redundancy_hunt` + this ledger). The prompt's floor_est ~900 is fantasy (premised on the stale v125
dense-plane family); the true forced floor is ~21,000.
