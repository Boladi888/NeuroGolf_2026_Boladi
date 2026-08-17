# task158 v131 paint-cell MASK render — WIN 22,841 (−346), dual-truth more-correct (2026-07-12, bo_ng_cld_worker1)

**Bar:** LIVE v130 = 23,187 (manifest, sha `84bf2db2…`).
**Candidate:** v131 = 22,841 (profile==static), sha `EA9AA9F4…`, crash-free. **−346.**

## The lever: the render only needs a BOOL mask + one scalar, not a per-cell COLOR gather
The rule paints only the legend's PAINT-color cells (all one color `pc` per grid) and PRESERVES
existing cells. v130 (and LIVE Sho before it) renders the WHOLE detected stamp:
`gat=Gather(TF,CPc)` looks up a per-cell legend COLOR (U8), gated by a full-canvas `pos=CP>0`
(BOOL [1,1,26,25] = **650 B**), `OC=Where(pos, gat, C8c)`. That is more machinery than the rule
needs. v131 renders the true rule directly:
- `maskP` (BOOL [9]) = legend paint cells = `NOT(E9f==bg OR E9f==X OR E9f==Y)` (legend = bg + pc + markers X,Y).
- `pc` (scalar) = `ReduceMax(Where(maskP, E9f, 0))` (all paint cells equal pc; others masked to 0).
- BOOL paint table: `idxSel=Where(Squeeze(m), IDXM, IDXA)` (orientation-select on the INT32 index —
  **ORT has NO Where kernel for BOOL data**, so you cannot select two BOOL tables directly);
  `TFm=Gather(maskP, idxSel)`; `TFb=And(TFm, nz77)`.
- `nz77` = const BOOL [77] = `[False]+[True]*76`. **Code 0 = "no stamp covers this cell"** (covered
  cells always have ConvInteger code ≥1 because every Wc2 tap and DET value is ≥1); forcing table
  entry 0 to False makes uncovered cells never paint. This 77-entry gate **REPLACES the deleted 650 B
  `pos` plane**.
- `gat_bool=Gather(TFb, CPc)` [BOOL 650]; `OC=Where(gat_bool, pc, C8c)`.

Net vs v130: delete `pos` (−650) and the U8 color table→BOOL; add ~304 B of tiny [9]/[77] machinery
(idxSel INT32 [77]=308 is the biggest single add, + nz77 77 param + paint-color extraction ~73).
**Net −346.** Clamp `CPc=Min(CP,76)` is untouched (still provably crash-free).

## Why it is NOT byte-equal — and why it is still shippable (dual-truth)
v130's whole-stamp render paints legend colors over the ENTIRE 3×3 scaled block, including the bg/marker
cells. Wherever a stamp's bg-cell overlaps a foreign non-bg component, v130 ILLEGALLY overwrites it
(the rule says "existing input cells are preserved"). v131 preserves it (correct). So the two RENDERS
differ only where whole-stamp ≠ paint-cells, and there v131 matches the rule and v130 does not.

Decisive check (`prototypes/_dualtruth_v130_v131_w1.py 40000`, vs the corpus's OFFICIAL cached
ground-truth Y — the same Y the official truth gate uses):
- v130 truth-misses = **36**, v131 truth-misses = **35** (v131 strict subset, −1).
- v130≠v131 divergences = 33: `v131_fixes=1` (v131 right, v130 wrong), `v130_only_correct(REGRESSION)=0`,
  `both_wrong=32`. The 32 both-wrong are **detection-ambiguity boards** (overlapping marker cross-pairs)
  where the shared DET/CP placement is itself wrong, so both renders miss the ill-defined truth; v131 is
  never worse there.
- **0 regressions over 40k boards.** Correctness(v131) ≥ correctness(v130) on every board; official test
  verifies 266/266.

Ship path = dual-truth (traps §DUAL-TRUTH), NOT byte-equal. `ab --n 2000` reports 2 divergences /
`byte_equal=False` — expected; the standard byte-equal gate does not apply to a deliberately-more-correct
render. Kaggle-scored board (official test) is clean (max CP=76 on all 266 officials → no overflow, no
cross-pairs → v131 == v130 == correct there), so this is a pure −346 cost win on the scored board.

## Status / flags
`candidate_class=PATH_B_RENDER_REWRITE_DUAL_TRUTH`, `status=PROBE_READY`, `x86_required=true`,
`kaggle_probe_required=true`, safety WARN parity with LIVE (`WARN_X86_QUANTIZED_OP_FAMILY`, 0 failures).

## Transferable facts (correct the older floor docs)
- Floor-doc fact "(b) runtime gather render forced, INT32 index" STILL HOLDS (`CPc` INT32 index is
  data-dependent), BUT the gathered TABLE can be BOOL (a paint-cell MASK) instead of U8 colors — the
  whole-stamp COLOR render was never required; only a per-cell paint-cell BOOL + scalar pc is.
- The `pos=CP>0` full-canvas plane is NOT forced: fold "code 0 = no paint" into the 77-entry table via a
  static `nz77` And (−650 for +77).
- ORT (this build, opset 17) has NO `Where` kernel for BOOL data — do orientation/branch selects on the
  INT32 index or via And/Or, never `Where(cond, boolA, boolB)`.
- v130/LIVE whole-stamp render is a *approximation* that overwrites overlapped foreign cells; the true
  paint-cells render is both cheaper here AND strictly more correct.

## Remaining floor (below 22,841)
`CP`(2,600)+`CPc`(2,600) INT32 double is the render elephant, forced (data-dependent Gather index +
provable clamp; table-extension crash-fix DEAD, see `table_extension_crashfix_dead.md`). `cidc` F32 read
2,600 forced (Conv float-only). 2 marker masks 2,600 forced (BOOL Equal + U8 Cast for QLinearConv). DET
1,656 (3 scales load-bearing). Sumtest locator minimal. No further byte-equal OR dual-truth lever found
below 22,841 this session.
