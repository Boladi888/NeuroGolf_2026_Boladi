# Input-as-active-mask: contract the FREE graph input INSIDE a free-output einsum

## What I discovered

For positional / input-independent-modulo-extent outputs (periodic fills, stamped patterns
cropped to the active grid), you do NOT need a separate width/height/active reduction. **The graph
input is memory-exempt** (named `input`), so feed it straight into the free-output Einsum and
**contract its channel axis there** — the resulting active mask `A[b,H,W] = sum_C input[b,C,H,W]`
(1 at every active grid cell, 0 at padding) is an **in-kernel temp (#89), never materialized,
costs 0 bytes**. Multiply it (in the same einsum) by the factored pattern.

Key: this kills the usual `ReduceMax/ReduceSum(input) -> [1,30] F32` width-signal intermediate
(120 B, otherwise the single largest charged tensor once the pattern is factored) AND any
row/column bounding — the input's own support bounds BOTH axes for free. Every grid cell in the
encoder one-hot has exactly one channel set, so `sum_C input` is exactly 0/1.

Example einsum (task176, period-12 colored fill cropped to width):
```
out[b,c,H,W] = (sum_C input[b,C,H,W]) * sum_{o,r,k} ChanSel[o,c]*RowSel[r,H]*U[o,r,k]*V[k,W]
equation:  'bCHW,oc,rH,ork,kW->bcHW'
```
`C` appears only in `input` (and the output drops it) => it's contracted => gives the active mask.
`H`/`W` appear in both `input` and the placement bases (`rH`, `kW`) and in the output => the
masks line up pointwise. ONE Einsum node, ZERO charged memory.

## Where I encountered it

- task176 (Pattern_Recognition / "Filling Regions", gen `7447852a`). Period-12 column tile of a
  fixed 3x12 base, 3 active channels (ch0 bg, ch2 red, ch4 yellow), cropped to width.
- Builder: `The_400_Tasks/Task_by_Task/task176/prototypes/build_v6.py` (direct, cost 390) and
  `build_v7.py` (rank-6 ColTile, cost 354).
- Staged: `staged_submits/task176_einsum_input_active_rank6_cost354/`.

## Why it matters

- Took task176 from LIVE 3613 (and the team-best bar 450) to **354**, byte-equal vs LIVE over
  200025 inputs + 100% over 200000 generator pairs, profile==static (gate-5 exact).
- The progression shows the win: separate-ReduceMax-active einsum = **512**; same einsum but the
  input contracted in-kernel for the active mask = **390/354**. The ~120-150 B delta is exactly
  the eliminated width-reduction chain (colsum F32 120 + bool/cast).
- Generalizes any "free input + multiplicative active mask" — the input is a FREE einsum operand,
  so reach for it instead of a charged reduction whenever the active region is `sum_C input > 0`.

## Supporting reusable facts (verified this session)

- **rank-6 ColTile exact integer factorization**: a period-P colored field `[colors*rows, cols]`
  often has small col-mode multilinear rank; an EXACT `U[.,k]@V[k,.]` factorization with
  U in {0,1}, V in {-1,0,1} is F32-exact (small integers), so `>0` decode is safe even with -1
  entries (they cancel exactly). Cuts the stored field params (270 -> 234 here). Compute via
  sympy rref pivots + `(U^T U)^-1 U^T M`.
- **Don't cast the input to F16** to match F16 factors: casting the full `[1,10,30,30]` frame =
  18000 B intermediate. Keep all einsum factors F32 so the free input feeds with no Cast node
  (all-F32 einsum, output F32 decodes via `>0` fine).
- **bg-as-complement / red-passthrough is a TRAP for this structure**: ch2==input ch2 and
  ch0 = active - ch2 - ch4 are true pointwise, BUT exploiting them needs ADDITIVE terms mixing
  input-derived and basis-derived contributions => two einsums + Add => materializes the
  [1,10,30,30] canvas. The UNIFIED multiplicative einsum (all colors in one ColTile) is strictly
  better. (Additive cross-source decomposition loses the free-output property.)

## What's missing from the library

`techniques.md` #118/#119 (Aditya's basis-matrix / deep-separable einsum) cover factoring the
PATTERN, but I didn't find an explicit note that **the graph input itself is a free einsum operand
whose channel-contraction yields the active/support mask at zero memory** — replacing a charged
`ReduceSum/Max(input)` width-signal. by_symptom has periodicity routes but none point at this
"input-as-mask" lever. Worth a one-liner under #119 or a new entry, plus the rank-6 exact-integer
field factorization as a costing trick for periodic stamped fields.
