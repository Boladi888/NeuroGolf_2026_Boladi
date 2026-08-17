# Free-output separable Einsum dissolves a kron(P,P) one-hot patch (task195, 1807 -> 1011)

## What I discovered
A `kron(P,P)` (Kronecker self-product) one-hot output is FULLY free-output-able via a single
multi-operand Einsum (#109/#119) — the "patch + Pad" 810-B intermediate that two prior versions
treated as forced is not forced at all.

`kron(P,P)[r,c] = P[r//3,c//3] * P[r%3,c%3]` is separable: the row index splits as `r = 3i+a`
(block i, within a) and likewise the column. So the whole `[1,10,30,30]` one-hot can BE the output:

```
output = Einsum('gtij,gtad,tk,ir,ar,jc,dc->gkrc', Q, Q, Ksel, Hb, Lb, Hb, Lb)
```

Three reusable moves that combined here:

1. **Self-product via repeated operand (#119):** P appears TWICE in the same Einsum
   (`gtij` and `gtad`), so `P[i,j]*P[a,d]` is computed in-kernel, never materialized.

2. **Two-term window/figure decomposition through a `t` axis with a stacked data operand.**
   `Q = concat(ones(3,3), P)` -> `[1,2,3,3]`. Term t=0 (Q=ones) sums to the 9x9 WINDOW indicator;
   term t=1 (Q=P) is the kron figure. A `Ksel[2,10]` channel selector routes term0 -> ch0 and
   term1 -> (e5 - e0), so ch5 = kron and ch0 = window - kron = NOT-kron inside the 9x9, zero outside.
   This is the #109 "fold the negative background into the channel selector instead of adding a term"
   trick applied to a 2-term (window vs figure) case — no separate background plane, no sentinel.

3. **Factor a dense `[3,3,30]` placement basis into two `[3,30]` matrices reused on BOTH axes (#118).**
   `Hb[i,r] = (r//3==i & r<9)`, `Lb[a,r] = (r%3==a & r<9)`; then `Hb[i,r]*Lb[a,r] = (r==3i+a)`.
   Hb and Lb are each reused for the row AND the column axis. This took the placement params from
   540 (two dense `[3,3,30]`) down to 180 (two `[3,30]`) — a 3x param cut on the basis, and the single
   biggest lever after the patch deletion. Generalizes: any `out_index = base*block + within` placement
   factors into a block-basis x within-basis product, reusable across axes.

## Where
- build: `The_400_Tasks/Task_by_Task/task195/prototypes/build_task195_freeoutkron.py`
- staged: `staged_submits/task195_freeoutkron_cost1011/` (manifest has all gate results + sha256)
- task notes: `The_400_Tasks/Task_by_Task/task195.md` (CURRENT BEST + floor note)

## Why it matters
1807 -> 1011 (memory 1732 -> 760 by deleting the 810-B `out9_oh` patch + the kron Reshape ladder +
Where/Equal/Pad tail; params 75 -> 251 from the bases). Net -796 cost, expected +0.58 LB points.
Gates all green: visible 265/265, truth 200000/200000, ab byte-equal 200265/200265 vs LIVE, and
**profile == static == 1011** (the Einsum contraction temp is uncharged at runtime; the #109
profile==static discriminator held, so no reprice). Adversarial corner r0=7,c0=9 (corpus max) clean.

## Gate decode note (already in #67/task295, reconfirmed)
Output is F32 {0,1}, not BOOL. `ng_harness.session.run_raw` applies `(out > 0.0)` to BOTH candidate
and reference before comparing, so visible/truth/ab all gate correctly on a {0,1} float free-output
Einsum — no need to end on a comparison op for this case. Confirmed the scorer credits it (profile run).

## What's missing from the library
#109/#119 already cover free-output separable Einsum and self-reuse, and #118 covers basis-matrix
factors. What's NOT explicitly spelled out anywhere I found: the **"window minus figure via a 2-element
term axis on a stacked data operand (ones | P)"** idiom for emitting `figure` in one channel and
`window AND NOT figure` in another, with ZERO background plane and ZERO sentinel. That specific pattern
(any "paint X in color A, paint the complement-within-a-bounded-region in color B") is worth a one-line
addendum under #109 or #119. The kron-self-product-as-free-output case is also a clean new exemplar for
the "is this patch really forced?" floor-falsification list (#68).
