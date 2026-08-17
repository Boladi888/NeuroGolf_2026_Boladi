# Two-term free-output Einsum breaks a U8 bit-AND wall (task065: 641 -> 625, AB byte-equal)

## What I discovered

A 3-operand **bitwise-AND tree** (`output = chan_bits & row_bits & col_bits`, all U8) is NOT
automatically the floor for a "fill region minus one cell, plus that cell in another channel"
pattern — even though it stays 1-byte U8 throughout. Its dominant charge is the FIRST AND's
intermediate, whose size is the **association-order minimum** = (channels) x (one spatial 30-dim).
For task065 that is `chan_row_bits [1,10,30,1]` U8 = 300 B; spatial-first (row[30] & col[30] =
[30,30] = 900 B) is worse. So the bit-AND family floor was mem 582 -> cost 641.

The output pattern is **rank-2 per channel**: the background channel = full size x size quadrant
MINUS the single marker cell (= a rank-1 "full" minus a rank-1 "hole"); the marker channel = the
marker cell only. The bit-AND scheme encodes this rank-2 structure cheaply via carry-free bit-codes,
which is why naive single separable products fail. BUT rank-2 = exactly **two** separable terms:

```
output[1,c,r,w] = A[0,c]*valid_r[r]*valid_c[w]      # bg quadrant (full)
                + A[1,c]*row_hot[r]*col_hot[w]       # marker correction (the hole / the dot)
   A[0] = bg_mask                  (1 at bg channel)
   A[1] = dot_mask - bg_mask       (+1 at marker channel, -1 at bg channel)
```

Marker cell: bg = 1-1 = 0 (suppressed), dot = 0+1 = 1 (set). Quadrant elsewhere: bg = 1.
Implemented as ONE free-output Einsum `kc,rk,wk,z->zcrw` (k=2), F16 output (output tensor is free).

## Why it matters

- **641 -> 625**, and the result is **byte-equal to LIVE** (AB 200266/200266, 0 div; visible 266/266;
  truth 200000/200000). static==profile==625, no dynamic gap.
- The win is purely the **term count**: k=2 stacks `[30,2]` F16 = 120 B each (240 B) beat the bit-AND's
  300 B chan-row intermediate (+60 B row/col). A prior k=3 attempt at this task profiled 1016 because
  its `[30,3]` F16 stacks were 180 B each — the lesson is **count the Einsum rank exactly** (rank of the
  per-channel spatial mask), don't default to one-term-per-bit-condition.
- Free reuse: when row-validity == col-validity (`pos < period` both axes), share the selector (-7 B).

## Transferable rule for the library

When you see a U8/bitwise-AND "tree" graph whose dominant charge is a `[C, S, 1]` or `[C, 1, S]`
broadcast intermediate (channels x one spatial axis): compute the **matrix rank of the per-channel
spatial target mask**. If it is rank-K with K small, a K-term free-output separable Einsum
(`sum_k A[k,c]*R[k,r]*W[k,w]`, F16) charges only `K*S` F16 per spatial axis. That beats the bit-AND
intermediate `C*S` U8 whenever `2*K*S(f16) + K*C(f16) < C*S(u8)` — for task065 K=2,S=30,C=10 gave
240+40 < 300+60. The signed channel-weight `A[1]=present_other - present_self` is what lets one extra
term carve a single-cell hole out of a filled region (rank-2), which a one-term product cannot do.

## Where

- Task: task065 (marker-quadrant extract). Builder:
  `The_400_Tasks/Task_by_Task/task065/prototypes/build_task065_w065_two_term.py`
- Staged: `staged_submits/task065_two_term_einsum_cost625/` (manifest + sha256).
- Floor proof + component table: `The_400_Tasks/Task_by_Task/task065.md` ("Two-term break" / 2026-06-22).

## What was missing from the library

`techniques.md #109/#119` cover free-output separable Einsum but frame the term count as a cost
liability ("three dynamic factor stacks exceed the bit-code"). The missing nuance: **the term count
equals the rank of the per-channel spatial mask**, and rank-2 (fill-minus-hole) is common and cheap.
The bit-AND association-order-minimum law (charge = C x smallest spatial dim, not the full C x S x S)
is also not stated anywhere I could find.
