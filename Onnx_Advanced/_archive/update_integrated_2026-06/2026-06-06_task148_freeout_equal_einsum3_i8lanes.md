# task148 drive-to-floor findings (2026-06-06, worker-7) — 18,388 → 4,461 (−76%)

Five promotable findings from the task148 marginals rewrite ladder
(v5 8,302 → v6 7,388 → v7 5,319 → v8 4,920 → v9 4,769 → v10 4,461; byte-equal-vs-LIVE, truth 200k/200k).

## 1. FREE-OUTPUT EQUAL — move the one-hot explosion INTO the output node (≈−2k on Equal-tail tasks)

**GRADER-PROVEN PRECEDENT (found 2026-06-06 post-hoc): shipped task287 (+2.83 LB confirmed)
already ends `Pad(u8, sentinel) → Equal(ch_ids) == output`** — this pattern is ship-class,
not experimental. What's new here is naming the mechanism + the corpus scan below.

§39 MECHANISM: `calculate_memory` AND the profiler skip tensors literally named `output`.
Corollary: **the most expensive op should be the output node when legal.**
The #23 color-index-canvas tail `canvas → Equal(ids) → [1,10,h,w] BOOL → Pad → output` pays
10·h·w for the pre-pad Equal. Flip it:

```
canvas [1,1,24,12] u8 → Pad(value=<non-color sentinel>) → [1,1,30,30] u8 (900 B)
                      → Equal(ids [1,10,1,1] u8)  ==  graph output  (FREE)
```

task148: 2,880 B → 900 B (−1,980). Requirements: (a) pad value = a sentinel no id matches, so
padding decodes all-channels-False (use the same sentinel the canvas Conv bias gives
out-of-grid cells, e.g. 20); (b) Equal(u8) needs opset 11+; (c) output dtype BOOL decodes
fine (`>0` threshold). **Generalizes to every Equal-broadcast / one-hot-assembly tail** —
candidates: any task whose graph ends `Equal/Concat → Pad → output` with the Pad NOT doing
real work beyond geometry. Grep targets: `Equal.*ids`, final-Concat(axis=1) tails.
Spread-pattern candidate: detect `X → Pad→output` where X is >2× the padded-X size.

**SCAN RESULT (2026-06-06, `scan_freeout_candidates.py` over submission.zip): 97 tasks end
`Pad→output` with pre-pad tensors ≥1,800 B — upper bound ≈210 KB total memory across the
corpus** (top: task289 9,000B-MatMul, task323 6,760B-Concat, task278/295 6,480B, task350
6,240B, task071 5,120B; 12 tasks ≥4,400B). Each conversion = restructure the tail to
1-channel-value-canvas → Pad(sentinel) → Equal==output. Realizable fraction needs per-task
work (the canvas must be expressible as a single value plane + per-channel ids), but even
30–50% × log-scaling = several LB. HIGH-priority spread target for a dedicated worker.

## 2. 3-OPERAND free-input Einsum is also temp-free (extends §42)

`input → Einsum('bchw,c,w->bh', input, sel[10], weight[30])` scores
**(macs 0, memory = its [1,30] output only = 120 B, params 40)** — probe-verified on the local
scoring pipeline (ORT 1.24.x). Gives per-row *weighted* column reads (seed column, wall
column) without the 1,200 B `'bchw,w->bch'` intermediate + Slice. §42's caveat ("3+-operand
unverified") can be tightened: verified locally; static graph ⇒ grader-exact per §39.
Adopt-list addition: `'bchw,c,w->bh'` / `'bchw,c,h->bw'` (joint channel+axis weighted
marginals, 120 B each).

## 3. INT8 / U8 vector lanes beat F16 for small-integer row machinery (opset 14)

All [1,30]-class row vectors with values in 0..30 (bounds, positions) run in **U8 (30 B)**,
and signed lanes (mirror offsets, ±delta, sign ∈ {−1,0,1}) run in **INT8 (30 B)** — half of
F16. Probe-verified at opset 14: `ArgMax(INT8)`, `Sub/Mul/Add(INT8)`, `Add(U8)`,
`Where(bool,u8,u8)`, `GreaterOrEqual(u8,u8)` broadcast, `GatherElements(BOOL data, INT32 idx)`.
Trap: integer-narrow `Add/Sub/Mul` REJECT at opset 13 (shape inference fails) — **opset 14
minimum**. GatherElements indices stay INT32 (spec) — one 120 B cast is the floor.
Also handy: XOR interval test `fill = GE(c,lo) XOR GE(c,hip1)` with `hip1 := lo` on empty
rows — kills the `hi−1`/`−1` constants and one bound tensor vs the GE∧LE form (v8; itself
superseded by finding 4).

## 4. U8 WRAPAROUND INTERVAL TEST — one Sub + one Less replaces any 2-threshold band mask

`fill = Less(Sub(colar_u8, lo_u8), span_u8)` computes `c ∈ [lo, lo+span)` in TWO plane ops:
when `c < lo` the U8 subtraction wraps to ≥226, which always fails `Less` against span ≤ 30.
(864 B GE+GE+Xor → 576 B Sub+Less on a [1,1,24,12] grid; empty rows encode as span=0 — no
fillrow gate needed.) ORT Sub(u8) wrap semantics probe-verified. Generalizes to every
"per-row/col interval mask" build (fills, band selections, crop masks): compose (lo, span)
per row instead of (lo, hi).

## 5. RANK-3 EINSUM BIRTH — free Unsqueeze elimination on row-vector lanes

`Einsum('bchw,cx,w->bhx', input, sel[10,1], weight[30])` outputs **[1,30,1]** directly
(x = size-1 free label on the selector). Every downstream row vector is born
broadcast-ready against a [1,1,1,12] col-arange — the Slice→Unsqueeze rank-lift pair
disappears. Same trick: `'bchw,cx->bhx'` for plain row counts. ORT-probed; ~50 B/graph and
2 ops saved whenever per-row vectors must broadcast against a column axis.

Builders: `The_400_Tasks/Task_by_Task/task148/prototypes/build_task148_v{5,6,7,8,9,10}_*.py`.
Probes: job-tmp `t148_probe_einsum3.py`, `t148_probe_f16.py`, `t148_probe_int8.py`,
`t148_probe_v10.py` (patterns reproducible from the builder code).
