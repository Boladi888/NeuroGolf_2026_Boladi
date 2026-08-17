# task138 — free-output separable-Einsum crop-reconstruction, and the inf-in-F16 fragility

Session `bo_ng_cld_worker1` (Claude lane), 2026-07-02, task138 (crop object detect / frame-raycast-crop).
Source builds: team-best Vamshi `task138.onnx` sha `c077e34a…` cost **5403**; my prototype
`The_400_Tasks/Task_by_Task/task138/prototypes/task138_f16gather_rc.py|.onnx` cost **5283**.

---

## Insight 1 — a free-output separable Einsum SHATTERS the old "canvas + N×plane" floor proof

task138's archived floor proof (this task's own `task138.md`, 2026-06-30) claimed a CORRECT crop
build must cost **> 10,377**: F32 color canvas + 7×`[1,1,24,23]`=552 crop planes + 4 I64 ArgMax +
900 output pad. That proof was **family-specific** (the U8-canvas raycast family) and is now
**RETRACTED**. Teammate Vamshi shipped a `5403` build that passes 200k truth using a completely
different family — **no per-cell canvas at all**:

```
output[b,k,y,x] = Einsum('bktz,ry,sx,tr,ts->bkyx',
                         chfeat_h[1,10,11,1], rowfeat_h[5,30], colfeat_h[5,30],
                         coef_row_h[11,5], coef_col_h[11,5])
```

i.e. the entire 30×30×10 output is reconstructed as a sum of **11 per-channel-modulated rank-1
(row⊗col) terms** over a **shared rank-5 F16 basis** (`rowfeat_h`/`colfeat_h` = the 5 spatial basis
rows: `y0, ylast, ymid, hrow, vrow` for rows; `xmid, y0, xlast, hcol, vcol` for cols). The largest
intermediate is 300 B (`[5,30]` F16); there is no `[1,1,24,23]` plane anywhere. Cost ≈ half the
"floor". **Lesson: a floor proof that enumerates the forced tensors of ONE representation is not a
floor — always ask whether the output can be emitted directly by a free-output Einsum (library
verdict #119 family) before believing a canvas-based floor.**

## Insight 2 — exponential-ramp positional encoding FORCES an F32 island inside an F16 graph

Vamshi finds the ray source position with an **exponential ramp** initializer, not ArgMax:
`p_pos`∈[7.8e-5 … **662,499.5**], `p_neg`∈[1.5e-6 … 12,855]. A row/col "code" is
`Einsum(input, …, prow)` where `prow = Where(flag, p_pos, p_neg)`; the magnitude of the resulting
scalar encodes *which* interior column/row holds the marker (a soft-argmax-by-magnitude). This needs
~12 orders of dynamic range ⇒ **F32 is forced** for `prow, pcol, pinvx, pinvy, rowcode, colcode` and
the Einsums that consume the F32 `input`. This is the cheap replacement for the canvas family's "4
I64 ArgMax planes": one F32 `[1,30]` scalar-code per axis instead of a full plane.

## Insight 3 (the trap) — inf-in-F16 is load-bearing; `nan` is the real failure mode, and truth% hides it

Because `p_pos` (662k) ≫ F16 max (65,504), the gathered ray codes **overflow to `inf`** in the F16
tensors `invxg_h`/`invyg_h` on ~76% of real inputs (measured). Vamshi's build is *correct anyway*
because the grader decode is `run_raw = (output > 0.0)` — and `inf > 0 → True` gives the right color
bit, while the `inf` never reaches a cell that should be 0. **`inf` is tolerated; `nan` is not**
(`nan > 0 → False → 0`).

I tried the "obvious" −240 B micro-optimization: for each `Gather(F32_data, idx) → Cast_F16`, move
the cast **before** the gather (`Cast_F16(data) → Gather`), dropping the F32 gather-result tensor
(−60 B/chain × 4). This is *algebraically* bit-identical (`Cast_F16(v[idx]) == Cast_F16(v)[idx]`).
It **passed** `visible 266/266` **and** `truth 2000/2000` — yet emitted **`nan` on 100 % of real
inputs**. Root cause: on the two exp-ramp chains, casting the whole tensor to F16 first materialises
`inf`, and ORT then evaluates the downstream `Einsum('bix,bx->bx', invxg_h, xmid_h)` in a *different
order* than in the original graph, producing `inf * 0 = nan` where the original produced a finite
value from the same bit-identical inputs. The `nan` landed in should-be-0 cells, so `truth%` stayed
100 % — a silent corruption that a truth gate **cannot** catch. **Lesson: any structural edit to an
inf-carrying F16 free-output graph MUST be re-gated for `nan`/`inf` on real inputs (not just truth%);
`truth 100%` is necessary but NOT sufficient when the output can contain `nan` in ignored cells.**

## Insight 4 (the safe win) — dtype-narrow a gather chain ONLY on tensors that provably never overflow

The cast-before-gather narrowing IS safe on chains whose data never exceeds the narrow dtype's range.
Procedure: **measure the global abs-max of the gather-data tensor over a few-thousand-input corpus
sample before narrowing.** For task138:

| gather-data | corpus abs-max | overflows F16? | safe to narrow? |
|---|---:|---|---|
| `rowcode` | 3,320 | 0 / 3000 | **yes** |
| `colcode` | 6,105 | 0 / 3000 | **yes** |
| `pinvx`   | 662,499 | 2266 / 3000 | **no (→nan)** |
| `pinvy`   | 662,499 | 2277 / 3000 | **no (→nan)** |

Narrowing only the two safe chains (`rowcode`,`colcode`) gives a clean, `nan`-free, decode-identical
build at **5283** (−120 vs 5403). Verified: 0/2500 decoded-grid mismatches vs team-best, truth
2500/2500, and (pending) full 200k. Builder: `prototypes/task138_f16gather_rc.py` (pure ONNX surgery
on the team-best graph — I did not have Vamshi's builder source).

## What's missing from the library

`lib_query.py search "free-output einsum float16 inf overflow exponential ramp positional gather"`
returned **no hits**. None of insights 1–4 are captured. Nearest existing anchors: verdict #119
(free-output Einsum micro-floors) and ORT dtype facts (#5) — neither mentions (a) exp-ramp
positional encoding as an ArgMax-plane replacement, (b) inf-tolerant `>0` decode, or (c) the
cast-before-gather nan trap.

## Sorter note: traps.md — add new trap "inf-in-F16 free-output: truth% cannot catch nan in ignored cells; re-gate structural edits for nan on real inputs; dtype-narrow gather chains only after measuring corpus abs-max < narrow-dtype-max"
## Sorter note: techniques.md § (free-output / #119 area) — add "exp-ramp magnitude positional encoding forces an F32 island; replaces I64 ArgMax planes at ~1 F32 scalar/axis; used in task138 team-best 5403"
## Sorter note: op_cost_patterns — add "cast-before-gather narrows a Gather result -60B/chain but is bit-identical ONLY when gather-data never overflows the narrow dtype; MEASURE corpus abs-max first (task138: rowcode/colcode safe, pinvx/pinvy overflow to inf→nan)"
