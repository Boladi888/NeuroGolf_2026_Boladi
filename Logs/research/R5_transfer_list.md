# R5 Transfer List — apply Aditya's mined techniques (#111–#127) to my highest-cost Boladi tasks

Built 2026-06-09 from the R5 teardown (`R5_deepdive_log.md` / `R5_aditya_idiom_catalog.md`). For each of my
highest-cost assigned tasks, the op-profile of MY current `submission.zip` onnx shows what I overspend on, mapped
to the Aditya technique that would cut it. **Worker prompts for these tasks should cite the suggested technique.**
Cost = current scorer cost (memory+params); LB headroom if floored to Aditya's class (~median 130p) ≈ `ln(cost/150)`.

## The doctrine (the lever behind every row)
Aditya is 5–15x cheaper because he **computes** outputs instead of **learning/looping** them: a 1-node separable
einsum (#119) over place-value (#113) + basis-matrix (#118) factors, instead of canvas+Equal-tail, stacked
QLinearConv detectors, or per-cell Gather/Mod loops. The over-noded nets below (233 @ 585n, 101 @ 427n, 277 @ 246n)
are the loudest tells — he'd collapse them to a handful of nodes.

## Tier 1 — clearest / biggest wins
| task | my cost | I overspend on | → technique | note |
|---|---:|---|---|---|
| **task351** | 31,271 | Slice/Greater/Gather (28n) | **#121 GridSample** | Aditya SOLVES this exact task with a computed-grid `GridSample` warp at ~tens of params — near-direct port. |
| **task366** | 52,001 | 18× QLinearConv + 19× Where | **#119 + #120 + #113** | stacked quantized detectors → one separable-einsum output; bucket the few needed convs via `y_scale` (#120). Biggest single opportunity. |
| **task233** | 25,658 | **585 nodes**: 61 Gather, 23 Mod, 49 Less | **#113 + #123 + #118** | a per-cell Gather/Mod loop unrolled → `GatherND` stencil reads + `Mod` periodic indexing + basis-matrix placement. Should drop an order of magnitude. |
| **task101** | 18,236 | **427 nodes**: 62 Unsqueeze, 16 Mod | **#119 + #123** | per-element construction bloat → one separable einsum + `Mod` tiling. |
| **task169** | 16,106 | 170n: 67 Pad, 28 QLinearConv | **#119 + #120** | (already staged a candidate; this is the deeper floor.) |

## Tier 2 — conv/detector-heavy → separable einsum + moments
| task | my cost | overspend | → technique |
|---|---:|---|---|
| task158 | 34,217 | 13× QLinearConv | #119 / #120 / #118 |
| task093 | 30,403 | Conv + Greater/Where | **#93** color-gated marginals (`bchw,c->bh`/`bw`) replace the row/col conv; #119 for output |
| task102 | 28,959 | 8× Conv + Equal | #119 / #84 |
| task110 | 22,059 | 149n: 26 Gather, 21 Where | #113 stencil + #119 |
| task277 | 16,748 | 246n: Greater/Equal/ReduceSum/Where | #119 + #125 (collapse the boolean sprawl) |
| task324 | 18,545 | Conv + Greater | #119 / #93 |

## Tier 3 — arithmetic/moment & structural → #93 / #118 / #123 / #125
| task | my cost | overspend | → technique |
|---|---:|---|---|
| task300 | 32,854 | Reshape/Add/Mul | #93 moments / #119 separable |
| task281 | 30,342 | Reshape/Less/Equal | #119 / #125 |
| task011 | 29,790 | 14× Slice | structural #67 / #118 basis |
| task132 | 29,611 | Sub/Mul/ReduceSum | #93 weighted marginals |
| task154 | 29,015 | Mul/Sub | #93 / #119 |
| task091 | 28,989 | Add/Less/Sub + Mod | **#123** Mod-periodic + #118 |
| task284 | 25,855 | 131n: Sub/Equal/Add/ReduceSum | #119 + #125 |
| task376 | 24,626 | Sub + Mod | #123 / #93 |
| task184 | 23,667 | Equal/ReduceSum/Where | #114 CumSum-ordinal (it's a banding rule) / #93 |
| task025 | 23,495 | 32× Reshape | #119 (kill the per-element reshape build) |

## How to use
1. When writing a worker prompt for any task above, name the technique # and point at its `Onnx_Advanced/techniques.md`
   entry + the Aditya source task in `R5_deepdive_log.md`. The worker re-derives the cost target via `ln(cost_now/cost_after)`.
2. **task351 first** — it's a near-direct port of an analyzed Aditya net (#121), lowest-risk biggest-win.
3. The over-noded nets (233/101/277/110/025) are pure compression plays — the node count itself is the smell.
4. Re-rank as costs move; this is a snapshot of the 2026-06-09 task_state.
