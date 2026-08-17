# Insight: a single fused Conv can be UNbeatable — decomposition/rank-factoring loses on `memory+params`

## What I discovered
On the `memory + params` cost model (`neurogolf_utils.calculate_memory`/`calculate_params`), a graph that already expresses its whole computation as **one fused Conv whose output is named `output`** is often at the *forced floor*, and the intuitive "factor the sparse/rank-1 kernel / drop zero channels / split into arithmetic" rewrite makes cost **worse, not better**. Two independent reasons:

1. **`params` counts ELEMENTS, not nonzeros or rank** (`calculate_params`: `math.prod(init.dims)`). A dense Conv weight `[M, C/g, kh, kw]` costs its full element count no matter how sparse (task352: "27 of 450 nonzero" → still 450). ONNX has no sparse-Conv. Separable/rank-1 factoring only helps if it literally shrinks the *single* weight tensor's dims — and it can't, because factoring an 8-neighbour ring into row×col needs **two** conv nodes.

2. **`calculate_memory` charges every node-output tensor except the two literally named `input`/`output`** (line 234). The minimum spatial intermediate on a 30×30 grid is `[1,1,30,30]` = **900 B even as BOOL/INT8**. So *any* ≥2-node decomposition adds ≥900 B — which for a small graph (task352 bar = 460) alone exceeds the entire budget. A single fused op has zero charged intermediates.

Corollary / rule of thumb: **before attempting a structural "rank-factor / separable-split / channel-drop" rewrite of a single-Conv graph, check the bar against 900 B.** If bar < ~900 + realistic param count, the single fused Conv is the floor — declare a WALL, don't build variants.

## Where I encountered it
task352 (batch48, GroupedConvRankFactor). Bar 460 = one grouped Conv `W[10,5,3,3]`(450)+`b[10]`(10), mem 0. Prompt asked to factor the 2 rank-2 kernels to reach ~300 (a params-only mental model). Measured:
- rank-factored arithmetic graph (Split+box-Conv+Mul/Add/Sub+Concat), 17 params, byte-equal to LIVE, visible 266/266 → profile **64,817** (mem 64,800).
- minimal 2-node (Conv → Add-bias) → **36,460** (one F32 intermediate).
Full proof: `The_400_Tasks/Task_by_Task/task352/hard_records/rank_factor_wall.md`. Build scripts: `The_400_Tasks/Task_by_Task/task352/prototypes/{build_v1,build_2node}.py`.

## Why it matters
Avoids a whole class of wasted STRUCTURAL rewrites: prompts that estimate a floor from a params-only count of nonzero weights will systematically overstate the achievable saving whenever the live graph is already a single fused Conv. The memory charge on 30×30 intermediates is the dominant term for small graphs.

## What's missing from the library
traps.md §GRADER-GAP / §39 cover `max(static,runtime)` per-tensor charging, but I did not find an explicit statement of the **"single fused Conv vs decompose" floor** — i.e. that params count elements (sparsity/rank irrelevant) AND that the 900 B minimum-intermediate makes decomposition lose for small-bar tasks. Worth a named rule.

## Sorter note: traps.md — add as new §"fused-single-node floor": params=elements (sparsity/rank irrelevant, no ONNX sparse-Conv) + every non-`input`/`output` intermediate ≥900 B on 30×30 ⇒ decomposing a single fused Conv raises cost; check bar vs ~900 B before any rank-factor/separable/channel-drop rewrite. Cross-link task352 hard_record.
