# multistage_key_first_compositional.md

Multi-stage ARC-style rules are best written as explicit tensor compositions of the form `K = f(X)`, `S = g(X, K)`, `Y = h(X, K, S)`: a key is extracted first, later scoring or propagation depends on that key, and only then is the final grid assembled. ARC-AGI-2 explicitly frames this family of difficulty as multi-rule composition, multi-step composition, and contextual rule application, while Hodel’s ARC-DSL shows the same idea in symbolic-program form via compositions such as object extraction → filtering → painting.

*Assumption used throughout: when the card says reach-`N` propagation and treats one-shot `MaxPool(2N+1,2N+1)` as equivalent, I interpret propagation as square / Chebyshev-radius expansion on a fixed 2D grid; if the intended reach is Manhattan / cross-shaped, repeated cross-kernel updates are the faithful construction instead. This equivalence note is an engineering inference, not a published ONNX claim.*

## Inventory of multi-stage patterns

The pattern inventory below is a synthesis grounded in ARC-AGI-2’s published categories of multi-rule composition, multi-step composition, and contextual gating, plus Hodel’s published examples of sequential object extraction, filtering, and painting.

| Pattern | Stage 1 | Stage 2 | Stage 3 | Composition style | Use case |
|---|---|---|---|---|---|
| Key-first + count-score | Extract a key object/cell/property `K` by histogram, area, position, or marker match | Score every candidate against `K` by count, color, shape, distance, or feature similarity | Select top-`K` or argmax survivors and render them | Global key tensor drives later ranking | “Find the dominant object, then keep the objects most similar to it” |
| Anchor + propagate | Detect anchor(s) by location, marker, border status, or unique feature | Expand anchor influence into a seed mask or directional control mask | Paint propagated region or use it to gate original content | Context object becomes a control tensor | “Start from the central/marked object, then extend or flood outward” |
| Filter-then-paint | Build a first-pass candidate mask from static predicates | Apply a second filter that depends on stage-1 outputs or contextual cues | Paint only the surviving mask | Cascaded masking with late rendering | “Find all candidates, then keep only those compatible with the chosen key” |
| Group-then-aggregate | Partition objects/cells by a property such as color, hole count, or size bucket | Aggregate within each group by count, sum, mean, or max | Reduce groups to one output mask/object/color | Group tensor feeds reduction tensor | “Group by color first, then select the largest or most frequent group member” |
| Two-stage with reach-`N` propagation | Mark seeds or anchors | Expand seeds for exactly `N` steps or radius `N` | Compose expanded mask with original grid or a paint color | State tensor is updated before final write-out | “Mark eligible seeds, then dilate them by a fixed radius” |

The practical signature of all five families is the same: stage-2 inputs are not just the original grid, but also stage-1 results. That is exactly the kind of “previous placements determine later placements” dependency ARC-AGI-2 calls out as multi-step composition, and it is also the kind of intermediate-variable program that ARC-DSL examples make concrete.

## ONNX-static staging patterns

In ONNX, this is still a single static graph. ONNX defines models as graphs of operator nodes connected by intermediate tensors, and an operator in a graph follows the newest schema whose version is less than or equal to the graph opset. For an opset-18 model, the relevant definitions here are ArgMax-13, TopK-11, Conv-11, MaxPool-12, Where-16, MatMul-13, and ScatterElements-18.

**Tensor passing.** The default staging pattern is simply: stage-1 tensors become inputs to stage-2 nodes. Example chain: `Equal -> ReduceSum -> ArgMax` to get a key; `Equal/MatMul/Where` to compute key-conditioned scores; `TopK` to keep survivors; `ScatterElements` or one-hot `MatMul` to rebuild a spatial mask; `Where` to assemble the output. Nothing about this requires `Loop` or dynamic branches; only the values are data-dependent, not the graph structure.

**Mask-then-Where.** If stage 1 yields a boolean mask, stage 2 can convert that into conditional computation with `Where`, using neutral tensors such as zeros or large negative sentinels. Because `Equal` and `Where` both support NumPy-style broadcasting, this scales naturally from `[H,W]` masks to `[M,H,W]` banks of candidate masks.

**MatMul-as-selection.** `Gather` is the canonical indexing operator, but when you want to stay in dense linear algebra, a runtime index can be converted into a one-hot selector and multiplied against a candidate bank: if the bank is `[B,D]` and the selector is `[1,B]`, then `MatMul(selector, bank)` returns the selected row. This is often attractive when the key domain is tiny, such as 10 colors or 8 directions. Its compute cost is dense, though: roughly `B*D` MACs instead of an index read. `Gather` remains cheaper when plain indexing is acceptable.

**Pre-computed combinations.** If stage-1 has a finite small domain known at graph-build time, you can pre-build every branch and choose afterward by masking. For `B` possible stage-1 outcomes and branch body cost `C_branch`, the static cost is approximately `C_shared + B*C_branch + C_select` if every branch is run in parallel; peak intermediate memory is likewise roughly `B` times a branch activation if those tensors are all materialized. This is an engineering inference from ONNX’s static DAG execution model, but it is usually the right way to think about the budget.

For ARC-sized grids, the key question is domain size. Enumerating 10 color branches is often fine. Enumerating 900 possible key cells on a `30×30` grid is usually not fine unless each branch is extremely tiny and reduced immediately. In practice, “compute all branches in parallel” is preferable when `B` is small, while explicit one-hot selection over a bank is preferable when the branch bodies are large but the selected bank rows are compact. This tradeoff is inferred; ONNX does not publish a dedicated guideline for it.

A minimal reference for the canonical three-stage pattern is:

```python
# numpy reference for a 3-stage key-first rule
hist = (grid[..., None] == colors).sum((0, 1))     # stage 1: key extraction
key = hist.argmax()

score = np.where(obj_color == key, obj_count, -10**9)  # stage 2: key-conditioned scoring
keep = np.argpartition(score, -K)[-K:]

seed = obj_masks[keep].any(0)                      # stage 3: output assembly
out = np.where(seed, paint_color, grid)
```

The ONNX analogue is a straight tensor chain: `Equal -> ReduceSum -> ArgMax -> Equal/Where -> TopK -> ScatterElements or one-hot MatMul -> Where`. `TopK` takes `K` as a 1-D `int64` tensor with one positive value, and `ScatterElements` in opset 18 can use reductions including `max/min`, which is convenient for building a binary keep-mask from top-`K` indices.

## Key extraction primitives

`ArgMax` is the cleanest exact key extractor for categorical counts. In opset-18 models it uses the ArgMax-13 schema, returns integer indices, and exposes `select_last_index`, which is valuable when a task’s tie-break must be “first occurrence” versus “last occurrence.” That makes it a good fit for “most common color,” “largest pre-materialized object,” or “rightmost maximal score” rules.

`TopK` is the right primitive for “top-`K` objects,” “`k`-th most common color,” and “`k`-th smallest extent.” In the TopK-11 schema, `largest` switches between largest and smallest selection, `sorted` controls output ordering, and `K` is supplied as a one-element `int64` tensor. For deterministic ARC behavior, if ties matter beyond value ranking, it is safest to build a secondary numeric tie-break into the score tensor before `TopK` runs; that advice is inferred, because the spec documents value ranking but not a special ARC tie policy.

Centroids are best computed as weighted coordinates, not by premature floating-point means. `ReduceSum` is exact for integer numerators and masses, while `ReduceMean` is defined but yields undefined behavior on empty reductions. For ARC grids, a robust pattern is to keep `(sum_row, sum_col, mass)` as integer tensors and compare cross-products or divide only once, late, after a nonzero-mass check. That avoids small float drift when centroids are used only for comparisons or ranking.

Marker detection is naturally encoded as template matching: one-hot encode the relevant colors, run a constant `Conv` with a fixed template kernel, then use `Equal` or `GreaterOrEqual` to detect exact or partial hits. This keeps the graph static and avoids ad hoc indexing. The main caveat is that exact multi-color templates should be matched on one-hot channels, not raw color IDs, to avoid aliases; that alias warning is an engineering inference rather than a published ONNX note.

A practical caveat for “largest connected component”: ONNX has rich tensor ops, but no dedicated standard connected-components primitive in the operator set cited here. In static ARC graphs, “largest CC” usually means either a task-specific object bank already compiled into masks, or a finite set of candidate masks extracted by earlier deterministic logic. That limitation is inferred from the operator inventory and from the way ARC-DSL / RE-ARC solve tasks by explicit object programs and generators.

## Reach-N propagation

The costs below assume a single binary mask on a fixed `30×30` grid. They are derived from ONNX operator semantics plus direct arithmetic on kernel sizes; comparison counts and bit-operation counts are engineering estimates, while MAC counts are exact for the stated convolution formulas.

| Strategy | N=3 cost | N=7 cost | N=14 cost | Ops | Notes |
|---|---:|---:|---:|---|---|
| N-step 3×3 depthwise convolution dilation | `3 * 30 * 30 * 9 = 24,300` MACs | `56,700` MACs | `113,400` MACs | `N × (Conv + threshold)` | Faithful for genuinely sequential state updates; with a square `3×3` kernel it matches iterative square dilation. |
| One-shot `MaxPool` with kernel `(2N+1, 2N+1)` | `44,100` max-comparisons, `0` MACs | `202,500` max-comparisons, `0` MACs | `756,900` max-comparisons, `0` MACs | `1 × MaxPool` | Equivalent to square / Chebyshev-radius dilation in one pass (inferred); very attractive if the score counts MACs but not comparisons. |
| Separable `1×(2N+1)` then `(2N+1)×1` pooling | `12,600` max-comparisons, `0` MACs | `27,000` max-comparisons, `0` MACs | `52,200` max-comparisons, `0` MACs | `2 × MaxPool` or `2 × Conv` | Same square reach at `O(N)` window cost instead of `O(N^2)`; usually the best static compromise for large `N`. |
| Bit-packed propagation | about `540` word shifts/ORs + pack/unpack | about `1,260` word shifts/ORs + pack/unpack | about `2,520` word shifts/ORs + pack/unpack | pack + `BitShift` + `BitwiseOr` + unpack | Lowest arithmetic budget, but more graph plumbing and higher EP-support risk; counts are inferred word-level estimates. |

Two engineering consequences matter for NeuroGolf-style scoring. First, if your score is `params + intermediate_memory_bytes + MACs`, one-shot or separable `MaxPool` can dominate iterative `Conv` even when wall-clock runtime is not obviously better, because pooling contributes no MACs in the strict arithmetic sense while convolution does. Second, if you need *exactly* the state after each step because later stages inspect intermediate fronts, iterative `Conv`/threshold or iterative pooling is the right choice; a one-shot `(2N+1)` kernel only gives the final radius-`N` envelope.

If the desired reach is Manhattan rather than square, repeat a cross-shaped kernel for `N` iterations. A one-shot square `MaxPool` is then not equivalent; it over-expands corners. That geometric warning is inferred from the pooling and convolution definitions, not directly stated by ONNX.

Bit-packed propagation is legal in opset 18 because `BitShift` is available and `BitwiseOr` is standardized in opset 18, but it is the least portable option across execution providers. ONNX Runtime notes that unsupported ops may fall back to CPU, which can erase the speed benefit even if the model stays functionally valid.

## Composition reasoning and verification

The cleanest correctness method is stage-by-stage testing: temporarily expose `key_idx`, `score_vec`, `seed_mask`, `prop_mask`, and final `Y` as outputs, then compare each against a small NumPy reference. This aligns with ONNX’s model-as-graph semantics, where intermediate tensors are explicit node outputs, and with ARC’s task format of a few visible input/output demonstrations per task. Promoting intermediates to outputs is an engineering pattern rather than a cited API recipe, but it follows directly from ONNX graph structure.

For visible-pair verification, every demonstration pair should check three things, in order: exact stage-1 key identity, exact stage-2 survivor set, and exact final output grid. That decomposition prevents a later paint step from hiding an earlier selection bug. ARC-AGI tasks are explicitly defined as a handful of demonstration pairs plus one or more test inputs, so stagewise replay over those visible pairs is the natural harness.

A NumPy “golden program” is still the easiest oracle. Hodel’s ARC-DSL and RE-ARC work are strong precedents for expressing ARC transformations as compact symbolic/algorithmic programs and example generators, and they are useful templates for the reference implementation you compare the ONNX graph against. In practice, compare the unoptimized graph first, then the ORT-optimized graph second, because ONNX Runtime deliberately rewrites graphs through constant folding and fusions, and its own quantization docs recommend separating optimization from quantization when debugging mismatches.

For numeric stages, use exact equality for booleans and integers, and tolerance-based checks only where you knowingly introduced floats, such as centroid division or normalized similarity scores. This matters because `ReduceMean` on empty sets is undefined, while count-based integer logic remains exact.

## Compression strategies

Stage fusion helps only when the “key” does not need to exist as a separately inspected tensor. For example, if stage 2 only asks “does candidate color equal winning color,” you may be able to rewrite histogram + equality into a direct matrix score over candidate color one-hots without ever materializing a scalar key index. But when later stages truly consume the selected key object, fusion usually stops being algebraic and becomes branch enumeration instead. The general idea is a synthesis; ONNX Runtime’s documented fusions are simpler graph rewrites such as Conv+Add, Conv+Mul, MatMul+Add, and related node combinations.

Constant folding is the easiest win whenever any stage depends only on initializers. ONNX Runtime’s basic graph optimizations explicitly include constant folding, and its troubleshooting guide notes that leaving initializers in graph inputs can block const-folding and fusion because those inputs are no longer treated as constant. For hand-built ARC graphs, that means all immutable coordinate grids, permutation banks, template kernels, and one-hot codebooks should be model initializers, not overridable inputs.

Per-stage quantization is most useful for the numeric heavy parts of the graph, namely score banks and propagation kernels built from `Conv` or `MatMul`. ONNX Runtime supports both QDQ and QOperator-style quantized graphs, and the standard operator set includes `QLinearConv`, `QLinearMatMul`, `QuantizeLinear`, and `DequantizeLinear`. In staged ARC rules, the usual split is: keep boolean / index / control subgraphs (`Equal`, `ArgMax`, `TopK`, `Where`) in their natural domains, and quantize only the dense numeric subgraphs where arithmetic dominates.

For this project’s cost model, quantization is attractive only when it reduces real memory and arithmetic in the numerically heavy stages. It usually does not help the pure-control parts much, because `ArgMax`, `TopK`, and mask plumbing are already low-parameter and mostly logic-bound. That last point is an engineering inference from the operator mix, not a published ORT rule.

## Worked example for task285-style graph

A correction is needed here: the public NeuroGolf competition states that its 400 files are drawn from the ARC-AGI public training set v1, community visualization notebooks map train task 285 to ARC task `4e7e0eb9`, and public hint notebooks describe `4e7e0eb9` as a symmetry-expansion task over sparse colored seed pixels; therefore the card’s label “keyfirst + countscore + two-stage `K=23` with reach-7 propagation” should be treated as a project-side factorization, not a published canonical annotation.

Under that project-side factorization, a practical opset-18 graph sketch is:

**Input encoding.** Keep the raw grid as `[1,1,30,30]` for late assembly, and also derive a 10-channel one-hot color tensor with `Equal` against a constant color codebook. If the task-specific builder already knows a small candidate object bank `obj_masks[M,30,30]` and object descriptors `feat[M,d]`, treat those as constant initializers or deterministic early-stage outputs. The object bank assumption is inferred, but it is the most realistic way to avoid generic CC extraction in a strict static graph.

**Stage 1: key extraction.** Build key scores from count-like features. The simplest version is a histogram or per-object area vector `count[M]`, then `key_idx = ArgMax(count)`. If ties matter, use `select_last_index` or perturb with a fixed lexicographic epsilon encoded in the score initializer. Convert `key_idx` to a one-hot selector either by equality against `arange(M)` or by scattering a `1` into a zero vector.

**Stage 2: count-score against the key.** Extract the key feature row with one-hot `MatMul`, then score all candidates against it. A canonical static form is `score = feat @ W @ key_feat`, or, for exact count matching, `score_i = -abs(count_i - count_key) + bonus_same_color + bonus_same_shape`. The important point is that `count_key` and `key_feat` are tensors produced by stage 1, so stage 2 is still static but data-dependent. If `M` is small, this dense banked formulation is usually cheaper than branching over all possible keys.

**Stage 3: `K=23` selection.** Supply `K` as a constant one-element `int64` tensor `[23]` to `TopK(score, K)`. Convert the returned indices into a keep-vector with `ScatterElements` or a one-hot equality bank, then reduce the selected object masks into a single binary seed mask of shape `[30,30]`. In notation: `top_idx -> keep[M] -> seed = reduce_max(keep[:,None,None] * obj_masks, axis=0)`. `TopK` and `ScatterElements` make this pattern particularly clean in opset 18.

**Stage 4: reach-7 propagation.** If the target propagation is square radius 7, the lowest-MAC static implementation is either a single `MaxPool` with kernel `15×15`, stride 1, same-size padding, or two separable pools `1×15` then `15×1`. If you care about exact seven-step intermediate fronts rather than just the final radius-7 envelope, use seven repeated `3×3` updates instead. Under a MAC-count score, the pool version is usually preferred.

**Output assembly.** Use `Where(prop_mask, paint_value, X)` if the propagated region should overwrite the original grid, or combine `prop_mask` with the unpropagated `seed` when only the front or only the final region should be painted. Because `Where` broadcasts, both scalar paint values and full replacement tensors are easy to support.

The dominant cost term in this sketch is usually not `ArgMax` or `TopK`; it is the object bank itself, especially if `M` approaches the number of cells. When `M` is small and task-specific, the graph is compact. When `M = 900`, the keep-vector and object-mask bank become the main parameter and memory burden, and a color/domain-level factorization is usually better than a cell-level one. This budget observation is inferred from the static-bank construction.

## Decision tree

Use the table below as a compile-time chooser. The mapping is a design synthesis built from the published ARC-AGI-2 categories and the ONNX operator semantics cited above.

| Rule structure | Stage-1 primitive | Stage-2 primitive | Output assembly |
|---|---|---|---|
| Key is a small categorical value such as color, direction, or marker type | `Equal + ReduceSum + ArgMax/TopK` | broadcast `Equal`, `Where`, or one-hot `MatMul` against a small bank | `Where` paint or `TopK` survivor mask |
| Key is one object from a small compiled object bank | object score vector + `ArgMax` | one-hot `MatMul` to extract key features; dense score over candidate bank | `ScatterElements`/one-hot keep-vector, then mask reduction |
| Need one filter, then a context-dependent second filter | first-pass mask with `Equal`, range tests, or template `Conv` | `Where` with later key/context tensor | late paint with `Where` |
| Need radius-`N` growth from seeds | seed mask from marker/key selection | `MaxPool`, separable pooling, or repeated `Conv` | overwrite / union with original grid |
| Need grouping before selection | group one-hot tensor or grouped masks | `ReduceSum` / `ReduceMean` / `ArgMax` inside groups | paint representative group or reduce to one mask |
| Need truly stepwise dependence where state after step `t` affects step `t+1` | initial seeds/anchors | repeated fixed updates, unrolled for known `N` | final state or final front only |

**Open questions / limitations.** Two pieces remain inference-heavy because I did not find a primary published source for them: a generic no-`Loop` connected-components construction in standard ONNX, and a public source that annotates NeuroGolf task285 with the exact internal label “keyfirst + countscore + `K=23` + reach-7.” I therefore treated both as design assumptions and marked them explicitly when used.

**Verification checklist**

- [x] At least four distinct multi-stage patterns are surveyed in the inventory table.
- [x] Static-graph encoding is explicit, and the note keeps everything within a single opset-18 ONNX DAG without `Loop`.
- [x] Reach-`N` propagation costs are tabulated for `N ∈ {3, 7, 14}` on a `30×30` grid.
- [x] Composition correctness is described with stagewise testing, visible-pair replay, and NumPy golden-code comparison.
- [x] A task285-style graph is sketched, with a correction that distinguishes the public task mapping from the project-side decomposition.
- [x] All recommended primitives are compatible with an opset-18 model, and ONNX Runtime’s compatibility policy covers opsets 7 and higher; thus opset 18 is within ORT’s supported range. 