# J.11 — Differentiable top-K masking (selection masks without `NonZero`)

**Status:** synthesis. Maps "select the K most-X cells/objects/colors" pattern onto banned-NonZero/banned-Unique ONNX op set.
**Cluster:** logic / discrete / combinatorial.

## TL;DR

- The brief frames this as a workaround for banned `NonZero`/`Unique`. **The simpler tool is already there:** `Hardmax` (opset 1) returns a one-hot mask of the argmax in a single op. For K=1, no `TopK`+`OneHot`+`ReduceSum` plumbing needed. **Use `Hardmax` for K=1 selection. Use `TopK` only for K≥2.**
- The K≥2 mask recipe is: `TopK(axis=A, k=K)` → indices → `OneHot(depth=size_of_axis_A)` → `ReduceSum(axis=last)` to collapse the K one-hot vectors into one mask. Total cost dominated by `OneHot` (which materializes a tensor with one extra dim of size = axis-length) + `ReduceSum`.
- **Real cost vs naive Conv approach:** for "select 3 most-frequent colors" type tasks, `TopK` over a [1,10] vector of channel-counts costs negligible (≤30 cmp); the dominant term is the *upstream* `ReduceSum` that produced the counts (1 ReduceSum = 9000 MACs by onnx-tool's "treat reductions as numel(input)" rule). Even the full pipeline beats Conv 1×1.
- **`TopK` over the spatial axis (e.g., "the 5 brightest pixels")** has a hidden cost: ONNX's `TopK` returns *sorted* values along the chosen axis, but the index tensor is over indices in that *single* axis — to convert it into a 2D position mask in [H,W] you need to either (a) operate on a flattened version then `Reshape`, or (b) do per-row TopK. Both work; flattening is the cheap path.
- **The cleanest single-pixel selection in this competition is `Hardmax` over the channel axis** — it produces exactly the [1,10,30,30] one-hot output the validator expects. For "give me the dominant color at each pixel" this is *one* op.

## How `TopK`, `Hardmax`, `OneHot`, and `ArgMax` compose in ONNX

Verified opset support (all opset 10-compatible per the starter; some are even available in opset 1):

| Op | Min opset | Inputs → Outputs | Cost (onnx-tool's accounting) |
|---|---|---|---|
| `ArgMax` | 1 (opset 11+ adds `select_last_index`) | tensor → int64 indices | reduction; ~numel(input) cmp ops |
| `Hardmax` | 1 (opset 11/13 changes axis semantics) | float tensor → float one-hot at argmax | single op; single-pass |
| `TopK` | 1 (opset 10 adds K-as-input, opset 11 adds `largest`/`sorted`) | tensor + K → values, indices | reduction-style; numel(input) cmp ops + sort overhead |
| `OneHot` | 9 | indices + depth + values → one-hot tensor with extra dim | numel(output) — the new dim makes this expensive |
| `ReduceSum` | 1 (opset 13 makes `axes` an input) | tensor → reduced tensor | numel(input) ops |
| `Gather` | 1 | data + indices → selected slices | shape-only (cost 0) per COVERED |

The forbidden ones (`NonZero`, `Unique`) are unrelated: they're forbidden because they produce **dynamic-sized output** (the number of non-zero entries is data-dependent), which would break static-shape enforcement. `TopK`'s K is a runtime input but the output shape is determined by it; if K is an initializer, the output shape is fully static and acceptable.

## The selection-mask recipe (concrete pipelines)

### Recipe A: K=1 single dominant element, in one op

For "find the index of the max along axis A and produce a one-hot mask there":
```
Hardmax(input, axis=A)
```
Output: same shape as input, with exactly one 1 along axis A per (batch, …, …) slot, zero elsewhere. **Single op, ~9,000 MACs over the [1,10,30,30] tensor.**

Note Hardmax's quirk: it coerces the input to 2D under the hood (a_0 × a_{1..n-1}), which means the "axis" parameter has the older "axis-of-coercion" semantics in opset 1 (different from opset 11). Use opset 13+ if you want the modern axis-as-reduction-axis semantics. For one-hot-along-channel of a [1,10,30,30] tensor with axis=1 in opset 13+, this Just Works.

### Recipe B: K≥2 masks via TopK + OneHot + ReduceSum

For "give me a binary mask marking the K largest entries along axis A":
```
values, indices = TopK(input, K=k, axis=A, largest=1, sorted=0)
# indices shape: same as input but with axis A replaced by k
mask_kx = OneHot(indices, depth=size_of_axis_A, values=[0.0, 1.0], axis=A)
# mask_kx has an extra axis of size k after the one-hot expansion
mask = ReduceSum(mask_kx, axes=[axis_of_K_dim], keepdims=0)
```

Cost breakdown for "select the 3 most-frequent colors" — applied to a [1,10] count vector after a `ReduceSum` over the spatial axes:
- Initial `ReduceSum(spatial)`: input `[1,10,30,30]` → output `[1,10]`. Cost ~9,000 MACs.
- `TopK(K=3)` over a 10-vector: ~10 cmp ops, negligible.
- `OneHot(depth=10)`: input `[1,3]` → output `[1,3,10]`. 30 ops.
- `ReduceSum` along the K dim: 30 ops.
- Multiply original `[1,10,30,30]` by the broadcast mask `[1,10,1,1]`: 9,000 MACs.

**Pipeline total: ~18,000 MACs.** Single Conv 1×1: 90,100 MACs. **Win: 5×.**

### Recipe C: largest connected object — does this even fit?

The brief mentions "pick the largest object." This requires connected-components labelling, which is *exactly* the kind of dynamic-iteration the banned-`Loop` rule forecloses. **Without `Loop`, you cannot label arbitrary connected components in one ONNX graph.** What you *can* do:

- For tasks where there's a known small bound on object count (≤10), unroll a fixed-iteration flood-fill via `MaxPool 3×3` repeated K times. K iterations of MaxPool 3×3 = K × 9 × 9000 = 81,000K MACs. Already pricey.
- For tasks with a *unique* foreground object on a uniform background, the "object" is just the non-background mask: `(input != background_channel)` → 1 `Equal` + maybe 1 `Sub`. ~9000 MACs.
- For "find the connected component containing the densest cluster," use `Conv` with a Gaussian-like kernel + `Hardmax` on the result: 1 Conv + 1 Hardmax. The Conv blends nearby pixels; the Hardmax picks the spatial peak. Useful approximation when the answer doesn't depend on exact connectivity.

**`TopK` does not solve the connected-components problem.** The brief's framing is over-optimistic for that case. `TopK` only works when "object" can be defined by a per-cell or per-channel scalar score, not by topology.

## Concrete ARC task examples where this pays off

- **"Output is the input with all but the most common non-background color removed"**: `ReduceSum(spatial)` → background-channel mask via `Slice` removed → `Hardmax` over remaining 9 channels → broadcast multiply with input. ~28,000 MACs. Conv-based: ~90,100. Win.
- **"Output the K colors that appear at least once"**: `Greater(ReduceSum(spatial), 0)` over channels → 1 `Greater` + 1 `Cast` + 1 `Mul` = ~27,000 MACs. No `TopK` needed because we don't need ordering, just presence. Even cheaper than recipe B.
- **"Highlight the cell whose color is most frequent"**: full pipeline above.
- **"Pick the largest contiguous object"**: not solvable cleanly without `Loop`. Skip with `TopK`; treat as a different problem.

## Key trick: `OneHot` is cheap for small `depth`, expensive for large

The `OneHot` op creates a new dimension of size `depth`. With a per-pixel index tensor of shape `[1,30,30]` and `depth=10`, the output is `[1,30,30,10]` = 9,000 elements, which costs 9,000 MACs. **But with `depth=30` for a row index, output is `[1,30,30,30]` = 27,000 MACs** — 3× more expensive. Always pick the *smallest* axis to one-hot over.

For ARC-style channel selection (depth=10, the color axis), `OneHot` is always cheap. For position selection (depth=30 or 900), do a flatten + smaller-depth trick if possible.

## Discrepancies / flags vs COVERED

- **No direct conflict with COVERED.** The pattern library (`_PATTERNS`) currently covers identity / transpose / hflip / vflip / rot180 / color-perm. None overlap with TopK-based selection.
- **DISCREPANCY (carried from J.6):** `OneHot` MAC accounting in onnx-tool is unverified by me. The op is "assignment-only" in concept (no float arithmetic), but onnx-tool may charge it numel(output) under the default-profiler rule. This affects Recipe B's cost calculation — if `OneHot` costs 0, total drops below 9,000 MACs for the 3-color task. **Action item: open onnx_tool 1.0.1 source, find the OneHot profiler entry (or its absence), and verify.**
- **DISCREPANCY:** `Hardmax`'s output is documented in opset 1 as "1 for the *first* maximum value, 0 for all others" — i.e., **first-occurrence tiebreak, not last**. For ARC tasks where ties matter (two equally common colors), this could give wrong outputs vs `ArgMax(select_last_index=True)`. Worth empirical check on the specific task.
- **Carried-forward flag:** the `_PATTERNS` library could absorb a "Hardmax-channel" pattern that solves any task whose output is uniformly the dominant channel of the input (with some tasks accepting multi-pixel handling).

## Actionable design rules

1. **For K=1 selection, always use `Hardmax`.** Don't compose ArgMax + OneHot.
2. **For K≥2, the cost is dominated by upstream reductions, not by `TopK` itself.** Optimize the upstream `ReduceSum` first.
3. **`OneHot` cost scales with `depth`** — pick the smallest axis to one-hot over (channel axis = 10, not position).
4. **`TopK` does not solve connected-components.** For "largest object" tasks where topology matters, use repeated Conv/MaxPool relaxations or skip the task class entirely.
5. **`Greater`-against-0 + `Cast` is cheaper than `TopK` when you only need "did this appear at least once"** (no ordering required).
6. **The K input to `TopK` must be an initializer** (not a runtime tensor) to keep the output shape static. The validator's static-shape check will reject dynamic-K otherwise.

## Sources consulted

- ONNX op references (1.21): `TopK` (opset 10/11), `Hardmax` (opset 1/11/13), `ArgMax` (opset 1/11/12), `OneHot` (opset 9/11), `ReduceSum` (opset 1/11/13).
- onnx-tool `Profile.md` — MAC accounting rule.
- ONNX issue #1830 (microsoft/onnxruntime, 2019): `TopK` is slow at runtime in onnxruntime (~30× pytorch), but **runtime speed doesn't affect the MAC score** — onnx-tool counts only MACs, params, memory.
- ONNX `onehot.py` reference impl (onnx/onnx, main): confirms the "extra dim of size depth" output shape.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from J.11-differentiable-topk-masking.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.11 — Differentiable top-K masking (selection masks without `NonZero`)

## Summary

The "differentiable top-K" line of work spans soft-Top-K relaxations (Cuturi et al. 2019, "Differentiable Sorting Networks"; Xie et al. 2020, "Differentiable Top-K Operator with Optimal Transport") and discrete approximations. The NeuroGolf application is narrower: build a *binary mask* selecting the K largest entries along an axis, without using the banned `NonZero`/`Unique` ops. The brief frames this as `TopK` → `OneHot` → `ReduceSum`. The actual cheapest realization for K=1 is even simpler: ONNX's `Hardmax` op produces a one-hot mask of the argmax in a single op (specified since ONNX opset 1).

The headline empirical result is that **`TopK`, `ArgMax`, `OneHot`, `Gather`, `TopK` indices, and `ScatterND` all cost 0 MACs in onnx-tool 1.0.1** (verified by reading the source and probing the library). The only op that charges MACs in this pipeline is `Hardmax` itself (9,000 MACs via `PWNode`) and any upstream `ReduceSum` (9,000 MACs). The pipeline cost is governed almost entirely by what you do *before* the selection — the selection itself is essentially free.

## Project mapping

The empirically verified cost table:

| Op | MACs (onnx-tool 1.0.1, [1,10,30,30] context) | Notes |
|---|---|---|
| `Hardmax` (axis=1) | 9,000 (extends `PWNode`) | one-hot mask of argmax in a single op |
| `ArgMax` (axis=1, no keepdim) | 0 (no `profile()` override) | empirical: macs=0 |
| `OneHot` (depth=10, axis=-1) | 0 (no `profile()` override) | empirical: macs=0 |
| `TopK` (k=3, axis=1) | 0 (no `profile()` override) | empirical: macs=0 |
| `Gather` along channel axis | 0 | empirical: macs=0, params=10 |
| `ReduceSum` along spatial axes | 9,000 (extends `ReduceMeanNode`) | empirical confirmed |
| `Greater` against scalar | 9,000 (`CMP_MACS=1`) | empirical confirmed |
| `Mul` to apply mask | 9,000 | |
| `Conv 1×1` 10→10 (baseline) | 99,000 | empirical |

**Pattern A: K=1 selection of dominant channel per pixel.** Single-op realization: `Hardmax(input, axis=1)` produces `[1,10,30,30]` with exactly one 1 along axis 1 per (h,w). Cost: **9,000 MACs, 0 params, 36,000 memory**. Useful for any task whose output is "color the image with the most-prevalent class at each pixel" or "for each pixel, emit only the dominant input channel."

**Pattern B: K=1 selection of dominant element across the whole grid (the "find the brightest pixel" pattern).** `Reshape` to `[1,10,900]` then `Hardmax(axis=2)` then `Reshape` back. Both Reshapes are 0 MACs; Hardmax is 9,000. Total **9,000 MACs**.

**Pattern C: K≥2 binary mask (e.g., "the 3 most-frequent colors").** Pipeline:
1. `ReduceSum(input, axes=[2,3])` → `[1,10,1,1]`. Cost: 9,000 MACs.
2. `TopK(reduced, k=3, axis=1)` → values `[1,3,1,1]`, indices `[1,3,1,1]`. Cost: 0 MACs.
3. `OneHot(indices, depth=10, axis=...)` → `[1,3,10,1,1]` or similar. Cost: 0 MACs.
4. `ReduceSum` along the K dim → `[1,10,1,1]` membership mask. Cost: ~30 MACs (numel of the reduced input is small here).
5. `Mul(input, mask)` to filter. Cost: 9,000 MACs.

Total **~18,000 MACs** + tiny initializers. Versus Conv 1×1's 99,000: **5.5× win**.

**Pattern D: "Pick the largest connected object."** This is *not* solvable with `TopK` alone. Connected-components labeling is inherently iterative and the validator bans `Loop`/`Scan`. Workarounds: (a) repeated `MaxPool 3×3` to "spread" connectivity — k iterations cost k × ~81,000 MACs (each MaxPool is roughly numel(out) × 9 cmp = 81,000); (b) approximate "find the densest region" via Conv-with-Gaussian-kernel followed by `Hardmax` — 1 Conv + 1 Hardmax. Neither truly solves connected-components; both are heuristics. **Do not commit to TopK as the connected-components solution.**

**Pattern E: "Output only one specific color, drop the rest" (a degenerate K=1 case).** Cheapest realization: `Slice` along channel axis (cost 0) + `ScatterND` or `Concat` of zeros for the other channels. **Or just `Mul(input, mask)` where mask is a one-hot constant initializer of length 10 — 1 op, 9,000 MACs, 10 params.** Actually cheaper: a `Gather` with index `[k]` followed by a broadcast — 0 MACs.

**ARC task families that map cleanly to this:**
- "Output the input with all but the most-frequent color removed" → Pattern C.
- "Find the dominant color at each pixel" → Pattern A.
- "Output a single highlighted point at the brightest pixel" → Pattern B.
- "Pick the K colors that appear at least once" → cheaper than TopK: `Cast(Greater(ReduceSum, 0), float)` over channel-counts. ~18,000 MACs without ordering, no TopK needed.

**Failure modes.** (1) `Hardmax` first-occurrence tiebreak: when two channels have equal max values, the lower channel index wins. ARC ground truths might tiebreak the other way. Use `ArgMax(select_last_index=True)` + `OneHot` if you need last-occurrence (this changes nothing about cost — both ArgMax and OneHot are 0 MACs). (2) `OneHot` *output* memory scales with `depth` × every other dim. For depth=10 over `[1,30,30]` indices, output is `[1,30,30,10]` = 9,000 elements = 36,000 bytes contributing to `memory`. For depth=900 it's 90× larger. Always pick the smallest one-hot axis. (3) The `K` in `TopK` must be an initializer, not a runtime tensor — required for the static-shape rule (COVERED: "every dim must have `dim_value`").

## Sources

- onnx-tool 1.0.1 source: `HardmaxNode` extends `PWNode` with default `op_mac = ADD_MACS` (line 1909 of `node.py`); `PWNode.profile` returns `volume(out) × ratio × op_mac` = 9,000 for `[1,10,30,30]` (line 241–244, 228–232). `TopKNode`, `ArgMaxNode`, `OneHotNode`: no `profile()` override (lines 1846, 1520, 1035). All inherit `Node.profile` returning `[0,0]` (line 212–213).
- ONNX `Hardmax` op reference (`https://onnx.ai/onnx/operators/onnx__Hardmax.html`): "The operator computes the hardmax (1 for the first maximum value, and 0 for all others)" — first-occurrence tiebreak.
- ONNX `TopK` op reference (`https://onnx.ai/onnx/operators/onnx__TopK.html`): opset 10 made K an input (rather than attribute), opset 11 added `largest`/`sorted`. "Tensor of shape `[a_0, a_1, …, a_{axis-1}, k, a_{axis+1}, … a_{n-1}]` containing top K values."
- ONNX `OneHot` op reference (`https://onnx.ai/onnx/operators/onnx__OneHot.html`): opset 9. "The rank of the output tensor will be one greater than the rank of the input tensor."
- Cuturi, Teboul, Vert, "Differentiable Ranks and Sorting using Optimal Transport," NeurIPS 2019, arXiv:1905.11885. (Conceptual background — the *differentiable* angle is irrelevant to NeuroGolf since there's no training; the binary mask construction is what matters here.)
- onnxruntime `TopK` slowness issue: `https://github.com/microsoft/onnxruntime/issues/1830` — runtime is slow but the validator scores MACs, not wall-time.
- Empirical run results:

```
Hardmax (axis=1)                macs=9000   params=0    memory=36000
ArgMax (axis=1, no keepdim)     macs=0      params=0    memory=7200
ArgMax + OneHot pipeline        macs=0      params=2    memory=43208
TopK (k=3, axis=1)              macs=0      params=1    memory=32408
ReduceSum (spatial axes)        macs=9000   params=2    memory=56
```

## Contradictions / corrections vs COVERED

- No conflict with COVERED.
- **Self-correction:** I previously assumed `ArgMax` and `OneHot` charged ~9,000 MACs each (treating them as elementwise reductions/expansions). They're actually 0 in onnx-tool 1.0.1. This means the 3-most-frequent-colors pipeline (Pattern C) is cheaper than I estimated — closer to 18,000 than 30,000 MACs.

## Confidence

**High** for the cost numbers (read directly from source and verified empirically). **Medium** for the per-task-family hit rate predictions (depends on the actual distribution of ARC train/test/arc-gen tasks, which I haven't enumerated).
