# radius_encoded_weighted_cover.md

Assumption: costs below use same-padding, single-channel detectors on a `30×30` grid producing a `30×30` score map; `1 MAC ≈ 2 FLOPs` is used when converting compute to FLOPs, and numeric cost totals are arithmetic inferences from operator layouts.

## Inventory of patterns

A useful unifying form is a per-cell score `score(p) = Σδ w[δ] · x[p+δ]` or, for binary / one-hot templates, `score(p) = Σδ w[δ] · 1{x[p+δ] matches condition}` followed by a threshold or category selection. In the literature, the closest standard families are hit-or-miss template matching for foreground/background patterns, weighted local-distance masks such as chamfer neighborhoods, and radial / annular symmetry operators that accumulate evidence over discrete radii. Crosses, T-junctions, corners, circles, and multi-scale shells all fit this template-bank view.

| Pattern | Kernel shape | Per-cell weights | Use case | Cost (FLOPs) |
|---|---|---|---|---|
| Cross / star / hit-or-miss template | `3×3`, `5×5`, or `k×k` | Binary hit/miss, or signed cover / anti-cover weights | Detect `+`, `*`, corners, junctions, or exact foreground/background motifs | `O(k²)` per cell; dense `5×5` on `30×30` is `22,500` MACs ≈ `45,000` FLOPs |
| Distance-bounded square / diamond / disk coverage | Usually `k×k` with radius `r=(k-1)/2` | Uniform counts, Manhattan/Chebyshev weights, or chamfer weights like `3-4`, `5-7-11` | Paint cells whose neighborhood contains enough support within a chosen metric radius | Same dense `5×5` example: ≈ `45,000` FLOPs for one radius |
| Ring / annulus / radial symmetry | `5×5`, `7×7`, …, often one kernel per radius | Outer-ring positive weights, optional center penalties, or radial vote weights | Detect circular halos, annuli, or “edge-at-radius-r” patterns | `O(k²·R)` for `R` radii; one dense `5×5` annulus is ≈ `45,000` FLOPs |
| Asymmetric L / arrow / T template | `3×3` or `5×5`, usually rotated banks | Positive weights on expected arm(s), negative/zero on forbidden cells | Direction-sensitive object or junction detection | `O(k²·O)` for `O` orientations; four rotated `5×5` templates are ≈ `180,000` FLOPs |
| Concentric multi-scale shells | Multiple `k×k` kernels or multi-output bank | Distinct shell weights per radius band | Classify by radius, thickness, or “best matching shell” | `O(Σj k_j²)`; three dense `5×5` shells are ≈ `135,000` FLOPs |

Hit-or-miss remains the canonical exact-pattern detector, including rotated banks for corners and junctions; Weber et al. explicitly show cross detection and discuss applying multiple orientations and sizes, while Borgefors shows how local weighted masks implement different neighborhood metrics with different accuracy / cost tradeoffs. Loy and Zelinsky’s fast radial symmetry transform is the standard published example of a radius-indexed template family whose score changes with scale. A dense `5×5` kernel always stores `25` weights; that is `100 B` in `fp32` or `25 B` in `int8`.

## ONNX-static implementation strategies

For hand-built opset-18 graphs, the core operator specs you actually lean on are `Conv-11`, `QLinearConv-10`, `MatMul-13`, `ReduceSum-13`, `ArgMax-13`, and `Pad`’s current constant / edge / reflect semantics. ONNX Runtime documents kernel support centrally and its reduced/mobile operator tables include the main ingredients used here, including `Conv`, `QLinearConv`, `MatMul` / `QLinearMatMul`, `Pad`, `ReduceSum`, `ReduceMax`, `Reshape`, `Transpose`, and `ArgMax`. Correction: ONNX has `Col2Im-18`, but there is no standard core `Im2Col` operator, so “Im2Col-equivalent” in pure ONNX means a synthetic extractor bank or an external/custom preprocessor.

| Strategy | ONNX ops | Memory cost | Compute cost | Quantization-friendly | Notes |
|---|---|---|---|---|---|
| QLinearConv with custom integer weights | `QLinearConv` | `25 B` weights for one `5×5 int8` template + `900 B` output map for `uint8/int8`; scales / zero-points are tiny constants | `22,500` MACs for one `5×5` template on `30×30`, plus requantization | High | Best for small exact integer-weight detectors when the output dynamic range fits the chosen `y_scale` |
| Conv + threshold | `Conv`, `Greater` / `Less`, optional `Where` | `100 B` weights for one `5×5 fp32` template + `3.6 kB` float score map + `900 B` mask | `22,500` MACs + `900` comparisons | Medium | Easiest exact path for signed “anti-cover” weights and wide score ranges |
| MatMul + reshape | Synthetic extractor `Conv`, then `Reshape`, `Transpose`, `MatMul` | If extractor is synthesized literally for `5×5`: `625` extractor weights (`2.5 kB fp32`) + `90 kB` patch buffer (`900×25 fp32`) + template vector | Literal synthetic extractor costs `562,500` MACs, then `22,500` MACs for `MatMul`; near-conv cost only if extraction is external or fused | Medium | Useful only when patch extraction is shared, fused, or external; otherwise it duplicates work |
| Depthwise + reduce | `Conv(group=...)`, second `Conv` or `ReduceSum` / `1×1 Conv` | Exact rank-1 `5×5` separable kernel uses `10` weights (`40 B fp32`) + one temp map (`3.6 kB`) + output map (`3.6 kB`) | Rank-1 separable `5×5`: `9,000` MACs; rank-`R`: about `R·9,000` MACs plus cheap combine | Medium to High | Excellent for box / rectangular cover and low-rank approximations; not the default for arbitrary rings or stars |

The full dense formulas are straightforward from the ONNX `Conv` weight layout `W.shape = (M, C/group, kH, kW)`: parameters `= M·C·k²/group`, visible intermediate bytes `= Hout·Wout·M·Bout`, and MACs `= Hout·Wout·M·C·k²/group`. If you synthesize patch extraction as a one-hot `Conv`, the extractor itself costs `C·k⁴` weights and `Hout·Wout·C·k⁴` MACs, which is why a literal pure-ONNX `MatMul + reshape` path is usually inferior unless that patch matrix is reused across many downstream templates. ONNX’s own reference runtime notes that optimized `Conv` kernels use an internal `im2col + Gemm` decomposition, which is another reason not to materialize patch columns yourself unless you truly need them as a reusable intermediate.

A direct `QLinearConv` kernel is the cleanest layout when the detector really is “weighted cover = one convolution score.” `QLinearConv` supports scalar or per-output-channel weight scales, grouped convolution, and `int8/uint8` inputs, weights, and outputs.

```python
import numpy as np
from onnx import helper, TensorProto

# 3x3 weighted cross detector, int8 weights
w = np.array([[[[0, 2, 0],
                [2, 4, 2],
                [0, 2, 0]]]], dtype=np.int8)

qconv = helper.make_node(
    "QLinearConv",
    inputs=["x", "x_scale", "x_zero_point",
            "w", "w_scale", "w_zero_point",
            "y_scale", "y_zero_point"],
    outputs=["score_q"],
    pads=[1, 1, 1, 1]
)
```

The float path is the most exact and easiest to reason about when thresholds are tight or weights include negative “forbidden region” penalties; it pairs naturally with `Greater` / `Less`, which broadcast in ONNX.

```python
import numpy as np
from onnx import helper

w = np.array([[[[0., 1., 0.],
                [1., 3., 1.],
                [0., 1., 0.]]]], dtype=np.float32)

conv = helper.make_node("Conv", ["x", "w"], ["score"], pads=[1, 1, 1, 1])
gt   = helper.make_node("Greater", ["score", "tau"], ["mask"])
```

If you really need a reusable patch matrix, the pure-ONNX way is a fixed extractor bank whose output channels each pick one spatial offset. That is static and legal, but it is almost never the cheapest graph unless the extracted patches feed many templates.

```python
import numpy as np
from onnx import helper

# 3x3 one-hot patch extractor: output channels enumerate offsets
patch_w = np.array([
    [[[1,0,0],[0,0,0],[0,0,0]]],
    [[[0,1,0],[0,0,0],[0,0,0]]],
    [[[0,0,1],[0,0,0],[0,0,0]]],
    [[[0,0,0],[1,0,0],[0,0,0]]],
    [[[0,0,0],[0,1,0],[0,0,0]]],
    [[[0,0,0],[0,0,1],[0,0,0]]],
    [[[0,0,0],[0,0,0],[1,0,0]]],
    [[[0,0,0],[0,0,0],[0,1,0]]],
    [[[0,0,0],[0,0,0],[0,0,1]]],
], dtype=np.float32)

template = np.array([[0.0], [1.0], [0.0],
                     [1.0], [4.0], [1.0],
                     [0.0], [1.0], [0.0]], dtype=np.float32)

extract = helper.make_node("Conv", ["x", "patch_w"], ["cols"], pads=[1,1,1,1])
# Then: Transpose/Reshape cols -> [N*H*W, 9], MatMul with template -> [N*H*W, 1]
matmul  = helper.make_node("MatMul", ["cols_flat", "template"], ["score"])
```

For truly separable cover detectors, grouped 1D convolutions are the right static pattern. A square / box cover is exact under a vertical 1D pass followed by a horizontal 1D pass; with channel banks, use `group=C` and reduce afterward.

```python
import numpy as np
from onnx import helper

# three color channels, same 5x1 then 1x5 box-count kernel per channel
v = np.array([
    [[[1],[1],[1],[1],[1]]],
    [[[1],[1],[1],[1],[1]]],
    [[[1],[1],[1],[1],[1]]],
], dtype=np.float32)

h = np.array([
    [[[1,1,1,1,1]]],
    [[[1,1,1,1,1]]],
    [[[1,1,1,1,1]]],
], dtype=np.float32)

dw_v  = helper.make_node("Conv", ["x", "v"], ["tmp"], pads=[2,0,2,0], group=3)
dw_h  = helper.make_node("Conv", ["tmp", "h"], ["score_ch"], pads=[0,2,0,2], group=3)
merge = helper.make_node("ReduceSum", ["score_ch", "axes"], ["score"], keepdims=1)
```

## Radius parameterization

The most practical way to choose radius `r` from data is to treat `r` as a discrete hyperparameter over a small bank of candidate neighborhoods, compute each candidate score map on training / calibration examples, and select the smallest `r` that yields a stable positive-vs-negative margin or validation accuracy. That is exactly the same spirit as Weber et al.’s structuring-element design: they describe setting structuring-element shape / size from expert knowledge, from scribbles, or directly from a machine-learning algorithm that identifies the best decision functions. For CNN-style integer detectors, “smallest sufficient radius” is usually the right bias because params, intermediate bytes, and MACs all grow with neighborhood size.

When `r` gets large, the first compression question is whether the kernel is exact-separable or nearly low-rank. A `k×k` box detector is exact with two 1D passes, so cost drops from `k²` MACs per cell to `2k`. On a `30×30` grid, an `11×11` dense box detector costs `108,900` MACs, while two `11`-tap 1D passes cost only `19,800` MACs. A plus detector can also be compressed exactly as “vertical line score + horizontal line score - center correction,” giving roughly `2k+1` work per cell rather than `k²`, inferred. For generic large-radius masks, the standard approximation is a rank-`R` separable basis; Jaderberg et al. give the reduced cost form `O(R·k·(C+M)·H·W)` versus dense `O(C·M·k²·H·W)`, while MobileNets show the same depthwise-separable principle as a mainstream efficiency pattern.

If multiple radii must coexist in one detector, the best static layout is usually a parallel radius bank in output channels, followed by `ReduceMax`, `ArgMax`, or threshold bins. For concentric square shells, an especially efficient design is to compute cumulative box counts and subtract adjacent radii to get annuli, so the outer shells reuse inner work; for general rings, low-rank sharing is weaker, so a direct multi-output bank is simpler. Published radial-symmetry detectors already use multi-radius accumulation as a first-class design choice, so this is a standard rather than exotic pattern.

## Quantization considerations

For binary or one-hot occupancy inputs, the cleanest `QLinearConv` setup is usually `uint8` activations with `x_zero_point = 0` and `x_scale = 1.0`, because `0` and `1` remain exact and zero padding stays exact as well. `QLinearConv` allows scalar or per-output-channel weight scales and zero-points, so a bank of templates can each use its own `w_scale`. For signed “anti-cover” penalties, keep `w_zero_point = 0` and encode weights in `int8`. This makes the effective real score proportional to the centered integer accumulator, inferred from the `QuantizeLinear` and `QLinearConv` definitions.

ONNX Runtime’s quantization guidance matters here because `QLinearConv` is the operator-oriented `QOperator` format. ORT says CPU supports `U8U8`, `U8S8`, and `S8S8`; it also states that `S8S8` with `QOperator` is slow on x86-64 and should generally be avoided. For weighted cover detectors, `U8S8` is the natural single-node choice when activations are unsigned occupancy maps and weights need negative penalties. If x86 saturation becomes visible, ORT explicitly recommends either `reduce_range` or `U8U8`; in this niche, `U8U8` generally means splitting the detector into a positive branch and a negative branch and subtracting later, which doubles the conv work but avoids signed-weight saturation.

The real precision bottleneck for `QLinearConv` is rarely the inner accumulation; it is the final requantization to `int8/uint8`. A useful engineering rule is: if the integer score range is `[-Smax, Smax]`, choose `y_scale >= x_scale·w_scale·Smax/127` for signed `int8`, or `y_scale >= x_scale·w_scale·Smax/255` for nonnegative `uint8`, and quantize the threshold as `τq = round(τreal / y_scale) + y_zero_point`, inferred. This means some large neighborhoods are still perfectly exact: an `11×11` unit-count cover detector has maximum count `121`, so a nonnegative `uint8` output can represent every count exactly. But a `121`-cell signed kernel with weights up to `±3` can reach `|score| = 363`, forcing coarser bins if you insist on a single `int8` output channel. That is where float `Conv + threshold`, multi-branch partial sums, or a less aggressive quantized `y_scale` become preferable.

If threshold margins are small, use ORT’s standard static quantization workflow: calibrate on representative examples, inspect which score tensors drift most, and either widen the score margin or exempt that layer from quantization. ORT’s quantization docs explicitly provide activation / weight matching and intermediate-tensor debugging for this purpose.

## Multi-template selection

The direct static design for “pick the best of several templates per cell” is one output channel per template, then `ArgMax` over the channel axis. For `T=3` templates, a dense `5×5` bank on a `30×30` grid costs `3 × 22,500 = 67,500` MACs. The score stack is `3×30×30 = 2,700` elements, which is only `2.7 kB` in `int8` or `10.8 kB` in `fp32`. ONNX `ArgMax` supports deterministic tie choice by first occurrence or last occurrence through `select_last_index`, so you do not need ad hoc logic just to make ties reproducible.

A hierarchical detector is still useful in a static graph, but not because it can skip runtime work without control flow. Its value is parameter sharing. Instead of separate full kernels for cross, T, and ring, compute reusable primitive score channels first—center occupancy, horizontal line support, vertical line support, annulus support, forbidden-diagonal occupancy—then combine them in a second stage with a `1×1` or tiny conv. That mirrors Weber et al.’s observation that families of template configurations differ mainly by orientation, size, and local substructure; it also matches the general low-rank / basis-sharing logic used in efficient convolution papers.

For tie-breaking, there are three robust patterns. First, use `ArgMax(select_last_index=0)` or `1` for a simple priority convention. Second, add tiny fixed per-template biases before `ArgMax`; this works well when templates should have a strict lexical order and thresholds are coarse, inferred. Third, if the semantic tie-break itself has structure—“prefer exact match count, then larger radius, then lower color id”—encode that as a second compare stage with `Greater`, equality masks, and a `Where` cascade, inferred. `ArgMax` solves the common case cleanly; the other two solve policy-heavy cases.

## Edge handling

Pad-then-detect is the simplest boundary policy because `Conv` / `QLinearConv` already accept explicit padding. `constant` zero pad means “outside the grid contributes nothing,” which is often the right semantics for occupancy maps but systematically lowers scores near borders. `edge` pad replicates the nearest border value, which can preserve constant backgrounds but can also hallucinate support along the boundary. `reflect` mirrors the interior and is the least biased for smooth signals, but for symbolic ARC-style grids it can fabricate mirror objects that were never present. ONNX `Pad` formally supports all three modes.

If the rule requires a full neighborhood, mask-out-boundary is usually superior to any padding trick. Precompute a valid-center mask with ones only where the full radius fits, and set invalid scores to a sentinel before thresholding or `ArgMax`. For `r=2` on a `30×30` grid, the fully valid interior is `26×26 = 676` cells, leaving `224` boundary cells to suppress. This costs one extra static mask tensor and avoids semantic ambiguity completely, inferred.

Shrinking the radius near boundaries is the most faithful option when the intended rule is “use whatever part of the neighborhood exists,” but it is also the heaviest static implementation. Without `Loop`, the exact approach is to build a finite bank of truncated kernels for each border-distance class and splice them with selector masks. That is lossless, but kernel count balloons quickly for asymmetric templates. In quantized graphs, ORT’s warning that zero must be exactly representable is directly relevant here because constant-zero padding only behaves correctly when quantized zero is exact.

## Worked example for task118-style detector

Public NeuroGolf pages and notebook snippets show the competition is built around public ARC tasks and refer to files such as `task118.json`, but I did not locate a public primary rule write-up for ordinal `task118` in the sources reviewed. The example below is therefore an explicit engineering inference from the rumor “kernel encodes radius/weight per-cell,” not a claim about a verified ground-truth rule.

A plausible compact kernel family is:

| candidate | weights |
|---|---|
| `3×3` compact plus cover | `[[0,2,0],[2,4,2],[0,2,0]]` |
| `5×5` radius-coded plus with anti-cover bias | `[[0,-1,2,-1,0],[-1,0,3,0,-1],[2,3,5,3,2],[-1,0,3,0,-1],[0,-1,2,-1,0]]` |

The `3×3` version says “reward center-adjacent arm support, especially the center.” The `5×5` version says “reward the Manhattan-radius arm cells more strongly as they approach the center, but penalize off-axis near-center clutter.” Structurally, that is exactly what “radius-encoded weighted cover” sounds like in a static detector: the radius class is stored in the weight value itself, inferred.

A QConv-style detector for that family would be: one-hot or binary occupancy input for the candidate source color; `QLinearConv` with one output channel per template (or per color-template pair); optional second branch for a forbidden-color / forbidden-background channel if the task depends on specific colors; `Greater` against one or more thresholds to produce a hit mask; and `Where` to paint the target color. If the task chooses “best radius wins,” stack the `3×3` and `5×5` score channels together and use `ArgMax`; if it is “paint when enough weighted cover exists,” a single threshold is enough. Because `QLinearConv` supports per-output-channel scales, each radius can be calibrated separately even in a shared template bank.

For a practical threshold, start from the positive support total and back off for tolerated defects. In the `3×3` example the maximum score is `12`; a threshold near `8` says “center plus most of the arm support is present.” In the `5×5` example the positive support sums to `22` and the anti-cover penalties sum to `-6`; a threshold around `14–16` says “most arm cells are present and near-center off-axis clutter is limited,” inferred. That kind of threshold is stable enough to calibrate with a few training examples and much easier to port than a custom rule engine.

## Decision tree and verification checklist

### Decision tree

| Rule signature | Recommended strategy | Kernel layout | Threshold tactic |
|---|---|---|---|
| Small dense weighted mask, binary or one-hot input, deployment wants integer graph | `QLinearConv` with custom integer weights | One output channel per template; `k×k` dense kernel | Compare quantized score to quantized threshold |
| Small or medium mask with signed anti-cover weights, or threshold margin is tight | `Conv + threshold` | Dense `fp32` kernel with negative penalties allowed | `Greater` / `Less`, or exact-bin test via narrow score interval |
| Many templates share the same patch vector and patch extraction is external, fused, or reused broadly | `MatMul + reshape` | Synthetic / external patch matrix `[H·W, C·k²]` times template bank | Threshold each template score or `ArgMax` across bank |
| Kernel is exact-separable or well-approximated by low rank | `Depthwise + reduce` | Two 1D passes for box / rectangular cover, or rank-`R` basis | Threshold merged score; for multi-rank banks use `ReduceMax` / `ArgMax` |
| Multiple radii or multiple shapes compete per cell | Parallel template bank + `ArgMax` | One output channel per radius / shape | `ArgMax`, optionally with deterministic tie policy |
| Boundary semantics require full neighborhoods only | Any of the above + valid-center mask | Same kernel, plus precomputed interior mask | Set invalid scores to sentinel before threshold / `ArgMax` |
| Boundary semantics require partial neighborhoods to count | Bank of truncated kernels | Radius- or border-distance-specific kernels | Select by mask, then threshold per bank |

### Verification checklist

- ✅ At least four distinct cover-pattern families were surveyed: hit-or-miss cross/star templates, distance-bounded coverage, ring / annulus detectors, asymmetric L/T/arrow templates, and concentric multi-scale shells.
- ✅ Memory and compute were computed for a `5×5` template on a `30×30` grid: `25` weights, `22,500` MACs, and concrete intermediate sizes for `QLinearConv`, `Conv + threshold`, `MatMul + reshape`, and separable paths.
- ✅ Quantization strategy was made explicit: zero-points, scales, per-output-channel weight scales, `U8S8` / `U8U8` tradeoffs, and threshold quantization.
- ✅ Multi-template selection covered `≥3` templates with parallel detection, `ArgMax`, hierarchical primitive sharing, and tie-breaking strategies.
- ✅ Edge handling covered more than two strategies: pad-then-detect, mask-out-boundary, and shrinking-radius banks.
- ✅ A task118-style worked example was provided; limitation noted explicitly: the kernel and threshold are inferred because no public primary rule description for ordinal `task118` was located.