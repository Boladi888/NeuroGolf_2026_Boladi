# Direct detection via multi-QLinearConv (Research Card F4 deliverable)

> **Provenance**: Deep-research deliverable for card F4 (`Todo/Done/card_F4_direct_detection_multi_qlinearconv.md`). Project key-insight extraction lives in `Onnx_Advanced/techniques.md` under "Multi-QLinearConv detector banks cost reference". Citation markers `` are deep-research-tool provenance pointers (broken as URLs but document source-set composition).

This note treats **direct detection via multi-`QLinearConv`** as a hand-authored template bank: each kernel is a structural rule, each detector yields a local integer score map, and downstream ONNX ops turn those scores into paint masks. Correction for opset compatibility: for an **opset 18** model, use **`Cast-13`**, not `Cast-19`; ONNX distinguishes model opset imports from individual operator introduction versions, and `Cast-19` was introduced after opset 18.

Compatibility note for **ORT 1.24**: ONNX Runtime’s compatibility page explicitly states that newer ORT versions keep support for models that worked in prior versions, and its published table already shows **opset 18** support in ORT **1.14**. ORT **1.24.1** is a public release and upgraded ONNX to **1.20.1**, so an opset-18 graph made from `QLinearConv-10`, `ArgMax-13`, and `Cast-13` is comfortably inside the documented compatibility envelope. That last step is an inference because the public compatibility table currently stops before 1.24.

## Inventory of multi-detector patterns

The closest published analogs are classical **template matching**, **morphological hit-or-miss**, and **hand-crafted filter banks**. Classical lecture notes describe convolution/correlation as comparing a template with each local patch; the hit-or-miss literature describes pattern detection with a bank of structuring elements; and vision texts describe filter banks as collections of filters tuned to different features, historically often hand-crafted rather than learned. Multi-template matching papers then add the key idea of searching with **several templates** and combining results, including two-step coarse-to-fine pipelines. The ONNX translation below is therefore partly direct and partly design synthesis.

| Pattern | # detectors | Kernel shape | Combination | Memory cost | Use case |
|---|---:|---|---|---|---|
| Template library | D | `D x C_in x k x k` in one node, or D separate `1 x C_in x k x k` nodes | `ArgMax`, threshold, or priority chain | `~D·H·W` output bytes + `D·C_in·k²` weight bytes | Finite library of local motifs such as corners, L-shapes, T-shapes, hull witnesses |
| Multi-class detection | C classes | One `QLinearConv` with C output channels | `ArgMax` across channel axis, optional threshold | `~C·H·W` output bytes + `C·C_in·k²` weight bytes | Mutually exclusive structural classes with one winning paint class per cell |
| Hierarchical detection | `1 + F` | Coarse detector bank plus fine detector bank | Threshold coarse stage, then fine stage only in ROI or via masked chain | Best case `~H·W + Σ ROI`; worst case `~(1+F)·H·W` | Object-then-part, hull-then-fill, region detect then detail refine |
| Per-color detection | `P` or `P·D_c` | Grouped/depthwise bank on one-hot palette planes, or full multi-channel kernels | Per-color threshold, then palette selection | `~P·D_c·H·W` bytes if fully expanded | Same geometry means different rule depending on color identity |

In ONNX terms, the first and second rows are the most natural layouts: `QLinearConv` already exposes an output-channel axis `M`, and its weight tensor is explicitly `M x (C/group) x kH x kW`, so “N parallel detectors” and “C class channels” are both first-class encodings. The third row maps well to the published two-step template-matching idea of detecting an ROI and then matching subregions inside it. The fourth row follows from the same channel semantics, but uses one-hot color planes so kernels can reason about color-conditioned structure instead of raw scalar cell values.

## QLinearConv kernel design for structural detection

`QLinearConv-10` consumes quantized input `x`, quantized weights `w`, scalar input/output scales and zero-points, optional **int32** bias `B`, and standard convolution attributes such as `pads`, `strides`, `dilations`, and `group`. The spec allows **per-output-channel** `w_scale` and `w_zero_point`, while `y_scale` and `y_zero_point` are scalar for the whole output tensor. Inputs may be `int8` or `uint8`, weights may be `int8` or `uint8`, outputs may be `int8` or `uint8`, and bias must use `scale = x_scale * w_scale` with zero-point `0`.

For an **exact-match binary template** on a one-hot plane `x∈{0,1}`, the cleanest hand-built recipe is:

\[
w_i =
\begin{cases}
+1,& i \in P \text{ and the template requires a 1} \\
-1,& i \in Z \text{ and the template requires a 0} \\
0,& i \in D \text{ and the template does not care}
\end{cases}
\]

\[
b = 1 - |P|
\]

Then the detector score is

\[
s(x)=\sum_i w_i x_i + b
\]

and it has a useful property: an exact match gives `s=1`, while any missing required positive or any forbidden positive drives `s<=0`. This is a synthesis, not a quoted ONNX recipe, but it matches the operator’s integer interface exactly.

For a canonical **3×3 L-shape** with required ones at
`[(0,0), (1,0), (2,0), (2,1), (2,2)]` and required zeros everywhere else, a valid int kernel is:

\[
W=
\begin{bmatrix}
+1 & -1 & -1 \\
+1 & -1 & -1 \\
+1 & +1 & +1
\end{bmatrix},
\quad
B = -4
\]

because `|P|=5`, so `b = 1 - 5 = -4`. Exact match gives `+1`; one missing leg or one forbidden extra pixel gives `0`; multiple mismatches go negative. The same construction works for **T-shapes** and **crosses** by changing the support set `P`.

```python
import numpy as np
from onnx import helper

# exact-match 3x3 L-shape on a binary uint8 plane X in {0,1}
W = np.array([[[[ 1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1,  1]]]], dtype=np.int8)   # [M=1, C=1, 3, 3]
B = np.array([-4], dtype=np.int32)                # exact match -> +1, mismatch -> <=0

node = helper.make_node(
    "QLinearConv",
    inputs=["x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp", "B"],
    outputs=["score"],
    pads=[1, 1, 1, 1],
    strides=[1, 1],
)

# Practical scalar constants for this detector:
# x_scale = 1.0, x_zp = 0 (uint8)
# w_scale = 1.0, w_zp = 0 (int8)
# y_scale = 1.0, y_zp = 0 (int8)
```

For detector banks, the safest score domain is usually **signed output**: `uint8` activations for one-hot planes, `int8` weights for positive/negative evidence, and `int8` output scores so negative evidence survives until thresholding. If a later stage insists on `uint8`, shift the score by an output zero-point such as `128`, so “non-positive” becomes `<=128` and “positive” becomes `>128`. Because small structural detectors rarely exceed a few dozen in absolute score, 3×3, 5×5, and 7×7 exact-match kernels fit comfortably in `int8` output range when normalized this way. This output-range budgeting is inferred from the operator’s types and the explicit score formula above.

For **multi-input-channel kernels**, use one-hot palette planes or feature planes. The spec’s weight shape `M x C/group x kH x kW` means a single detector can demand “red here, blue there, background elsewhere” by placing positive weights on the wanted channel-position pairs and negative weights on disallowed channel-position pairs. If that becomes too dense, a cheaper pattern is **grouped/per-color detection**: one detector bank per color plane, then combine later. This is often the best trade when the color palette is small but rule count is large.

The main published integer-arithmetic caveat is **ORT’s x86 U8S8 fast path**: ONNX Runtime documents that AVX2/AVX512 backends may use `VPMADDUBSW`, which can suffer from intermediate saturation, and explicitly recommends trying **U8U8** if accuracy drops. The same doc also notes that **S8S8 with QOperator** is typically slow on x86-64, whereas CPU quantization in ORT can run **U8U8, U8S8, and S8S8** overall. For tiny ±1 structural kernels the risk is usually low, but it is the important published edge case to keep in mind.

## Combination strategies

**ArgMax across detectors.** Put all detector scores in a shared tensor of shape `[N, D, H, W]`, then run `ArgMax(axis=1)` to get the winner index at each cell. ONNX `ArgMax-13` is deterministic on ties: by default it returns the **first** max, and `select_last_index=1` flips that to the **last** max. If detector IDs map to paint colors, add `Gather` on a constant palette tensor; if background suppression is needed, pair `ArgMax` with `ReduceMax` and a threshold compare. This has the cleanest “winner-take-cell” semantics, but without a threshold it will choose something everywhere, including weak negatives.

**Max with threshold.** If the rule only needs to know whether **any** detector fired, use `ReduceMax` over the detector axis, compare against `T` with `Greater` or `GreaterOrEqual`, then use `Where` to paint or keep background. This is the cheapest combination path when all positive detectors mean the same paint action. It is also useful as a gating stage before a second pass, but it does not by itself preserve detector identity.

**Priority chain.** Convert each detector to a boolean mask `score_i > T_i`, then nest `Where(mask1, color1, Where(mask2, color2, ... background))`. Cost grows roughly linearly in detector count, but behavior is extremely easy to reason about: detector 1 overrides detector 2 regardless of relative numeric score, which is often exactly what exception-first rule systems want. This is the most literal ONNX encoding of “if A then X; else if B then Y.”

**Sum plus lookup.** After booleanizing detectors, cast them to integers, add either plain weights or bitfield weights, and `Gather` the final paint from a lookup tensor. Plain sums only work when code collisions are impossible; powers of two avoid collisions but make the LUT size exponential in the number of jointly active detectors. So this is excellent for **small composite rule sets** and poor for large unconstrained overlaps. It is an engineering synthesis rather than a published ONNX pattern, but it maps directly to `Cast`, integer adds, and `Gather`.

The strongest default for this project class is: **shared-score detector bank → `ReduceMax` threshold for background → `ArgMax` for color choice**. That gives explicit control over false positives and preserves detector identity only where the bank actually fired.

## Tiling and receptive-field tradeoffs

Assumption for the table below: **30×30 grid, batch 1, one input plane, one detector, int8 output, SAME-style padding unless the note says otherwise**. The `FLOPs` column reports **MAC counts** because that is the more useful score for your project; double them only if your tooling defines one multiply and one add as two separate FLOPs. The formulas are inferred from standard convolution arithmetic plus the `QLinearConv` stride/dilation/kernel attributes.

| Kernel size | Stride | Output shape | Memory | FLOPs | Notes |
|---:|---:|---|---:|---:|---|
| 3 | 1 | `30 x 30` | `900 B` | `8,100` | Dense per-cell detection |
| 5 | 1 | `30 x 30` | `900 B` | `22,500` | Dense per-cell detection |
| 7 | 1 | `30 x 30` | `900 B` | `44,100` | Dense per-cell detection |
| 3 | 3 | `10 x 10` | `100 B` | `900` | Tile-wise detector; good when paint is block-aligned |
| 5 | 5 | `6 x 6` | `36 B` | `900` | Tile-wise detector; 30 not divisible by 5 only in alignment, not size |
| 7 | 7 | `5 x 5` | `25 B` | `1,225` | SAME-style estimate; VALID would drop to `4 x 4` and `784` |
| 3 | 1 | `30 x 30` | `900 B` | `8,100` | With dilation 2, effective receptive field is `5 x 5` at 3×3 cost |
| 3 | 1 | `30 x 30` | `900 B` | `8,100` | With dilation 3, effective receptive field is `7 x 7` at 3×3 cost |
| 3+5+7 | 1 | three maps | `2,700 B` | `74,700` | Multi-scale bank; concatenate or compare across scales |

The practical split is straightforward. **Stride 1** is the right default when every output cell may need an independent paint decision. **Stride k** is attractive when the task is naturally tile-based or when a coarse stage will be followed by an upsample-or-paint step. **Dilation** is the cheapest way to widen context when the template only checks sparse anchors instead of dense occupancy. **Multi-scale banks** are filter-bank style design: more robust to scale variation, but with linearly additive compute and memory.

## Quantization tactics

The best default for a multi-detector bank is to **share the input quantization** across all detectors and, if possible, **share the output quantization** across all detector outputs that will later participate in `ArgMax` or threshold comparisons. The spec directly supports this style: `x_scale` and `x_zero_point` are scalar for the input tensor, `w_scale` and `w_zero_point` may be **per output channel**, and `y_scale` / `y_zero_point` are scalar for the output tensor. That combination is ideal for a single multi-output-channel detector bank: detectors may carry slightly different weight quantization, but all scores land in one comparable integer domain.

That shared output domain matters a lot. If two detectors live in **different output scales**, raw integer `ArgMax` is not meaningful anymore, because the integers represent different real units. In that case either normalize the kernels so every detector shares one `y_scale`, or dequantize/upcast before comparing. For hand-authored structural banks, normalizing every exact-match detector to “exact = +1, mismatch <= 0” is usually the simplest way to avoid this problem entirely. This recommendation is inferred from ONNX quantization semantics rather than stated as a dedicated rule in the docs.

When to upcast back to float: only when detector ranges are genuinely heterogeneous, when thresholds are easier to express in real units than integer units, or when a later non-quantized subgraph already forces a dequantize boundary. Otherwise, keep the detector bank **fully quantized**. ORT’s quantization docs explicitly note that quantized QOperator models can run directly, so inserting float round-trips in the core detector path is normally self-defeating.

For direct paint output, I recommend **separating score from color**. Let detector scores stay in a comparison-friendly integer domain, then map detector IDs or masks to palette colors with integer `Gather` or `Where`. Trying to encode both “how confident is the detector?” and “which color should I paint?” into the same quantized score usually makes thresholds fragile. If all detectors paint the same color, `ReduceMax` plus `Where` is enough; if colors differ, use thresholded `ArgMax` plus palette lookup. This is design synthesis, but it is the most stable arrangement under ONNX static quantization.

For ORT CPU specifically, the documented practical order is: start with **U8S8** if it behaves, switch to **U8U8** if the x86 saturation path bites, and avoid **S8S8 QOperator** if latency matters on x86-64. On Arm, ORT’s docs say the same saturation issue does not apply.

## Detector pruning

Pruning in this setting is straightforward because the detector bank is explicit and symbolic. Run the bank on all visible training inputs, record `max(score_i)` and optionally `argmax-wins_i`, and mark detector `i` as dead if it never exceeds the chosen positive threshold or never wins a downstream selection. That is not proof of global irrelevance, but it is the right first pass for shrinking a hand-authored library. This logic is an inference; the ORT docs contribute the instrumentation hooks.

ORT’s quantization-debugging API is useful here even though it was written for float-vs-quantized comparisons. The official docs expose `modify_model_output_intermediate_tensors()`, `collect_activations()`, and `create_activation_matching()`, which together give a direct path to harvesting intermediate detector activations. In this project class, that makes detector-bank auditing and dead-channel discovery cheap and mechanical.

Once a detector is dead, prune it structurally. In a **single multi-output-channel `QLinearConv`**, remove the corresponding output channel and shrink `w`, `B`, and any per-output `w_scale` / `w_zero_point` tensors. In a **priority chain**, delete the branch and collapse nested `Where`s. In a **sum+lookup** scheme, recompute the integer codebook and shrink the LUT; dead bits are pure waste there. Those graph rewrites are synthesis, but they map directly onto ONNX initializer and downstream-index edits.

## Worked examples for task367 v2 and task077 rawwhere

The public material I found describes likely underlying ARC tasks, but **not** the project-local labels `structural direct-25`, `no MPH`, `rawwhere`, or `hull-8`. So the two sketches below are **explicitly inferred** from the labels plus public ARC-adjacent hints. For `task367`, a public hint describes a color-5 rectangular-boundary task over color-0 background. For `task077`, a public hint describes a color-2-centered region-expansion rule over a Manhattan neighborhood of radius less than 3.

### task367 v2

A plausible reading of **“structural direct-25 QLinearConv”** is a **25-detector direct-paint bank** for a fixed-size 5×5 structural output. If the underlying task is “detect a closed color-5 rectangular frame and fill its enclosed 5×5 region,” then the most literal direct bank is: one detector per **output offset** `(dy, dx)` in the 5×5 target patch. Detector `(dy, dx)` checks whether the input frame exists at the corresponding relative location and, if so, emits a positive score exactly at the cell that should be painted for that offset. All 25 detectors therefore share the same structural evidence, but are spatially shifted so each channel paints one different cell of the final 5×5 block. Combination is then **`ReduceMax` + threshold** if all 25 channels paint the same color, or a trivial priority chain if one wants deterministic suppression of overlaps. The attraction of this layout is that the detector bank itself is the paint program; there is no second-stage codebook collapse. The exact “25” interpretation is inferred from the label, while the rectangle-boundary task hint is public.

### task077 rawwhere

A plausible reading of **“rawwhere with hull-8 detection”** is: detect a local region or hull around color-2 seeds, booleanize the result, and feed it directly into `Where` rather than first collapsing to a detector ID. Public hints point to a color-2-centered region-growth rule, so a natural ONNX-static design is an **8-direction witness bank**: eight small kernels, one for each compass direction, each checking whether the current cell belongs to the directional hull of a color-2 seed. If the intended neighborhood is larger than 3×3, the same idea can be done with **dilated 3×3 kernels** or with 5×5 kernels centered on “reachable in <3 Manhattan steps.” The output masks are ORed via `ReduceMax` or elementwise `Max`, converted to a boolean condition by thresholding, and then used in **raw `Where`** to write the paint color directly. This is exactly the kind of rule where `ArgMax` is unnecessary: there is only one paint action, so the detector bank only needs to answer “paint here or not.” The “hull-8” phrase is project-local and the eight-direction bank is inference; the color-2 region-growth character of the task is public.

## Decision tree and verification checklist

### Decision tree

The table below is a design synthesis from the operator specs, the filter-bank/template-matching literature, and the ORT quantization/runtime constraints discussed above.

| Rule structure | # detectors | Kernel design | Combination strategy | Quantization tactic |
|---|---:|---|---|---|
| One exact local motif | `1–8` | `+1/-1/0` exact-match kernel on one-hot plane | Threshold or priority chain | `uint8 x` + `int8 w` + `int8 y`, exact match normalized to `+1` |
| Many mutually exclusive motifs | `D` | One multi-output-channel bank, one channel per motif | Thresholded `ArgMax` | Shared `y_scale/y_zp` across channels; optional per-output `w_scale` |
| Same geometry, color-dependent semantics | `P` or `P·D_c` | One-hot palette planes with per-channel penalties | Per-color threshold, then palette write | Shared input quantization; grouped/per-color banks if full multi-channel kernels get too dense |
| Coarse region then fine subpattern | `1 + F` | Cheap coarse hull/ROI detector, then fine exact-match kernels | Coarse threshold gating; then priority or `ArgMax` | Separate stages may have separate scales; dequantize only at the stage boundary if needed |
| Many detectors, one paint color | `D` | Structural bank only needs “fire / no fire” | `ReduceMax` + threshold + `Where` | Keep score domain integer; no need for per-detector comparable identity |
| Small composite logic over overlapping detectors | `2–6` | Binary detectors, then weighted integer code | Sum + LUT `Gather` | Common integer domain; powers-of-two codes if overlaps must remain collision-free |

### Verification checklist

- [x] At least four distinct multi-detector patterns were surveyed: template library, multi-class detection, hierarchical detection, and per-color detection.
- [x] `QLinearConv` kernel design was covered with an explicit integer-weight 3×3 L-shape example and bias rule.
- [x] Combination strategies covered at least three options: thresholded `ArgMax`, max-with-threshold, priority chain, and sum-plus-lookup.
- [x] The tradeoff table includes kernel sizes 3, 5, and 7.
- [x] Quantization tactics were made explicit, including shared-vs-per-detector scales, comparator stability, float upcast boundaries, and ORT’s documented U8S8/U8U8/S8S8 caveats.
- [x] `task367 v2` and `task077 rawwhere` were sketched as worked examples, with public-task hints cited and project-local parts marked as inference.
- [x] All examples are compatible with **ONNX opset 18** and **ORT 1.24** as presented here: use `QLinearConv-10`, `ArgMax-13`, and `Cast-13`; ORT publishes CPU kernels for those ops and states that newer versions retain prior-model compatibility. 