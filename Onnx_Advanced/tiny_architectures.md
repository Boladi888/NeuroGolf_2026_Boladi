# tiny_architectures.md — hand-crafted tiny-ONNX building blocks

> Patterns for building sub-1k graphs by hand (hand-crafted weights, LUT-as-network, weight sharing, nonlinearities, static-shape tricks, iteration without Loop).
> Consolidated from `Onnx_Advanced/Research/tiny_onnx_arch/` 2026-06-01. Compressed 2026-06-10 (pass 4): textbook tier → `_archive/tiny_architectures_evicted_2026-06-10.md`; pre-refactor snapshot → `_archive/tiny_architectures_pre_refactor_2026-06-10.md`.

## Contents
- [B1_hand_crafted_weights.md](#b1_hand_crafted_weights)
- [B2_lut_as_network.md](#b2_lut_as_network) — summary; bulk in _archive
- [B3_weight_sharing.md](#b3_weight_sharing)
- [B4_nonlinearities.md](#b4_nonlinearities)
- [B5_static_shape_strategies.md](#b5_static_shape_strategies)
- [B6_iterative_without_loop.md](#b6_iterative_without_loop)


---
<a id="b1_hand_crafted_weights"></a>

# B.1 — Hand-crafted weights

The lever: any transformation writable as a fixed linear filter (or a small composition) ships its weights as initializers — no training. onnx-tool's MAC counter is dtype-blind: cost depends only on the *shape* of the kernel and the *arrangement* (depthwise vs dense, group, dilation), not the values. Pick the smallest shape that does the job, analytically.

## The operating-cost reference (for `[1,10,30,30]` activations)

Given Conv weight shape `[Cout, Cin/groups, kH, kW]` and output shape `[1, Cout, 30, 30]`:

```
macs = Cout · (Cin/groups) · kH · kW · 30 · 30
```

| Conv variant | Cost on 10ch [1,10,30,30] | Score floor (alone) |
|---|---|---|
| 1×1 dense, Cout=10 | 90,000 | ~13.59 |
| 1×1 depthwise (groups=10), Cout=10 | 9,000 | ~16.20 |
| 3×3 dense, Cout=10 | 810,000 | ~11.39 |
| 3×3 depthwise, Cout=10 | 81,000 | ~13.69 |
| 5×5 dense, Cout=10 | 2,250,000 | ~10.37 |
| 5×5 depthwise, Cout=10 | 225,000 | ~12.68 |

**Depthwise gives a 10× discount** for free when the per-channel transformation is the same logical operation (typical for symmetry/edge/pattern detectors that act on the binarized color-presence channels). All numbers below assume depthwise unless otherwise specified.

## Canonical kernels and their ARC uses

### Edge / boundary detectors (gradient kernels)

| Kernel | 3×3 weights | ARC use |
|---|---|---|
| **Sobel-X** | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | object boundaries on x-axis; left/right edges |
| **Sobel-Y** | `[[-1,-2,-1],[0,0,0],[1,2,1]]` | object boundaries on y-axis; top/bottom edges |
| **Prewitt-X** | `[[-1,0,1],[-1,0,1],[-1,0,1]]` | same as Sobel-X but unweighted; cheaper noise but same MAC cost |
| **Prewitt-Y** | `[[-1,-1,-1],[0,0,0],[1,1,1]]` | same as Sobel-Y |
| **Scharr-X** | `[[-3,0,3],[-10,0,10],[-3,0,3]]` | better diagonal edge response; same cost as Sobel |
| **Scharr-Y** | `[[-3,10,3],[0,0,0],[-3,-10,-3]]` | y-axis with rotational symmetry |
| **Laplacian (4-conn)** | `[[0,1,0],[1,-4,1],[0,1,0]]` | second derivative; isotropic; detects all-direction edges |
| **Laplacian (8-conn)** | `[[1,1,1],[1,-8,1],[1,1,1]]` | diagonal + axial edges in one kernel |

**Cost (depthwise 3×3 across all 10 channels):** 81,000 MACs. Score from a single such Conv: ~13.69. Adding a `Greater(0)` threshold + `Cast` to discretize → +9,000 for Greater, free Cast → 90,000 total → ~13.59. Two-Conv Sobel-magnitude (`sqrt(Sx² + Sy²)`) is ~243,000 → ~12.60. **Edge detection alone won't get past 14 pts/task.**

### Symmetry detectors (no MACs needed!)

All shape-only: **horizontal/vertical flip** = `Slice` with `step=-1` (free — but careful with non-30×30 grids; see COVERED caveat); **diagonal mirror** = `Transpose(perm=[0,1,3,2])` (free, the project's working pattern); **anti-diagonal** = `Transpose` + `Slice` reversal. **"Is this image symmetric?"** = `Equal(image, transposed_image)` then `ReduceMin` → broadcast the scalar as a mask, ~18k MACs total. For "if symmetric do X else Y": compute both and `Where(symmetry_mask, X, Y)` — cheaper than literal branching.

**Boolean logic via min/max/mul/add:** boolean ops via arithmetic: techniques.md #57 scaffold / #125 micro-idioms (worked table → _archive evicted file).

### Counting, presence, frequency (E.6 prerequisites)

| Quantity | ONNX expression | Cost |
|---|---|---|
| **Count of color c** in grid | `ReduceSum(channel_c)` | 9,000 |
| **Is color c present?** | `ReduceSum(channel_c)` then `Greater(_, 0)` | 18,000 |
| **Most frequent color** | `ReduceSum` per channel, then `ArgMax` over channels | 9k (sums) + 12 (argmax over length-12) ≈ 9,012 |
| **All-12-color presence vector** | `ReduceSum` over (H,W) per channel | 9,000 (single ReduceSum over a (1,10,30,30)→(1,10,1,1)) |
| **Total non-pad cells** | sum over channels 0–9 of `ReduceSum`, or `Sub(900, ReduceSum(channel_10))` | 9,000 |

Counting tasks (E.6) all reduce to one or two reduces — cheaper than any conv.

### Color remap via Gather (the project's existing pattern)

For tasks that map color c to color π(c) for some permutation π:
- Initializer = 12-element 1D `int64` (or `int8`) tensor `perm = [π(0), π(1), …, π(11)]`.
- `Gather(input, perm, axis=1)` permutes the channel dimension.
- **Zero MACs.** Score: ~25 pts.

Storage: 12 entries × 1 byte (uint8) = 12 bytes for `perm`. Negligible. **This is the gold standard pattern — already in production.**

### Color overlap / set operations across two grids

AND = `Min`, OR = `Max`, **A−B** = `Mul(a, Sub(1, b))` ("a present where b is absent"), symmetric difference = XOR. All ~9k–18k per output channel, broadcast over channels.

### Shifted-input operations (translation, gravity)

Translation by a known offset (dx, dy): `Slice` the source region + `Pad` with color-10 (no-color) on the destination side. Both free.

Gravity (cells fall until they hit something) is iterative — B.6 unrolled; worst case 30 iterations of a "shift down one if cell below empty" depthwise conv, ~30 × 81k = 2.4M MACs → ~10.3 pts/task. Unrolling fewer steps from the visible worst-case fall distance is risky — may fail on the private set.

### Morphological dilate / erode

Dilate (binary OR of 3×3 neighborhood) on a binary channel: depthwise Conv with all-1s kernel → `Greater(0)` → `Cast`. Cost: 81,000 (Conv) + 9,000 (Greater) = 90,000 MACs per dilation step. Erode (binary AND): same Conv → `Equal(9)` → `Cast`, same cost. Uses: thickening/thinning, interior cells (erode then sub), connectivity checks (B.6).

### 3×3 fixed-template detection

For "find the 3×3 cross pattern" tasks: define template `T` as a 3×3 binary mask, convolve with depthwise Conv weights = `T` minus a center scalar, threshold on `Equal(_, sum(T))` → match positions 1. Cost: 81k (Conv) + 9k (Equal) = 90k per template — same as a binarized edge detector.

### Bitwise overlay (XOR/AND/OR of grid halves)

"Split the grid in half, XOR the two halves": `Slice` both halves (free), then `Sub` + `Abs` (or `Equal` + `NOT`) — ~18,000 MACs → ~15.2 pts/task. If output is `[1,10,30,15]`, `Pad` back to 30×30 with color-10 (free).

## Fixed-point / int8 weights with manual scale

Hand-crafted Sobel weights are `±1, ±2` — fits int4/int8. Sobel-X 3×3 = 9 values: float32 36 B, int8 9 B, uint4 5 B packed (4–7× savings). Ten depthwise 3×3 kernels = 90 values: float32 360 B, uint4 45 B.

Pipeline: store weights as int8 raw_data (or uint4 if opset 21 accepted — A.4 follow-up) → `Cast` to float32 immediately before the Conv (zero MACs) → Conv runs in float32, same MAC count, less `params`.

**Caveat:** the int8→float32 `Cast` produces a float32 *intermediate activation* counted in `memory` — net win only because the Conv input had to be float32 anyway.

## When NOT to bother with hand-crafted Convs

A single Conv stage costs **at minimum 81,000 MACs** for a depthwise 3×3 across 10 channels — score floor 13.7 pts/task. Anything we can solve with pure shape ops (Identity / Transpose / Slice / Gather / Reshape / Cast / ConstantOfShape / Concat / Split / Tile / Pad / Resize-nearest) is **always strictly better**. So:

1. **First**, attempt to express the transformation as a composition of zero-MAC ops (target: 25 pts).
2. **If that fails**, use a single hand-crafted depthwise Conv plus a threshold (target: 13–14 pts).
3. **Only if needed**, stack two Convs or use a dense (non-depthwise) Conv (target: 11–13 pts).

## Concrete weight specifications, ready to ship

Initializer templates (all 3×3 unless noted). The eight gradient/Laplacian matrices (`sobel_x/y`, `prewitt_x/y`, `scharr_x/y`, `lap4`, `lap8`) are printed once in the table above — binarize via `Greater(0)`. The rest:

```python
# Morphological 3×3 (binarize via Greater for dilate, Equal(9) for erode)
ones3     = [[ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1]]

# Identity center-pick (1×1 dense, Cout=Cin)
id1x1     = identity matrix of shape [10, 10, 1, 1]

# Dihedral group D4 (8 symmetries) — combinations of Transpose + Slice
# All zero-MAC. Use to detect/apply rotations and reflections.

# Diagonal kernels (45° edges)
diag_pos  = [[ 0, 1, 1], [-1, 0, 1], [-1,-1, 0]]
diag_neg  = [[ 1, 1, 0], [ 1, 0,-1], [ 0,-1,-1]]

# Center-vs-neighbors (high-pass)
hpf       = [[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]]

# Box blur (smoothing — for noise-robust pattern detection if needed)
blur3     = [[ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1]]   # divide by 9 = Mul(_, 1/9), or use un-normalized + threshold scaled accordingly
```

All of these fit `int4` *if pre-shifted by a known offset* (e.g. `sobel_x + 2` ranges 0..4, fits uint4) — but the recovering `Sub(2)` after the Conv costs 9k MACs. Usually keep `int8`/`int4` symmetric and skip the offset.

## Action items for build_submission.py extension

Add `_PATTERNS` entries: Sobel-X/Y + Laplacian-4 edge maps, full edge magnitude, AND/OR/XOR of two stacked grids, 3×3 dilate ("thicken the object"). Verify by `np.array_equal` on the visible examples — incorrect = 0, so a 90k-MAC pattern only pays if it actually solves the task; the safety check is mandatory.

---
<a id="b2_lut_as_network"></a>

# B.2 — LUT-as-network

Summary — worked constructions evicted → `_archive/tiny_architectures_evicted_2026-06-10.md`. The distilled verdict:

- ARC's domain is discrete, so `Gather` on a precomputed table is 0 MACs + tiny params. A LUT net beats a hand-constructed Conv only when ALL hold: function of discrete labels; irregular/non-monotonic (a monotonic rule = one Conv anyway); index domain ≤ ~1000 entries; index-encoding cheaper than the arithmetic it replaces.
- Size crossover (costed): 10-entry color-perm/color-fold Gather = 0 MACs → 25 pts — the only LUT patterns beating 16 pts (color-perm deployed). Count-indexed 901-entry table = 9k (ReduceSum) → ~16.2, constant-color outputs only. Color-pair LUT (2× ArgMax + index combine, 100-entry) = 36k → ~14.5. 3×3-binary-neighborhood rule (power-of-2 depthwise Conv encode + Cast + Gather, 512-entry) = 90k → ~13.6 — ties a single Conv, wins only when the rule is non-monotonic (corners, T-junctions, CA rules) and would otherwise need ≥3 Convs.
- Row/region LUTs explode (12^30 states per 30-cell row); viable only for binary rows ≤ 8 cells. LUT compression (factorization/symmetry/hashing) never pays at ≤1000 entries — encoding overhead dwarfs the params saved.

---
<a id="b3_weight_sharing"></a>

# B.3 — Weight sharing

`params = sum_{unique tensor names} bytes(weight_tensor)` — the same initializer name referenced N times counts **once**; `macs` and `memory` still scale with node count. So sharing pays only in iterative/unrolled stacks (B.6); single-Conv tasks have nothing to share. (Academic survey — G-CNN, HashedNets, Kronecker, steerable, literature pointers — evicted → _archive; none survive the ONNX cost model.)

## The big winner for us: repeated initializer reference

Concrete pattern for an unrolled depthwise Conv stack (e.g. iterated dilation in B.6):

```python
W = make_tensor("W", FLOAT, [10, 1, 3, 3], values=ones3x3)   # one initializer
graph.initializer.add(W)

prev = "input"
for i in range(6):
    node = make_node("Conv", inputs=[prev, "W"], outputs=[f"act_{i}"],
                     group=10, kernel_shape=[3,3], pads=[1,1,1,1])
    graph.node.append(node)
    # add a Greater + Cast to binarize after each Conv
    ...
    prev = f"binarized_{i}"
```

`params` counts `W` once (90 elements × dtype bytes). `macs` counts 6 × 81000 = 486k. `memory` counts 6 intermediate `[1,10,30,30]` activations + 6 binarized ones.

**Savings calculation** for a 6-deep stack: without sharing 6 × 90 × 4 = 2160 bytes (float32; uint4: 270 B); with sharing 360 bytes (uint4: 45 B). Against a base cost of ~480k MACs that is ~0.4% — a rounding error. **Useful only when params dominates**, which it usually doesn't.

## Tied-across-channels = depthwise Conv (reframing)

The classic "tie all output channels' kernels" idea, in our 10-channel setup, is exactly depthwise Conv: **set `group = Cin = Cout`**. Where it pays big: a `[100, 100, 3, 3]` dense Conv carries 90,000 weights = 360KB float32; the tied version is 9KB. Depthwise **is** the canonical weight-sharing pattern for our setting. (G-CNN rotation sharing doesn't transfer: ONNX has no symbolic kernel reuse across rotations — the savings evaporate; details in _archive.)

## Cyclic-shift-tied kernels

If kernel rows are cyclic shifts of one another (e.g. `[a,b,c],[c,a,b],[b,c,a]`): store one row of 3 values, reconstruct via Tile + Slice + Concat — all zero MACs. 3× params savings — but here that is 3 bytes instead of 9. **Negligible. Not worth the build complexity.**

## Low-rank Conv decomposition

`[Cout, Cin, kH, kW]` ≈ `[r, Cin, 1, kW]` then `[Cout, r, kH, 1]`. Costed on Cout=Cin=10, 3×3: dense single Conv = 810,000 MACs; two-pass dense (1×3 + 3×1) = 270k each = 540,000 (1.5× cheaper); two-pass depthwise = 27k each = 54,000 MACs + 36KB extra intermediate memory ≈ 90k effective → 13.59 pts vs plain depthwise 3×3's 13.69. **About a wash — depthwise alone is already the right answer. Skip low-rank decomposition.**

## Symmetry-reduced storage for symmetric kernels

Store the 4 unique values of a symmetric 3×3 kernel (`[[a,b,a],[c,d,c],[a,b,a]]`) + `Gather`-scatter into the 3×3 layout. But both Gather inputs are initializers, so the validator's `constant_folding=True` folds the reconstruction back into a 9-element initializer: **zero post-folding `params` savings.** Only *file bytes* (the 1.44 MiB cap is measured pre-folding) shrink. **Useful only at the file-byte cliff.**

## What weight sharing cannot do / practical guidance

- Cannot reduce MACs in non-iterative graphs, cannot share intermediate-activation memory, irrelevant to shape-only solutions (no weight tensors at all).
- **Always use depthwise (`group=10`)** for 10-channel transformations unless the task genuinely needs cross-channel mixing.
- **In any unrolled stack (B.6), reference the same initializer name across iterations.** Rare params savings, but free and harmless.
- Skip G-CNN, low-rank, Kronecker, HashedNets; symmetry-reduced storage only if at the file-size cap.

The single useful idea from this whole section is "share the same initializer name across nodes in an unrolled stack." Everything else is academic.

---
<a id="b4_nonlinearities"></a>

# B.4 — Nonlinearities

The cost question for any nonlinearity is binary: does it cost more than `output_elems = 9000` MACs per `[1,10,30,30]` activation? Most stdlib nonlinearities cost exactly `output_elems` (1 MAC per element in onnx-tool's accounting); a few cost more (Softmax, LayerNorm composites). ARC decisions are sharp thresholds ("is this pixel one color or another"), never soft. **Verdict: smooth activations (Sigmoid / Tanh / HardSigmoid / GELU / SiLU / Swish / Mish / Elu / Selu / LeakyRelu / PRelu) are never exact-safe here; use Greater+Cast / Clip / Relu.** Evicted menu rows → _archive evicted file.

## The MAC-cost menu — decision table (per `[1,10,30,30]` activation)

Source: A.1's per-op table. All numbers in MACs.

| Op | Cost | Behavior | Verdict |
|---|---|---|---|
| **Identity** | 0 | passthrough | trivially free |
| **Cast** | 0 | dtype convert (e.g. bool → float) | free |
| **Greater / Less / Equal / GreaterOrEqual / LessOrEqual** | 9k | hard threshold, output bool | **best per-element discriminator** |
| **And / Or / Not / Xor (bool)** | 9k | boolean logic on bool tensors | combine thresholds |
| **Where** | 9k | mask-multiplexer | best mux |
| **ReLU** | 9k | `max(0, x)` | continuous discriminator at 0 |
| **Abs** | 9k | `\|x\|` | useful before threshold |
| **Sign** | 9k | `-1, 0, +1` | three-way discriminator |
| **Clip** | 9k | clamp to [min, max] | limit range |
| **Min / Max** | 9k | element-wise (e.g. `Min(a, 1)`) | logical AND/OR on 0/1 |
| **Hardmax** | ≈ output_elems + reduce | one-hot of argmax | useful for "pick one channel" |
| **Softmax / LogSoftmax** | ≈ 3·input_elems = ~27k | smooth normalized | **avoid — 3× cost of Greater + Cast** |
| **Argmax / Argmin** | ≈ input_elems = 9k | index of max along axis | **useful** — see below |

Special functions (Erf/Exp/Log/Pow/Sin/Cos/Tan/Sqrt): billed 9k each, and onnx-tool's own doc warns "Special function's MACs are not accurate" — pointless for ARC regardless. Activation chains compound: every extra activation = +9k.

## Greater + Cast for hard threshold (the cheapest discriminator)

```
mask = Greater(activation, threshold_initializer)   # 9k MACs, output dtype = bool
result = Cast(mask, to=FLOAT)                        # 0 MACs
```

Cost: **9k MACs total** (Cast is free), output float32 0/1. Functionally equivalent to ReLU+threshold / Step / Sign — all also 9k. **No nonlinearity is cheaper than this** for binary discrimination.

**ReLU vs Greater:** ReLU preserves positive magnitude (use when downstream consumes magnitude, e.g. a second Conv summing edge strengths); Greater+Cast collapses to 0/1. ARC downstream is almost always presence/absence, and the validator's correctness check thresholds at `> 0.0` anyway. **Default to Greater + Cast.**

**Where as the universal multiplexer:** `Where(cond, a, b)` = branch-free if-else at 9k — cheaper than mask-multiply (4 ops, 36k). Key use: sentinel preservation — `Where(input_eq_color10, color10_grid, computed_output)` carries pad regions through any computation.

**Min/Max as logic on 0/1 masks:** `Min(a,b)` = AND, `Max(a,b)` = OR, 9k each. (`Min(Add(a,b),1)` is OR at 18k — prefer the single `Max`.)

**ArgMax for "pick the most-present color"** (E.6 counting):

```
sums = ReduceSum(input, axes=[2,3])     # 9k MACs, output [1,10,1,1]
winning_color = ArgMax(sums, axis=1)    # ~10 MACs, output [1,1,1,1]
```

Then `OneHot(winning_color, depth=10)` ≈ 9k back to a 10-channel mask; `Tile`/`Expand` to 30×30 free. Total ~27k MACs → "fill the whole grid with the most frequent color" at ~14.8 pts.

**Hardmax** = ArgMax + OneHot fused, ~18k. Sometimes cheaper than the separate path if onnx-tool fuses — **verify in 1.0.1**; otherwise the separate path is safer.

## Validator silence on Argmax / Hardmax

The COVERED list says "Argmax allowed (validator silent — verify)": not on the banned-op list, but we haven't observed the validator profiling it. **Before relying on Argmax/Hardmax in a cost-critical path, locally instrument a one-op model and compare onnx-tool's reported MACs against the ~9k expectation.**

## Smart nonlinearity stacking patterns

### Pattern: "is this value the max along an axis"

```
maxval  = ReduceMax(x, axes=[1], keepdims=True)   # 9k MACs
is_max  = Equal(x, maxval)                         # 9k MACs
mask    = Cast(is_max, FLOAT)                       # 0 MACs
```

Total: 18k MACs. Replaces ArgMax + OneHot (also ~18k) with a directly-broadcasting mask. Use when the result must be a mask, not an index.

### Pattern: "is this value above the second-max" (second most frequent color)

```
top2_vals, top2_idx = TopK(x, k=2, axis=1)
second_max = top2_vals[:, 1:2, ...]
is_secmax  = Equal(x, second_max)
```

Cost: ~9k for TopK + 9k Equal + 0 Cast = 18k. **TopK is allowed; useful for ranking tasks.**

### Pattern: "any pixel of this color present"

```
ch_sum = ReduceSum(input[:, c:c+1, ...], axes=[2,3])    # 9k
present = Greater(ch_sum, 0)                              # 9k
```

Cost: 18k. Single-bit per channel; multiple "is color X present" queries share the ReduceSum across channels.

### Pattern: "which color has count exactly N"

```
counts  = ReduceSum(input, axes=[2,3])                   # 9k, shape [1,10,1,1]
matches = Equal(counts, ConstantOfShape([1,10,1,1], N))   # 9k (assumes ConstantOfShape is folded)
```

Cost: 18k. 10-channel single-pixel mask of "which color has count N" — for "find the unique-count color" tasks.

## Score budget for nonlinearity-heavy chains

A two-Conv-with-binarization-between chain:
```
Conv₁ (depthwise 3×3)     → 81k
Greater + Cast            → 9k
Conv₂ (depthwise 3×3)     → 81k
Greater + Cast            → 9k
```
Total: 180k MACs → score ≈ 12.9 pts. Each binarization adds ~0.07 to log(cost), shaving ~0.07 pts from the score. Multiple binarization stages quickly compound — design for **at most one binarization** between Convs.

## Preference order for build_submission.py extensions

1. **No nonlinearity** if the operation is shape-only (Identity, Transpose, etc.).
2. **`Greater(x, 0) → Cast`** for binarization.
3. **`Where`** for conditional pass-through.
4. **`ArgMax/Hardmax`** for single-channel selection (verify in onnx-tool 1.0.1 first).
5. **`Min/Max/Sub/Mul`** for boolean logic on already-binary values.

Softmax replacement when channel selection is needed: Hardmax or `Greater(channel_max - epsilon)`.

---
<a id="b5_static_shape_strategies"></a>

# B.5 — Static-shape strategies

The validator enforces every dim has `dim_value` — no `dim_param`. Model IO is fixed at `[1,10,30,30]` float32 (`"input"` and `"output"`). All internal activations must also be statically shaped (constant_folding requires it; shape_infer must succeed; calculate_memory walks value_info byte sizes). Not a constraint — the operating reality; the question is how to exploit it.

## What "static-shape" means here

1. The submitted model declares `[1,10,30,30]` for IO. Both input and output. **Output shape is fixed, not derived.**
2. Every intermediate `value_info` entry has `dim_value` for every dim. No `?`, no symbolic `N`.
3. Constant-folded structure (Reshape with literal shape, Slice with literal indices, Tile with literal repeats) gets pre-evaluated.
4. The 30×30 grid encodes ARC inputs of *any size up to 30×30* via padding with color-10 ("no color"). The padding is right- and bottom-aligned: the grid lives at top-left, padding fills the bottom and right.
5. Decoder strips trailing pad-color rows only. So output transformations that move the active grid to the right or bottom edge break the decoder (the project's flip caveat).

## Static-shape constraints actually have an upside

- **Free shape engine:** all shapes known at build time → all shape-input ops fold, no runtime shape compute.
- **Predictable memory:** `memory` = sum of intermediate-activation byte sizes — computable exactly at assembly; reject over-budget builds before submitting.
- **No dynamic dispatch** (`If` banned); **constant-folded `Reshape` is free** (literal shape input folds to no-op data-aliasing — zero MACs, zero memory).

## Strategy 1: pad-color sentinel everywhere

Color 10 is the "no color" channel. The validator's preprocessor pads grids smaller than 30×30 with color 10 in the bottom-right. The decoder removes trailing color-10 rows.

**Use it offensively:**
- Output the grid with sentinel-aware structure. As long as the active region stays top-left and trails fill with color 10, the decoder reproduces the smaller intended grid.
- **For any task where the output is smaller than the input or arbitrary-sized**, design the network to emit the active output at top-left and color-10 elsewhere. Then the decoder strips and reveals the right answer.

This is why the project's **transpose H↔W works on all sizes** but the **flip operations only work on 30×30 grids** — flipping the canvas moves the active region away from the top-left.

**Implementation pattern for "transformation that may produce a smaller output":**
1. Compute the transformed grid in 30×30 space, with active region at top-left.
2. After the transform, mask: replace any cell outside the active region with color 10 (channel 10 = 1, others = 0).
3. The masking can be a fixed initializer (precomputed where the active region ends, if static across the task) or computed (more expensive).

If the active-region size is fixed for the task (every example agrees on output dimensions), use a **fixed mask initializer** of shape `[1,10,30,30]`, applied via `Where(mask, color10_grid, computed_output)` — 9k MACs, one-time cost. If output size varies across examples, this approach is risky.

## Strategy 2: per-task fixed shape when train/test agree

If every visible example of task K agrees on shape (e.g. always 5×5): the data lives in `[1,10,5,5]` space padded to 30×30, so `Slice` to the small region, work there, `Pad` back. Per-element ops shrink with area: 9000 MACs at 30×30 → 250 at 5×5 (36×).

**Risk:** the shape *isn't actually fixed* on private leaderboard — ARC-GEN-style variation may produce 5×5, 7×11, 30×30 versions of the same task; slicing to a fixed sub-shape breaks then. Detecting the active region at runtime requires data-dependent shapes, which static ONNX doesn't permit — so the size must be committed at build time.

**Recommendation:** only when train + test + arc-gen all agree on shape AND the op is much cheaper at the smaller shape AND we accept the private-set risk — likely **not worth** the brittleness.

## Strategy 3: output shape via Slice with fixed indices

For outputs always smaller than 30×30 in a known way (e.g. always 15×15 top-left): compute on the full 30×30, `Slice` to the smaller shape (free), `Pad` back to `[1,10,30,30]` with color 10 (free). Simplest assembly: `Concat([sliced_active, sliced_color10_pad], axis=2)` then `Concat(..., axis=3)` — all free.

## Strategy 4: pad sentinel via color 10 (multi-region task handling)

Inputs split into sub-grids by sentinel borders: `Slice` extracts each region, process (potentially with shared weights, B.3), `Concat` back. Shape ops free; per-region ops scale with region area.

## Strategy 5: multi-model fallback per task

One ONNX file per task — **no fallback within a single ONNX**. `build_submission.py` decides at *build time* which pattern fits and builds the smallest-cost ONNX that solves the visible benchmarks; if none fits, ship the safe Identity baseline (scores 25 on output==input tasks, 0 elsewhere).

**Key insight from the scoring rules:** incorrect = literal 0. A 50% pattern-match rate on a task is **worse** than always-shipping-Identity. Pattern detectors must err toward declining (output the safe Identity model) rather than guessing.

**Concrete recommendation:** every pattern in `_PATTERNS` should compute its predicted output on **ALL visible examples** (train + test + arc-gen subset) and only commit if all match. The COVERED note about the 7-of-10-colors threshold for color-perm confidence is exactly this principle.

## Strategy 6: shape-consistent task-family detection up front

Partition tasks by shape signature before pattern matching:

| Signature | Pattern candidates |
|---|---|
| `input.shape == output.shape` and same for all examples | identity, transpose, color-perm, edge detection |
| `output.shape != input.shape` and consistent across examples | crop, scale, half-grid extraction, tiling |
| `output.shape != input.shape` and varies across examples | hard — requires variable-output handling |
| Output is uniform color | constant-output, color-of-most-frequent, constant-grid generators |

Pre-filtering avoids expensive pattern checks on tasks they couldn't solve — **pure script-time optimization, no inference cost.**

## Strategy 7: declare smaller per-task shapes if validator allows

The validator hard-codes only `[1,10,30,30]` for IO; *internal* activations can be any static shape:

```
input  [1,10,30,30]
  → Slice → [1,10, k, k]                    (zero MACs, k smaller)
  → Conv on [1,10,k,k]                      (k²·10·9 MACs, way cheaper than 30² version)
  → Pad → [1,10,30,30]                      (zero MACs, color-10 padding)
output [1,10,30,30]
```

`k` is fixed at build time; works if the task's active region is always ≤ k×k. Best case: 10×10 across all visible examples → depthwise 3×3 there is `10·1·9·100 = 9,000 MACs` instead of 81k — 9× reduction, ~2.2 pts gain.

## What to actually do

1. **Default everything to 30×30.** No shape gymnastics unless a specific task demands it.
2. **Use color-10 sentinel masking aggressively** for output-region control.
3. **Prefer transpose / identity / shape-only ops** (insensitive to active-region shape) over flips (which break on non-square active regions).
4. **Resist per-task shape narrowing** unless train + test + arc-gen (1000+ samples) prove shape invariance.
5. **Pattern detectors must verify on all visible examples** before committing — incorrect = 0, guessing wrong is catastrophic.

## A note on the validator's comparison semantics

The validator compares `(result > 0.0).astype(float)` against the expected output via `np.array_equal`. So:

- The **comparison is binary** per cell: any positive value = "this cell has this color"; zero or negative = "does not."
- Small positives (0.01) still count as 1; negatives where the expected cell is 0 are fine; negatives where it should be 1 are wrong.

**Implication:** outputs don't need to be exact 0/1 — just sign-correct. **A cheaply scaled sign-preserving network output works.** This is why raw ±1/±2 Sobel kernels solve edge tasks without normalization — only the sign side of zero matters.

---
<a id="b6_iterative_without_loop"></a>

# B.6 — Iterative without Loop

`Loop` is banned (`calculate_memory()` returns None on subgraph attrs). Anything that would have been a `for` loop must become a fixed unroll: N copies of the same Conv stage stacked in series, N chosen at build time. Cost grows linearly with N for `macs` and `memory`; weight sharing (B.3) keeps `params` flat.

The strategic question: which iteration patterns, and what minimum N? For a 30×30 grid the diameter is bounded — most "iterate until convergence" algorithms converge in ≤30 steps, and a *few* converge in `O(log diameter)` ≈ 5–6 steps. Those few are the ones worth pursuing.

## Cost of an N-deep unroll

For depthwise 3×3 Conv per step:
- Per step: 81,000 MACs + 9,000 (binarize) = 90,000 MACs.
- Plus 36,000 bytes activation memory per step (one `[1,10,30,30]` float32).
- N steps total: `90,000·N` MACs + `36,000·N` memory + 1× weight (shared across N).

Score impact of various depths:

| N steps | macs+memory | Score |
|---|---|---|
| 1 | 126,000 | 13.25 |
| 3 | 378,000 | 12.15 |
| 6 | 756,000 | 11.46 |
| 10 | 1,260,000 | 10.95 |
| 30 | 3,780,000 | 9.85 |
| 60 | 7,560,000 | 9.16 |

**Cost dominates score above ~6 iterations.** Anything that needs `O(diameter)` iterations in linear time is not a good candidate. Anything with `O(log diameter)` is excellent.

## Convergence-bound table for ARC primitives

| Primitive | Linear iterations | Logarithmic schedule | Notes |
|---|---|---|---|
| Single dilation step | 1 | 1 | trivial |
| Flood-fill 4-conn convergence | up to 60 (worst diameter) | 6 (power-iter doubling radius) | use log schedule |
| Flood-fill 8-conn convergence | up to 30 | 5 | use log schedule |
| Erosion to convergence | similar to dilate | 6 | mirror of dilate |
| Conway's Life N steps | exactly N | not log-reducible | only for fixed N |
| Connectivity (component label) | diameter | log(diameter) ≈ 6 | via repeated max-dilation |
| Gravity (cells fall) | up to 30 | not log-reducible | use linear unroll, expensive |
| Reachability mask | diameter | 6 | dilate from a seed |
| Closest-distance map | diameter | log per bit; 6 iterations × 5 bits | expensive |

The cheap wins are everything that admits the "doubling radius" trick — power iteration applied to morphological dilation.

## Power iteration for dilation (the log-iteration trick)

**Problem:** "find all cells reachable from a seed within the 4-connected component."

**Naïve:** repeatedly dilate by a 3×3 cross until convergence. 60 iterations worst case → expensive.

**Power-iter:** at each step, dilate by a kernel that **doubles the previous reach**. After `k` steps, reach = `2^k`. For diameter ≤ 30, we need `k = ceil(log2(30)) = 5` steps.

Implementation:
```
mask = seed_mask    # shape [1,1,30,30]

# Step 1: dilate by 3×3 → reach = 1
mask = Conv(mask, kernel_3x3_ones, group=1) > 0

# Step 2: dilate by 3×3 → reach = 2 (radius compounding)
mask = Conv(mask, kernel_3x3_ones, group=1) > 0

# Step 3: dilate by 5×5 (or two 3×3 in succession with a doubled "step") → reach = 4
mask = Conv(mask, kernel_3x3_ones_dilation_2, group=1) > 0

# Step 4: dilate by 9×9 effectively, via 3×3 with dilation 4 → reach = 8
mask = Conv(mask, kernel_3x3_ones_dilation_4, group=1) > 0

# Step 5: dilate by 17×17, via 3×3 with dilation 8 → reach = 16
mask = Conv(mask, kernel_3x3_ones_dilation_8, group=1) > 0

# Now reach ≥ 16. For 30-diameter grid, reach=16 doesn't quite cover; add one more.
mask = Conv(mask, kernel_3x3_ones_dilation_16, group=1) > 0  # reach = 32, > 30, converged
```

**Cost analysis** (depthwise 3×3 with dilation, single channel):
- 6 × (Conv + Greater) = 6 × (81k/10 + 9k) = 6 × 17.1k = 102.6k MACs total.
- Wait — we said depthwise with 10 channels costs 81k. With 1 channel, it's `1·1·9·900 = 8,100 MACs per Conv`. So 6 × (8,100 + 9,000) = 102,600 MACs. **Score ≈ 13.5 pts.**

If instead we run full 10-channel dilations in parallel (same cost as 1 channel under depthwise group=10), the total is `6 × (81k + 9k) = 540k` → score ≈ 11.81. Per-channel reasoning is cheaper if we can target which channel to dilate.

**Key trick: `dilation` attribute on Conv.** `Conv` accepts a `dilations=[d_h, d_w]` attribute that spreads the 3×3 kernel over a larger effective region (e.g. dilation=4 means the 3×3 covers a 9×9 area). MAC count scales with kernel size × output size — but the kernel is still 3×3 with mostly zeros — so onnx-tool counts `kH × kW` regardless of dilation. **Dilated Conv is the same MAC cost as undilated 3×3.** This is what makes the power-iter trick cheap.

## Bounded-iter flood-fill via stacked depthwise Conv

For "fill the interior of an enclosed shape" tasks (E.4):

```
boundary = input == enclosing_color    # initial mask: where boundary cells are
exterior_seed = corners_of_grid        # known exterior

# Power-iter dilate exterior, blocked by boundary
exterior = exterior_seed
for k in [1, 2, 4, 8, 16]:
    new_dilation = Conv(exterior, ones_3x3, dilation=k) > 0
    exterior = And(new_dilation, Not(boundary))

interior = Not(Or(exterior, boundary))
```

Cost: 5 power-iter steps × ~17k MACs each = 85k MACs. Plus the AND/NOT operations: ~5×9k = 45k. Total ~130k. Score ≈ 13.

Compare to a 60-step linear flood-fill: ~6.4M MACs, score ≈ 9.4. **Log schedule wins by 4 pts.**

## Power-iter for connectivity

"Are pixels A and B in the same connected component?"

1. Set `seed_at_A` to mask with 1 at A, 0 elsewhere.
2. Power-iter dilate `seed_at_A` blocked by background (5–6 steps to cover diameter).
3. Check `mask[B]`.

Cost: same ~5×17k = 85k for the dilation, plus 9k for the blocking AND step per iteration = 130k total. Score ≈ 13.

## Linear unroll patterns (when log doesn't apply)

- **Conway's Life N steps:** the rule isn't monotonic so you can't compress steps. Run exactly N. For N=2: 2 × 90k = 180k MACs → score ≈ 12.9.
- **Gravity / falling:** "if cell below is empty, swap." Each step moves cells by 1 row. 30 worst-case → 2.7M MACs → score ≈ 10.2. **Bad — better to skip these tasks.**
- **Counter increment** (count something, output the count colored): `ReduceSum` + Greater is `O(1)`, no iteration needed.
- **Sequential transformation** (apply rule R to leftmost X, then second-leftmost, …): no log trick. Linear cost. Likely unsolvable cheaply.

## Depth-vs-iteration tradeoff

Width is forced at 10 channels; depth is the only knob. **For ARC, deeper beats wider** — the transformations are about spatial reach, not channel count. (Counter-example: one-shot multi-channel comparisons, e.g. "per-pixel max across colors" = a single ReduceMax at depth 1.)

## Practical iteration patterns for the project

### Pattern: "fill the inside of enclosed shape" (E.4)
- Detect background color. Detect boundary color.
- Power-iter dilate from corners, blocked by boundary.
- Output: pixels not reached by exterior dilation.
- Cost: ~130k MACs → 13 pts.

### Pattern: "connect A to B if they're in the same component"
- Per-pixel dilation from each pixel is too expensive. Better: label propagation (Hoshen-Kopelman in convolutional form), probably 6 power-iter steps + a final `Equal` test.

### Pattern: "expand each colored cell to fill its quadrant"
- 30×30-kernel Conv: 810,000 MACs per output channel — too much. Two-pass separable dilation (1×30 then 30×1): 540,000 — still ~11 pts, bad.
- **Actual answer:** if the fill is uniform, use `Tile` after a `Slice` that picks one cell. Free.

### Pattern: "iterate the rule N times" (small fixed N)
- Inline the rule N times with shared weights (B.3); linear in N, small N (≤6) tolerable.

## Memory pressure of unrolling

Each unrolled stage produces an intermediate `value_info` entry counted by the validator's `memory` metric (`[1,10,30,30]` float32 = 36,000 bytes). 6-deep unroll: 216k bytes (plus binarized intermediates) vs `macs` ~540k — the smaller term but non-negligible.

Bool-cast relief does NOT work: bool tensors can't feed Conv, so the value must be Cast back to float at every conv input — nothing actually saved. **Better pressure-relief:** keep each stage as small as possible — if only one color's binarized presence matters, slice to `[1,1,30,30]` first; intermediates are 1/10 size.

## Summary

- **Log-schedule power iteration is the killer move** for any reach/connectivity/flood-fill task. 5–6 stages of dilated Conv get you to `score ≈ 13` for 30-diameter problems.
- **Linear unrolls are tolerable up to ~6 stages**, painful past that.
- **Always share weights across iterations** via repeated initializer references (B.3).
- **Binarize between stages** when feasible; saves nothing on memory but limits numerical drift in the stack.
- **Tasks needing >10 unrolled steps are likely best skipped** (better to ship the safe Identity baseline).

---
*Evicted textbook tier (LUT construction, activation menu, weight-sharing survey): `_archive/tiny_architectures_evicted_2026-06-10.md`.*
