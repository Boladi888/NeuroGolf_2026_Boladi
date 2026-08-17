# tiny_architectures.md — evicted textbook tier (boolean-op table, LUT construction, weight-sharing survey, activation menu)
Evicted 2026-06-10, refactor pass 4 — charter: the live tree keeps only what a Sonnet-class model doesn't already know; these sections are preserved verbatim under their original headings so title-grep still finds them.
Live home: `Onnx_Advanced/tiny_architectures.md` · full pre-refactor snapshot: `_archive/tiny_architectures_pre_refactor_2026-06-10.md`.

---

# B.1 — Hand-crafted weights (evicted block)

> Live file keeps a one-line pointer (boolean ops via arithmetic: techniques.md #57 scaffold / #125 micro-idioms); this block duplicated that content.

### Boolean logic via min/max/mul/add

Color comparisons in ARC reduce to per-pixel boolean ops on the 10 color channels (which are one-hot after the validator's preprocess). The operators:

| Logic | ONNX expression | Cost on `[1,10,30,30]` |
|---|---|---|
| **AND**(a,b) | `Min(a, b)` if 0/1; or `Mul(a, b)` | 9,000 |
| **OR**(a,b) | `Max(a, b)` if 0/1; or `Add(a, b)` then `Min(_, 1)` | 9,000–18,000 |
| **NOT**(a) | `Sub(1, a)` | 9,000 |
| **XOR**(a,b) | `Sub(Add(a,b), Mul(2, Mul(a,b)))` (3 ops); or `Equal(a,b)` then `Sub(1,_)` (2 ops, no broadcast); cheapest: `(a + b) % 2` via `Sub(Add, Mul(2,_))` | 18,000–27,000 |
| **MUX**(mask, a, b) | `Where(mask, a, b)` | 9,000 |
| **A implies B** | `Or(NOT(a), b)` = `Min(Add(Sub(1,a), b), 1)` | 18,000–27,000 |
| **A iff B** | `Equal(a,b)` then `Cast` | 9,000 |

These let us build any per-pixel boolean function in ~9k–27k MACs. **Threshold detection**: `Greater(x, 0.5)` then `Cast` is the most robust binarizer at 9,000 MACs — no Cast cost in MACs.

---

# B.2 — LUT-as-network

> Live file keeps the distilled summary (win conditions + size crossover); the worked constructions below are the evicted bulk.

The ARC input domain is *discrete*: each pixel is one of 12 channels, and the input arrives as a one-hot tensor `[1,10,30,30]` (channels 0–9 are colors, channel 10 is no-color pad, channel 11 is the "too many colors" decode error sentinel — but the validator's runtime input has 10 channels per the hard-coded shape). For discrete domains, lookup tables dominate over learned linear filters: a single `Gather` op on a precomputed table is **zero MACs and a tiny initializer**. This is the cheapest way to express any function from a small finite domain to anything.

## What "LUT-as-network" means here

For the project, three useful flavors:

1. **Channel/color LUTs.** Functions of the form `output_channel = f(input_channel)` — color permutations, color-pair logic, palette remappings. Single `Gather` on a 10-entry table.
2. **Pixel-pattern LUTs.** Functions of `(input_at_pos_i, input_at_pos_j)` → output, where indices are positions in a fixed neighborhood. A 3×3 binary neighborhood has 2^9 = 512 patterns; a 12-color 3×3 has 12^9 = 5.16 billion patterns (too many — more on this below).
3. **Region-encoded LUTs.** Encode a region (3×3 patch, row vector, etc.) as a single integer index, then `Gather` the precomputed answer. Encoding step costs MACs; lookup is free.

The project's existing `color-perm-via-Gather` is type (1). Types (2) and (3) are unexplored.

## Type 1: channel/color LUTs (the gold standard)

A 10-entry int8 permutation table addresses any color permutation in 10 bytes of params, zero MACs. Score: 25 pts/task.

**Variants:**
- **Permutation only** — `Gather` along the channel axis. Already deployed.
- **Color folding** (multiple input colors → same output color) — same Gather, but the table values repeat. Still 10 bytes, still 0 MACs.
- **Conditional remap** ("if color c, swap with d, else identity") — same Gather; the table is mostly identity with two swapped entries.
- **Dropping a color** ("blank out all blue cells") — Gather with that channel mapped to channel 10 (no-color).
- **Color count → output color** ("if there are exactly 5 reds, output green; else output black") — needs counting first (B.1's ReduceSum on the red channel), then a *value-indexed* lookup. The count is a scalar `[0..900]`; we'd Gather a 901-entry table. Cost: 0 MACs (Gather), 901 bytes (uint8 table). Actually beautiful — but only works when output is a constant (single color repeated everywhere). Most counting tasks have a more complex output.

## Type 2: 3×3 binary neighborhood LUT (the connectivity gateway)

Many ARC tasks reduce to "for each cell, look at its 3×3 binary neighborhood, output some function." Examples:
- Skeletonization, dilate, erode (B.1 covers Conv-based versions; LUT can be cheaper).
- Cellular automata steps (B.6).
- Corner detection, junction detection.
- "Hole filling" (interior detection).

A 3×3 binary neighborhood has 9 binary positions = 512 patterns. The LUT can be precomputed offline for any rule.

**ONNX implementation sketch:**
1. Encode the 3×3 neighborhood as a 9-bit integer per pixel, using a 3×3 depthwise Conv with weights `[[1,2,4],[8,16,32],[64,128,256]]`. Output shape `[1, ch, 30, 30]`, values in [0, 511].
2. `Cast` to int64 (or int32, validator-permitting).
3. `Gather` a 512-entry initializer holding the precomputed rule outputs.

**Cost:**
- Step 1: 81,000 MACs (depthwise 3×3) — same as any 3×3 Conv.
- Step 2: free Cast.
- Step 3: zero MACs.
- Initializer: 512 entries × 1 byte (uint8) = 512 bytes per channel-rule.

**Compare to direct B.6 unrolled stack:** for a *single rule application*, the LUT path costs the same as one Conv stage (~81k MACs). The LUT win is when **the rule is non-monotonic** (e.g. corners, T-junctions) and would otherwise need multiple Convs (Conv → threshold → AND with shifted versions, ~3 Convs minimum). LUT collapses that into 1 encoding Conv + 1 Gather.

For our scoring: 90,000 MACs (encoding + Greater binarization if needed) → ~13.6 pts/task. **No advantage over a single Conv** if the rule is monotonic. LUT pays off only for irregular boolean functions of the neighborhood.

**Risk:** a 3×3 *color* neighborhood (12^9 = 5.16B states) is impossibly large. We'd need to factorize: encode "is each of the 9 cells a particular color?" as 9 separate one-bit channels (one per asked color), then run a 9-bit binary LUT per query color. Costs scale with the number of color queries.

## Type 3: row/column/region encoding LUTs

For tasks where each row of the grid maps to a row of the output independently (e.g. "swap the first and last columns"):
- Encode each row as an index (e.g. via Gather + linear-combination if values fit).
- Lookup answer rows from a precomputed table.
- Output assembled via Concat / Reshape.

For 30×30 grids with up to 12 colors per cell, a row has 12^30 ≈ 4×10^32 distinct values — too many for a literal LUT. **We'd need to factor by row structure**: encode position-of-distinct-colors, run-length encoding, etc. Quickly becomes more expensive than a Conv solution for any non-trivial task.

**When this works:** binary (or near-binary) rows, length ≤ 8. 2^8 = 256 entries per LUT. Doable if the task constrains rows that small. Rare.

## OneHot + MatMul color encoding

The validator's input is already one-hot per color (10-channel `[1,10,30,30]`). So we don't need an explicit `OneHot` op — the encoding is given. But for *output* generation:

- If we computed an integer index per cell (e.g. via the 3×3 binary LUT), we'd want to convert it back to a 10-channel one-hot grid for the final output.
- `OneHot(indices, depth=10, values=[0,1])` produces the channel-axis-expanded tensor. Cost: ~9000 MACs (1 per output element).
- Alternative: pre-bake the one-hot output values into the LUT itself. Lookup directly returns a 10-vector. Initializer becomes 10× larger but no `OneHot` step needed. Often cheaper overall.

## Pixel-pair LUTs (100-entry)

For tasks involving two-grid combinations (XOR of stacked grids, "color result based on input pair"), we can encode `(color_A, color_B)` as a 10×10 pair-index = 100 entries.

**Implementation:**
1. Two-grid case: input is conceptually two stacked `[1,10,30,30]` activations, one per grid. ARC tasks present this as two halves of a single input, separable via `Slice`.
2. ArgMax along channel axis on each grid → integer color labels in `[0..9]`.
3. Combine: `index = colorA * 10 + colorB` via `Add(Mul(colorA, 10), colorB)`. Cost: 9k (Mul) + 9k (Add) = 18k MACs.
4. Gather from a 100-entry palette of (output) one-hot vectors. Free.

**Cost:** 9k (ArgMaxA) + 9k (ArgMaxB) + 18k (combine) + 0 (Gather) = **36,000 MACs** → ~14.5 pts/task.

Plus ~100 × 10 = 1000 bytes for the pair-output table (one-hot encoded).

**Compare:** without LUT, the same pair logic via per-pixel conditionals would be Greater + Where chains, easily 100k+ MACs. LUT wins by ~2×.

## Pixel-triple LUTs (1000-entry)

Three-color combinations: 10^3 = 1000 entries. Same construction as pairs but with three index combines (Mul, Add, Mul, Add). Cost: ~54k MACs. Edge cases.

## LUT compression: factorization, Kronecker, hashing

When the LUT is large but structured:

- **Factorization**: if `f(a,b) = g(a) * h(b)` (separable), use two small LUTs and one Mul. 100→20 entries, +9k MACs for the Mul. Worth it if `g`, `h` are different per query.
- **Kronecker products**: similar idea for higher-dimensional LUTs.
- **Hashing (HashedNets)**: not applicable — we want exact, not lossy, lookups for ARC's discrete correctness.
- **Symmetry reduction**: if `f(a,b) = f(b,a)`, store only the upper triangle (55 entries instead of 100). Need a "min/max symmetric encoding" step before lookup, costing 18k MACs (Min + Max + linear combine).

For the project's scale (≤ 1000-entry LUTs), compression rarely pays — the params we'd save are dwarfed by the encoding overhead.

## When LUT-as-network beats Conv-as-network

A LUT win condition:

1. The function being computed is **inherently discrete** (operates on color labels, not continuous values).
2. The function is **irregular / non-monotonic** (otherwise a single Conv is just as cheap).
3. The input domain is **small enough** (≤ ~1000 entries for params, ≤ ~10000 for the output expansion).
4. The encoding step (mapping inputs to indices) is **cheaper than the alternative arithmetic**.

For ARC, condition 1 is almost always met. Condition 2 holds for: cellular-automaton rules, junction/corner detection, boolean combinations with many cases, color-pair semantics. Condition 3 is the bottleneck — color-triples and beyond are too big.

## Concrete LUT patterns to add to `_PATTERNS`

| Pattern | Index domain | Encoding cost | Lookup cost | Total | Score |
|---|---|---|---|---|---|
| Color permutation | 10 colors | 0 (already one-hot) | 0 | 0 | 25.0 |
| Color folding (many→one) | 10 colors | 0 | 0 | 0 | 25.0 |
| Color → output-color via count | 1 scalar (0..900) | ~9k (ReduceSum) | 0 | 9k | ~16.2 |
| Color-pair → output color | 100 entries | ~36k (2 ArgMax + combine) | 0 | 36k | ~14.5 |
| 3×3 binary neighborhood rule | 512 entries | ~81k (Conv encode) + 9k (binarize) | 0 | 90k | ~13.6 |
| Boolean function of two binary masks | 4 entries | ~18k (combine) | 0 | 18k | ~15.2 |

The first two are the only LUT patterns that beat 16 pts/task. The rest tie or lose to a single hand-crafted Conv. **LUTs win cleanly only for pure-color-permutation/folding tasks** where they already match the project's deployed pattern.

## Action items for build_submission.py

- Extend the existing color-perm Gather to detect "color folding" cases (where the inferred map collapses multiple inputs to one output). Same operator, more solved tasks.
- Add a "constant output" LUT pattern: if the expected output is the same color in all visible cells regardless of input, emit `ConstantOfShape` (or a shape-matched broadcast) of that color. Zero MACs (modulo the constant-folding caveat in A.4).
- Skip pair/triple LUT patterns until a specific task is identified that needs them; they don't win on cost alone, only on solving tasks Conv can't.

---

# B.3 — Weight sharing (evicted blocks: literature/survey prose)

> Live file keeps the empirical bits (params-dedup rule, repeated-initializer pattern, depthwise reframing, cyclic-shift, low-rank cost math, symmetry-storage folding finding).

Original framing: Two angles matter for our cost metric: (a) onnx-tool counts each *unique* initializer once toward `params`, so reusing the same tensor name across multiple nodes is free in `params`; (b) a smaller filter shape gives proportionally fewer MACs. Weight sharing exploits both. The neural-network literature mostly cares about training-data efficiency; we care about deploy-time bytes/MACs. Most of the academic ideas don't transfer — the few that do are the ones with clean ONNX implementations.

## Weight-sharing techniques and their ONNX viability

| Technique | Idea | ONNX implementation | Useful for us? |
|---|---|---|---|
| **Repeated initializer reference** | Same `name` referenced N times | trivial — emit two Conv nodes pointing to one initializer | **Yes — free params reduction in unrolled stacks (B.6)** |
| **Depthwise separable Conv** | `groups = Cin = Cout` reduces MACs by `Cin×` factor | native ONNX (`Conv` with `group` attribute) | **Yes — already covered in B.1, big MAC win** |
| **Group Equivariant CNN (p4/p4m/D4)** | One kernel parameterizes 4 (or 8) symmetric copies | extra channel dimension (×4 or ×8); requires `Concat` + `Transpose` of rotated copies | **Limited — G-CNN multiplies activation tensors by 4, ballooning `memory`. Net loss for us unless the task NEEDS rotational equivariance** |
| **Cyclic-shift-tied weights** | Conv kernel = cyclic shifts of one row | one initializer + Concat/Transpose | **Niche — shifts cost MACs that often outweigh the savings** |
| **HashedNets / ArbNets** | Hash bucket many weights into few storage slots | not native — needs extra Gather ops at inference | **Loss — Gather adds bytes to graph, no MAC help** |
| **Kronecker-factored weights** | `W = A ⊗ B` with `A`, `B` small | reconstruct via Reshape + matmul tricks | **Loss — reconstruction MACs > savings** |
| **Low-rank decomp (`W = U V^T`)** | Replace one big matmul/Conv with two small | two Convs/Gemm in series | **Possibly — see below** |
| **Steerable / circular kernels** | Parameterize kernel by basis functions | needs custom op or more Convs | **No — extra cost** |
| **Symmetry-reduced storage (half kernel)** | Store one half, reconstruct via flip | initializer + Concat/Slice | **Possibly — only meaningful for symmetric kernels** |

## Tied-across-channels — the G-CNN angle (non-depthwise sharing)

For non-depthwise sharing patterns:
- **All output channels use kernel rotated 90°/180°/270°/flipped** — this is the G-CNN angle. Implementation: store one kernel `[Cout/4, Cin, kH, kW]`; emit 4 Conv nodes, each referencing the *same* kernel after a hand-precomputed rotation. But "rotation of an initializer" requires either pre-rotating offline (then storing 4 separate but identical-byte-count tensors — *no params savings if they're all distinct names*) or computing the rotation at graph time (Concat+Transpose+Slice on the kernel — then it's not foldable back to an initializer, blowing up memory).
- **Verdict:** G-CNN's parameter savings only materialize if the implementation can *symbolically* reuse a kernel across rotations. ONNX has no such mechanism. The savings evaporate.

## Key literature pointers (cited; minimal value to project)

- Cohen & Welling 2016 (G-CNN, arXiv 1602.07576) — formal foundation, but ONNX cost model nullifies the savings.
- Lengyel 2021 (arXiv 2106.04914) — empirical study of G-CNN parameter redundancies; relevant only if we ever train.
- HashedNets (Chen et al. 2015) — irrelevant for our deploy-time cost metric.
- Deep Compression (Han et al.) — pruning + quantization + Huffman; the quant + pruning parts overlap with A.2; Huffman is not natively ONNX.

---

# B.4 — Nonlinearities (evicted block: generic modern-activation menu)

> Live file keeps the decision table with discrete/useful rows + the verdict "smooth activations are never exact-safe here; use Greater+Cast / Clip / Relu". These are the evicted generic rows of "The MAC-cost menu (per `[1,10,30,30]` activation)".

| Op | Cost | Behavior | Verdict |
|---|---|---|---|
| **Neg** | 9k | `-x` | flip sign |
| **Floor / Ceil / Round** | 9k | discretize float to int | rarely useful, we get Greater for free |
| **Sigmoid** | 9k | smooth `1/(1+e^-x)` | wasteful — costs same as Greater but soft |
| **Tanh** | 9k | smooth `(-1, +1)` | same — wasteful |
| **HardSigmoid** | 9k | piecewise-linear sigmoid | no advantage over Greater |
| **LogSoftmax** | ≈ 3·input_elems | log of Softmax | avoid |
| **PRelu** | 9k + slope param | parametric ReLU | not useful for ARC |
| **LeakyRelu** | 9k | `max(αx, x)` | no advantage over ReLU |
| **Elu** | 9k | `exp`-based smooth ReLU | wasteful |
| **Selu** | 9k | scaled Elu | wasteful |
| **GELU / SiLU / Mish** | composite ops, ≈ 18-27k | smooth modern activations | avoid |
| **Erf, Exp, Log, Pow, Sin, Cos, Tan, Sqrt** | 9k each | special functions | one-shot transforms; rarely useful |

## Things to avoid (original prose)

- **Softmax / LogSoftmax** — 3× the cost of `Greater + Cast` for the same downstream effect (selecting one channel). Replace with `Hardmax` or `Greater(channel_max - epsilon)`.
- **Sigmoid / Tanh / GELU / Swish / Mish** — same cost as ReLU, all of which can be replaced by Greater + Cast for our use case. Don't use unless we need calibrated probabilities (we don't).
- **Erf, Exp, Log** — useful only in cost-insensitive networks. Each is 9k MACs and onnx-tool's billing for special functions is "1 MAC per element" but the doc warns "Special function's MACs are not accurate." That said: from the validator's perspective they're 9k regardless. Just pointless for ARC.
- **Activation chains** — every activation = 9k. Don't stack them. ReLU + Sigmoid would be 18k for what should be one threshold.


---

## [completeness sweep 2026-06-10] snapshot lines not carried into live or the sections above

> Restored verbatim from the pre-refactor snapshot after a containment audit found the
> compression pass had dropped them from both live and this file. Mostly MAC-era points math
> and dtype-packing claims (dead under current scoring — see scoring_internals.md) plus
> stray prose; preserved here so nothing exists only in the snapshot. Lines appear in
> snapshot order, context-free; the snapshot has the surrounding text.

> Consolidated from `Onnx_Advanced/Research/tiny_onnx_arch/` on 2026-06-01 for discoverability (see INDEX.md). Originals removed; this is the single home.
The lever: for any task whose transformation can be written as a fixed linear filter (or a small composition of them), we can ship the weights as initializers, set MACs explicitly via the Conv structure, and skip training. Because onnx-tool's MAC counter is dtype-blind, the cost depends only on the *shape* of the kernel and the *arrangement* (depthwise vs dense, group, dilation), not on the values. So we pick the smallest shape that does the job, and we do it analytically.
These are pure shape-only operations:
- **Horizontal flip** = `Slice` with `step=-1` (free) — but careful with non-30×30 grids; see COVERED caveat.
- **Vertical flip** = same, different axis.
- **Diagonal mirror** = `Transpose(perm=[0,1,3,2])` — free, the project's working pattern.
- **Anti-diagonal mirror** = `Transpose` + `Slice` reversal.
- **Detection of "is this image symmetric?"** = `Equal(image, transposed_image)` then `ReduceMin` → boolean. Cost: 9k (Equal) + 9k (ReduceMin output is scalar). Then mask the whole image with the broadcast scalar. Total ~18k MACs.
For tasks asking "is this symmetric, if so do X else do Y", we don't need the literal answer — we can do both transformations and `Where(symmetry_mask, X, Y)`. That's cheaper than literal branching.
If a task involves combining an "object" channel with a "shape" channel:
- **AND**: `Min` (per-pixel, per-channel where both are 1).
- **A−B**: `Mul(a, Sub(1, b))` = "a present where b is absent".
- **Symmetric difference** = XOR.
- `Slice` to crop the source region.
- `Pad` with color-10 (no-color) on the destination side.
Gravity (cells fall down until they hit something):
- Iterative — falls into B.6 (unrolled). Worst case: 30 iterations of a "shift down by one if cell below is empty" depthwise convolution. ~30 × 81k = 2.4M MACs → ~10.3 pts/task.
For tasks where the gravity step is *bounded by the grid size*, we can unroll fewer steps if we know the worst-case fall distance from the visible examples. Risky — might fail on private set if the fall distance varies.
- Depthwise Conv with kernel of all-1s, then `Greater(0)` then `Cast`.
- Cost: 81,000 (Conv) + 9,000 (Greater) = 90,000 MACs per dilation step.
Erode (binary AND of 3×3 neighborhood):
- Depthwise Conv with kernel of all-1s, then `Equal(9)` then `Cast`.
Useful for: object thickening/thinning, finding interior cells (erode then sub), connectivity checks (B.6).
For tasks like "find the 3×3 cross pattern":
- Define a template `T` as a 3×3 binary mask.
- Convolve with depthwise Conv weights = `T` minus a center scalar.
- Threshold on `Equal(_, sum(T))`.
- Match positions are 1; non-matches are 0.
Cost: 81k (Conv) + 9k (Equal) = 90k per template. Same as a binarized edge detector.
If the task is "split the grid in half, XOR the two halves":
- `Slice` left half, `Slice` right half (free).
- `Sub` then `Abs` (or `Equal` + `NOT`) — 18k MACs.
- Output shape needs handling — if output is `[1,10,30,15]`, must `Pad` back to 30×30 with color-10. Free.
Total: ~18,000 MACs → ~15.2 pts/task. Plus shape-only operations stay free.
Hand-crafted Sobel weights are `±1, ±2`, fits trivially in int4 (signed range −8..7) or int8 (−128..127). Storage:
- Sobel-X 3×3 = 9 values. As int8: 9 bytes. As float32: 36 bytes. As uint4: 5 bytes (packed). 4–7× savings on this initializer.
- For 10 channels each with its own (depthwise) 3×3 kernel: 90 values total. Float32 = 360 bytes. Uint4 = 45 bytes.
1. Store weights as int8 raw_data (or uint4 if opset 21 accepted — A.4 follow-up).
2. Insert `Cast` to float32 immediately before the Conv. Cast is zero MACs.
3. Conv operates in float32. Same MAC count. Less `params`.
**Caveat:** the `Cast` from int8 to float32 produces a float32 *intermediate activation*, which is what the validator counts in `memory`. So this saves `params` but spends `memory` only if we'd have had a float32 intermediate anyway — which we do, since the Conv input must be float32.
Listing exact weight matrices the project should bake in as initializer templates. All are 3×3 unless noted.
# Edge detection (binarize via Greater(0))
sobel_x   = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]                # 4-byte signed range
sobel_y   = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
prewitt_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]                # uint4-friendly via offset
prewitt_y = [[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]]
scharr_x  = [[-3, 0, 3], [-10,0,10], [-3, 0, 3]]                # bigger range, needs int8
scharr_y  = [[-3,10,3 ], [ 0, 0, 0], [-3,-10,-3]]
lap4      = [[ 0, 1, 0], [ 1,-4, 1], [ 0, 1, 0]]
lap8      = [[ 1, 1, 1], [ 1,-8, 1], [ 1, 1, 1]]
blur3     = [[ 1, 1, 1], [ 1, 1, 1], [ 1, 1, 1]]   # divide by 9 = Mul(_, 1/9), or use it un-normalized + threshold scaled accordingly
All of these fit in `int4` *if we pre-shift them by a known offset*: e.g. `sobel_x + 2` ranges from 0..4, fits in uint4. Then a single `Sub(2)` after the Conv recovers the true gradient — but `Sub` costs 9k MACs. Whether worth it depends on `params` vs `macs` tradeoff; usually keep `int8`/`int4` symmetric and skip the offset.
- Add `_PATTERNS` entries for: Sobel-X edge map, Sobel-Y edge map, Laplacian-4 edge map, full edge magnitude (both Sobels + abs + add). Detect by computing the pattern on the visible examples and `np.array_equal` against expected output.
- Add color-set-overlap patterns: AND/OR/XOR of two grids. Useful for tasks where the input has a clear "two grids stacked" structure.
- Add a 3×3 morphological dilate template for "thicken the object" tasks.
The detection logic in `build_submission.py` already does pattern→model mapping; new patterns plug in there. Cost-aware: a pattern that costs 90k MACs (one Conv) is only a win if it actually solves the task — incorrect = 0, so the safety check is mandatory before shipping.
## What "shared" means at the validator's eye
`params = sum_{unique tensor names} bytes(weight_tensor)`. Same name → counted once. So:
- Stack 6 Conv nodes that each reference an initializer named `"W"`. Param cost = 1 × bytes(W). MAC cost = 6 × per-Conv MACs (memory and macs both scale with depth; only `params` is dedup'd).
- Tie kernels across channels (all channels use the same 3×3 kernel inside one Conv) — this is just **depthwise Conv with a reshape trick** OR a **dense Conv with a tiled weight tensor**. The latter wastes bytes; the former is what we want.
Implication: weight sharing is most useful for **iterative/unrolled stacks** (B.6), where the same kernel is applied N times. For single-Conv tasks, there's nothing to share.
`params` counts `W` once (90 elements × dtype bytes). `macs` counts 6 × 81000 = 486k. `memory` counts 6 intermediate `[1,10,30,30]` activations + 6 binarized ones. `params` term scales with N — ours stays flat.
**Savings calculation** for a 6-deep stack with 10-channel depthwise 3×3:
- Without sharing: 6 × 90 × 4 = 2160 bytes (float32). With uint4: 270 bytes.
- With sharing: 90 × 4 = 360 bytes (float32). With uint4: 45 bytes.
In score terms, 2160→360 bytes off a base cost of ~480k MACs is about a 0.4% reduction — a rounding error against the macs term. **Useful only when params is the dominant cost** — which it usually isn't, except for very-shallow/wide models.
**Where sharing actually pays:**
- A single-Conv-with-many-channels model where Conv weights dominate. E.g., a `[100, 100, 3, 3]` dense Conv would have 90,000 weights = 360KB float32. If we tied the channels (use a `[1, 100, 3, 3]` kernel broadcast — actually that's just depthwise!), we'd cut to 9KB. So depthwise Conv **is the canonical weight-sharing pattern** for our setting.
The classic "tie all output channels' kernels" idea, in our 10-channel setup, is exactly depthwise Conv. Both the literature insight and the ONNX implementation collapse to: **set `group = Cin = Cout`**. Already in our B.1 toolbox.
- Store one row of 3 values.
- Reconstruct via Tile + Slice + Concat — all zero MACs.
- Final kernel has 9 values from 3 stored values: 3× params savings.
For the project: 3-byte initializer instead of 9-byte. Saves 6 bytes. **Negligible.** Not worth the build complexity.
`Conv` with weight shape `[Cout, Cin, kH, kW]` ≈ Conv `[r, Cin, 1, kW]` then Conv `[Cout, r, kH, 1]` — separates the spatial filter into a horizontal and vertical pass.
For 3×3 dense Conv (Cout=Cin=10):
- Single Conv: 810,000 MACs, 360 bytes (uint4).
- Two-pass (1×3 then 3×1, both dense): each is `10 × 10 × 1 × 3 × 30 × 30 = 270,000` MACs → 540,000 MACs total. **Cheaper by 1.5×.**
- Two-pass depthwise: `10 × 1 × 1 × 3 × 30 × 30 = 27,000` MACs each → 54,000 total. **15× cheaper than the single dense 3×3.**
Plus separate intermediate activation memory of 36KB. Net effect: separable depthwise saves macs at the cost of one extra activation in memory.
- Dense 3×3: cost 810k → 11.39 pts.
- Depthwise 3×3: cost 81k → 13.69 pts.
- Two-pass depthwise (3×1 then 1×3): cost 54k + 36k memory = 90k → 13.59 pts. **About the same as depthwise, slightly worse due to memory term.**
So separable buys very little over depthwise; depthwise is already the right answer. **Skip low-rank decomposition.**
A horizontally-symmetric 3×3 kernel `[[a,b,a],[c,d,c],[a,b,a]]` has 4 unique values out of 9. Storage saving: 4 vs 9 = ~55% reduction.
- Store the 4 unique values as an initializer `[a, b, c, d]`.
- Use `Gather` to scatter into a 3×3 layout via a precomputed index tensor.
- The reconstruction is folded by `constant_folding=True` since both Gather inputs (data + indices) are initializers — back into a 9-element tensor as an initializer.
**Result: zero post-folding savings.** The reconstruction folds away and the materialized weight is 9 bytes, same as without symmetry encoding.
**However**, the *file bytes* (what counts toward the 1.44 MiB cap) before constant folding are smaller because the saved file holds 4 values + a small Gather node + index initializer instead of 9 values. But the validator runs `constant_folding=True` *before* measurement, and the cap is on file bytes (pre-folding measurement). So:
- File bytes = bytes pre-folding (lower with symmetry encoding).
- Params (folded) = bytes post-folding (same as direct storage).
For a 3×3 symmetric kernel: file might shave a few bytes via the symmetry encoding, then `params` is the same after folding. **Useful only if we're at the file-byte cliff.**
## What weight sharing cannot do for us
- **Reduce MACs in non-iterative graphs.** A single-Conv solution has no shared weights to dedupe. The trick only helps when we already have multiple Convs.
- **Reduce activation memory.** Sharing weights doesn't share intermediate activation tensors — those are produced fresh per node.
- **Help with shape-only solutions.** Identity/Transpose/Slice/Gather solutions have no weight tensors at all.
## Practical guidance for the project
1. **Always use depthwise (`group=10`) for 10-channel transformations** unless the task genuinely requires cross-channel mixing.
2. **In any unrolled stack (B.6), reference the same initializer by name across iterations.** Rare params savings, but free and harmless.
3. **Don't bother with G-CNN, low-rank decomp, Kronecker, or HashedNets** — they don't translate to ONNX cost savings.
4. **Symmetry-reduced storage is a file-byte trick only**, useful at the 1.44 MiB cliff and not for `params`. Skip unless we're hitting the cap.
The cost question for any nonlinearity in our setting is binary: does it cost more than `output_elems = 9000` MACs per `[1,10,30,30]` activation? Most stdlib nonlinearities cost exactly `output_elems` (1 MAC per element per onnx-tool's accounting). A few cost more (Softmax, LayerNorm composites). The interesting ones are those that achieve a **discrete decision** at `output_elems` cost or below. For ARC, the decisions are almost always "is this pixel one color or another" — sharp threshold, not soft. So we want the cheapest hard-discrete operator.
## The MAC-cost menu (per `[1,10,30,30]` activation)
Source: A.1's per-op table. All numbers below in MACs.
| **Softmax** | ≈ 3·input_elems = ~27k | smooth normalized | **avoid — 3× cost of Greater + Cast** |
## What we should reach for first
### Greater + Cast for hard threshold (the cheapest discriminator)
For "binarize this activation":
Cost: **9k MACs total** (Cast is free). Output: float32 0/1.
This is functionally equivalent to ReLU+threshold or Step or Sign — all of which also cost 9k. **No nonlinearity is cheaper than this** for binary discrimination. Use Greater whenever a sharp 0/1 output is needed.
For the project's existing color-perm-via-Gather pattern, no nonlinearity is needed — the input is already discrete. For tasks where we'd otherwise insert ReLU between Convs, prefer `Greater + Cast` if the Conv output's sign matters (edge-detected → present/absent).
### ReLU vs Greater: when to use which
- **ReLU** (`max(0, x)`): preserves positive magnitude, zeros negative. Useful when downstream depends on the magnitude (e.g. a second Conv summing edge magnitudes).
- **Greater + Cast**: collapses to 0/1. Loses magnitude. Use when downstream just needs presence/absence.
In ARC, downstream is almost always presence/absence (the validator's correctness check thresholds at `> 0.0` anyway). **Default to Greater + Cast.**
### Where as the universal multiplexer
`Where(cond, true_val, false_val)` is the cheapest replacement for `If`. Cost: 9k MACs (1 per output elem). Cheaper than mask-multiply (4 ops, 36k MACs).
- Conditional color application: `Where(mask_red, green_grid, blue_grid)`.
- Sentinel handling: `Where(input_eq_color10, output_eq_color10, computed_output)` — preserve pad regions through any computation.
- Branch-free if-else logic in graph form.
- `Min(a, b)` = AND. 9k MACs.
- `Max(a, b)` = OR. 9k MACs.
For continuous values, `Min(x, 1.0)` clamps to a max — useful as a saturation step after addition of two binary masks (`Min(Add(a, b), 1)` = OR, 18k MACs total — but since `Max(a, b)` is one op at 9k, prefer Max).
### ArgMax for "pick the most-present color"
Counting tasks (E.6) often boil down to:
Then `OneHot(winning_color, depth=10)` to expand back to a 10-channel mask, ~9k MACs, then `Tile` or `Expand` to 30×30 — free.
Total: ~27k MACs for "fill the whole grid with the most frequent color." Score: ~14.8 pts.
### Hardmax as ArgMax alternative
`Hardmax(x, axis)` produces a one-hot vector of the argmax position. Cost roughly = ArgMax + OneHot fused, ~18k MACs.
Sometimes cheaper than `ArgMax → OneHot` separately if onnx-tool fuses them — **verify in 1.0.1**. Otherwise the separate path is safer.
The COVERED list says "Argmax allowed (validator silent — verify)." Means: the banned-op list doesn't include Argmax, but we haven't observed the validator profiling it. Risk: subtle differences between onnx-tool's MAC formula for Argmax (~input_elems) and what we'd predict.
**Action item:** before relying on Argmax, locally instrument an Argmax-only model and compare onnx-tool's reported MACs against our 9k expectation. Same for Hardmax.
Total: 18k MACs. Replaces ArgMax + OneHot (which is also ~18k) with shape-matching that broadcasts directly. Use when result must be a mask, not an index.
Useful for "second most frequent color":
Cost: 18k. Yields a single-bit per channel. Multiple "is color X present" queries can share the ReduceSum across channels.
Cost: 18k. Produces a 10-channel single-pixel mask of "which color has count N." Useful for "find the unique-count color" tasks.
## Implementation notes for build_submission.py extensions
When adding new patterns, prefer the following order of nonlinearity choices:
Avoid Softmax, Sigmoid, Tanh, and the smooth-modern activations entirely. They cost the same as the simpler discrete ops without delivering anything useful for ARC's discrete domain.
## Top-line recommendations
- **Default to Greater + Cast** for any threshold need.
- **Default to Where** for any branching need.
- **Default to depthwise + Greater** for any "is this region X" detection.
- **Avoid Softmax** — replace with Hardmax or Greater.
- **Verify Argmax/Hardmax MAC accounting locally** before relying on them in cost-critical paths.
The validator enforces every dim has `dim_value` — no `dim_param`. The model IO is fixed at `[1,10,30,30]` float32 (`"input"` and `"output"`). All internal activations must also be statically shaped (constant_folding requires it; shape_infer must succeed; calculate_memory walks value_info byte sizes). This isn't really a constraint — it's the operating reality. The strategy question is how to use the static-shape requirement to our advantage.
- **Free shape engine.** All shapes are known at build time → all shape-input ops fold → no runtime shape compute.
- **Predictable memory.** Validator's custom `memory` metric is the sum of all intermediate-activation byte sizes. Static shapes mean we can compute memory exactly during model assembly and reject builds over budget.
- **No dynamic dispatch.** Branching by data is not allowed (`If` banned). All ops act on fully-shaped tensors.
- **Constant-folded reshape is free.** A `Reshape` node with a literal shape input is constant-folded to a no-op data-aliasing — zero MACs, zero memory.
If the active-region size is fixed for the task (every example agrees on output dimensions), use a **fixed mask initializer** of shape `[1,10,30,30]`. Cost: 0 MACs (the mask is multiplied with the result, but we apply it via `Where(mask, color10_grid, computed_output)`, which is 9k MACs — and a one-time cost).
For tasks where output size varies across examples, this approach is risky.
If for task K, every visible example has the same input shape (e.g. always 5×5) and same output shape (e.g. always 5×5):
- The actual data lives in `[1, 10, 5, 5]` space, padded to `[1, 10, 30, 30]`.
- We can `Slice` to extract just the `[1, 10, 5, 5]` region, do work on that smaller tensor, then `Pad` back to `[1, 10, 30, 30]`.
**Cost win:** any per-element op that scales with output_elems gets cheaper proportionally to the area:
- Full 30×30 op: 9000 MACs.
- 5×5 op after Slice: 250 MACs.
- 36× MAC reduction on per-element ops.
**Risk:** the input shape *isn't actually fixed* on private leaderboard. ARC-GEN-style variation may produce 5×5, 7×11, 30×30 versions of the same task. **Slicing to a fixed sub-shape breaks if the actual grid is larger or differently shaped.**
**Safer pattern:** detect the active region size by counting non-color-10 cells, then slice — but this requires data-dependent shape, which ONNX doesn't permit in static mode. So we'd have to commit to a maximum size at build time.
**Recommendation:** use this strategy ONLY when (a) train + test + arc-gen subset all agree on shape, AND (b) the operation is much cheaper at the smaller shape, AND (c) we accept the private-set risk. Likely **not worth** the brittleness.
For tasks where the output is always smaller than 30×30 in a known way (e.g. always 15×15 in the top-left because the task halves the grid):
- Compute on the full 30×30.
- `Slice` to the smaller shape (free).
- `Pad` back to `[1, 10, 30, 30]` with color 10 (free).
The Pad fills with channel 10 = 1, others = 0. **Three Pad ops to achieve color-10 padding** — one per spatial axis with the right `constant_value`. Or one Pad with mode="constant" and value=0 (if we Cast everything correctly).
Actually, simpler: build the 30×30 output as `Concat([sliced_active, sliced_color10_pad], axis=2)` and then `Concat(..., axis=3)`. All free.
Some tasks have the input split into multiple sub-grids separated by sentinel borders. For each sub-grid:
- `Slice` extracts each region.
- Process each sub-region (potentially with the same shared weights, see B.3).
All shape-only operations are free, but the per-region ops scale with the region area.
Submission format is one ONNX file per task — so per-task models are mandatory. **No fallback within a single ONNX.** This was a queued question in the brief; here's the answer:
The detection logic in `build_submission.py` decides at *build time* (not inference time) which pattern fits which task. Per-task it builds the smallest-cost ONNX that solves the visible benchmarks, then ships it. If no pattern fits, ship the cheapest "always wrong" baseline (Identity, scoring 25 on tasks where output==input and 0 elsewhere).
**Key insight from the scoring rules:** incorrect = literal 0. So a 50% pattern-match rate on a task is **worse** than always-shipping-Identity (which scores 25 on the few identity tasks and 0 elsewhere). Pattern detectors must err toward declining (output the safe Identity model) rather than guessing.
**Concrete recommendation:** every pattern in `_PATTERNS` should compute the pattern's predicted output on **ALL visible examples** (train + test + arc-gen subset) and only commit if all examples match. The COVERED note about the 7-of-10-colors threshold for color-perm confidence is exactly this principle.
Before pattern matching, partition tasks by their shape signatures:
Pre-filtering by signature avoids running expensive pattern checks on tasks they couldn't solve. **Pure script-time optimization, no inference cost.**
Hypothesis: the validator hard-codes only `[1,10,30,30]` for IO. *Internal* activations can be any static shape. So:
This requires choosing `k` at build time. If we know the active region of the task is always ≤k×k, this works.
**Best case:** the active region is fixed at, say, 10×10 across all visible examples and we trust the private set respects this. Then a 3×3 depthwise Conv on the 10×10 region is `10·1·9·100 = 9,000 MACs` — instead of 81k. 9× MAC reduction → score gain of ~2.2 pts.
3. **Prefer transpose / identity / shape-only ops** (which are insensitive to active-region shape) over flips (which break on non-square active regions).
4. **Resist the temptation of per-task shape narrowing** unless we can prove from train + test + arc-gen (1000+ samples) that shape is genuinely invariant.
5. **Pattern detectors must verify on all visible examples** before committing — incorrect = 0, so guessing wrong is catastrophic.
- The **comparison is binary** per cell: any positive value counts as "this cell has this color." Zero or negative counts as "this cell does not have this color."
- We can produce small positive values (0.01) and they still count as 1.
- Negative outputs in cells that should be 0 are fine.
- Negative outputs in cells that should be 1 are wrong.
**Implication:** we don't need our model output to be exact 0/1. Just sign-correct. **A cheaply scaled sign-preserving network output works.**
This is why hand-crafted Sobel kernels with raw +1/-1/+2/-2 values can solve edge tasks without normalization — the only thing that matters is which side of zero each output cell lands on.
`Loop` is banned (`calculate_memory()` returns None on subgraph attrs). Anything that would have been a `for` loop in PyTorch must become a fixed unroll: N copies of the same Conv stage stacked in series. The depth N must be chosen at build time. The cost grows linearly with N for `macs` and `memory`; weight sharing (B.3) keeps `params` flat.
The strategic question: which iteration patterns can we use, and what's the minimum N for each? For a 30×30 grid, the diameter is bounded — most "iterate until convergence" algorithms converge in ≤30 steps, and a *few* converge in `O(log diameter)` ≈ 5–6 steps. Those few are the ones worth pursuing.
Some iterations don't admit the doubling trick:
- **Counter increment** (some ARC tasks count something then output the count colored): `ReduceSum` + Greater is `O(1)`, no iteration needed.
- **Sequential transformation** (apply rule R to leftmost X, then to second-leftmost, etc.): no log trick. Linear cost. Likely unsolvable cheaply.
For a fixed total compute budget B (in MACs), we have two knobs: `(width × per-step cost)` and `(depth × per-step cost)`. Width is forced by our 10-channel input. Depth is the choice we have.
**For ARC, deeper is almost always better than wider.** The transformations are about spatial reach, not about more channels. A network with depth 6 and 10 channels (60 logical compute steps) > a network with depth 1 and 60 channels (also 60 compute steps but no spatial reach across iterations).
Counter-example: tasks requiring multi-channel comparisons in one shot (e.g., "for each pixel, output the max across colors"). One ReduceMax suffices, depth 1.
- Per-pixel connectivity test via dilation from each pixel — too expensive.
- Better: count components first via repeated label propagation (Hoshen-Kopelman in convolutional form). Probably 6 power-iter steps + a final `Equal` test.
- Single dilation with a large kernel, no iteration.
- Or: 1× Conv with a 30×30 kernel and `dilation=1` and `pads=15`. Cost: `Cout × 1 × 30 × 30 × 30 × 30 = 810,000` per output channel — too much.
- Better: two-pass separable dilation (1×30 then 30×1). `2 × 9000 × 30 = 540,000` total. Still ~11 pts. Bad.
- Inline the rule N times using shared weights (B.3).
- Cost: linear in N. Small N (≤6) tolerable.
Each unrolled stage produces an intermediate `value_info` entry that the validator's custom `memory` metric counts (`[1,10,30,30]` float32 = 36,000 bytes). For depth N:
- `memory` = N × 36,000 bytes (plus the binarization output, also 36k each).
**6-deep unroll:** 6 × 36k = 216k bytes of memory (plus binarized intermediates). Compare to `macs` (~540k): memory is the smaller term but non-negligible.
For a binary-only iteration we could `Cast(..., BOOL)` to make intermediate activations 1 byte/elem instead of 4, potentially cutting `memory` 4×. But:
- Bool tensors don't natively support Conv.
- Cast to bool costs 0 MACs but the bool intermediate would need Cast back to float for the next Conv — still float in the conv-input position.
- Nothing actually saved.
**Better pressure-relief:** keep each stage as small a tensor as possible. If the operation only needs the binarized presence of one color, slice to `[1,1,30,30]` first; intermediates are 1/10 size.
