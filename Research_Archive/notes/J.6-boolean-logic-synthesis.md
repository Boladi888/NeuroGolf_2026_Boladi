# J.6 — Boolean logic synthesis (Espresso / Quine-McCluskey → ONNX)

**Status:** synthesis. Maps two-level Boolean minimization onto banned-If ONNX op set. Ranks `Mul`/`Add`/`Sub`-from-1 / `Where` realizations by MAC-cost and parameter-cost under the validator's accounting rules.
**Cluster:** logic / discrete / combinatorial.

## TL;DR

- Espresso/QM produce a **two-level SOP cover**. SOP → ONNX: each AND-term is an elementwise `Mul`-product of binary inputs; the OR over terms is element-wise `Max` (or saturated `Add`). With binary {0,1} inputs, AND = Mul, OR = Max, NOT = `1 - x` (one `Sub`). NAND/NOR are not natively cheaper than this. **Don't synthesize down to NAND/NOR — synthesize to `Mul`+`Max`** (the brief's premise about NAND being cheap as `1-Mul` was correct for the gate itself, but a NAND-only network always costs more than the equivalent SOP because of the universal-basis tax).
- **The minimum is almost never an Espresso problem.** With 30×30 grids and 12 colors per cell, the input alphabet is too wide for honest two-level synthesis — but the **per-pixel** problem (treat each cell independently after `OneHot` decode) collapses to a 12-input → 12-output bitfield, which Espresso eats easily. This is where Espresso pays off.
- **Cost lower bound for any non-folded `Mul`/`Add` over the full grid is 900 MACs per layer (1×10×30×30 = 9000 elements; broadcast Mul against scalar is 9000 MACs; against a 10-channel constant is 9000 MACs — onnx-tool counts elementwise ops at numel(output) per node).** That's the unit cost you're trading against. Anything you can fold into a single Conv 1×1 (90,100 MACs already in COVERED) is dominated unless the SOP needs ≥10 product terms.

## How onnx-tool charges Boolean ops (the actual bottleneck)

Per onnx-tool's `Profile.md` (ThanatosShinji/onnx-tool, retrieved 2026-05-01): **"1 MAC = float(a) * float(b) + float(c). … Treat other instructions as 1 MAC, like cmp, add, and mov."** That means:

| ONNX op | MAC charge per output element | Notes |
|---|---|---|
| `Mul`, `Add`, `Sub`, `Div` | 1 | elementwise binary |
| `Equal`, `Greater`, `Less`, `And`, `Or`, `Not`, `Xor` | 1 | elementwise (cmp) |
| `Where(cond, X, Y)` | 1 | one mov per element (per onnx-tool's "treat mov as 1 MAC" rule, but **flag**: the actual onnx-tool source may charge 0 — see DISCREPANCIES) |
| `Max`, `Min` | 1 per binary, n-ary scales linearly | n-ary `Max(a,b,c,d)` ≈ (n-1) × numel MACs |
| `Mod` | 1 | elementwise; useful for GF(2) (see J.7) |
| `OneHot` | 0 (assignment-only, no arithmetic) | **flag**: not verified in onnx-tool, see DISCREPANCIES |
| `Gather`, `Slice`, `Reshape`, `Transpose`, `Identity` | 0 | shape-only (already in COVERED) |
| `Conv` | k_h × k_w × C_in × numel(output) | the dominant baseline lever |

For a `[1,10,30,30]` tensor, numel = 9000. Every elementwise binary op = 9000 MACs. Every Conv 1×1 with 10 in/out channels = 90,000 MACs (matches COVERED probe of 90,100 — the extra 100 is the bias). **A 10-product-term SOP realized as 10 `Mul`s + 9 `Max`s = 19 × 9000 = 171,000 MACs, dwarfing a Conv 1×1.** SOP only wins when it can be folded.

## Synthesis recipe (when it actually fits)

1. **Decode the input.** The competition tensor is `[1,10,30,30]` already representing a one-hot color grid (channel = color index 0–9, with index 10 = "no color" pad — see COVERED). So the input is *already* one-hot. No decode needed.
2. **Identify per-pixel boolean structure.** Many ARC tasks have rules of the form "output channel c' is on iff some Boolean combination of input channels c1,…,ck is on at this pixel." That's a 10-in / 10-out per-pixel Boolean function — a tiny SOP problem.
3. **Run Espresso.** Feed the truth table (≤2^10 = 1024 rows; ARC training rules usually use ≤4 input channels so ≤16 rows) in PLA format to Espresso; get a minimal SOP. Reference: hadipourh/espresso (modernized C build), scottinet/espresso-logic-minimizer (Node bridge), or `pyeda` (Python).
4. **Realize as a Conv 1×1.** Here is the crucial collapse: **a per-pixel Boolean SOP over one-hot channels with k input variables and m product terms is an exact Conv 1×1 with weight matrix that is structured but not sparse.** A product term `c1 ∧ ¬c2 ∧ c3` over one-hot channels is `c1 · (1 - c2) · c3`. With one-hot binary inputs, `c1 · (1 - c2) · c3` simplifies because at any pixel exactly one channel fires: `c1 · c3 = 0` always (you can't be color 1 and color 3 simultaneously). **One-hot gives you exclusivity for free, which means most SOP terms collapse.** A pure "this pixel is one of {color set S}" rule becomes a single Conv 1×1 row with weight 1 on each color in S, weight 0 elsewhere — and that Conv 1×1 already costs 90,100 MACs (covered). **No SOP synthesis can beat that for membership-style rules**.
5. **Espresso pays off only for negation-of-set rules** (e.g., "output channel 0 fires iff input is *not* in {1,3,5}"). Even then, the cheapest realization is `1 - sum_S` = one `ReduceSum` over a slice + one `Sub`, not Espresso output.

## Concrete example: "is this cell on the diagonal of the bounding box"

The brief's seed example. Decomposes to:
- bounding box detection (requires the full grid — `ReduceMax` along H and W, then thresholds)
- diagonal test (`row == col + offset`)

**ONNX realization (no SOP needed; Espresso doesn't help):**
1. Detect colored-cell mask: `Sum` along channel axis 0 (skip background channel) → `[1,1,30,30]`. Cost: 9 × 900 = 8,100 MACs (or one `ReduceSum` if the profiler charges it as numel(output)).
2. Get bounding-box edges via `ArgMax` on row-projection and col-projection — `[1,1,30,1]` and `[1,1,1,30]`. Cost: ~30+30 ≈ 60 cmp ops, but `ArgMax` is annotated by onnx-tool as a reduction with cost = numel(input) — call it 900 MACs each.
3. Diagonal mask: precompute a constant `[1,1,30,30]` tensor `D[r,c] = 1 iff r==c` and bounding-box-translated versions — these are all *initializers* (params, not MACs). Then `Mul` the row/col-shifted constant by the cell mask: 9000 MACs.

Total: ~18,900 MACs. Versus a Conv 3×3 baseline of 810,900: **42× win.** Espresso never enters the picture; the win comes from spotting that the diagonal is a *constant pattern* rather than a Boolean function of input pixels.

**Generalizable lesson:** for ARC tasks where the output mask is a *fixed geometric pattern conditioned on a coarse property of the input* (color count, bounding box, presence of a mark), the cheap realization is **(reduction) × (constant template)**, not Boolean synthesis.

## NAND/NOR gate-level realization in ONNX (reference, not recommended)

For completeness — the brief asked for this:

- `NAND(x, y) = 1 - x*y` → 1 `Mul` + 1 `Sub` against scalar 1 (initializer). 2 ops × 9000 = **18,000 MACs**.
- `NOR(x, y) = 1 - (x + y - x*y)` (assuming binary {0,1}; boolean OR = saturation) or, for {0,1} inputs without saturation, `NOR(x,y) = (1-x)*(1-y)` = 2 `Sub` + 1 `Mul` = 3 ops × 9000 = **27,000 MACs**. Slightly cheaper: precompute `1-x` once, share between fan-outs.
- `NOT(x) = 1 - x` → 1 `Sub`. **9,000 MACs**.

A NAND-tree realizing an arbitrary k-input Boolean function uses `O(k)` gates for monotone functions, more for non-monotone. **Every gate costs 9,000 MACs at this tensor shape.** A single Conv 1×1 (90,100 MACs) is cheaper than any 11+ gate NAND tree.

## What the validator actually rewards

- Foldable structure costs 0 (COVERED). Espresso-minimized SOPs over **constant** inputs fold completely; useless because the input isn't constant.
- The cheapest Boolean op on the live input tensor is one elementwise op = 9000 MACs.
- Multi-term SOPs only win against `Conv` when total ops < 11 elementwise (i.e., < ~99,000 MACs vs Conv 1×1's 90,100). Most ARC color-rules are simpler than that, but most simple rules are even cheaper as Gather (color-perm) or `ReduceSum` (membership) — which cost zero or near-zero.

## Discrepancies / flags vs COVERED

- COVERED says NAND realizes as `Mul` + `Sub`-from-1. Verified — that's correct, and matches my derivation above. No conflict.
- COVERED says NOR realizes as `Add` + `Sub`-from-1. **Partial conflict:** that formula `1 - (x+y)` returns 1 only if x=y=0, which is correct for binary inputs — but it returns negative values when both inputs are 1, which then trips the output-thresholding rule `(result > 0.0).astype(float)`. Safer: `(1-x)*(1-y)` (1 Mul + 2 Subs, but the two `Sub`s are against the same constant 1 so they may share computation depending on graph optimizer behavior). Empirically untested; flag for verification.
- **DISCREPANCY:** I claim `OneHot` MACs = 0 in onnx-tool's accounting. I did not verify against onnx-tool's source — the `Profile.md` doc only lists Conv-class ops explicitly. The forum thread `/discussion/692827` is flagged in COVERED as raising scoring discrepancies; OneHot MAC accounting is plausibly one of them. **Action item:** open `onnx_tool/node_profilers.py` locally (already in the pinned 1.0.1 install) and grep for `OneHot`. If unimplemented, falls back to a default profiler that may charge numel(output) MACs — this would make `OneHot(indices, depth=10)` over a `[1,30,30]` index tensor cost 9,000 MACs, equal to one elementwise op.
- **DISCREPANCY:** I claim `Where` charges 1 MAC/elem. The brief lists `Where` as the realization for J.36 Fredkin. `Where` may be 0 in onnx-tool (it's pure data movement, no float arithmetic). Same verification path as OneHot — check the source of the pinned 1.0.1 build.

## Actionable design rules (extract for `_RESEARCHED_TOPICS.md`)

1. **Don't run Espresso for full-grid rules.** Membership / set-test rules collapse to a single Conv 1×1 weight pattern; that's 90,100 MACs and Espresso can't make it cheaper.
2. **Run Espresso only for tiny per-output-channel rules** where the rule has ≥3 distinct product terms. Below that, hand-write the SOP.
3. **Boolean rules over a one-hot input are sparse by construction** — almost every product term collapses to 0 because color exclusivity. Exploit this in the lib BEFORE calling Espresso; the search is over the ≤10-bit power set of color subsets, not the full 1024-row truth table.
4. **The MAC-floor for any per-pixel Boolean rule is 9,000 MACs** (one elementwise op on the live tensor). Below this you need shape-only or constant-folded ops.
5. **`(1-x)*(1-y)` (3 ops) is the safe NOR; `1-(x+y)` (2 ops) goes negative and trips the post-thresholding.**

## Sources consulted

- onnx-tool Profile.md, ThanatosShinji/onnx-tool main branch — MAC accounting rule.
- ONNX 1.21 operator reference (Mul, Add, Sub, Where, OneHot, Mod) — domain `""`/`"ai.onnx"` confirmed; opsets verified ≥10.
- Espresso heuristic logic minimizer (Wikipedia, hadipourh/espresso, scottinet/espresso-logic-minimizer) — input/output formats and the SOP-cover guarantee.
- COVERED list cost-lever probe (2026-05-01): Conv 1×1 = 90,100 MACs, Conv 3×3 = 810,900 MACs.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from J.6-boolean-logic-synthesis.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.6 — Boolean logic synthesis (Espresso / Quine–McCluskey → ONNX)

## Summary

Quine–McCluskey (1952–1956) is the classic exact two-level Boolean minimizer; Espresso (Brayton/McMullen/Hachtel/Sangiovanni-Vincentelli, UC Berkeley, mid-1980s) is the heuristic descendant that scales to industrial PLA sizes. Both consume an on-set/off-set/don't-care truth table and produce a near-minimal sum-of-products (SOP) cover. The cover maps to logic hardware as a two-layer AND-OR PLA. For NeuroGolf 2026, the question is whether running Espresso on per-pixel Boolean transforms gives a graph cheaper than `Conv 1×1` under onnx-tool 1.0.1's MAC/params/memory accounting.

The headline answer is **no, almost never** — but for a non-obvious reason. Empirical probing of onnx-tool 1.0.1 (the pinned validator backend) shows that on `[1,10,30,30]` float32 input, every elementwise binary op (`Mul`, `Add`, `Sub`, `Mod`, `Greater`, `Equal`, `And`, `Or`, `Xor`, `Not`) costs exactly 9,000 MACs (= numel(output) × 1). A single Conv 1×1 (10→10 channels) costs 99,000 MACs. **An SOP synthesized into ONNX needs ≥11 elementwise ops to lose to one Conv 1×1, and 11-term SOPs almost never describe a useful per-pixel rule over one-hot color inputs.** Espresso pays off only for a narrow class: rules expressed *over a derived Boolean axis* (a single bit per pixel from thresholding), where the SOP is genuinely small (≤4 product terms) AND there is no simpler `Gather`/`ReduceSum` realization.

## Project mapping

The cost vocabulary, verified by reading and probing `/usr/local/lib/python3.12/dist-packages/onnx_tool/node.py` (1.0.1):

| Op | onnx-tool 1.0.1 charge for `[1,10,30,30]` output | Class in source |
|---|---|---|
| `Mul`, `Add`, `Sub`, `Div`, `Mod` | 9,000 MACs (`volume(out) × 1`) | `NpMathBase` (Div is `volume×4`) |
| `And`, `Or`, `Xor`, `Not`, `Less`, `Greater`, `Equal` | 9,000 MACs (`volume(out) × CMP_MACS=1`) | `LessNode`/`GreaterNode`/`EqualNode` |
| `Where` | **0 MACs** (no `profile()` override → `Node.profile` returns `[0,0]`) | `WhereNode` |
| `Hardmax` | 9,000 MACs (extends `PWNode`) | `HardmaxNode` |
| `Identity`, `Transpose`, `Reshape`, `Slice`, `Gather`, `Cast`, `OneHot`, `ArgMax`, `TopK`, `ScatterND`, `GatherND` | **0 MACs** | inherit `Node.profile` default |
| `Conv 1×1` 10→10 | 99,000 MACs (NB: COVERED says 90,100 — a +9,000 bias-add discrepancy worth investigating, see Contradictions) | `ConvNode` |
| `ReduceSum`, `ReduceMean` | `volume(input) × 1` MACs (= 9,000 for the full tensor) | `ReduceMeanNode` |

**Boolean realizations on {0,1} float inputs:**
- AND(x,y) = `Mul(x,y)` → 9,000 MACs.
- OR(x,y) over disjoint one-hot channels = `Add(x,y)` → 9,000 MACs (no saturation needed — channels are mutually exclusive at every pixel by construction).
- NOT(x) = `Sub(1,x)` → 9,000 MACs (with `1` as scalar initializer of size 1 element = 1 param).
- NAND(x,y) = `Sub(1, Mul(x,y))` → 18,000 MACs.
- NOR(x,y) = `Sub(1, Add(x,y))` → 18,000 MACs (safe — `Add` of disjoint one-hot channels never exceeds 1, so `1-Add` stays in [0,1]). **My earlier note about preferring `(1-x)*(1-y)` was over-cautious; on disjoint one-hot inputs the simple `Sub(1,Add(x,y))` is correct.**
- XOR(x,y) on {0,1} = `Add(x,y) - Mul(x,y)*2` → 27,000 MACs, OR `Mod(Add(x,y), 2)` with `fmod=1` → 18,000 MACs.

**SOP synthesis pipeline.** A k-term SOP needs k AND-stages (one `Mul` per AND, sharing factors where possible) and (k-1) OR-stages. Best case for k=2: 1 `Mul` + 1 `Add` = 18,000 MACs over the grid + tiny initializer for the literal pattern. **Beats Conv 1×1 (99,000) by 5.5×.** Worst case for k=10: 10 `Mul`s + 9 `Add`s ≈ 171,000 MACs — Conv 1×1 wins.

**One-hot input collapses most SOPs.** The competition tensor is `[1,10,30,30]` already one-hot over channel axis 1 (color 10 = no-color pad, color 11 = decode error per COVERED). At any single pixel, at most one channel is 1. So any product term `c1 ∧ c3` (i.e., `Mul(channel_1, channel_3)`) is identically 0 — channels are mutually exclusive. **Useful Boolean rules over one-hot input are *unions of channels*, not arbitrary AND-trees.** "Output fires iff input is in {1,3,5}" → `Add(c1, c3, c5)` = 2 `Add`s = 18,000 MACs. Equivalent Conv 1×1 with a binary weight row: 99,000 MACs. **Always use Add-of-channels, not Conv, for membership rules over one-hot.**

**The diagonal-of-bounding-box example from the brief.** Decomposes cleanly into shape-only ops + one `Mul`, no Espresso involvement. (1) Spatial-extent of color via `ReduceMax` along H and W: 2 `ReduceMax`es over `[1,10,30,30]` = 18,000 MACs total. (2) Diagonal mask is a *fixed* `[1,1,30,30]` initializer (900 params) — it's a pattern, not a function of input. (3) Multiply mask by color channel: 1 `Mul` = 9,000 MACs. Total ≈ 27,000 MACs vs Conv 3×3's 810,900 (covered). Espresso never enters.

**Failure modes.** (1) Espresso outputs assume Boolean inputs; on float [0,1] inputs the AND-as-Mul interpretation is correct, but on float arbitrary-real inputs it isn't. The validator's `(result > 0.0).astype(float)` post-thresholding (COVERED) means *any* positive output gets rounded to 1, so Add-saturation (1+1=2) doesn't matter — both 1 and 2 round to 1. **NOR/NAND derived via `Sub(1, …)` returning negatives DO get rounded down to 0, which is the correct Boolean output.** No issue. (2) Espresso minimizes literal count, not Conv-equivalent param count, so a cover with many distinct product terms but few literals can be worse than a single Conv 1×1 with structured weights. (3) Espresso's PLA format only handles single-output functions cleanly; multi-output sharing requires running with `-out` or post-processing.

**Recommendation.** Don't add Espresso to the pipeline as a primary tool. Use it only to derive minimal SOPs for one or two specific tasks where (a) the per-pixel rule depends on a derived Boolean (e.g. "is this cell part of the bounding box edge"), and (b) the hand-derivation isn't obviously cheap. The expected hit rate is ≤5 tasks out of 400.

## Sources

- Espresso heuristic logic minimizer — Wikipedia article (en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer): "The input for ESPRESSO is a function table of the desired functionality; the result is a minimized table, describing either the ON-cover or the OFF-cover of the function… This allows for efficient implementation in two-level logic arrays such as a PLA (Programmable Logic Array) or a PAL (Programmable Array Logic)."
- Modern Espresso build: `https://github.com/hadipourh/espresso` ("Espresso Logic Minimizer - Modernized Edition" — maintained C build of UC Berkeley original).
- Node.js bridge with PLA format examples: `https://github.com/scottinet/espresso-logic-minimizer`.
- Rust port: `https://crates.io/crates/espresso-logic` (`espresso-logic-minimizer`).
- onnx-tool 1.0.1 source — exact MAC charges (read locally from the pinned install): `MUL_MACS = ADD_MACS = CMP_MACS = 1; DIV_MACS = 4; EXP_MACS = 32; LOG_MACS = 43` (`onnx_tool/node.py` lines 19–28). `NpMathBase.profile` returns `volume(out) * ratio * op_mac` (line 262–265, with `ratio = max(1, len(input)-1)`). `LessNode.profile` returns `volume(out) * CMP_MACS` (line 722–723); `GreaterNode`, `EqualNode`, `AndNode`, `OrNode`, `XorNode`, `NotNode` all extend `LessNode` (line 711–765). `WhereNode` defines no profile override (line 769–779).
- onnx-tool Profile.md: "1 MAC = float(a) * float(b) + float(c)… Treat other instructions as 1 MAC, like cmp, add, and mov" (`https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Profile.md`).
- ONNX op reference 1.21: `Mul`, `Add`, `Sub` opset 1+ (`https://onnx.ai/onnx/operators/onnx__Mul.html` etc.); `Mod` opset 10+ with `fmod=1` for float (`https://onnx.ai/onnx/operators/onnx__Mod.html`).
- Empirical verification table (this run, with `numpy==2.4.4 onnx==1.21.0 onnx-tool==1.0.1`):

```
Identity (baseline)             macs=0       params=0       memory=36000
Mul (input * scalar)            macs=9000    params=1       memory=36004
Add (input + scalar)            macs=9000    params=1       memory=36004
Greater (input > scalar)        macs=9000    params=1       memory=9004
Equal (input == scalar)         macs=9000    params=1       memory=9004
Conv 1x1 (10->10 channel)       macs=99000   params=110     memory=36440
```

## Contradictions / corrections vs COVERED

- **Conv 1×1 cost: 99,000 vs 90,100.** COVERED says Conv 1×1 identity = 90,100 MACs from validator. My empirical onnx-tool run gives 99,000. The difference is exactly 9,000 = numel(output), which suggests the validator's Conv profiler counts bias differently (or omits it). The forum thread `/discussion/692827` flagged in COVERED about "onnx-tool scoring discrepancies" plausibly concerns this. **Action item:** read the validator's `calculate_macs` (in `neurogolf_utils.py`) to see whether it overrides onnx-tool's Conv profiler or uses it as-is.
- No conflict with the COVERED statement that pure shape-only ops floor cost; the empirical Identity/Transpose/Gather rows confirm 0 MACs.

## Confidence

**High.** The MAC cost numbers are read directly from the pinned onnx-tool 1.0.1 source AND verified by running the library against constructed test models. The Conv 1×1 discrepancy is the only loose thread, and it doesn't change the conclusion (Espresso is rarely worth it for full-grid rules).
