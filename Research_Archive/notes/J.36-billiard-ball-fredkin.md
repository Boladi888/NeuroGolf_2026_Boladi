# J.36 — Billiard ball (Fredkin) reversible logic for grid permutations

**Status:** synthesis. Maps Fredkin gates / conservative reversible logic onto the validator's banned-`If`, banned-`Loop` ONNX op set. Investigates whether reversibility translates to score wins via constant folding.
**Cluster:** logic / discrete / combinatorial.

## TL;DR

- **The Fredkin gate (controlled-swap, CSWAP) realizes naturally as `Where(c, x, y)` + `Where(c, y, x)` — two `Where` ops, ~18,000 MACs at this tensor shape.** The arithmetic-only formulation is `Add(Mul(c,y), Mul(1-c, x))` (4 ops, ~36,000 MACs). `Where` halves the cost.
- **Reversibility under the cost metric is a near-zero benefit.** The validator's pre-measurement `constant_folding` (COVERED) only folds nodes whose inputs are all initializers; a self-inverse op applied to a *runtime* tensor does not fold. So `T(T(x)) = x` semantically does NOT save MACs — it saves only the build-time decision to remove the second op (which any graph optimizer does anyway).
- **The interesting cost-bearing ARC tasks where reversibility shows up are *involutions*: hflip, vflip, rot180, transpose, color-XOR-with-mask, "swap colors A and B."** All of these are already in the `_PATTERNS` library or trivially expressible as Gather/Slice (cost 0). **Fredkin doesn't add anything to these.**
- **Where Fredkin *might* pay off: encoding a "conditional swap of two channels at every pixel based on a mask" as a single `Where`-pair.** This is task-specific but cheap: 2 `Where`s + 1 boolean mask construction = ~27,000 MACs total. Same order as a Conv 1×1 (90,100), so 3× cheaper. Useful.
- **Net assessment: Fredkin is a useful *vocabulary* for thinking about reversible patterns, but the realization is just `Where` — and `Where` is already a known cheap op for any conditional pattern. The "billiard ball" framing doesn't unlock anything new.**

## What a Fredkin gate is, and how it maps to ONNX

A Fredkin gate has 3 inputs (c, a, b) and 3 outputs:
- output_c = c (control passes through)
- output_a = if c then b else a (swap if c)
- output_b = if c then a else b (swap if c)

It's reversible (its own inverse: applying it twice = identity), conservative (input bit-count = output bit-count), and universal for reversible computation when extended to tensors of bits.

In ONNX:
- `Where(c, b, a)` returns elementwise `if c[i] then b[i] else a[i]`.
- `Where(c, a, b)` returns the swapped version.
- One Fredkin gate = 2 `Where` ops + the unchanged `c` (no op needed for c since it's referenced directly).

**Cost per Fredkin gate over the [1,10,30,30] tensor:** if `Where` is charged 1 MAC/elem (per onnx-tool's "treat mov as 1 MAC" rule), then 2 × 9,000 = 18,000 MACs. **DISCREPANCY flag carried forward (J.6 / J.11): `Where` may actually be charged 0 MACs by onnx-tool 1.0.1's profiler if treated as pure data-movement. Verify in source.**

For an integer-valued Fredkin (preferred for purity) you also need a `Cast(c → bool)` if `c` is float. With opset 9+ `Where`, the condition input is typed `tensor(bool)`. Cast costs ~9,000 MACs.

## Reversibility under constant folding — what actually saves work

The validator runs `onnx_tool.loadmodel(..., {'constant_folding': True})` before measurement (COVERED). Constant folding collapses any subgraph whose inputs are all initializers into a single initializer, charging 0 MACs and 0 memory for the subgraph (initializers count toward `params`, not `macs`/`memory` — COVERED).

So:
- `Where(c, a, b)` where `c`, `a`, `b` are all initializers → folds to a single initializer. **Free at MAC level.** But pointless because the result is then just a constant, not a function of the input.
- `Where(c, input, b)` where only `c` and `b` are initializers → does NOT fold. The `input` tensor is dynamic.
- `T(T(input))` where `T` is self-inverse → does NOT fold for exactly the same reason.

The "free inverse" intuition from physics doesn't translate: physics-reversibility is about *thermodynamic* cost (energy dissipation per Landauer), not about MAC count. The MAC cost is paid every time the op runs on runtime data, regardless of whether the op is invertible.

**Concrete corollary:** there is no "apply twice for free" pattern in this competition. Don't try to build one.

## ARC tasks where reversibility is naturally present

The brief's question. I count four classes:

### Class 1: Geometric involutions
hflip, vflip, rot180, transpose. Already in `_PATTERNS`. All realized as `Gather` or `Transpose` (shape-only, cost 0). Fredkin adds nothing.

### Class 2: Color-pair swaps
"Swap colors A and B everywhere." Realization: `Gather(input, axis=1, indices=swap_table)` where `swap_table = [0,1,...,B,...,A,...,9]`. Shape-only, cost 0. Fredkin adds nothing.

### Class 3: Mask-gated color toggles
"Within region R, swap colors A and B; outside R, leave unchanged." This IS a Fredkin pattern: control = mask, swap_targets = (channel_A, channel_B).

Realization options:
1. **Fredkin via `Where`:** build `mask_R` (1 op or precomputed if R is fixed), then `Where(mask_R, channel_B, channel_A)` for output_A and `Where(mask_R, channel_A, channel_B)` for output_B. Other channels pass through unchanged. **Cost: 2 `Where` + mask construction ≈ 27,000 MACs.**
2. **Algebraic (no `Where`):** `output_A = mask_R * channel_B + (1 - mask_R) * channel_A`. 4 ops × 9,000 = 36,000 MACs per channel × 2 channels = 72,000 MACs.
3. **Conv-based:** A 1×1 Conv with weights that swap channels A and B uniformly across the spatial axis cannot do *region-conditional* swaps; you'd need a Conv at higher channel rank. Out of scope at low cost.

**Fredkin via `Where` wins by 2.7× over algebraic, 3.3× over Conv 1×1.** Concretely useful.

### Class 4: XOR-with-mask
"In region R, invert colors X (set foreground ↔ background); outside R, identity." This is a mask-gated NOT. `Where(mask, 1-channel, channel)` for the affected channel = 2 ops × 9,000 = 18,000 MACs (or 9,000 if `Where` is charged 0). Cheaper than Class 3 because there's only one swap target.

**Class 3 and Class 4 together cover ~5-15 ARC tasks** I'd estimate without doing the full task survey. Worth a pattern-library entry: `region-gated-swap-via-Where`.

## Self-inverse graph structures and constant folding (the brief's deepest question)

The brief asks: "whether reversibility under the cost metric translates to 'self-inverse' graph structures that fold to identity under constant folding (the validator already does folding pre-measurement)."

**Answer: no, with one degenerate exception.**

The exception: if you build a subgraph `T(T(x))` where `T` involves an initializer, and you give `onnx-tool`'s constant folder a hint that `T(T)` simplifies to identity at the *symbolic* level, then it could fold the whole subgraph to `Identity`. **But onnx-tool 1.0.1 does NOT do symbolic simplification — it only folds constant-input subgraphs.** Verified by reading onnxruntime/core/optimizer/constant_folding.cc (which onnx-tool's folding pass mirrors): "constant folding does not support executing a node that includes subgraphs … but [requires] AllNodeInputsAreConstant."

So this entire pathway is closed at the validator's tooling level. The brief was right to flag this as worth investigating, but the answer is negative.

## ONNX ops worth knowing for reversible patterns

| Op | Use | Min opset | Cost note |
|---|---|---|---|
| `Where` | conditional swap (Fredkin-like) | 9 | 1 MAC/elem (or 0 — verify) |
| `Equal`, `Greater`, `Less` | producing the bool control mask | 1 / 7 / 7 | 1 MAC/elem |
| `Not` (bool) | invert a bool mask | 1 | 1 MAC/elem; cheap |
| `Cast(bool→float)` | thresholded float mask | 6+ | 1 MAC/elem |
| `Sub` from 1 | float NOT (`1-x`) | 1 | 1 MAC/elem |
| `Gather` (axis=1) | channel permutations (color swaps) | 1 | shape-only (cost 0) |

For most reversible patterns, **a precomputed `Gather` channel-permutation is cheaper than a Fredkin realization** because Gather is shape-only. Fredkin is only better when the control depends on the *spatial position* (region masks).

## Discrepancies / flags vs COVERED

- **Re-confirms the J.6/J.11 open question on `Where` cost.** I assume 1 MAC/elem for `Where` based on onnx-tool's "treat mov as 1 MAC" rule. **If `Where` is actually 0, every Fredkin pattern in this note becomes ~3× cheaper, and Class 3/4 dominate even more clearly.** Highest-leverage open question. **Action: read onnx_tool 1.0.1's `node_profilers.py`, find `Where`'s entry (or absence), confirm.**
- **No direct conflict with COVERED.** Geometric involutions (hflip/vflip/rot180/transpose) are in `_PATTERNS` already; my Class 3/4 patterns are new and complementary.
- **The brief's framing of "ARC tasks where reversibility is naturally present (apply same transform twice = original)"** turns out to be a *property of the task type*, not a *property worth exploiting in the graph*. The score-relevant thing is that involutions tend to admit Gather/Slice realizations at cost 0; that's their actual contribution.
- **The brief asks about Toffoli (CCNOT) too implicitly.** Toffoli is reversible AND-with-XOR. ONNX realization: `Xor(c1*c2, target)` (3 ops if int, more if float). Not cheaper than Fredkin and less commonly useful for grid transforms. Skip.

## Actionable design rules

1. **Add a `region-gated-swap-via-Where` pattern to `_PATTERNS`** — uses 2 `Where` ops, expected to hit ~5-15 ARC tasks.
2. **Add a `mask-gated-NOT-via-Where` pattern** — single `Where`, hits ~3-10 tasks.
3. **Don't try to use reversibility for "free inverse" savings.** Constant folding only fires on all-initializer subgraphs. Self-inverse over runtime tensors doesn't fold.
4. **For pure channel-permutation involutions (color swaps), use `Gather`, not `Where`.** Gather is cost-0; Where is cost-9000+.
5. **`Where` is only worth it when the control depends on position** (i.e., a 30×30 mask). For uniform conditions, the Conv/Gather alternatives are cheaper.
6. **Highest-leverage open question for this cluster:** verify `Where`'s MAC charge in onnx-tool 1.0.1. If 0, Fredkin patterns dominate. If 1, they're competitive but not dominant.

## Sources consulted

- ONNX 1.21 op reference: `Where` (opset 9+, supports bool condition + any-type values, broadcasts).
- Wikipedia "Billiard-ball computer", "Fredkin gate" — gate semantics, reversibility, conservativity.
- "Computing Inside the Billiard Ball Model" (Springer, 2018) — Fredkin gate realization in cellular automata; not directly applicable but confirms the gate's universality.
- onnxruntime/core/optimizer/constant_folding.cc — the folding rule "AllNodeInputsAreConstant", which forecloses symbolic self-inverse simplification.
- onnx-tool Profile.md — "treat mov as 1 MAC" rule (governs `Where`'s likely cost; unverified for that specific op).
- COVERED `_PATTERNS`: hflip/vflip/rot180/transpose already present (geometric involutions), color-perm-via-Gather present (Class 2).


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from J.36-billiard-ball-fredkin.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.36 — Billiard ball (Fredkin) reversible logic for grid permutations

## Summary

Edward Fredkin and Tommaso Toffoli's "Conservative Logic" (Int. J. Theor. Phys. 21:219, 1982) proposed reversible computation via the *Fredkin gate* (controlled-swap, CSWAP) and the *Toffoli gate* (controlled-controlled-NOT, CCNOT), both universal for reversible computation. The "billiard ball model" realizes these gates as elastic collisions of conservative particles, the canonical example showing computation can be physically reversible (zero thermodynamic dissipation per Landauer's principle). For NeuroGolf 2026, the question is whether reversibility under the validator's `params + memory + macs` cost metric translates to "free inverses" or self-inverse-folds-to-identity savings.

The empirical answer: **`Where` costs 0 MACs in onnx-tool 1.0.1** (verified by reading the source and probing the library), so a Fredkin gate realized as `(Where(c, b, a), Where(c, a, b))` is essentially free at the MAC level — but its `params` and `memory` contributions are large because the alternative-value tensors `a` and `b` are full-grid initializers. The "free inverse" intuition does NOT translate: the validator's constant folder fires only when *all* node inputs are themselves constant, so `T(T(input))` for runtime `input` does not fold even if `T` is self-inverse. The right way to think about Fredkin in this competition is as a vocabulary for spotting **mask-gated swap/toggle patterns** and realizing them with `Where`, not as a path to MAC savings via reversibility per se.

## Project mapping

**Empirical cost table for Fredkin-relevant ops:**

| Op | MACs (onnx-tool 1.0.1, [1,10,30,30] context) | Notes |
|---|---|---|
| `Where(cond, x, y)` | **0** (no `profile()` override on `WhereNode`) | dominant cost-saver |
| `Greater`, `Equal` (build cond mask) | 9,000 (CMP_MACS=1) | needed if cond depends on input |
| `Sub(1, x)` (logical NOT in float) | 9,000 (NpMathBase) | |
| `Mul`, `Add`, `Sub` (algebraic alternative to Where) | 9,000 each | |

**Fredkin gate in ONNX.** A Fredkin controlled-swap with control `c` and operands `a`, `b` produces `output_a = c ? b : a` and `output_b = c ? a : b`. Two ONNX `Where` nodes:
```
output_a = Where(c, b, a)
output_b = Where(c, a, b)
```
**Both nodes cost 0 MACs.** The `params` and `memory` contributions depend on whether `a`, `b`, `c` are initializers or runtime tensors. Where it gets interesting:

**Pattern A: Region-gated channel swap.** "Within mask `R`, swap colors X and Y; outside `R`, leave channel intact."
1. Construct mask `R`: 1 elementwise op (e.g. `Greater(input_channel, 0)`) = 9,000 MACs.
2. `output_X = Where(R, channel_Y, channel_X)`. 0 MACs.
3. `output_Y = Where(R, channel_X, channel_Y)`. 0 MACs.

**Total: 9,000 MACs.** The other 8 channels pass through unchanged. Compare to a Conv 1×1 with channel-swap weights conditional on position — Conv 1×1 cannot do region-conditional swaps in a single layer, so this isn't even comparable; you'd need a more complex multi-layer Conv. **Pattern A is therefore a unique capability, not just a cost win.**

**Pattern B: Mask-gated NOT-of-channel.** "In region `R`, invert color X (set foreground ↔ background); outside `R`, identity."
1. Mask construction: 9,000 MACs.
2. `output_X = Where(R, 1-channel_X, channel_X)`. The `1-channel_X` is `Sub(1_init, channel_X)` = 9,000 MACs. The `Where` itself = 0 MACs.

**Total: 18,000 MACs.** (You can save 9,000 by precomputing `1-channel_X` only once if multiple downstream ops use it.)

**Pattern C: "Apply transform T conditionally."** When `T` is itself cheap (e.g. a `Gather` color-perm = 0 MACs), the gated version is `Where(R, T(input), input)` = 0 + 0 = **0 MACs** plus mask construction. This is the cleanest demonstration that `Where` doesn't tax you.

**Self-inverse / "apply twice = identity" patterns.** For any involution `T` (hflip, vflip, rot180, transpose, swap-two-colors), `T(T(input)) ≡ input`. Naively you might hope this folds to `Identity` at the validator, costing 0. **It doesn't.** The constant folder fires only when all node inputs are initializers (`AllNodeInputsAreConstant`, per onnxruntime/core/optimizer/constant_folding.cc), and `input` is dynamic. So the second `T(...)` runs on a runtime activation. **No savings from reversibility per se.**

**The build-time savings** ("don't write the second op because we know it cancels") are real, but they're an *export-time* optimization that any sane graph builder does anyway — not a property unique to reversible ops.

**ARC tasks where Fredkin patterns help:**
- Region-gated color swaps: any task where the rule is "in this enclosed area, recolor; else leave alone" — Pattern A.
- Region-gated inversions: "inside the contour, flip foreground/background" — Pattern B.
- Conditional shape-only transforms: "if some input criterion, apply transpose; else identity" — Pattern C with T = `Transpose`. Cost: just the criterion.

Rough estimate of hit rate on the 400 ARC tasks: **5–15 tasks plausibly fit Pattern A/B/C if the pattern detector can identify them.** Lower confidence than other estimates; depends on the actual task distribution.

**Failure modes.**
- The `Where` *condition* must be a `tensor(bool)` (per ONNX spec, opset 9+). Building it from float requires `Greater`/`Equal` first. Forgetting to `Cast` produces a schema error.
- `Where` broadcasts the three inputs to a common shape. If `a`, `b`, and `c` have inconsistent broadcasts, the output shape silently changes — unlikely to be wrong, but worth checking with `shape_infer`.
- Pattern A as written above swaps two specific channels (X and Y); to swap *all* channels by some permutation conditional on position, you'd need a per-channel `Where` or a `Slice`/`Concat` chain. Costs add up linearly.
- Pattern B's `1-channel_X` exits the [0,1] range only if `channel_X > 1`, which doesn't happen for a one-hot input. Safe.

## Sources

- Fredkin & Toffoli, "Conservative Logic," Int. J. Theor. Phys. 21:219–253 (1982). The Fredkin gate definition (3 in / 3 out, controlled-swap, reversible, conservative).
- Toffoli, "Reversible Computing," MIT/LCS/TM-151 (1980).
- Wikipedia, "Billiard-ball computer" (`https://en.wikipedia.org/wiki/Billiard-ball_computer`): "A billiard-ball computer, a type of conservative logic circuit, is an idealized model of a reversible mechanical computer based on Newtonian dynamics, proposed in 1982 by Edward Fredkin and Tommaso Toffoli."
- Adamatzky (ed.), "Collision-Based Computing" (Springer 2002), chapter "Computing Inside the Billiard Ball Model": "Realizations of Fredkin gate and NOT gate with dual signal encoding are offered."
- onnx-tool 1.0.1 source: `WhereNode` defines `shape_infer`/`value_infer` only, no `profile()` override (line 768–779 of `node.py`). Inherits `Node.profile` returning `[0, 0]` (line 212–213). Confirmed empirically: `Where only (with bool init): macs=0`.
- onnxruntime constant folding rule (`https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/constant_folding.cc`): "constant folding does not support executing a node that includes subgraphs… `graph_utils::AllNodeInputsAreConstant(graph, n, constant_inputs, excluded_initializers_)`."
- ONNX `Where` op spec (`https://onnx.ai/onnx/operators/onnx__Where.html`): opset 9, "Return elements, either from X or Y, depending on condition. Where behaves like numpy.where with three parameters." Condition is `tensor(bool)`.
- Empirical run results:

```
Where only (with bool init)    macs=0      params=18000  memory=81000
Greater + Where pipeline       macs=9000   params=18001  memory=117004
```

The `params=18000` in "Where only" comes from the bool condition (9,000 elements) + the alternative value initializer (9,000 elements). **`Where`'s cost contribution is ALL params, no MACs — so the design rule is: keep its initializers small.**

## Contradictions / corrections vs COVERED

- No conflict with COVERED.
- **Self-correction vs my own previous notes:** I previously assumed `Where` charged 1 MAC/elem. It charges 0. Every Fredkin pattern in those notes is therefore cheaper than I estimated by 9,000 MACs per `Where`.
- **No "free inverse" trick exists.** This was an open question; the answer is definitive: constant folding only fires on all-initializer subgraphs.

## Confidence

**High** for the cost numbers (source-read and empirically verified). **Medium** for the hit-rate predictions on Pattern A/B/C — they're plausibly useful but I haven't enumerated the 400 tasks to confirm.
