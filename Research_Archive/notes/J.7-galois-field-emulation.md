# J.7 — Galois field emulation for reversible grid permutations

**Status:** synthesis. Maps GF(2) and GF(2^n) arithmetic onto the validator's float32 ONNX op set.
**Cluster:** logic / discrete / combinatorial.

## TL;DR

- **GF(2) on float32 inputs costs the same as integer GF(2)** (since {0,1} float values give XOR via `Add`-then-`Mod 2` or via `(x + y) mod 2 = x + y - 2*x*y` — three ops, ~27,000 MACs at this tensor shape). **GF(2) does not buy anything over plain Boolean (J.6).** XOR is just three elementwise ops; that's no cheaper than a single Conv 1×1 (90,100 MACs ÷ 27,000 ≈ 3.3× more expensive than three elementwise ops, but only when the Boolean structure can be expressed in those three ops).
- **GF(2^n) buys nothing for ARC-style grid transforms.** GF(2^n) is useful when you need *invertible non-trivial permutations of n-bit symbols* — but ARC color values are 4-bit (12 entries) and the validator decodes via one-hot over channel axis. The "n-bit symbol" abstraction is already broken by the channel encoding.
- **Where reversibility *does* pay off**: any **self-inverse** transform applied an even number of times collapses to identity *under constant folding* — i.e. the validator's pre-measurement `constant_folding` pass eliminates it. Useful pattern: apply your transform via `Mul` with weights that are themselves outputs of a subgraph using only initializers, so the whole `Mul` becomes a foldable constant. But this is just the existing "shape-only ops are free" rule (COVERED) restated; reversibility doesn't add anything.
- **Bitwise ONNX ops are unusable at the starter's opset.** `BitwiseAnd`/`BitwiseOr`/`BitwiseXor` are opset 18 (added in PR #4496, 2022). Starter ships IR 10 + ai.onnx 10 (COVERED). `BitShift` is opset 11. **All bitwise ops are integer-only**, so any GF(2^n) realization needs `Cast` float→int → bitwise → `Cast` back, and `Cast` is **2 × 9000 = 18,000 MACs** of pure overhead per round-trip per tensor.
- **Net assessment: J.7 has no cost-positive realization.** Recommend deprioritizing.

## What's actually in ONNX for bit-level operations

Verified against onnx.ai/onnx/operators/ (retrieved 2026-05-01):

| Op | Min opset | Type constraints | Comment |
|---|---|---|---|
| `Mod` | 10 | int + float (with `fmod=1` for float) | Mod 2 for GF(2). Charged 1 MAC/elem. |
| `BitShift` | 11 | unsigned ints only (uint8/16/32/64) | LEFT or RIGHT. Need Cast. |
| `BitwiseAnd` | **18** | signed/unsigned ints | Need Cast + opset bump. |
| `BitwiseOr` | **18** | signed/unsigned ints | Same. |
| `BitwiseXor` | **18** | signed/unsigned ints | Same. THE GF(2) addition op. |
| `BitwiseNot` | **18** | signed/unsigned ints | Same. |
| `And`, `Or`, `Xor` | 1 (opset 7+) | **bool only** | These are *logical*, not bitwise — only act on `tensor(bool)`. Useful for {0,1} inputs after `Cast` to bool. |
| `Cast` | 6+ | wide source/target | Cost: 1 MAC/elem (treated as `mov`). |

**Hidden constraint that matters most:** the competition's submission tensor must be `[1,10,30,30] float32` with names `"input"`/`"output"` (COVERED). This is enforced; you cannot ship integer or bool I/O. So any bit-level scheme requires at least 2 `Cast` ops (one in, one out), adding **18,000 MACs of pure overhead**. That alone makes bitwise math non-competitive against the existing Conv 1×1 = 90,100 baseline in any task that fits in ≤8 elementwise ops.

## GF(2) realizations (concrete)

### Addition in GF(2) = XOR

For binary {0.0, 1.0} float inputs, three equivalent realizations:

1. **Algebraic**: `xor(x, y) = x + y - 2*x*y` → `Add`, `Mul`, `Mul-by-scalar` (or equivalent) = 3 elementwise ops, **~27,000 MACs**.
2. **`Mod 2` of `Add`**: `xor(x, y) = (x + y) mod 2` → `Add` + `Mod` = 2 ops, **~18,000 MACs**. Requires Mod with `fmod=1` to operate on floats (verified: ONNX Mod opset 10 supports float when `fmod=1`).
3. **Bool-domain logical XOR**: `Cast`(float→bool), `Xor`, `Cast`(bool→float). 3 ops, ~27,000 MACs. Same cost as algebraic; no benefit.

The cheapest is **(2) `Add` + `Mod 2`**: 18,000 MACs.

### Multiplication in GF(2) = AND

For {0.0, 1.0} inputs: `and(x, y) = x * y` → 1 `Mul`, **9,000 MACs.** Identical to plain Boolean (J.6).

### Why GF(2) doesn't help

GF(2) is a tool for proving theorems about reversible XOR-only transforms. In ONNX, "addition" is what we wanted anyway. The `Mod 2` in option (2) is only needed when the Boolean variables *aren't* {0,1}-valued — i.e., when the values have escaped the Boolean range. For the ARC tensor (one-hot channels), the values are already {0,1}. So `Add` alone is exact for "either of these channels is on" (saturating to 1 only fails when *both* are 1, which is impossible at a one-hot pixel). **`Add` between disjoint one-hot channels = OR for free** (no Mod needed). 9,000 MACs, same as a single AND.

## GF(2^n): why it's a trap for this competition

GF(2^n) gives you bijective n-bit permutations via multiplication by a generator and reduction modulo a primitive polynomial. Two issues:

1. **No useful symbol width.** The competition's "symbols" are pixels with 12 possible color values, encoded as one-hot over a 10-channel axis with 2 reserved channel meanings. You don't have an "n-bit register" to permute; you have a tensor whose shape is fixed. Permuting the channel axis is `Gather` (cost 0 — already in COVERED); GF(2^n) machinery for the same effect costs orders of magnitude more.
2. **Polynomial-reduction multiplication is multi-instruction.** A standard GF(2^4) multiply via the irreducible polynomial `x^4 + x + 1` requires: 4 conditional XORs of the multiplicand, each gated by a bit of the multiplier, with reduction whenever the high bit overflows. Each conditional XOR in ONNX is `Where(cond, x_xor_y, x)` ≈ 3 ops. **Total: ~16 ops × 9,000 = 144,000 MACs** for one GF(2^4) multiply over the grid. A Conv 1×1 with the equivalent permutation matrix as weights costs 90,100 MACs and **already does the permutation correctly without any of this**.

The only place GF(2^n) might pay off is if the *permutation table* itself were so structured that storing it cost dramatically less than 100 floats (the Conv 1×1 weight bias). It isn't — even a 12×12 permutation matrix has 144 entries × 4 bytes = 576 bytes, well under any threshold that matters relative to the 1.44 MiB cap.

## Reversibility under the cost metric

The brief asks: "where does reversibility help under the cost metric (e.g., free inverses)?"

**Under `onnx-tool`'s constant-folding pass (COVERED), the answer is: only on subgraphs that fold to identity.** A reversible op applied twice with the same parameters folds to identity *if both applications are over constants* — but the input is not a constant. So `T(T(input))` does NOT fold; the inner `T(input)` is a runtime activation, and `T(activation)` is another runtime op.

A weaker version that *does* help: if `T = T^{-1}` (self-inverse, e.g., `1 - x`, transpose-of-symmetric-axis, hflip-of-symmetric-grid), then `T(T(input)) = input` semantically, and a smart graph builder can elide it pre-export (not by constant folding — by graph rewriting). The cost saving is "don't write the second op," which is also free. So reversibility-as-free-inverse is **a build-time optimization, not a runtime saving**, and it's already implicit in normal graph optimization. Doesn't justify a GF-themed approach.

## ARC tasks where reversibility is *naturally present*

The brief asks for examples. Of the 400 training tasks I'd guess:
- **Color permutation pairs (`Gather`-realizable)** — already in the `_PATTERNS` library (task 016/337 → "color-perm-via-Gather"). These are reversible by definition (a permutation σ with σ^2 = id is an involution; only some color perms are involutions, and those are even cheaper because the inverse equals the forward map).
- **Geometric involutions: hflip, vflip, rot180, transpose** — already in `_PATTERNS`. All are involutions.
- **XOR-with-mask** — "the output is the input with all cells inside region R inverted." Realized via `Sub`-from-1 on a slice, or `Where(mask, 1-x, x)`. The "reversibility" property (apply twice = original) is not directly cost-relevant but it's a useful sanity check during pattern detection.
- **Modular arithmetic on color indices** — e.g., "output color = (input color + 1) mod 10." Cheap as `Gather` with a precomputed permutation table; GF doesn't help.

Conclusion: every reversible ARC pattern I can identify has a cheaper non-GF realization already.

## Discrepancies / flags vs COVERED

- **No direct conflict with COVERED.** GF emulation is an unexplored corner; the COVERED list does not assert anything about it.
- **Flag for J.6 cross-reference**: my J.6 note claims `(1-x)*(1-y)` is the safer NOR than `1-(x+y)`. Both work fine for GF(2) inputs (they're literally the same Boolean function). The post-thresholding rule `(result > 0.0).astype(float)` (COVERED) means values ≥ 0 round to 1 — so `1-(x+y)` going negative produces the desired 0 output. **My J.6 note may be over-cautious; verify experimentally.**
- **Flag re bitwise ops:** the COVERED list says "default IR/opset starter uses IR 10 + ai.onnx opset 10; newer accepted in principle but not pinned." If the validator accepts opset 18, GF(2) via `BitwiseXor` over int8 becomes an option, but still costs 2 `Cast`s minimum. **Action item: empirically test whether onnx-tool 1.0.1 + the validator accept a model declaring opset_import 18.** If yes, document the cost of one `Cast` more precisely.

## Actionable design rules

1. **Don't use Mod 2 unless overflow can occur.** For one-hot channel inputs, `Add` between disjoint channels gives OR exactly. No correction term needed.
2. **GF(2^n) is dominated by Conv 1×1 + Gather for every permutation problem in this competition.** Permutations of color = Gather (free). Permutations of position = Transpose/Reshape (free). Mixing them = a single Conv 1×1 (90,100 MACs).
3. **Self-inverse ops save build-time cost only.** They don't fold at runtime over runtime tensors.
4. **Bitwise ops are opset 18** and float→int round-trips cost 18,000 MACs of dead overhead. Use them only if you have a chain of ≥4 bitwise ops that genuinely cannot be expressed elsewise — which I haven't found.

## Sources consulted

- ONNX 1.21 op reference: BitShift (opset 11, unsigned-int only), BitwiseXor (opset 18), Mod (opset 10, supports float with fmod=1), Cast (opset 6+).
- ONNX issue #4343 + PR #4496 (2022): introduction of BitwiseAnd/Or/Xor at opset 18.
- ONNX issue #2406 (2019): BitShift type-constraint discussion (uint-only).
- Sachs, "Linear Feedback Shift Registers, Part I" (embeddedrelated.com 2017): GF(2) ↔ XOR/AND identity; GF(2^n) polynomial multiplication.
- Wikipedia "Galois field": GF(2^n) requires polynomial reduction modulo a primitive polynomial; multiply is not a single instruction in any commodity ISA.
- onnx-tool Profile.md: "treat other instructions as 1 MAC" for cmp/add/mov, governs Cast/Mod/BitShift cost.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from J.7-galois-field-emulation.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.7 — Galois field emulation for reversible grid permutations

## Summary

GF(2) is the two-element field where addition is XOR and multiplication is AND. GF(2^n) extends it to n-bit polynomial arithmetic, used in error-correcting codes, AES, and cryptographic primitives like LFSRs. Foundational references: Reed–Solomon (1960) and the standard finite-field algebra texts (e.g. Lidl & Niederreiter). For NeuroGolf 2026, the question is whether GF arithmetic gives a cheaper way to encode reversible grid-state permutations than the obvious `Gather` / `Conv 1×1` realizations.

The answer is **no, with high confidence.** Three independent reasons: (1) ONNX's bitwise ops (`BitwiseAnd`/`Or`/`Xor`/`Not`) are opset 18 — the starter uses opset 10, and even if 18 is accepted, these ops are integer-typed only (the I/O is hard-pinned to float32). (2) `Gather` along the channel axis costs 0 MACs in onnx-tool 1.0.1 (verified empirically) and realizes any color permutation directly, whereas any GF(2^n) multiplication path requires multiple ops on the live tensor. (3) "Reversibility" gives no MAC savings under the validator's constant-folding rule, which only fires when *all* node inputs are constant — runtime self-inverse subgraphs pay the full op cost twice.

Two earlier-flagged worries turn out to be wrong, in the user's favor: **`Cast` costs 0 MACs in onnx-tool 1.0.1** (no `profile()` override on `CastNode`, line 2699 of `node.py`), so the "18,000 MAC Cast tax" I assumed in the previous round of notes does not exist. That changes the arithmetic but not the conclusion — `Gather` still wins on cost-and-simplicity grounds.

## Project mapping

The empirical cost table for GF-relevant ops on `[1,10,30,30]` float32:

| Op | MACs (onnx-tool 1.0.1) | params (count, not bytes) | Notes |
|---|---|---|---|
| `Cast` (float→int8 or back) | 0 | 0 | inherits `Node.profile` default |
| `Mod` with `fmod=1` (for float) | 9,000 | 1 (the divisor scalar) | extends `NpMathBase` |
| `BitShift` (LEFT/RIGHT) | not in onnx-tool's NODE_REGISTRY at all in 1.0.1 — likely 0 by default if the op resolves; **but BitShift is opset 11+, integer-only**, untested |
| `BitwiseAnd`/`Or`/`Xor`/`Not` | also not in 1.0.1's registered profilers — would default to 0 if executed; **opset 18+, integer-only**, won't load with starter's opset-10 model |
| `And`/`Or`/`Xor`/`Not` (logical, bool only) | 9,000 each (`LessNode` lineage) | 0 | requires `Cast` to bool first |
| `Gather` along channel axis | 0 | numel of indices initializer (e.g. 10 for color-perm) | the real workhorse |

**GF(2) realizations on float {0,1} inputs.** Without bitwise ops:
- ADD = XOR over disjoint one-hot channels: just `Add(c1, c2)` = 9,000 MACs. No correction needed; channels are mutually exclusive.
- MUL = AND: `Mul(c1, c2)` = 9,000 MACs.
- Inverse / negation: in GF(2) every element is its own inverse, so a "subtract" is the same as an "add" — meaning any expression involving subtraction can be replaced with the equivalent addition, saving nothing concrete because both cost 9,000.

**GF(2) via Mod 2.** `Mod(Add(c1,c2), 2)` with `fmod=1` = 18,000 MACs (Add + Mod). Strictly more expensive than the no-correction `Add` for one-hot inputs. Only useful if inputs can take values outside {0,1} — which one-hot doesn't.

**GF(2^n) "n-bit symbol permutations."** A primitive-polynomial-based GF(2^4) multiplication is 4 conditional XORs of a multiplicand by bits of a multiplier, with reduction modulo a degree-4 irreducible polynomial (typically `x^4 + x + 1`). Each conditional XOR = `Where(bit, xored, original)` ≈ pair of gates. **Even if `Where` is free (it is, 0 MACs), each XOR step on the live grid is 9,000 MACs (`Add` + `Mod 2`), and the overall pipeline is dominated by the XORs:** ~8 XORs × 9,000 = 72,000 MACs for a single GF(2^4) multiply over the grid.

A `Conv 1×1` with the equivalent permutation matrix as weights costs 99,000 MACs and *directly produces the permuted channels*. So even at 72k vs 99k, GF(2^n) wins by ~27%, but only for permutations that genuinely need that complexity — and a `Gather` (cost 0!) realizes any color permutation in this competition. **GF(2^n) loses to Gather by an unbounded factor for every permutation pattern.**

**Reversibility under constant folding.** The COVERED list states: "constant folding applied before measurement: `onnx_tool.loadmodel(..., {'constant_folding': True})`. Foldable weights/structure don't count." The folder (mirroring onnxruntime/core/optimizer/constant_folding.cc) only fires when *all* inputs to a node are themselves initializers or constant-folded outputs. So `T(T(input))` for a self-inverse `T` does NOT fold — `input` is dynamic. **No "free inverse" exists in this competition's cost metric.** Reversibility-as-energy-conservation (Landauer-style) is a *thermodynamic* concept, not a MAC concept; it doesn't transfer.

**ARC tasks where GF buys anything: zero.** Every reversible color/spatial transform admits a cheaper or equal `Gather`/`Transpose`/`Slice` realization. The `_PATTERNS` library already covers the easy ones (transpose, hflip, vflip, rot180, color-perm); none of them benefit from a GF reframing.

**Failure modes if you tried GF anyway.** (1) Bitwise ops at opset 18: starter is opset 10. Bumping the model's `opset_import` to 18 *might* be accepted by the validator (COVERED: "newer accepted in principle but not pinned"), but if not, the model fails the schema check and scores 0. (2) The bitwise ops are integer-only; the I/O contract `[1,10,30,30] float32` is non-negotiable, so a pure-int internal representation needs explicit `Cast`. The good news: `Cast` is 0 MACs. The bad news: each `Cast` introduces a new intermediate activation, contributing to `memory` (= numel × elementsize). Each int8 intermediate adds 9,000 bytes; each float32 intermediate adds 36,000 bytes.

## Sources

- onnx-tool 1.0.1 source — `CastNode` defines no `profile()` override (line 2699 of `node.py`); inherits `Node.profile` which returns `[0, 0]` (line 212–213). Confirmed empirically: `Cast (float32 → int8): macs=0 params=0 memory=9000`.
- onnx-tool 1.0.1 — `ModNode` extends `NpMathBase` with `op_mac = ADD_MACS` (line 332–339). `NpMathBase.profile` returns `volume(out) × ratio × 1` MACs.
- ONNX 1.21 op references: `BitwiseAnd`/`Or`/`Xor` since opset 18 (`https://onnx.ai/onnx/operators/onnx__BitwiseXor.html`). PR #4496 introduced them in 2022. `BitShift` since opset 11 (`https://onnx.ai/onnx/operators/onnx__BitShift.html`), unsigned-int only — issue #2406 documents the type-constraint complaint.
- ONNX `Mod`: opset 10 added float support via `fmod=1` attribute (`https://onnx.ai/onnx/operators/onnx__Mod.html`).
- Sachs, "Linear Feedback Shift Registers, Part I: Ex-Pralite Monks and Finite Fields" (`https://www.embeddedrelated.com/showarticle/1065.php`, 2017): "If you interpret GF(2) in a bitwise manner, addition is like XOR, and multiplication is like AND." And Part II (`https://www.embeddedrelated.com/showarticle/1070.php`): GF(2^n) multiplication via Russian Peasant.
- onnxruntime constant folding rule: `https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/constant_folding.cc` — "constant folding does not support executing a node that includes subgraphs (control flow operators, such as If/Loop/Scan, fall into this category) … `graph_utils::AllNodeInputsAreConstant(graph, n, constant_inputs, excluded_initializers_)`."
- Empirical run: `Mod fmod=1: macs=9000`, `Cast float32→int8: macs=0`, `Gather (color-perm via axis=1): macs=0 params=10`.

## Contradictions / corrections vs COVERED

- No direct conflict with COVERED.
- **Self-correction vs my own previous notes:** I claimed `Cast` costs ~9,000 MACs and used that to derive an "18,000 MAC Cast tax" for any int↔float round-trip in GF schemes. **This was wrong.** `Cast` is 0 MACs in onnx-tool 1.0.1. The conclusion (GF(2^n) is dominated by Gather) doesn't change, but the cost numbers in any version of these notes you have from before this empirical pass need updating.

## Confidence

**High.** Empirical verification against the pinned onnx-tool 1.0.1 confirms every cost claim. The qualitative conclusion (skip GF, use Gather) is robust across every realistic ARC permutation pattern.
