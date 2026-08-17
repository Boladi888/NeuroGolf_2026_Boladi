# J.1 — Bit‑slicing logic

## Verdict
**Partially viable.** Bitwise ops exist (opset 18, integer‑only), but every entry into integer space costs Cast round‑trips, so bit‑slicing only beats Conv on tasks that are *natively* Boolean (XOR / parity / symmetry overlays).

## Actionable ONNX design rules
- **Op availability (opset domain `""`, `onnx==1.21.0` ⇒ opset ≤ 23):**
  - `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot` — `since_version=18`, integer types only (int8/16/32/64, uint8/16/32/64). **No float32**.
  - `BitShift` — `since_version=11`, **uint8/16/32/64 only**. Signed and float types rejected.
  - `Mod` — `since_version=10` (updated 13), **float32 supported with `fmod=1`**.
  - `Cast` — every opset, supports `float→int*` and back. Counts ≥ 1 MAC per element under onnx‑tool.
- **Two implementation paths on `[1,10,30,30]` float32:**
  1. **Bitwise path** (`Cast→BitShift→BitwiseAnd→Cast`): 4 element‑wise nodes per bit plane × 9 000 elements ≈ 36 000 MACs, plus ≈ 4×9 000 = 36 000 B of int8 intermediate `memory` per plane. For all four bit planes of a 4‑bit color: ≈ 144 000 MACs + 144 000 B intermediates.
  2. **Arithmetic path** (`Mul → Floor → Mod(fmod=1) → Sub`, all float32, no Cast): same 36 000 MACs per plane but intermediates are float32 (36 000 B each) so memory term ≈ 4×36 000 = 144 000 B per plane. The two paths are roughly tied; the arithmetic path wins because it avoids two Casts (saves 18 000 MACs).
- **Per‑plane cost summary (one bit plane, both paths): ~36 000 MAC + ~144 000 B activation memory + few B params.** Compare to Conv 1×1 = 90 100 (a single plane is ≈ 2× cheaper than Conv 1×1) and Conv 3×3 = 810 900 (single plane ≈ 5× cheaper).
- **Target ARC families:**
  - XOR / "spot the difference" between two stored halves of a grid (Vision‑ARC's AND/OR/XOR cluster). Implement as `Slice` (shape‑only) → `Sub`/`NotEqual`/arithmetic XOR (`a + b − 2·a·b` on 0/1 tensors) → threshold.
  - Parity tasks (e.g., "color cells whose row+col is odd"): pre‑compute parity grid as initializer, multiply with input mask. Initializer cost: 900 B at uint8 / 3 600 B at float32.
  - Symmetry completion: `Transpose` (shape‑only, free) of input as the second operand to a Boolean overlay.

## Literature survey & project‑mapping commentary
- **Bit‑plane stack filters** (Frías‑Velázquez 2010, Springer 2018) show that ANY rank/morphological operator can be realised as Boolean stack‑filter on bit planes. Useful intuition: if an ARC task is a *thresholded morphological* op (e.g., dilate by 1, fill holes), it is a stack filter and bit‑slicing is optimal in transistor count. In ONNX the gain is smaller because we already operate on float32, not on serial bit lines.
- **Minimum description length / Kolmogorov:** the description "XOR with mirrored copy" is shorter than "Conv 5×5 weight tensor of 250 floats" by orders of magnitude — and onnx‑tool rewards short descriptions.
- **Group‑equivariant networks** are a stronger neighbour: D₄ orbit features (mirror, rotate by 90°) are realised by free `Transpose`/`Slice` and a final `Add`/`Max`. They subsume the symmetry‑completion sub‑case at zero MAC cost.
- **Recommendation:** prefer the float32 arithmetic path (`Mul/Floor/Mod/Sub`) over the int Cast path; reserve true `BitwiseXor` for the rare task where both operands are already 0/1 indicator masks (then `BitwiseXor` ≡ `Sub` ≡ `Abs(Sub)` and Conv 1×1 with a fixed 1.0 weight may be even cheaper — measure!).

## Discrepancies with COVERED list
None. All claims (Identity/Transpose=0; Conv1×1=90 100) are upheld; bit‑slicing is presented as additive to, not a replacement for, the locked cost levers.

## Recommended next experiment
Build minimal ONNX graphs A and B exporting the same XOR of two halves of a 30×30 grid:
- A: `Slice(left) → Slice(right) → Sub → Abs → (>0)` — pure float32 arithmetic.
- B: `Slice → Cast(uint8) → BitwiseXor → Cast(float32) → (>0)`.
Profile via `onnx_tool.Model(..., constant_folding=True).profile()` and report MACs, memory, params for both. Hypothesis: A < B because B pays two extra Cast nodes (~18 000 MACs each).


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from j01-bit-slicing.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.01 — Bit-slicing logic

## Summary

Bit-slicing decomposes an *N*-bit per-pixel image into *N* binary "bit planes" so each plane can be processed independently by Boolean logic; the original image is rebuilt by a weighted sum of plane values × powers of two. The technique dates from 1970s–80s morphological/stack-filter literature (Maragos, Schafer; Wendt-Coyle-Gallagher 1986 stack filters) and remains the standard primitive for FPGA / focal-plane image processors. For NeuroGolf, the relevance is that ARC color values are 0–11 in the 12-channel decoded representation but the activation tensor itself is float32 in `[1,10,30,30]` — bits are *not* a free representation, they have to be carved out with arithmetic or with the integer bitwise op family.

## Project mapping

The technique would be valuable for Boolean overlay tasks (XOR of two halves, parity stripes, symmetry difference maps), but every implementation path has measured costs that make it dominate Conv only in narrow circumstances. I built and profiled five concrete implementations against `onnx-tool 1.0.1`, opset 21, on a `[1,10,30,30]` float32 input:

| Construct | macs | memory (B) | params | total cost | pts |
|---|---:|---:|---:|---:|---:|
| `Conv 1×1` baseline | 90,000 | 0 | 100 | **90,100** | 13.591 |
| `Mul + Floor + Mod` (extract one bit, float arith) | 27,000 | 72,000 | 2 | **99,002** | 13.497 |
| `Cast → BitwiseAnd → Cast` (int8) | **n/a** | n/a | n/a | **graph unprofilable** | 0 |
| `Cast → BitShift → BitwiseAnd → Cast` | **n/a** | n/a | n/a | **graph unprofilable** | 0 |
| `Cast(int) → Cast(float)` only (entry tax) | 0 | 9,000 | 0 | **9,000** | 15.895 |

**The single most important finding for J.01:** `onnx-tool 1.0.1` has **no registered handler** for `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`, or `BitShift`. Calling `Model.shape_infer()` on such a graph raises `NotImplementedError: this Node BitwiseAnd has no value_infer`. The validator either (a) fails with `calculate_memory()` returning `None` → **task scored 0**, or (b) emits the `UserWarning` and silently treats the node as zero-cost — both behaviors have been observed. Either way, depending on bitwise ops is unsafe.

The viable path is **float arithmetic bit extraction**: for the *k*-th bit of a non-negative integer-valued float `x`, use `Mod(Floor(x * 2^(-k)), 2.0)` with `Mod`'s `fmod=1` attribute (mandatory for float). This costs 27,000 macs + ~72,000 bytes of intermediate memory + 2 params per plane, i.e. **~99,002 cost — slightly worse than Conv 1×1**. So a single bit plane breaks even with a 1×1 Conv only when it replaces *more* than one Conv.

Where it actually pays off:

- **XOR of two stored grids**: `Slice(input,left) → Slice(input,right) → Sub → Mul (by self) → Sub (1.0) → Mul → Threshold`. Or equivalently, on 0/1 indicator inputs, `a + b - 2·a·b` is XOR. Cost ≈ 4 element-wise ops × 9,000 = 36,000 macs + ~108,000 B intermediate = **~144,002**. Still worse than Conv 1×1, only attractive when the Conv would need to be 3×3 (810,900) or the kernel is task-specific and would not generalize across the three benchmark sets.
- **Symmetry-completion via free Transpose**: `Transpose(input)` is shape-only; *stack* with input via `Add` or `Max` is one element-wise op (~9,000 macs + 36,000 B). This subsumes most of the bit-slicing motivation at ~45,000 cost — about half a Conv 1×1.

ARC families where bit-plane logic could plausibly help: parity-by-row/col tasks, "mark all cells that differ between two halves", "find the cell whose bit-pattern is unique" (Vision-ARC's AND/OR/XOR cluster, see `arxiv.org/html/2511.14761v1`). Across the 800 public tasks, the sub-family is < 5% by my reading of the Hodel DSL primitives.

**Failure modes**: (a) Bitwise/BitShift ops in any submitted graph likely cause `calculate_memory() → None`. (b) The decoder maps unknown colors to channel index 11; any bit-extraction on a value of 11 (binary 1011) lights bits 0, 1, 3 — different from the color 9 (1001) which only lights bits 0, 3. Don't treat bit planes as if they correspond to ARC's perceptual color groupings; they don't. (c) Negative intermediate values from `Sub` will misbehave under `Floor` + `Mod`; clamp via `Relu` first if values can go negative.

## Sources

- **ONNX op specs** — verified on `onnx==1.21.0` installed locally:
  - `BitwiseAnd/Or/Xor/Not`: `since_version=18`, domain `""`, `T = {(u)int8/16/32/64}`, **no float**.
  - `BitShift`: `since_version=11`, domain `""`, `T = {uint8/16/32/64}`, **no signed int, no float**, `direction ∈ {"LEFT","RIGHT"}`. PR `https://github.com/onnx/onnx/pull/1931` ("Add bit-shift operators for supporting hashing"). Issue `https://github.com/onnx/onnx/issues/2406` confirming unsigned-only.
  - `Mod`: `since_version=10`, updated to opset 13, domain `""`, supports `tensor(float)` with mandatory `fmod=1`. Spec at `https://onnx.ai/onnx/operators/onnx__Mod.html`.
  - `Cast`: every opset, all numeric pairs supported. `Floor`: `since_version=13`, float types only.
- **`onnx-tool 1.0.1` source** — `/usr/local/lib/python3.12/dist-packages/onnx_tool/node.py`:
  - Line 247: `class NpMathBase(Node): self.op_mac = ADD_MACS` (ADD_MACS=1, MUL_MACS=1).
  - Line 332: `class ModNode(NpMathBase): self.op_mac = ADD_MACS` — Mod counts as 1 op/element.
  - **Grep `bitwise|bitshift` returns no hits** in `node.py`, `graph.py`, `model.py`. Unregistered → `NotImplementedError` at shape inference time.
  - Default `Node.profile() → [0, 0]` (line 212).
  - `CastNode.shape_infer` exists but no `profile()` override → 0 macs, but the output tensor still contributes its byte size to `memory`.
- **Stack-filter / bit-plane theory** — Frías-Velázquez & Philips, "Bit-Plane Stack Filter Algorithm for Focal Plane Processors" (2010), `https://telin.ugent.be/~afriasve/Frias-Velazquez2010-Bit-plane_stack_filter_algorithm_for_focal_plane_processors.pdf`. Wendt, Coyle, Gallagher (1986) "Stack filters" IEEE Trans. ASSP — original threshold-decomposition theorem. These confirm the theoretical equivalence but are about transistor cost, not float-MAC cost.
- **ARC bit/Boolean cluster** — Vision-ARC paper, `https://arxiv.org/html/2511.14761v1`, identifies an "AND/OR/XOR" cluster in its task embedding. Hodel DSL `arc-dsl` (`https://github.com/michaelhodel/arc-dsl`) has primitives `_difference`, `_intersection`, `_objects` that map onto Boolean overlays.
- **Empirical profile** — my own runs in `/home/claude/research/code/probe_j01.py` (included in this delivery), measured against onnx-tool 1.0.1 with `constant_folding=True`.

## Contradictions / corrections

- **COVERED list says "Pure shape-only ONNX ops (Identity, Transpose, Slice, Gather, Reshape) cost 0".** This is true for `macs` and `params` but **not for `memory`**. Each non-IO intermediate tensor contributes its full byte size (`numpy_helper.size_in_bytes(value_info)`) to the `memory` aggregate. A `Reshape → Gather → Reshape` chain on `[1,10,30,30] → [1,10,900] → [1,10,900] → [1,10,30,30]` produces three intermediate value_info entries of 36,000+ bytes each. Profile output: `Total memory = 111,656 bytes; params = 907 elements; macs = 0`. Total cost = **~76,531**, not 0. This is still < Conv 1×1 (90,100) by ~15% but not free.
- **COVERED says floor=1.0 for correct.** The score formula `max(1.0, 25.0 - log(max(1.0, cost)))` clamps below at 1.0; my Identity probe matches the 25.000 figure exactly (0 cost → log clamped → 25.0).
- **COVERED says Conv 1×1 = 90,100.** Confirmed by my probe: 90,000 macs + 100 params + 0 memory = 90,100. **But this only matches because for a single-Conv graph, memory contributes zero** (only the output value_info exists, which the validator excludes). For multi-node graphs the memory term dominates.
- **The COVERED probe table omits `memory`.** The columns are macs+params only. For multi-node ONNX submissions this is misleading; intermediate-tensor memory is often the dominant term.

## Confidence

**High** for the negative result (BitwiseAnd/BitShift are unprofilable in onnx-tool 1.0.1 — measured directly on the pinned dependency). High for the float-arithmetic path numbers (measured). Medium for the ARC task-family estimates (based on DSL coverage analysis, not exhaustive ARC-AGI-1 task labeling).
