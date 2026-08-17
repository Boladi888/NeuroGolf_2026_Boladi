# Research Card O6 — Exact Boolean-circuit minimization compiled to opset-18 BITWISE ops on packed lanes

**Deliverable**: one copyable Markdown research note (no file writes, no code execution). Save the note under `Research/Done` with basename `o6_boolean_exact_synthesis_to_bitwise`. **Priority: MEDIUM-HIGH.** A rigorous crossover verdict (when this beats our incumbents) is the goal; a well-supported "does not beat K4's Conv-compile" is acceptable.

## Researcher contract

Internet access, NO repo access. Produce ONE self-contained Markdown note; do not ask follow-up questions. Human-readable citations (author-year-title / canonical tool names: ABC, EPFL mockturtle, ESPRESSO). Tag every nontrivial claim `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Project facts (all `[PROMPT]`)

- **Task**: minimal static ONNX graph that EXACTLY computes a KNOWN per-cell discrete rule. Many ARC rules are per-cell boolean functions of a small neighborhood: `out[r,c] = f(in[r-1..r+1, c-1..c+1])` for a `k`-cell window (`k = 4..25`), or `f` of a few boolean feature planes.
- **Cost** (minimize): `cost = memory + params`. `params` = Σ ELEMENT COUNTS of initializer + Constant tensors (unique counted once; **dtype-blind — element count, not bytes**). `memory` = Σ bytes of intermediate `value_info` tensors (`max(static, runtime)`); the single graph `input` and single `output` are NOT charged. A `[1,1,30,30]` u8 intermediate = 900 B; a `[1,1,30,1]` lane = 30 B; bool = 1 B/elem.
- **Incumbents to BEAT (already in our library — do NOT re-derive):**
  - **K4 verdict**: the one sanctioned logic-synthesis transfer is offline ESPRESSO/ABC two-level/AIG minimization mapped onto a **1×1 `Conv` + threshold** (one charged output plane + small kernel). This card asks a DIFFERENT compile target.
  - **opset-18 BITWISE ops are GRADER-VALIDATED live levers**: `BitwiseAnd/BitwiseOr/BitwiseXor/BitwiseNot` (integer dtypes) and `BitShift`. They exist at opset 18 ONLY (verified: checker+ORT reject at ≤17). Logical `And/Or/Xor/Not` are bool-only kernels.
  - **Row/lane bit-packing is established**: width ≤8 packs a whole ROW into one `uint8` (bit c = column c); a `uint32` packs 32 cells. A bitwise op on a packed `[1,1,30,1]` u8 lane then processes 8 cells/element; on u32, up to 32. Neighbor access via `BitShift`/u8-overflow lane shifts.
  - A 512-entry neighborhood LUT (3×3 binary → depthwise-Conv encode + `Gather`) prices ~90k and only TIES a single Conv; it wins only for non-monotone rules needing ≥3 Convs otherwise.
- **Environment**: ORT 1.24.4 CPU, `ORT_DISABLE_ALL` (no fusion/folding). opset bumpable to ≥18 (ORT loads through opset 25). Banned: `Loop/Scan/NonZero/Unique/Compress`, any `"Sequence"`-substring op, Functions, `If`/subgraphs. Static shapes. Input `[1,10,30,30]` F32 one-hot; output decoded `> 0.0`. Exactness mandatory on the full generated distribution.

## The gap this card fills (vs the rest of the board)

K4 mapped minimized boolean forms only onto **Conv/threshold**; the bitwise ops were grader-validated *after* K4. The board's O1 (general algorithm mining) explicitly declines "bit-packing as a general topic" and does not cover logic synthesis. So nothing on the board evaluates **XOR-rich exact circuits (XAG = AND+XOR graphs, MIG = majority-inverter graphs) compiled to our bitwise ops on PACKED lanes**, where one gate processes 8–32 cells per charged tensor. XOR is nearly free spatially (parity, CA, symmetry) yet expensive as a Conv/threshold; a `BitwiseXor` on a packed lane is one cheap node. That crossover is unexplored.

## Required answer order

1. **Exact-synthesis tooling for small boolean functions (≤25 inputs).** Survey PRACTICAL, runnable tools emitting a PROVABLY MINIMAL (or near-minimal) multi-level circuit for a fully-specified boolean function: ABC (`&exact`, `if -K`, `&deepsyn`), EPFL **mockturtle** (`exact_aig/xag/mig`, SAT-based exact synthesis), ESPRESSO (two-level only). For each: input format, gate library minimized for (AIG / **XAG** / **MIG**), practical input-size ceiling for EXACT (not heuristic) results, output format. Which produce XOR-rich (XAG/MIG) minimal circuits?
2. **Gate → ONNX bitwise mapping + per-gate cost `[DERIVED]`.** Map {AND, OR, XOR, NOT/INV, MAJ} → ONNX `{BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, BitShift, +composition}`. Price each gate on a FULL `[1,1,30,30]` u8 plane (900 B/gate) vs a ROW-PACKED `[1,1,30,1]` u8 lane (30 B/gate, 8 cells/byte) vs a u32 lane (32 cells/elem). MAJ(3) → how many bitwise ops? Neighbor windowing on packed lanes via `BitShift` — cost per shifted operand. Price the pack/unpack steps once.
3. **Crossover: minimal-XAG-on-packed-lanes vs incumbents.** For representative per-cell rules — (i) a parity/CA rule (XOR-heavy, e.g. Life-like), (ii) a monotone threshold ("≥k of 8"), (iii) a corner/T-junction detector (non-monotone), (iv) a small symmetry predicate — compare total `cost` of: (A) minimal XAG → bitwise on packed lanes, (B) K4 ESPRESSO→1×1-Conv+threshold, (C) 512-entry LUT-Gather, (D) direct hand-Conv stack. Where does XAG-on-lanes WIN vs LOSE? Quantify how XOR-density and window size `k` move the crossover.
4. **Exact-synthesis as a DESIGN-TIME artifact.** Confirm the workflow is offline (run the synthesizer once to DESIGN a fixed graph; no runtime search) and name the transcribable artifact (a gate netlist). Note any gate (wide MAJ, large XOR tree) whose bitwise expansion erases the win.
5. **Overall verdict** — one blunt paragraph: the single most actionable item (tool + gate library + rule class) with its win regime, or "does not beat K4 Conv-compile / hand-construction" with the arithmetic.

## Required tables

Section 1: `| tool | gate library (AIG/XAG/MIG/SOP) | exact vs heuristic | practical input ceiling | output format | citation |`

Section 2: `| gate | ONNX bitwise realization | cost on full [30,30] plane | cost on packed [30,1] u8 lane | cost on u32 lane | notes |`

Section 3: `| rule class | XAG-on-lanes cost | K4 Conv-compile cost | LUT-Gather cost | hand-Conv cost | winner | crossover driver |`

## Demands

- `params` = element count (dtype-blind); the bitwise advantage must come from FEWER/SMALLER charged INTERMEDIATES (packed lanes), not dtype narrowing.
- Bitwise ops are opset-18-only and need integer dtypes (not bool); logical And/Or are the bool-only alternative — say when each is correct.
- Distinguish a circuit's GATE count from its ONNX NODE/TENSOR count (each gate = one charged intermediate unless packed); the packed-lane amortization (cells-per-element) is the crux.
- This refines K4 — do NOT re-survey CompressARC / DreamCoder / threshold-circuit theory; focus on EXACT XAG/MIG synthesis → BITWISE-on-lanes, which K4 did not cover.
- No code beyond ≤15-line netlist/op sketches.

## Source policy

Primary: tool docs/papers for ABC, EPFL mockturtle, ESPRESSO; ONNX spec for each bitwise op; ORT docs for integer-dtype kernel support. Mark cost-model deductions `[DERIVED]`.

## Final checklist

- [ ] At least one runnable EXACT-synthesis path for XAG and for MIG, with input-size ceilings.
- [ ] Per-gate cost priced on full-plane AND packed-lane representations (with cells-per-element amortization).
- [ ] Crossover table covers XOR-rich, monotone-threshold, non-monotone, and symmetry rule classes vs all four incumbents.
- [ ] Offline-design workflow confirmed (no runtime search); transcribable artifact named.
- [ ] Verdict names ONE tool+gate-library+rule-class win, or refutes with arithmetic.
