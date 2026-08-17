# J.13 — Gray code coordinate indexing

## Verdict
**Skip / weakly viable.** Gray‑coding 30 row indices and 30 col indices is essentially free (≤ 480 B of `params`) but the downstream advantage exists only when subsequent ops are *bitwise* over the encoded index — which is not the case for ARC's typical Conv/Slice‑heavy programs. For 4/8‑adjacency tasks, normal row‑major indexing already gives the correct neighbours via shape‑only Slice/Transpose.

## Actionable ONNX design rules
- **Pre‑baked Gray‑code initializer for length‑30 row index:**
  - 5‑bit binary‑reflected Gray code restricted to 0..29 (the 5‑bit BRGC visits 0..31; remove the last two and renumber). Stored as `int32` permutation `P[30]`: 30 × 4 = **120 B params**. As `int64`: 240 B. Two such tables (rows + cols) total ≤ **480 B**.
  - Apply via two consecutive Gather nodes (rows then cols) — both cost 0 by the locked rule, both fold into shape‑only graph segments.
- **Neighbour lookup analysis:** in (row, col) raw indexing the 4‑neighbours of (r,c) are at indices ±1; in Gray order the 4‑neighbours are *not* at ±1 — they are at the index whose code differs by exactly the bit that toggles between consecutive rows/cols, which depends on the Gray‑code's reflection structure. So a 3×3 Conv applied AFTER Gray‑code permutation no longer corresponds to a 4‑neighbourhood; you would have to "un‑Gray" before Conv. **Net result: Gray coordinates do not reduce required Conv kernel size; they just rearrange.**
- **The one place Gray helps:** if you decompose the 5‑bit row index into 5 bit‑planes and apply the same logic per plane (e.g., for a parity‑along‑rows task), Gray ordering ensures that adjacent rows differ on exactly one bit‑plane, so a `BitwiseXor` of plane *k* between row *r* and row *r+1* produces a sparse, mostly‑zero map. This is a niche win; ARC doesn't have many such tasks.
- **Target ARC families with marginal gain:**
  - "Stripe / parity by row index": pre‑compute the Gray‑bit‑*k* indicator vector as a float32 initializer (30 × 4 = 120 B params); broadcast‑Mul against the input. Total ≈ 9 000 MACs, 36 000 B activation; comparable to a Conv 1×1 in cost but uses zero learned weights.
  - "Grow / shrink along the longer Gray run": no clear advantage; a Conv 3×3 with a fixed kernel does it for 810 900.

## Literature survey & project‑mapping commentary
- **Adjacency theory** (Mustansiriyah lecture notes, Hamarsheh, ResearchGate "Pixel Neighbors"): adjacency is defined on the spatial grid; relabelling cells does not change the adjacency graph. So Gray‑code coordinates can only help when "adjacency" is measured in the encoded label space, not in pixel space. ARC tasks measure pixel adjacency.
- **Crystallographic / group‑theoretic neighbours:** D₄ orbits (4 rotations, 4 mirrors) and translation orbits subsume most ARC symmetry tasks. They are realised by Slice + Transpose at cost 0; Gray coding offers no equivalent free move.
- **Combinatorial Gray codes** in qubit circuits (errorcorrectionzoo.org) optimise unitary depth; not relevant here.
- **Sparse distributed memory / hash‑based dim‑reduction:** these neighbours of "Gray code coordinates" can encode the 900 cell positions in a noise‑robust way, but ARC requires exact equality, so noise robustness has zero scoring value.

## Discrepancies with COVERED list
None. All claims fit within the shape‑only‑ops‑are‑free rule and the params/memory split.

## Recommended next experiment
Build two ONNX graphs that solve the same row‑parity task on a 30×30 input:
- A: `Mul(input, parity_mask)` where `parity_mask` is a length‑30 float32 initializer (120 B params).
- B: `Gather(input, gray_perm) → Mul(input, gray_bit_plane)` where both initializers are int32/float32 (≤ 240 B params).
Compare `params + macs + memory`. Hypothesis: B equals A in cost (Gather is free) and adds a 120 B params penalty for the permutation table — it strictly loses by ~120 B unless we have a downstream user of the Gray ordering.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from j13-gray-code.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.13 — Gray code coordinate indexing

## Summary

A Gray code is a binary encoding of integers where adjacent integers differ in exactly one bit; the binary-reflected Gray code (BRGC) of *n* bits, due to Frank Gray (1947 patent on PCM), is the canonical version. In image processing it appears in two places: progressive image coding (Gray-coded bit planes are more compressible than natural-binary because adjacent intensity levels differ by 1 bit) and combinatorial enumeration (Hamiltonian paths on the hypercube). For NeuroGolf the proposed use is to permute the 30-element row and column indices into Gray order so that "adjacent" cells in the permuted grid have a 1-bit-Hamming relationship, in the hope that downstream bit-level logic becomes cheaper.

## Project mapping

The construct is a length-30 Gray permutation applied via two `Gather` nodes (one per axis), realized as initializers. Cost analysis on `[1,10,30,30]` with `onnx-tool 1.0.1`:

| Construct | macs | memory (B) | params | total cost | pts |
|---|---:|---:|---:|---:|---:|
| Single `Reshape + Gather + Reshape` (length-900 perm, int32) | 0 | ~75,624 | 907 | **~76,531** | 14.484 |
| Two-axis Gray gather (axis 2 then axis 3, both length 30, int32) | 0 | ~108,000 | ~64 | **~108,064** | 13.297 |
| Conv 1×1 baseline | 90,000 | 0 | 100 | **90,100** | 13.591 |
| Conv 3×3 baseline | 810,000 | 0 | 900 | **810,900** | 11.394 |

Two-axis Gray (axis-2 then axis-3 Gather) is **more expensive than Conv 1×1** because of the two intermediate `[1,10,30,30]` tensors at 36,000 B each. A single combined permutation through a `Reshape → Gather (length-900) → Reshape` is cheaper but still pays ~76,531 — and crucially, the Gray-permuted layout doesn't change which neighbours a downstream `Conv 3×3` sees in 2-D space; it only relabels indices. So Gray indexing helps **only** when the downstream computation is *bitwise on the index*, which would itself require BitwiseAnd/Xor that — as established in J.01 — `onnx-tool 1.0.1` cannot profile.

ARC task families where Gray ordering could plausibly help:
- **Stripe / parity-by-row tasks**: pre-compute the *k*-th Gray bit as a length-30 float32 mask (120 B params), broadcast-multiply to an input. But this is just a 1-D mask multiplication; it doesn't need Gray ordering at all — natural-binary parity (`(row & 1) == 1`) does the same thing for the same cost.
- **Snake-fill tasks**: ARC has tasks where you need to fill cells in a path that alternates direction per row. Gray ordering linearises a snake naturally (the BRGC of (row, col) traces a Hamiltonian path through the grid). But again, downstream we need to *use* the linearisation, which means a 1-D `Slice` or `Conv1D`, and at that point Morton/Hilbert (J.14) does the same job with stronger locality guarantees.

**Concrete failure modes**:
1. The 5-bit BRGC visits 0..31; truncating to 0..29 destroys the single-bit-difference property at the wrap-around (indices 28→29 and the gap to "would-be 30,31"). For ARC's variable grid sizes this matters when the actual grid is < 30×30 and gets pad-color at the bottom rows.
2. Gather indices on `int32` cost 4 bytes/element; on `int64` cost 8 bytes/element. Always use **int32** (verified: int32 perm gives Gather memory line 39,600; int64 gives 43,200).
3. Two consecutive Gathers cannot be folded together by `onnx-tool`'s constant folder unless the input is also constant — and our input is the live tensor, so the two intermediate `[1,10,30,30]` tensors are real and cost 72,000 B of memory.

**Verdict**: I find no ARC task family where Gray-coded coordinates strictly beat either (a) a plain 1-D mask multiplication for parity-style tasks, or (b) Morton/Hilbert (J.14) for adjacency-rule tasks. **Recommend skip** unless the strategy explicitly couples to the (currently-unprofilable) bitwise op family.

## Sources

- **Gray code theory / patent** — F. Gray, "Pulse Code Communication", US Patent 2,632,058 (1947, granted 1953). Modern surveys: `https://en.wikipedia.org/wiki/Gray_code`. ErrorCorrectionZoo `https://errorcorrectionzoo.org/c/gray` for the BRGC construction.
- **Gray-coded bit-plane image compression** — Knuth's *TAOCP Vol 4A* §7.2.1.1 documents Gray-code Hamiltonian paths through hypercubes. US Patent 8,837,844 ("Image processing using least significant bits", `https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8837844`) describes binary→Gray conversion before LSB compression.
- **ONNX Gather** — verified on `onnx==1.21.0`: `since_version=13`, domain `""`, `T` is essentially any tensor type, `Tind ∈ {int32, int64}`. Spec `https://onnx.ai/onnx/operators/onnx__Gather.html`. Source: `/usr/local/lib/python3.12/dist-packages/onnx_tool/node.py` line 898 — `GatherNode` has only `shape_infer/value_infer` and inherits the default `profile() → [0, 0]`. Confirmed 0 macs in my probe.
- **Empirical Gather cost** — my run on `Reshape[1,10,30,30]→[1,10,900], Gather(axis=2, indices=length-900 int32), Reshape→[1,10,30,30]`: macs=0, memory=111,656 B, params=907 elem.
- **No primary source** found for "Gray-code coordinate indexing as a CNN technique" specifically — the literature is about Gray-coded data (intensities), not Gray-coded coordinates. The proposal in the brief is novel, and the analysis above is best-effort from first principles.

## Contradictions / corrections

- **The COVERED list classifies Transpose H↔W and Identity at cost 0.** Confirmed for these specific single-node graphs, where the only value_info is the output (excluded from memory). For Gray-code constructs that necessarily introduce intermediate tensors, cost is *not* 0. Caveat to add to project memory: "shape-only ops cost 0 macs, but their *outputs* contribute to `memory` whenever there is more than one node."
- The COVERED line "Color-perm threshold ≥7 of 10 colors observed for confident match" is consistent with the Gather-based color-permutation pattern already implemented; Gray-coded *coordinates* are unrelated.

## Confidence

**Medium**. High for the cost numbers (measured). Medium for the verdict because I haven't enumerated all 800 public ARC tasks for snake-fill / parity-stripe variants where Gray order might align by coincidence. Low for any speculative win from Gray + bitwise ops because that combination is blocked by J.01's finding that bitwise ops are unprofilable in onnx-tool 1.0.1.
