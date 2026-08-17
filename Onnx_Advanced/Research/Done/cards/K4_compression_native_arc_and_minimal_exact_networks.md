# Card K4 — Compression-native ARC solvers + minimal exact networks for KNOWN discrete rules (2024–2026)

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `k4_compression_native_minimal_nets.md`. **Priority: LOW — run after K1/K2/K3/K5.** A well-supported "nothing transfers" verdict is an acceptable, useful outcome; do not pad.

**Researcher contract**: internet access, NO repo access; self-contained card. Human-readable citations (author-year-title / repo paths). Tag claims `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Environment (all facts [PROMPT])

- Kaggle contest: build the SMALLEST static ONNX network per task that EXACTLY solves a fully-known grid transformation. Per-task `points = max(1.0, 25 − ln(memory + params))`; `params` = initializer bytes; `memory` = Σ node-output tensor bytes (max of static/runtime), ORT 1.24.4 CPU, optimizations off, opsets 10–18, no Loop/Scan/If/NonZero/Unique/Compress, static shapes only, file ≤1.44 MB.
- Crucial framing: unlike ARC-the-benchmark, we are NOT doing few-shot inference. For each task **we already know the rule** (decoded by humans/agents from 200k+ generated examples). The open problem is purely: *the minimal exact network computing a known discrete function* on `[1,10,30,30]` one-hot float grids (10 colors, ≤30×30), output binarized by `> 0.0`, exactness mandatory on the full input distribution.
- We hand-construct everything today (Conv counters, comparisons, Einsum contractions, scatter/gather, boolean planes). Typical shipped graphs: 100–25,000 bytes total.

## The questions, in required answer order

1. **CompressARC and descendants** — the 2025 "ARC-AGI without pretraining" compression-based line (Liao et al. / iliao2345, and anything citing it through 2026): faithful technical summary (what is compressed, decoder architecture, equivariance layers, parameter budgets, runtime), reported results, and the EXTRACTION question: do its equivariance/decoder design ideas suggest reusable fixed-graph constructions for single known rules? Distinguish ideas usable in a static ONNX (no training at inference) from ones needing optimization loops.
2. **MDL / program-synthesis-as-compression for ARC** (DreamCoder-on-ARC, Hodel DSL search, 2024–2026 follow-ups): does any work emit MINIMAL imperative/dataflow programs whose primitives map onto tensor ops? We care about the minimality machinery (program priors, library learning), not the search-for-unknown-rule part.
3. **Minimal exact networks for known boolean/discrete functions** — the sharpest section:
   a. Threshold-circuit synthesis: known size bounds and PRACTICAL synthesis tools for depth-2/3 threshold circuits computing given boolean functions (our Greater/Conv stack IS a threshold circuit). Anything actionable for 9–25-input window predicates?
   b. Weight-agnostic networks (Gaier & Ha 2019 + successors): topology-search yielding tiny exact discrete behaviors — practical tooling status.
   c. Lottery-ticket / magnitude-pruning to EXTREME sparsity with EXACT functional preservation on a finite input set — any literature on exact (not approximate) pruning for discrete tasks?
   d. Logic-synthesis crossover: ESPRESSO/ABC-style two-level/multi-level minimization of boolean functions, then mapping minimized SOP/AIG forms onto Conv/MatMul+threshold — any prior art doing this mapping? If none, sketch the mapping cost yourself `[DERIVED]`: AIG node → which ONNX ops, bytes per gate.
4. **Anything else 2024–2026** explicitly optimizing model SIZE as the objective for exact tasks (model-golf contests, "smallest MNIST" style work, tiny-model leaderboards) — list with one-line relevance each.
5. **Overall verdict** — one blunt paragraph: name the single most actionable item for a team hand-building 100–25,000-byte exact networks, or state plainly that nothing beats current hand construction.

## Required table schema (sections 1–4)

| approach | core mechanism | size regime | exact or approximate? | needs runtime optimization? | transferable element for static ONNX | citation |

## Demands

- Separate cleanly: (a) ideas needing training/search at inference (unusable — we ship fixed graphs), (b) ideas usable offline to DESIGN a fixed graph (usable).
- For section 3d, if no prior art exists, the `[DERIVED]` mapping sketch is mandatory: bytes per AND/OR/NOT/threshold gate under our cost model (e.g., a 1×1 Conv threshold gate over k input planes costs ~? params + 900–3,600 B output plane).
- No code beyond ≤15-line sketches.

## Final verification checklist

- [ ] CompressARC summarized from primary sources (paper/repo), not secondhand blogs alone.
- [ ] Section 3 covers all four sub-areas (a–d) with explicit verdicts.
- [ ] Every "usable offline" claim states WHAT the offline output artifact would be (a circuit, a topology, a weight set).
- [ ] Verdict names ONE most-actionable item or says "nothing transfers" plainly.
