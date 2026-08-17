# Card L2 — ORT 1.24.4 CPU Einsum internals: operand limits, contraction order, in-kernel memory, latency model

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `l2_ort_einsum_internals.md`.

**Researcher contract**: internet access, NO repo access; self-contained card. Cite onnxruntime source at the **v1.24.4 tag** (`file::function`/class names; say so if you must use a nearest browsable 1.2x tag) plus the ONNX Einsum spec. Tag claims `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Why this matters now (all facts [PROMPT])

- Our winning graph pattern is **multi-operand Einsum-as-the-output-node** on CPU EP, optimizations disabled, static shapes, float32. Shipped/validated: a 6-operand einsum-final (scored exactly, ran 200k+ inferences); locally run: 9-operand. Operands are the [1,10,30,30] input (sometimes appearing 2-3 times) plus small selector initializers.
- Our scorer charges only NODE OUTPUTS (per-tensor max of static/runtime bytes from the profiler) — **whatever Einsum allocates internally is free for scoring but real for RAM/latency**. One worker found operand ORDERING changes internal allocation ("smallest-first keeps pairwise internals KB-scale") — we want the source-level truth.
- Grading runs ~thousands of inferences per task with an (unpublished) time budget; one prior task hit runtime limits. We need a latency model for einsum-finals before designing 6-9-operand graphs routinely.

## The questions, in required answer order

1. **Decomposition strategy** — from `einsum.cc` / `EinsumComputePreprocessor` and related v1.24.4 sources: how does CPU Einsum execute a multi-operand equation? Strict left-to-right pairwise contraction? Any reordering/optimization of contraction order? Where do Transpose/MatMul/ReduceSum sub-calls come in?
2. **Operand-count limits** — any hard cap on inputs (spec or implementation)? Known issues/PRs about high operand counts?
3. **Intermediate allocation model** — for a pairwise chain, what tensors are materialized between steps (shapes as a function of the running contraction)? Are they ORT temporary allocations (invisible to node-output profiling) — confirm at source. Derive: the rule for operand ORDERING that minimizes peak intermediate size (is the worker's smallest-first heuristic optimal, or is it interleave-by-shared-indices?).
4. **Latency model** — per pairwise step, the implementation path (MatMul-mapped vs naive loops): which equation shapes hit the fast GEMM path vs element-loop fallbacks? What makes an einsum SLOW (diagonal subscripts? repeated labels? broadcast dims?). Give a practical complexity formula per step.
5. **Dtype + special-subscript support** — float32 confirmed; which others have CPU kernels; repeated-subscript (diagonal) handling at source; ellipsis handling; zero-dim edge cases.
6. **Failure modes** — known einsum bugs/issues in the 1.2x line relevant to: many operands, large rank, repeated labels, output-subscript permutations. Cite issue numbers.
7. **Overall verdict** — one paragraph: the design rules for einsum-finals (operand ordering rule, when to split one big einsum into two chained einsums for latency, max sane operand count, slow-path triggers to avoid).

## Required table schema (answers 1, 4, 5)

| aspect | source-level behavior | citation (file::function / issue#) | design implication |

## Demands

- Source citations for every implementation claim; spec citations for semantics; [SPECULATIVE] for anything unrecovered.
- The ordering rule (answer 3) must come with a worked example: 4 operands with given shapes, both orderings, peak intermediate bytes each.
- ≤15-line pseudocode max, only for the decomposition loop.

## Final verification checklist

- [ ] Decomposition order stated from source, not assumed.
- [ ] Operand-count limit answered definitively (number, or "none found in source/spec").
- [ ] Intermediates confirmed as profiler-invisible temporaries (or refuted!) with citation.
- [ ] Latency: fast-GEMM vs fallback trigger conditions enumerated.
- [ ] Verdict gives the ordering rule + the split-threshold heuristic.
