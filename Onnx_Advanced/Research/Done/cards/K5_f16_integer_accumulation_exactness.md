# Card K5 — F16/F32 integer-exactness envelopes in ORT 1.24 CPU kernels (accumulation precision)

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `k5_accumulation_exactness.md`.

**Researcher contract**: internet access, NO repo access; self-contained card. Cite onnxruntime source (kernel registration + the actual compute path, e.g. MLAS function names) for every accumulation claim; tag claims `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Environment (all facts [PROMPT])

- onnxruntime **1.24.4**, CPU EP, `ORT_DISABLE_ALL`, single session, opsets 10–18. Models must be bit-exact after `> 0.0` binarization — one wrong cell on one input = task scores zero.
- We aggressively use **float16 tensors holding small non-negative integers** (counts, codes, indicator sums) because F16 halves memory cost vs F32. The arithmetic stays exact only while every represented value is an exactly-representable integer.
- Representability facts `[PROMPT]`, to build on: F16 represents all integers exactly up to 2048 (then steps of 2 up to 4096, etc.); F32 up to 2^24 = 16,777,216.
- Load-bearing internal claim we need VERIFIED OR REFUTED at the source level: *"ORT F16 Conv accumulates integer partial sums with zero rounding (used for positional-hash Convs with window codes up to 2^11)."* If CPU F16 Conv actually upcasts to F32 accumulation and only rounds the final output, the safe envelope is "final output ≤ 2048". If it truly accumulates in F16, every PARTIAL sum must stay ≤ 2048 (order-dependent!). These imply different design rules — we have shipped models relying on the optimistic reading.

## Required answer order

1. **Conv (float16)** — does ORT 1.24 CPU register a native MLFloat16 Conv kernel, and what is its compute path (MLAS half-precision GEMM? upcast-to-float32 wrapper?). Cite kernel registration + the MLAS routine. State the resulting integer-exactness rule: (a) "exact iff final value ≤ 2048" or (b) "exact iff every partial sum ≤ 2048 under accumulation order O — and state O" or (c) other.
2. **MatMul / Gemm / Einsum (float16)** — same question per op. Note Einsum's decomposition path (does it lower to MatMul?).
3. **ReduceSum / ReduceL1 / CumSum (float16)** — accumulator dtype and ordering; same exactness rule format.
4. **QLinearConv / ConvInteger / MatMulInteger** — accumulator width (int32?), saturation behavior on requantization, exact-count envelope (we use QLinearConv with scale=1, zero_point=0 as an exact ≤255 counter `[PROMPT]` — confirm the int32-accumulate + requantize path makes this exact and state the true bound).
5. **Float32 versions** of 1–3: accumulator dtype (FMA? extended precision?), and whether intra-op parallelism can change summation order run-to-run at `intra_op_num_threads=1` and at default threading. We need run-to-run determinism — state the conditions under which ORT CPU guarantees it (cite docs/issues).
6. **F16 elementwise chain semantics** — for Add/Mul/Sub chains in F16: computed in F16 directly or upcast per-op? (Eigen path vs MLAS path.) Exactness rule for integer chains.
7. **Practical design-rule table** — one row per op family: "to keep F16 exact, ensure ___"; "to keep F32 exact, ensure ___"; max safe count for a k×k window sum at each dtype.
8. **Overall verdict** — one paragraph: is our optimistic F16-Conv claim safe as stated, must it be tightened to partial-sum bounds, or should F16 counting Convs be capped at a different bound than 2048? Be blunt; this gates real submissions.

## Required table schema (answers 1–6)

| op (dtype) | kernel/registration citation | compute path (MLAS fn / Eigen / custom) | accumulator dtype | accumulation order | integer-exactness rule | run-to-run deterministic? |

## Demands

- Source citations to onnxruntime 1.24.x (or nearest browsable 1.2x tag — say which) for every compute-path claim. GitHub issues/PRs that changed F16 paths between 1.20–1.26 are highly relevant — search for them and cite.
- Counterexample construction: for each "exact iff X" rule, give the smallest violating example (values + op) showing where rounding first appears.
- If the answer differs across CPU architectures (e.g., MLAS fp16 kernels only on ARM, x86 falls back to upcast), state both paths — our grader runs x86-64 `[PROMPT]`.
- No code beyond ≤10-line illustrations.

## Final verification checklist

- [ ] The load-bearing Conv-F16 claim has an explicit VERIFIED/REFUTED/REPLACED-BY verdict with source citation.
- [ ] x86-64 path explicitly addressed (not just ARM fp16 kernels).
- [ ] QLinearConv exact-counter bound confirmed with the accumulator/requantization path.
- [ ] Design-rule table present and self-sufficient (a builder could follow it without reading the rest).
- [ ] Determinism conditions stated for default threading AND intra_op=1.
