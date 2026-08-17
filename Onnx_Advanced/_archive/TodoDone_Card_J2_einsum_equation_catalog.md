# Card J2 — Two-operand Einsum equation catalog for grid puzzles (diagonals via repeated subscripts, traces, banded/outer forms)

**Researcher contract (read first)**: You are doing INTERNET research only. You have NO access to our repository, code, files, or machine — everything you need is in this prompt; do not ask for files and do not assume any project detail not stated here. Produce ONE copyable Markdown research note as your entire answer (no code execution, no file writes). Follow the required answer order and table schemas exactly. Tag every claim `[PROMPT]`, `[SPEC]`, `[ORT-DOC]`, `[ORT-SRC]`, `[DERIVED]`, or `[SPECULATIVE]`. Cite human-readably. End with a one-paragraph blunt verdict.

**Deliverable**: a research note (`einsum_equation_catalog.md`).

**Context (project facts — all [PROMPT])**: ONNX graphs on onnxruntime CPU, `ORT_DISABLE_ALL`, opset ≤ 20 grader-verified; cost = trace-visible intermediate bytes + initializer element count; graph input (F32 [1,10,30,30] one-hot) and outputs free. We ship 1–2-operand `Einsum` heavily and they land grader-exact (`bchw,c->bh`, `bchw,h,w->bc` *as a 3-operand it executes pairwise — prior research*, `bkrc,k->br`); ORT CPU Einsum is confirmed for 1–2 operands as fused (Transpose/ReduceSum/MatMul internally; ≥3 operands decompose pairwise with partial tensors). ORT CPU Einsum dtypes: double/float/int32/int64. What we have NOT mapped: the full space of useful **single- and two-operand equations** — especially **repeated-subscript forms** (diagonal semantics).

**Research questions (answer in this order)**:
1. **Repeated-subscript legality**: does ONNX `Einsum` (opset 12) spec support REPEATED subscripts on ONE operand (numpy-style diagonal extraction: `ii->i`, `bcii->bci`, `bchh->bch`)? Cite the spec text. Then: does ORT CPU's implementation support it (the EinsumTypedComputeProcessor source — diagonal handling exists? cite file::function)? If yes, what does `bchh->bch` cost in trace bytes (output [1,10,30] = 1,200 B F32, zero params, ONE op) vs our current EyeLike+Mul+Reduce route?
2. **Diagonal/trace family**: assuming Q1 legal — catalog: main-diagonal extract, anti-diagonal (needs a flip first — total cost with H1's Slice-negstep?), trace (`bcii->bc`), diagonal MASK construction (can einsum WRITE a diagonal — `i,i->ii`-style outer with a ones vector?), banded sums (k-offset diagonals via padded/shifted operand + diagonal extract).
3. **Two-operand workhorse catalog** (table): for each grid need — per-row/col color marginals, per-color counts, weighted coordinate sums (centroids), masked selections (`bchw,bhw->bc...`), outer products (row-mask ⊗ col-mask), correlation against a small template, axis-broadcast multiply-reduce — give the equation, output shape/bytes at 30×30×10, params, and any ORT CPU caveat. Mark which we can compute DIRECTLY from the free input (zero upstream intermediates).
4. **Integer Einsum**: ORT CPU supports int32/int64 Einsum — when is integer-exact Einsum preferable to F32 (counts >2^24 never occur at 900 cells, so F32 exact — confirm this reasoning; when do int64 outputs cost 8 B/elem unnecessarily)?
5. **Failure modes**: equations that ORT CPU rejects or silently decomposes expensively (cite source/issues); the diagonal-subscript edge cases (repeated subscript on BOTH operands? output-repeated subscripts?).
6. **Blunt verdict**: the 5–8 equations worth adopting immediately, and whether the repeated-subscript diagonal is real (it would obsolete EyeLike/Trilu routes for diagonal READS).

**Required table schema (Q3)**: | need | equation | output shape | trace bytes | params | caveats |

**Source policy**: ONNX Einsum spec + numpy einsum semantics for equation meaning; ORT source (`einsum*.cc/h`) for CPU behavior; [SPECULATIVE] tags where ORT behavior is inferred.

**Verification checklist**: Q1 answered with spec text AND ORT source reference; Q3 covers ≥8 needs; Q4 confirms the 2^24 reasoning explicitly; blunt verdict names the adopt-list.
