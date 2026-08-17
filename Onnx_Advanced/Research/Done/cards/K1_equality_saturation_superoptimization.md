# Card K1 — Equality-saturation / superoptimization frameworks for tiny ONNX graphs under a memory+params cost

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `k1_superoptimization_survey.md`.

**Researcher contract**: you have internet access but NO access to our repository. Everything you need about our environment is stated in this card. Cite sources human-readably (author-year-title, or repo `file::function` paths). Tag every claim: `[PROMPT]` (fact given here), `[PRIMARY]` (peer-reviewed paper / primary source code), `[SECONDARY]` (docs, blog, expository), `[DERIVED]` (deduction from cited sources + the cost model below), `[SPECULATIVE]` (unproven idea — mark clearly).

## Environment (all facts [PROMPT])

- We compete in a Kaggle contest scoring tiny ONNX networks: per-task `points = max(1.0, 25 − ln(memory + params))`, natural log, summed over 401 independent tasks.
- `params` = total bytes of all initializer payloads in the graph.
- `memory` = Σ over every node-output (intermediate) tensor of `max(static_bytes, runtime_bytes)`, where runtime sizes come from ONNX Runtime profiling on real inputs. Graph optimizations are DISABLED (`ORT_DISABLE_ALL`), CPU EP, onnxruntime **1.24.4**, onnx **1.21.0**, opsets 10–18 in the default domain.
- I/O contract: input `[1,10,30,30]` float32 one-hot color planes (0.0/1.0); output the same shape; correctness = `(output > 0.0)` must equal the expected boolean grid **exactly, bit-for-bit on every test case**. So rewrites must be EXACT on the reachable input set — float-algebra rewrites that are only approximately equal are unusable unless provably exact for our integer-valued tensors.
- Typical graph: 5–80 nodes, ops drawn from: Conv, QLinearConv, MaxPool, Pad, Slice, Concat, Split, Gather, ScatterElements/ScatterND, Reduce\*, ArgMax, TopK, Equal/Greater/Less, And/Or/Not/Xor, Where, Add/Mul/Sub/Div/Mod, Cast, Reshape, Transpose, Tile, Expand, Resize, OneHot, CumSum, Einsum, MatMul, Clip, Relu, Min/Max, Trilu. Dtypes: bool, uint8, int32, int64, float16, float32.
- Validator rejection gates (a model failing any scores zero): banned node types {Loop, Scan, NonZero, Unique, Script, Function, Compress}; any `model.functions` entry; any opset domain other than `""`/`"ai.onnx"`; any node attribute of type GRAPH/GRAPHS (so no If); any symbolic/dynamic tensor dimension; file > 1.44 MB.
- We currently hand-optimize each graph. ~176 of our tasks sit in a 10k–50k cost band; the win condition is grinding many graphs down by 2–10×, so an automated rewriter with our exact cost function would be transformative.

## The question

Can an existing equality-saturation or tensor-graph-superoptimization framework be adapted to minimize `memory + params` (as defined above) over equivalence classes of tiny static ONNX graphs — with EXACT (bit-level or integer-exact) equivalence?

## Required answer order

1. **Framework inventory** — table (schema below) covering at least: TASO (Jia et al. 2019), TENSAT (Yang et al. 2021, egg-based), egg / egglog ecosystems, Glenside, PET (partially-equivalent transformations — note its equivalence is NOT exact; assess accordingly), MLIR-based rewriters relevant to small graphs, Tensor Algebra SuperOptimizer successors 2022–2026, and anything ONNX-native (onnx-rewriter? onnxoptimizer pass framework? onnxscript rewriter API). Include post-2024 work — anything newer than TENSAT in the e-graph-for-tensors line.
2. **Cost-model pluggability** — for each framework: can a user-supplied cost function of the form "Σ over node outputs of byte-size + Σ initializer bytes" be expressed? E-graph extraction with such a cost is a weighted-DAG-extraction problem — discuss exact vs greedy extraction (egg's extractors, ILP extraction).
3. **Rewrite-rule sources** — what rule sets exist at the tensor-op level (TASO's ~130 generated rules; TENSAT's rule set; any ONNX-level rule collections)? Which of our op list are covered vs not (logical ops, Scatter, OneHot, CumSum, Resize are unusual in these frameworks)?
4. **Equivalence guarantees** — per framework: formally verified rules? randomized-testing-validated (TASO's approach)? For our use, classify each framework's rules into (a) exact for integer-valued float tensors, (b) exact only in real arithmetic (reassociation — usable for us ONLY when values are small integers; state float32 integer-exactness bound 2^24 and float16 bound 2048), (c) approximate (unusable). 
5. **Constraint handling** — can banned ops / dtype legality / static-shapes-only be enforced as extraction-time filters in each framework?
6. **Port-effort estimate** — for the single best candidate: concrete integration sketch (ONNX → framework IR → saturate → extract with our cost → ONNX), with the 5 hardest expected obstacles, and an S/M/L/XL effort grade for one experienced engineer.
7. **Prior art on "model golf"** — any published work or competition writeups on minimizing model size/memory as the objective (not latency/FLOPs)?
8. **Overall verdict** — one blunt paragraph: is there a framework worth a 1–2 week port, or is hand-optimization + a homegrown peephole pass the rational choice? No hedging.

## Required table schema (answer 1)

| framework | IR level | rule count & origin | custom cost fn? | extraction method | equivalence guarantee | dtype coverage | last release/commit | port effort | citation |

## Demands

- For the top recommendation: a worked micro-example (in prose or pseudocode, ≤30 lines) showing one nontrivial rewrite our hand process might miss (e.g., Conv+Pad fusion, Concat-of-Slices elimination, Einsum factoring) and how the cost function would score it.
- Counterexample-or-proof: if you claim a rule class is integer-exact, state why (closure of integer-valued tensors under the op, no division/rounding); if approximate, give a one-line counterexample.
- Code snippets only for recommended integration patterns, ≤30 lines each.

## Final verification checklist (researcher must confirm each)

- [ ] Every framework row has a citation a human can locate without clicking a link token.
- [ ] Equivalence classification (exact / integer-exact-only / approximate) is explicit per framework.
- [ ] The cost model used throughout is OURS (memory+params as defined), not FLOPs/latency.
- [ ] Banned-op and static-shape constraints addressed for the top recommendation.
- [ ] Post-2024 developments searched for and either included or explicitly declared absent.
- [ ] Overall verdict is one paragraph and answers "port it or skip it."
