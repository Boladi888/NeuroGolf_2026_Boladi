## Card G3 — N-ary / fused-op fan-in catalog (beating the plane-conservation law)

**Researcher contract (read first)**: You are doing INTERNET research only. You have NO access to our repository, code, files, or machine — everything you need is in this prompt; do not ask for files and do not assume any project detail not stated here. Produce ONE copyable Markdown research note as your entire answer (no code execution, no file writes). Follow the required answer order and table schemas exactly, and tag every claim with its source class.

**Deliverable**: a copyable Markdown research note (`nary_fanin_catalog.md`). Research note only.

**Context (project-verified)**: Under our cost model (memory = sum of intermediate-tensor bytes in the trace; params = initializer element count, dtype-blind), a per-cell decision core of the form "2 ternary decisions x leg-comparison x parity x 3-state output" was measured to cost ~19 plane materializations under **2-input ops**, and EVERY 2-input reformulation priced within ~150 B of the same total (a "plane-conservation law"). We broke parts of it with 3-operand `Einsum` (equation `bchw,h,w->bc` corner reads with ZERO intermediate planes) and variadic ops. We want the systematic catalog of ops that fold MULTIPLE logical inputs/decisions into ONE materialized output.

**Research questions (in order)**:
1. Catalog every ONNX op (opset <= 20) with **arity > 2 or internal fan-in**: variadic (Sum, Max, Min, Mean, Concat), 3-operand (Where, Clip, Range), N-operand Einsum, GatherElements/GatherND with computed indices, ScatterND with reduction, and ops with implicit reduction fan-in (Conv/MatMul/Einsum contractions, ReduceX, GlobalPool). For each: ORT CPU support + dtype matrix at opset <= 20, and what it can FUSE into one materialization (e.g., Conv = weighted sum over C*k*k inputs; Einsum = product-then-sum across operands).
2. Which boolean/decision structures map to single fused materializations: k-way mux (nested Where vs Gather-on-stacked vs arithmetic select via dot-product with a one-hot), majority/threshold of n masks (Sum+Greater vs Conv), piecewise-constant functions of an integer code (Gather LUT — count its true cost including the INT32 index plane), parity/mod classes (Mod), min/max-of-many (variadic).
3. **Multi-operand Einsum specifics**: ORT CPU's actual support for >= 3-operand Einsum equations — which equation forms execute fused vs decomposed-into-pairwise-temporaries under ORT_DISABLE_ALL (temporaries would appear in the profile trace and destroy the saving). Cite the ORT Einsum implementation behavior.
4. **Recipes** (table): for each decision structure in Q2, the cheapest known ONNX realization with plane count and caveats.

**Constraints**: banned Loop/Scan/NonZero/Unique/Compress; gather indices >= INT32 (an INT32 index plane costs 4 B/elem — count it); zeros-input profiling (no SIGFPE: clamp divisors with Max(x,1)).

**Required table schema** (Q4): | decision structure | realization | materialized planes | params | dtype/opset caveats |

**Source policy**: ONNX spec for arity/semantics; ORT source/docs for fusion-vs-temporaries behavior; SPECULATIVE tags mandatory where ORT internals are inferred.

**Verification checklist**: Q1 table covers >= 15 ops; Q3 cites the ORT Einsum implementation; >= 2 recipes in Q4 that beat a 2-input-op plane count for the same structure.
