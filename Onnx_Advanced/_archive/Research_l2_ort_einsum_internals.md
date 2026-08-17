# ORT 1.24.4 CPU Einsum Internals for Multi-Operand Finals

## Provenance

[PRIMARY] This note is grounded in the ONNX Runtime CPU Einsum implementation at the **GitHub `v1.24.4` tag**: `einsum.cc`, `einsum_compute_preprocessor.h/.cc`, `einsum_typed_compute_processor.h/.cc`, and `einsum_auxiliary_ops.h/.cc`. ONNX semantics come from the ONNX `Einsum` operator spec. I did **not** use repo checkout access, only browsable tagged source pages. citeturn29view0turn41view0turn24view0turn20view0turn17view0

[PROMPT] Your current use case is a **multi-operand, output-node Einsum on CPU EP**, typically `float32`, static shapes, graph optimizations disabled, with repeated appearances of a `[1,10,30,30]` input and small selector initializers. The key question is whether ORT’s internal temporaries and latency behavior make 6–9 operand finals systematically safe or risky.

## Decomposition strategy and operand limits

[PRIMARY] `einsum.cc::Einsum::Compute` gathers all inputs, acquires a **temporary-space allocator** via `OpKernelContext::GetTempSpaceAllocator`, and dispatches to `Einsum::DeviceCompute`. `DeviceCompute` constructs `EinsumComputePreprocessor`, runs it, then creates a typed processor and calls `EinsumTypedComputeProcessor<T>::Run`. The CPU kernel registration in `einsum.cc` is opset 12 and is specialized at runtime by dtype. citeturn29view0turn29view1turn30view0

[PRIMARY] `einsum_compute_preprocessor.cc::EinsumComputePreprocessor::Run` performs a fixed pipeline: `ProcessSubscripts()`, `PostProcessBroadcastedDims()`, `ParseOrCreateOutputSubscript()`, `CalculateOutputShape()`, then `PreprocessInputs()`. This stage computes global label metadata, ellipsis handling, output shape, and any per-input diagonal/transpose preprocessing before the actual contractions begin. citeturn41view0turn41view1

[PRIMARY] The actual multi-input execution order is a **strict left fold** in `einsum_typed_compute_processor.cc::EinsumTypedComputeProcessor<T>::Run`: it may first reduce dimensions that appear only in input 0 and not in the output, then it iterates `for (int input = 1; input < num_inputs; ++input)` and calls `PairwiseOperandProcess` once per next operand. There is **no contraction-order search** or reassociation in this loop. citeturn11view5turn12view3turn13view0

[PRIMARY] The source itself explicitly says ORT **could** rewrite the equation to lower intermediate-array cost using something like `numpy.einsum_path`, but does not do so: `einsum_compute_preprocessor.h::EinsumEquationPreprocessor` comments that such optimizations are “very advanced” and “not required for the average use-case.” That is the source-level answer to whether ORT reorders/optimizes contraction order: **it does not**. citeturn28view1turn40view0

[PRIMARY] Within each pairwise step, `EinsumTypedComputeProcessor<T>::PairwiseOperandProcess` uses **Transpose**, **ReduceSum**, and **MatMul** “under the hood,” exactly as declared in the class comment in `einsum_typed_compute_processor.h`. The pairwise contraction is normalized into a MatMul by permuting the left operand into `[lro, lo, reduce, ro]`, the right into `[lro, reduce, ro, lo]`, and then calling the helper `EinsumOp::MatMul<T>(..., {lro_size, lo_size, reduced_size}, ..., {lro_size, reduced_size, ro_size}, ...)`. citeturn9view8turn12view0turn13view1

[PRIMARY] ONNX spec allows **between 1 and 2,147,483,647 inputs** for `Einsum`. In the ORT CPU source I found **no additional hard cap on input count**: `einsum.cc::Einsum::Compute` only rejects `num_inputs == 0`, then loops over all inputs, and `einsum_compute_preprocessor.cc::ProcessSubscripts` only checks that the number of comma-separated input subscripts matches the number of tensors. citeturn17view0turn29view0turn41view0

[PRIMARY] There **is** a hard cap on the explicit label alphabet in ORT source: `einsum_compute_preprocessor.h` defines `EinsumOp::num_of_letters = 52` and `LetterToIndex` accepts only `a-z` and `A-Z`. Ellipsis dimensions are then added as extra “virtual” subscript indices during `PostProcessBroadcastedDims`, so the effective rank can exceed 52 only via ellipsis-expansion, not via more unique explicit letters. citeturn40view0turn15view4

[DERIVED] So the definitive operand-count answer is: **no Einsum-specific hard cap on input count was found in ORT CPU or in the ONNX spec beyond the variadic-input schema itself**; the real practical limits are memory, latency, the 52-letter explicit alphabet, and the fact that ORT always evaluates a left-deep pairwise chain. citeturn17view0turn29view0turn41view0turn40view0

| aspect | source-level behavior | citation (file::function / issue#) | design implication |
|---|---|---|---|
| decomposition order | [PRIMARY] Inputs are processed as a **left-to-right pairwise chain** after preprocessing; `Run()` loops from input 1 to `num_inputs-1` and calls `PairwiseOperandProcess` each time. | `einsum_typed_compute_processor.cc::EinsumTypedComputeProcessor<T>::Run` citeturn11view5turn13view0 | Input order is the contraction order. If you change operand order in the graph, you change the internal path. |
| order optimization | [PRIMARY] ORT source comments that it *could* rewrite using `numpy.einsum_path`, but does not. | `einsum_compute_preprocessor.h::EinsumEquationPreprocessor` citeturn28view1turn40view0 | There is no internal path optimizer to rescue a bad operand order. |
| pairwise implementation | [PRIMARY] Pairwise steps use `Transpose`, `ReduceSum`, and `MatMul`; the contraction is normalized to a MatMul call on collapsed 3D overrides. | `einsum_typed_compute_processor.h::PairwiseOperandProcess`; `einsum_typed_compute_processor.cc::PairwiseOperandProcess` citeturn9view8turn12view0turn13view1 | Latency comes from MatMul plus any needed transpose/diagonal/reduce glue. |
| input-count limit | [PRIMARY] ONNX spec allows 1..2,147,483,647 inputs; ORT CPU source adds no tighter cap besides rejecting zero inputs. | ONNX `Einsum` spec; `einsum.cc::Einsum::Compute` citeturn17view0turn29view0 | 6–9 inputs are legal. Operand count alone is not the blocker; path width is. |
| explicit label limit | [PRIMARY] ORT supports 52 explicit letters (`a-zA-Z`), then ellipsis dims are turned into virtual outer labels. | `einsum_compute_preprocessor.h::LetterToIndex`; `einsum_compute_preprocessor.cc::PostProcessBroadcastedDims` citeturn40view0turn15view4 | No many-operand cap, but there is a unique-label vocabulary cap. |

[DERIVED] Simplified pseudocode for the actual decomposition loop, faithful to `EinsumTypedComputeProcessor<T>::Run` and `PairwiseOperandProcess`, is:

```text
preprocess equation and inputs
result = maybe_reduce_dims_last_seen_in_input0(input0)

for next_input in inputs[1:]:
    reduce_dims = labels whose last occurrence is next_input and not in output
    result = pairwise(result_or_input0, next_input, reduce_dims)

pairwise(left, right, reduce_dims):
    reduce one-sided reduce_dims immediately with ReduceSum
    permute/reshape left  -> [lro, lo, reduce, ro]
    permute/reshape right -> [lro, reduce, ro, lo]
    output = MatMul([B,M,K], [B,K,N])
    reshape output -> [lro, lo, 1(reduced), ro]
    transpose/reshape to fixed order or final output order
```

## Intermediate allocations and ordering rule

[PRIMARY] The source is unambiguous that Einsum’s explicit intermediates are **temporary tensors**, not the op output itself. `einsum.cc::Einsum::Compute` acquires a temp allocator; then `einsum_auxiliary_ops.cc::Transpose` and `MatMul` both allocate `std::unique_ptr<Tensor>` with that allocator; `ReduceSum` returns a fresh tensor as well. By contrast, the true node output is allocated via `context_->Output(0, output_dims)` only in `FinalizeOutput`. citeturn29view0turn19view2turn20view0turn11view6

[DERIVED] Source therefore confirms the key memory fact you care about: **Einsum’s pairwise internals are ORT temp-space allocations, while the final node output is a distinct execution-frame output tensor**. I did **not** audit ORT’s profiler pipeline here, so the statement “profiler-visible node output vs profiler-invisible temporary” is safest as a **derived** conclusion from the allocation sites rather than a direct claim about profiler code. citeturn29view0turn19view2turn20view0turn11view6

[PRIMARY] `PreprocessInputs()` can also materialize **persistent preprocessed copies** of original operands for this node execution: diagonals are parsed for repeated input labels, and inputs may be transposed into a homogenized common axis order. The processor later reuses `preprocessed_inputs[input]` when available. citeturn15view0turn15view2turn15view3turn13view0

[PRIMARY] The pairwise output shape is explicitly constructed in `PairwiseOperandProcess` as `[lro dims, lo dims, 1 for each reduced dim, ro dims]`, where: `lro` = labels present nontrivially in both operands and **not** reduced at this step, `lo` = left-only survivor labels, `ro` = right-only survivor labels, and reduced labels are kept as size-1 placeholders until finalization. A non-final pair is then permuted back into the fixed global homogenized order for the next step; a final pair is passed to `FinalizeOutput`. citeturn24view0turn11view4turn12view1turn13view1

[PRIMARY] Reduce-on-last-use is decided from `mapped_indices_to_last_input_index`: before the loop, input-0-only dead labels are reduced; during the loop, `reduced_dims` is exactly the set of labels whose `last_input_index == current input` and that are not in the output. If a reduced label is nontrivial on only one side, the code immediately applies `ReduceSum` on that one operand before MatMul. citeturn12view3turn13view0turn11view2

[DERIVED] That means the nontrivial element count of the running pairwise result after step *t* is:

\[
|R_t| = \prod_{i \in \text{surviving live labels after step } t} d_i
\]

where “surviving live labels” means labels that have already been introduced by the consumed prefix and are either still needed by some unconsumed input or appear in the final output. Labels reduced at this step remain size-1 placeholders, so they do **not** contribute to bytes. This is a direct consequence of the `[lro, lo, 1(reduced), ro]` output shape construction. citeturn12view1turn13view1turn13view0

[DERIVED] The **ordering rule** follows immediately: because ORT never reorders, the best operand order is **not** “smallest-first” in general. The memory-optimal order is the one that minimizes the **maximum live-label product over prefixes**, i.e. it introduces as few new long-lived labels as possible and closes as many large soon-to-die labels as possible at each step. In practical human terms: **interleave by shared indices and last-use reductions**, not by raw operand byte size. This is exactly the contraction-path problem that ORT explicitly says it does not solve for you. citeturn28view1turn40view0turn13view0turn24view0

[DERIVED] A useful greedy rule for left-deep ORT paths is: **pick the next operand that maximizes “large labels that die now” and minimizes “new survivor labels introduced now.”** The worker heuristic “smallest-first” is only optimal when the smallest operand also happens to have that live-set property; otherwise it can be badly wrong. citeturn13view0turn24view0turn12view1

[DERIVED] A step-level memory upper bound visible in these sources is larger than just `|R_t|`: a bad step can simultaneously hold the previous result, optional reduced copies, optional transposed left/right copies, the MatMul output candidate, and—on the final pair—the real `context_->Output` tensor plus possibly a final transposed candidate. ORT can skip some copy cost when a transpose can be implemented as a reshape, but only when the order of dimensions with value `> 1` does not change. citeturn24view0turn19view2turn20view0turn11view6

[PRIMARY] `IsTransposeReshapeForEinsum` makes that reshape-only case explicit: “As long as the dims with values > 1 stay in the same order, it’s a reshape.” So moving only `1`-valued axes is allocation-free at the tensor-buffer level; moving nontrivial axes requires an actual transposed tensor allocation. citeturn24view0

[DERIVED] Worked example showing why “smallest-first” is not the right rule:

Equation: `ab,bc,cd,da->` with `float32` shapes  
`A[a,b]=[1000,2]`, `B[b,c]=[2,1000]`, `C[c,d]=[1000,2]`, `D[d,a]=[2,1000]`.  
All four inputs are the **same byte size**: `1000*2*4 = 8,000` bytes. The only thing that changes is order.

| ordering | pairwise live labels after each step | peak intermediate elements | peak bytes |
|---|---:|---:|---:|
| `A, B, C, D` | after `A·B`: `a,c` = `[1000,1000]`; after `(... )·C`: `a,d` = `[1000,2]`; then scalar | 1,000,000 | 4,000,000 |
| `B, C, D, A` | after `B·C`: `b,d` = `[2,2]`; after `(... )·D`: `b,a` = `[2,1000]`; then scalar | 2,000 | 8,000 |

[DERIVED] In the bad order, ORT creates a **4 MB** intermediate; in the better order, the biggest pairwise intermediate is **8 KB**. Since all inputs were equally small, “smallest-first” had nothing useful to say; the winning strategy was to **contract along the large `c` and then `a/d` cycles as early as possible**, i.e. an interleave-by-shared-indices order. The same logic is what your worker observed empirically in “smallest-first keeps pairwise internals KB-scale,” but the source-level refinement is: **smallest-first works only insofar as it approximates early closure of large reduced labels**. citeturn13view0turn24view0turn12view1

## Latency model

[PRIMARY] At the source level, there is **no generic naive nested-loop Einsum kernel** in ORT CPU for the pairwise contraction itself. `PairwiseOperandProcess` always normalizes the two-operand contraction to the helper `EinsumOp::MatMul<T>`, and `einsum_auxiliary_ops.cc::DeviceHelpers::CpuDeviceHelpers::MatMul` then loops over batches and calls `math::MatMul<T>(M, N, K, ...)` once per synthetic batch. So the “fast path” is not a fused tensor-contraction kernel; it is a **pairwise transpose/reduce/diagonal-to-MatMul lowering**. citeturn13view1turn19view0turn20view0

[PRIMARY] The collapsed MatMul dimensions are exactly: `B = lro_size`, `M = lo_size`, `K = reduced_size`, `N = ro_size`. The helper enforces 3D overrides, one synthetic batch dimension, matching batch sizes, and standard matrix compatibility `K_left == K_right`. citeturn13view1turn19view1turn20view0

[DERIVED] A practical per-step compute model is therefore:

\[
T_{\text{step}} \approx T_{\text{reduce-left/right}} + T_{\text{transpose-left}} + T_{\text{transpose-right}} + T_{\text{matmul}} + T_{\text{transpose-back/finalize}}
\]

with the dominant FLOP term

\[
\text{work}_{\text{matmul}} \sim B \cdot M \cdot K \cdot N
\]

and tensor-movement terms proportional to the sizes of any reduced or transposed tensors. This is the right latency model for CPU ORT Einsum from the sources you asked for. citeturn11view2turn12view0turn13view1turn19view0turn20view0

[PRIMARY] There are several clear **slow triggers** in source: repeated labels in an input trigger `Diagonal` preprocessing; one-sided dead labels trigger explicit `ReduceSum`; many actual axis permutations trigger full `Transpose`; and each shared live batch product in `lro` becomes another loop iteration around `math::MatMul<T>`. In other words, big `lro_size` means many batched MatMul calls; big `K` means a heavy contraction; and moving nontrivial axes means real data movement. citeturn15view2turn11view2turn12view0turn19view0turn20view0

[PRIMARY] Diagonal handling is especially expensive when the repeated axes are not already in the innermost positions. `einsum_auxiliary_ops.cc::Diagonal` may transpose the tensor so the diagonal dims become innermost, call `DiagonalInnermostDims`, then transpose back. The actual diagonal extraction loop carries a `TODO: Parallelize this operation`, which is a red flag for large repeated-label tensors. citeturn19view4turn19view5turn18view3

[PRIMARY] Final-output permutations can be costly too. `FinalizeOutput` allocates the actual output via `context_->Output`, then, if needed, creates an intermediate transposed candidate and copies it into the true output because the output buffer may be user-provided. A public performance issue, **#5897**, reported equations whose CPU decomposition incurred multiple transposes plus a MatMul and observed the final transpose as a major cost center. citeturn11view6turn12view2turn37view2

[SPECULATIVE] I did **not** audit `math::MatMul<T>` internals in this note, so I am using “fast GEMM path” in the precise ORT-Einsum sense of “successfully lowered to the MatMul helper,” not guaranteeing a particular MLAS/BLAS microkernel downstream. The source you asked for proves the lowering into `math::MatMul<T>`; it does not, by itself, prove which low-level kernel family each dtype hits. citeturn19view0turn20view0

| aspect | source-level behavior | citation (file::function / issue#) | design implication |
|---|---|---|---|
| contraction core | [PRIMARY] Every pairwise contraction ends in a MatMul helper on collapsed 3D overrides `{B,M,K}` and `{B,K,N}`. | `einsum_typed_compute_processor.cc::PairwiseOperandProcess`; `einsum_auxiliary_ops.cc::MatMul` citeturn13view1turn20view0 | There is no “many-operand fused Einsum” kernel on CPU EP; latency is the sum of pairwise lowerings. |
| batch behavior | [PRIMARY] CPU helper loops `for (size_t i = 0; i < num_batches; ++i)` and calls `math::MatMul<T>` per batch. | `einsum_auxiliary_ops.cc::DeviceHelpers::CpuDeviceHelpers::MatMul` citeturn19view0 | Large `lro_size` is not free; it becomes many batched MatMul invocations. |
| reduce-only-on-one-side | [PRIMARY] If a dead label is nontrivial on only one operand, ORT applies `ReduceSum` immediately before MatMul. | `einsum_typed_compute_processor.cc::PairwiseOperandProcess` citeturn11view2 | Broadcast-heavy equations can pay extra reduction cost before the contraction itself. |
| diagonal path | [PRIMARY] Repeated input labels trigger diagonal parsing; non-innermost diagonal axes may force transpose → diagonal extraction → transpose-back. | `einsum_compute_preprocessor.cc::PreprocessInputs`; `einsum_auxiliary_ops.cc::Diagonal` citeturn15view2turn19view4turn19view5 | Repeated-label equations are a slow-path trigger, especially on large tensors. |
| transpose elision | [PRIMARY] If nontrivial (`>1`) dims keep the same relative order, transpose becomes reshape-only. | `einsum_typed_compute_processor.cc::IsTransposeReshapeForEinsum` citeturn24view0 | Moving only unit dims is cheap; moving nontrivial dims is expensive. |
| final output permutation | [PRIMARY] Final transpose may allocate a candidate, then copy into the true output; public issue #5897 reported transpose-heavy CPU decompositions as slow. | `einsum_typed_compute_processor.cc::FinalizeOutput`; issue #5897 citeturn12view2turn37view2 | Avoid explicit output permutations that force a large final transpose. |

[DERIVED] The latency rule of thumb for your use case is: **the worst single pairwise step dominates**, and that worst step is governed by `(B*M*K*N)` plus whether ORT had to do large real transposes or diagonal extraction around it. For repeated `[1,10,30,30]`-style operands and small selectors, the winning path is the one that keeps `B`, `M`, and `N` small until the very end and uses selectors to kill labels early rather than merely introduce new survivor labels. citeturn13view1turn19view0turn24view0

## Dtype and subscript support

[PRIMARY] ONNX spec allows all numerical tensor types for `Einsum`, including `float`, `double`, `float16`, signed and unsigned integer widths up to 64-bit. ORT CPU `einsum.cc`, however, registers only `float`, `double`, `int32_t`, and `int64_t` for the CPU kernel. So the ONNX-spec answer and the ORT-CPU answer are different. citeturn17view0turn29view0turn30view0

[PRIMARY] Repeated labels in an **input** are allowed and are handled by diagonal extraction during `PreprocessInputs()`. Repeated labels in the **output** are explicitly rejected: `CalculateOutputShape()` returns `"Output subscript contains repeated letters"`. citeturn15view2turn16view0

[PRIMARY] Ellipsis is supported, but ORT enforces the ONNX rule that ellipsis must represent a **fixed number of dimensions across all inputs**. In explicit mode, if any input uses ellipsis, the output subscript must contain an ellipsis as well. In implicit mode, ORT constructs the output with ellipsis first, then the sorted once-seen labels—matching the ONNX spec statement that implicit-mode ellipsis dimensions go at the beginning of the output. citeturn14view8turn15view5turn15view6turn17view0

[PRIMARY] Scalars and empty-output scalar reductions are source-recognized cases: `IsTransposeRequired` returns false for rank-0 tensors, and `einsum_compute_preprocessor.h` documents that an empty output subscript means a scalar result. citeturn19view3turn40view0

[DERIVED] There is one subtle implementation detail around repeated-label diagonals: `einsum_auxiliary_ops.cc::Diagonal` dispatches by **element size** with only 4-byte and 8-byte cases. Under the CPU kernel’s actual registered dtypes, that covers `float`/`int32_t` and `double`/`int64_t`; it is one more reason not to infer broader dtype support from the presence of extra template instantiations elsewhere. citeturn19view4turn29view0turn30view0

| aspect | source-level behavior | citation (file::function / issue#) | design implication |
|---|---|---|---|
| CPU dtypes | [PRIMARY] CPU kernel registers only `float`, `double`, `int32_t`, `int64_t`. | `einsum.cc::ONNX_CPU_OPERATOR_KERNEL` and `Einsum::DeviceCompute` citeturn29view0turn30view0 | Your confirmed `float32` path is supported; `float16` is **not** a CPU-kernel dtype here. |
| ONNX spec dtypes | [PRIMARY] ONNX spec allows a broader numerical set, including `float16`, `int8`, `uint*`, etc. | ONNX `Einsum` spec citeturn17view0 | Portability to ORT CPU is narrower than ONNX operator legality. |
| repeated input labels | [PRIMARY] Repeated labels in an input trigger diagonal parsing in preprocessing. | `einsum_compute_preprocessor.cc::PreprocessInputs` citeturn15view2 | Repeated-label equations are supported but can add costly preprocessing. |
| repeated output labels | [PRIMARY] Repeated labels in output are rejected as invalid. | `einsum_compute_preprocessor.cc::CalculateOutputShape` citeturn16view0 | Do not express output diagonals as repeated RHS labels. |
| ellipsis | [PRIMARY] Ellipsis must denote the same rank-width across inputs; implicit mode places ellipsis dims first in output. | `einsum_compute_preprocessor.cc::ProcessSubscripts`, `ParseOrCreateOutputSubscript`; ONNX spec citeturn14view8turn15view5turn15view6turn17view0 | Ellipsis usually behaves like outer batch structure; keep it stable across inputs. |
| zero-dim / scalar | [PRIMARY] Scalar transpose is a no-op; empty output subscript denotes scalar result. | `einsum_auxiliary_ops.cc::IsTransposeRequired`; `einsum_compute_preprocessor.h` comment citeturn19view3turn40view0 | Scalar traces/reductions are intended to work and avoid needless transpose overhead. |

## Failure modes in the 1.2x line

[SECONDARY] The clearest public bug directly relevant to **output-subscript permutations after reductions** is **issue #18654**, reporting that `ijk->ki` failed with `Einsum op: The candidate output does not match the actual output's shape`. The issue body points specifically at `EinsumTypedComputeProcessor::FinalizeOutput`, which is exactly the part of source that handles output permutation after reduced dimensions have been kept as size-1 placeholders. citeturn37view1turn12view2

[SECONDARY] A long-standing **CPU performance** complaint is **issue #5897**, where a transpose-heavy equation was reported as slower than expected because ORT decomposed it into several transposes plus a MatMul and a final transpose. This issue is highly relevant to your latency model because the current v1.24.4 source still uses that same general lowering structure. citeturn37view2turn9view8turn12view0turn12view2

[SECONDARY] A historical syntax compatibility bug was **issue #4944**, where ORT CPU rejected uppercase labels such as `ZY,YX->ZX`. That specific problem is fixed in modern ORT source: `LetterToIndex` now accepts both lowercase and uppercase letters, and output parsing uses the same mapping. So this is best treated as historical evidence that Einsum edge semantics have had bugs before, not as a current v1.24.4 blocker. citeturn35view0turn40view0

[DERIVED] I did **not** find a public ORT issue specifically about **“many operands” as an input-count hard failure** on CPU EP during this research pass. The source is consistent with that absence: operand count is not specially capped, and the real many-operand risk is the lack of contraction-path optimization, which the source comments themselves acknowledge. citeturn17view0turn29view0turn28view1turn40view0

[DERIVED] I also did **not** find a public 1.2x issue specifically tagged to **repeated input labels/diagonals** on CPU EP. That does **not** mean the path is cheap: source shows repeated labels invoke diagonal preprocessing, and the diagonal helper contains a serial extraction loop with a `TODO: Parallelize this operation`. So the practical risk for repeated labels is more “slow path” than “known public correctness bug,” based on what I recovered here. citeturn15view2turn19view4turn18view3

[DERIVED] For **large rank**, the source comments that several vectors merely `reserve(10)` because the authors did not “expect” rank/output rank above 10, but those are comments about reservation strategy, not enforced limits. I found no hard rank cutoff in the CPU Einsum source; the meaningful limits remain label vocabulary, memory, and latency. citeturn14view0turn16view2turn40view0

## Overall verdict

[DERIVED] For CPU EP `Einsum` finals in ORT 1.24.4, the design rules are: **treat operand order as the contraction path**, because ORT runs a strict left-deep pairwise chain and does not optimize it; **order operands to kill large non-output labels as early as possible**, not merely to place the smallest tensor first; **avoid repeated-label diagonals and large explicit output permutations on big live tensors**, because those trigger extra diagonal/transpose work around the MatMul core; and **regard 6–9 operands as sane only when the best left-deep order keeps every prefix live-set small**, since there is no hard operand cap but there is linear step growth and potentially explosive intermediate width. For latency, the critical quantity per step is the worst `B*M*K*N` together with any real transposes or diagonal extraction; for graph design, keep one big Einsum if a good left-deep input order already exists, and only split into two chained Einsums when a **different tree shape** or isolated subexpression shrinks worst-step flops or peak temporary size by a **large margin**—roughly an order of magnitude is a reasonable threshold, but that specific numeric threshold is a heuristic rather than a source-stated rule. Because your scorer counts node outputs while ORT’s internal pairwise temporaries come from temp-space allocations, a split buys latency only when the new visible intermediate is worth the scoring cost; otherwise, stay single-node and fix the operand order. citeturn13view0turn28view1turn40view0turn19view0turn19view2turn20view0turn11view6turn37view2

[PRIMARY] Verification checklist:

- [x] Decomposition order stated from source, not assumed. citeturn11view5turn13view0
- [x] Operand-count limit answered definitively: **no extra ORT CPU hard cap found; ONNX spec says 1..2,147,483,647 inputs**. citeturn17view0turn29view0
- [x] Intermediates confirmed as temp-space allocations rather than real node outputs; profiler invisibility stated as a derived conclusion from allocation sites. citeturn29view0turn19view2turn20view0turn11view6
- [x] Latency fast-path vs slow-path triggers enumerated from source: MatMul lowering, diagonal preprocessing, one-sided reductions, and transpose/reshape behavior. citeturn13view1turn19view0turn19view4turn24view0
- [x] Verdict includes the ordering rule and a split-threshold heuristic. citeturn28view1turn13view0turn19view0