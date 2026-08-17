# ort_dynamic_shape_profiling.md

## Executive findings

In ONNX Runtime profiling, the per-node `output_type_shape` field is built from the **actual output tensors that were allocated and produced at execution time** for that specific run, not from `NodeArg` / `value_info` shape metadata. The key source path is `onnxruntime/core/framework/sequential_executor.cc`, where `CalculateTotalOutputSizes(...)` reads each output `OrtValue`, checks that it is a tensor, and serializes `tensor.Shape().ToString()` into `output_type_shape`; `KernelScope::~KernelScope()` then passes that JSON-ish string to the profiler, and `Profiler::EndProfiling()` only serializes the recorded event args. [ORT-src]

`ORT_DISABLE_ALL` does **not** skip ORT’s graph/type/shape resolution pass. ORT docs define `ORT_DISABLE_ALL` as “disables all optimizations,” but `InferenceSession::Initialize()` still calls `graph.Resolve()`, and `Graph::Resolve()` explicitly runs `PerformTypeAndShapeInferencing(...)`. That per-node path invokes the ONNX `TypeAndShapeInferenceFunction`, ONNX partial data propagation, and then updates output `NodeArg` type/shape information. [ORT-doc][ORT-src]

Because ONNX distinguishes **static/inferred tensor shape metadata** from a **runtime tensor shape**, inferred `value_info` / `NodeArg` shapes can exist under `ORT_DISABLE_ALL` and still differ from what the profiler records on a concrete input. In the trace, the runtime-observed tensor shape wins, because the trace is populated from executed `Tensor::Shape()` values, not from `NodeArg::Shape()`. [ORT-src][ONNX-spec]

For zero-input profiling specifically, the only dangerous constructions are the ones where **shape-driving tensors are computed from data values**. `Gather` is the important exception: changing the **values** of `indices` does not change `Gather` output shape; only changing the **shape of the `indices` tensor** does. By contrast, `Slice`, `ConstantOfShape`, `Resize` with `sizes`, and `Pad` with runtime `pads` all can change output extents when their shape-driving inputs change on zeros. [ONNX-spec][SPECULATIVE]

Given the project facts in your prompt, the strongest explanation for the task014 cost gap is that the grader credited the **actual zero-input extents emitted by the dynamic shape subgraph** at runtime, while your local “static cost” used a different extent basis. That interpretation fits both the ORT profiler source and the fact that your 24 fully-static models matched exactly. [ORT-src][SPECULATIVE]

## Profiler trace behavior

The direct answer to your first question is: **the profile trace records the shape observed at runtime for that specific input**, provided the node output is an allocated tensor. In current ORT source, `sequential_executor.cc::CalculateTotalOutputSizes(...)` iterates `OpKernelContextInternal` outputs, requires `p_output->IsTensor()` and `p_output->Get<Tensor>().IsAllocated()`, then serializes `tensor.Shape().ToString()` into `output_type_shape`. `KernelScope::~KernelScope()` records that field in the `NODE_EVENT` args as part of the `_kernel_time` event. [ORT-src]

That means a data-dependent `Slice`, `ConstantOfShape`, `Resize(sizes)`, or `Pad(pads)` contributes the **runtime-produced** output dimensions for the actual zeros run, not a symbolic placeholder. The profiler path does not ask `NodeArg` for a shape at record time. [ORT-src]

The source-level file/function references that answer Q1 are:

- `onnxruntime/core/framework/sequential_executor.cc::CalculateTotalOutputSizes(...)` and `onnxruntime/core/framework/sequential_executor.cc::KernelScope::~KernelScope()`, which build and attach `output_type_shape`. [ORT-src]
- `onnxruntime/core/common/profiler.cc::Profiler::EndTimeAndRecordEvent(...)` and `Profiler::EndProfiling()`, which store and serialize the event args without replacing them with graph metadata. [ORT-src]
- `onnxruntime/core/graph/graph.cc::NodeArg::Shape()`, which shows where graph-level shape metadata lives, separately from runtime `Tensor::Shape()`. [ORT-src]

There are two practical caveats. First, the field is not a pure shape vector; it is a serialized list of `{dtype: [dims...]}` objects, so it is really “runtime tensor type-and-shape” information. Second, non-tensor outputs or missing/unallocated optional outputs are not represented there, because the profiler only records tensor outputs that actually exist as `OrtValue` tensors in that run. [ORT-src]

`Gather` needs one special callout. ONNX defines `Gather` output shape from `data.shape`, `indices.shape`, and `axis`; it does **not** depend on the numeric index values themselves. So “Gather with runtime indices” only becomes shape-dynamic if the **shape of the `indices` tensor** is itself runtime-varying. If `indices` has fixed shape and only its contents change, the profiler still records the same output shape every run. [ONNX-spec]

## Shape inference with ORT_DISABLE_ALL

ORT documents `GraphOptimizationLevel::ORT_DISABLE_ALL` as disabling graph optimizations, but that is not the same thing as skipping graph/type/shape resolution. ORT’s profiling docs also describe `enable_profiling` as producing a JSON trace with detailed operator information, but they do not document `output_type_shape`; for that detail, the runtime source is the authoritative reference. [ORT-doc][ORT-src]

At session initialization, `InferenceSession::Initialize()` still performs graph work even under `ORT_DISABLE_ALL`: it adds predefined transformers for the requested level, calls `TransformGraph(graph, ...)`, and then unconditionally calls `graph.Resolve()`. `Graph::Resolve()` in turn says it performs “type/shape validation and inferencing,” and it calls `PerformTypeAndShapeInferencing(...)`. [ORT-src]

Within that pass, `Graph::InferAndVerifyTypeMatch(...)` constructs an `InferenceContextImpl`, runs the ONNX schema’s `TypeAndShapeInferenceFunction`, runs ONNX partial data propagation, and then updates output `NodeArg` types/shapes. ORT also has code for saving propagated shape values when they can be deduced. That is enough to materialize graph-level output shape metadata even when graph optimizations are disabled. [ORT-src]

ONNX’s own shape-inference documentation makes the separation explicit: a static tensor shape (`TensorShapeProto`) is distinct from a runtime tensor shape, and shape inference is not guaranteed to be complete for dynamic behaviors. It also notes that dynamic, value-driven constructions can block complete inference. [ONNX-spec]

So the precise answer to Q2 is:

- **Yes**, ORT still runs type/shape inference under `ORT_DISABLE_ALL`. [ORT-src][ORT-doc]
- **Yes**, that inference can populate `NodeArg` / `value_info` shapes that differ from observed runtime shapes. Whether those inferred shapes are true maxima, symbolic placeholders, unions, or exact values is operator-specific; calling them “max-shapes” generically would be too strong. [ONNX-spec][SPECULATIVE]
- **In the profiler trace, runtime shape takes precedence**, because the trace is emitted from actual `Tensor::Shape()` values after kernel execution. [ORT-src]

There is one source-level nuance worth noting. ORT’s transformer registration code includes a comment that “free dimension override” is enabled even when the graph optimization level is 0. That nuance does not change the core answer here, but it does mean `ORT_DISABLE_ALL` is not literally “no preprocessing of any kind.” [ORT-src]

## Zero-input shape drift patterns

The zero-input problem is best understood as a **single-sample runtime trace** problem: any subgraph that turns **data values** into **shape-driving tensors** will be profiled on the zeros-specific branch of that logic. [ORT-src][SPECULATIVE]

A common pattern is a **reduction-to-length** or **reduction-to-bound** chain. If you form a mask such as `Greater(X, 0)` or `NotEqual(X, 0)`, cast it to integers, and then `ReduceSum` it to produce a slice extent, a pad amount, or a resize target, an all-zero input tends to drive that count to 0. Feeding that count into `Slice`, `ConstantOfShape`, `Resize(sizes)`, or `Pad(pads)` can therefore collapse an output to a minimum or empty extent. Conversely, if the mask is `Equal(X, 0)` or `GreaterOrEqual(X, 0)`, the all-zero input can drive the count to a maximum extent and **over-state** real-data memory. [ONNX-spec][SPECULATIVE]

Another common pattern is an **ArgMax/ArgMin boundary extractor**. ONNX specifies that `ArgMax` returns the first occurrence of the maximum unless `select_last_index=1`; on a uniform all-zero tensor, ties are everywhere. If crop bounds, bbox edges, or pad margins are derived from `ArgMax`/`ArgMin` over score or mask tensors, the zeros input can systematically pin a boundary to 0 or to the edge, depending on the formulation. That can make a subsequent `Slice` collapse, or make a `Pad`/`Resize` chain saturate to a corner case not representative of real data. [ONNX-spec][SPECULATIVE]

`Slice` has an extra amplifier here because its effective starts/ends are initialized against input dims and then clamped according to the slicing rules. ONNX explicitly describes effective `starts`, `ends`, defaults, step handling, and clamping-to-dimension behavior. So when zeros drive a start/end computation to 0, below 0, or past the dimension limit, the realized slice can become empty, unit-length, or full-length in a way a symbolic estimate would not capture. [ONNX-spec]

`Resize` with runtime `sizes` is similarly direct. ONNX defines the output size from the `sizes` tensor values (or from `scales` when that path is used). If `sizes` is produced from reductions or bbox arithmetic over the data tensor, then the zeros input directly determines the profiled output size for that run. There is no symbolic “max shape” substitution in the profiler path. [ONNX-spec][ORT-src]

`Pad` with runtime `pads` is also value-driven. ONNX defines `pads` as the tensor that tells the operator how many elements to add or remove along each axis. If that tensor is built from zero-sensitive reductions or comparisons, zero-input profiling can under-state or over-state the padded intermediate, depending on whether zeros shrink or enlarge the computed pad vector. [ONNX-spec][SPECULATIVE]

`ConstantOfShape` is the most literal case: ONNX says its output tensor shape is specified by its 1D input tensor. If that input is computed from data values, all-zero inputs can produce a smaller or larger output tensor than real inputs, depending on the upstream formula. [ONNX-spec]

The `Gather` caveat is important enough to restate separately. ONNX defines `Gather` output shape from `indices.shape`, not from the numeric index contents. So zeros profiling only changes `Gather` memory if zeros change the **shape-producing subgraph upstream of `indices`**—for example, if the `indices` tensor itself was produced by a dynamic `Slice`, `Resize`, or `ConstantOfShape`. Runtime index values alone do not create a shape drift. [ONNX-spec]

## Static construction recipes

The opset entries below are **conservative lower bounds**, not claimed minimal lower bounds, unless the row text says otherwise. [SPECULATIVE][ONNX-spec]

| need | construction | ops used | profiled shape behavior | opset | caveats |
|---|---|---|---|---|---|
| crop-to-computed-bbox | Fixed `K_h x K_w` crop using **constant index templates** plus runtime offset and clamp | `Add`, `Clip`, `Gather`, optional mask ops | fixed `K_h x K_w` crop | 12+ | Clamping repeats border rows/cols near edges; if the true bbox is smaller than `K_h x K_w`, emit a validity mask too |
| crop-to-computed-bbox | **Fixed full canvas + mask** instead of a real crop | comparisons, `And`, `Cast`, `Mul`, optional `Where` | fixed input/canvas shape | 13+ | Memory is worst-case static; downstream must respect the mask rather than expecting a compact crop |
| emit `N x M` where `N, M` are computed from input | **Fixed `N_max x M_max` canvas** and separate valid extents or validity mask | constant/initializer canvas, elementwise ops, optional `Where` | fixed `N_max x M_max` | 13+ | You pay max-canvas memory every run; semantics become “ragged data in a fixed box” |
| variable pad/crop margins | **Constant worst-case pad** followed by fixed-size gather/mask, not runtime `pads` or runtime `Slice` extents | constant `Pad`, `Clip`, `Gather`, optional mask ops | fixed padded envelope or fixed recovered crop | 13+ | More memory than exact dynamic padding; choose the worst-case bound carefully |
| variable selection count or variable-length row/col emission | **Fixed-`K` selection** with clamped/padded index tensor and validity mask | `Clip`, `Where`, `Gather` | fixed `K` | 12+ | Duplicate indices or repeated edge indices are expected when the true count is `< K`; consumers must use the mask |

Row one is static because ONNX `Gather` output shape depends on the **shape of the `indices` tensor**, not on the index values, and ONNX `Clip` changes element values without changing tensor shape. Therefore a fixed-length index template, even after runtime offsetting and clamping, yields a fixed output extent. [ONNX-spec][SPECULATIVE]

Rows two and three are static because the tensor extent is chosen up front and variability is expressed as **values** (mask bits, valid extents, or payload values) rather than as shape-driving tensors. ONNX broadcasting rules and elementwise operator semantics keep the broadcasted canvas/input shape fixed. `ConstantOfShape` is safe here **only if its shape input is constant**, not if the shape input is data-derived. [ONNX-spec][SPECULATIVE]

Row four is static because the only `Pad` that affects shape is a **constant worst-case pad**, so the padded envelope is known in advance, and any subsequent fixed-template `Gather` again inherits shape from the template indices rather than their runtime values. [ONNX-spec][SPECULATIVE]

Row five is static for the same reason as row one: once the selection tensor has fixed shape `[K]` (or another fixed template shape), the `Gather` result has fixed extent even if the selected indices are computed from the input and then clamped or padded. [ONNX-spec][SPECULATIVE]

If the goal is specifically to stay compatible with your grader’s “single zeros run + sum of profiled intermediates” behavior, the robust design rule is: **never let data values determine tensor extents**. Let data values determine only element contents, masks, or values written into a preallocated fixed-shape box. [SPECULATIVE][ORT-src][ONNX-spec]

## Open questions and limitations

I did not find a public ONNX Runtime documentation page that specifies `output_type_shape` semantics directly; that part of the answer rests primarily on ORT source, which is the right authority for this question anyway. [ORT-doc][ORT-src]

The ORT source evidence here is from current `main` plus a `rel-1.20.0` check for the key executor/profiler pattern. The behavior appears stable across those source snapshots, but if the competition grader is pinned to a materially older ORT build, the exact file line numbers could differ. [ORT-src][SPECULATIVE]

For the zero-input drift patterns, the exact direction and magnitude of under- or over-statement still depend on the unseen arithmetic that produces your shape-driving tensors. The patterns above are the high-confidence classes implied by ONNX operator semantics, not a formal proof for any repository-specific graph. [ONNX-spec][SPECULATIVE] 