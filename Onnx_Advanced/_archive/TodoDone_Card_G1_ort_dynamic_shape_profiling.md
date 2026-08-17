## Card G1 — ORT profiler vs dynamic shapes (the task014 grader cost gap)

**Researcher contract (read first)**: You are doing INTERNET research only. You have NO access to our repository, code, files, or machine — everything you need is in this prompt; do not ask for files and do not assume any project detail not stated here. Produce ONE copyable Markdown research note as your entire answer (no code execution, no file writes). Follow the required answer order and table schemas exactly, and tag every claim with its source class.

**Deliverable**: a copyable Markdown research note (`ort_dynamic_shape_profiling.md`). Research note only — no implementation.

**Context (project-verified facts, do not re-derive)**: Our scorer replicates the competition grader: load ONNX in onnxruntime CPU EP with `ORT_DISABLE_ALL` graph optimizations, run once on a zeros input `[1,10,30,30]` F32 with profiling enabled, then **cost = memory + params**, where *memory* = sum of bytes of every intermediate tensor as recorded in the **profile trace** (output shapes per node), and *params* = initializer **element count (dtype-blind)**. We shipped a model whose locally-computed static cost was 4,380 but the grader credited about 6,093 (a −0.33 LB gap vs prediction): it used a **data-dependent Slice** (dynamic starts/ends from tensors) producing variable-shaped intermediates. 24 prior fully-static models scored EXACTLY as predicted.

**Research questions (answer in this order)**:
1. In ORT CPU profiling (`enable_profiling`), what shape is recorded for a node whose output is **data-dependent** (dynamic Slice, Gather with runtime indices, ConstantOfShape from a tensor, Resize with runtime sizes, Pad with runtime pads)? The shape OBSERVED at runtime for the specific input, the symbolic/max shape from shape inference, or something else? Cite ORT source (`profiler.cc`, `sequential_executor.cc`, NodeArg shape handling) or documentation.
2. Under `ORT_DISABLE_ALL`, does ORT still run **shape inference** that could materialize `value_info` max-shapes differing from observed shapes? Which takes precedence in the trace?
3. **On a zeros input specifically**: which ops produce DIFFERENT output shapes on zeros vs. real data (e.g., dynamic Slice whose bounds come from reductions of the input — on zeros, bounds collapse or hit clamps)? Enumerate the op patterns where zeros-profiling under- or over-states real shapes.
4. **Recommended constructions** (table): for each "variable-output" need (crop-to-computed-bbox; emit NxM where N,M computed from input), give a construction whose profiled shapes are provably STATIC (e.g., Gather with fixed-length clamped index vectors; fixed-max canvas + masking), with opset requirements and dtype caveats.

**Constraints**: opset <= 20 confirmed working on the grader; banned ops: Loop, Scan, NonZero, Unique, Compress; gather/scatter indices must be >= INT32; no training ops.

**Required table schema** (Q4): | need | construction | ops used | profiled shape behavior | opset | caveats |

**Source policy**: ORT source/docs for runtime behavior; ONNX spec only for op semantics. Mark every inference that is not directly sourced as SPECULATIVE.

**Verification checklist**: each claim tagged [ORT-src], [ORT-doc], [ONNX-spec], or [SPECULATIVE]; Q1 answered with a file/function reference; at least 3 static-construction recipes in Q4.
