# Compiler TODOs

Goal: make imported best-submission ONNX graphs progressively editable and recompilable in NeuroGolf Lab.

## Phase 1: Visual Import — DONE (2026-06-04)

- [x] Load best ONNX files as GUI nodes and edges (now sourced from the
      project's `submissions/submission.zip` via `/api/best-graph`).
- [x] Add every op type found in the best submission to the node palette
      (`/best/manifest.json` is built from the live zip).
- [x] Preserve ONNX attributes and initializer summaries for inspection.
- [x] Make unsupported compiler status explicit when raw ONNX graphs are loaded.

## Phase 2: Low-Risk Compiler Ops

Add direct compiler support for common ONNX ops whose output shapes are straightforward:

- `Or`, `Xor`
- `ReduceMax`, `ReduceMin`
- `Squeeze`, `Unsqueeze`
- `Reshape`, `Flatten`
- `Min`, `Max`, `Sum`
- `Relu`, `Abs`, `Neg`, `Floor`, `Clip`, `Sign`, `Sqrt`, `Mod`
- `GreaterOrEqual`, `LessOrEqual`

## Phase 3: Indexing And Tensor Construction

Add compiler support for ops that need careful static-shape and initializer handling:

- `Gather`, `GatherElements`, `GatherND`
- `ScatterElements`
- `OneHot`
- `Expand`
- `Split`
- `CumSum`
- `TopK`

## Phase 4: Spatial And Neural Ops

Add support where ONNX Runtime can run the op but the compiler needs more shape inference:

- `MaxPool`, `AveragePool`
- `ConvTranspose`
- `MatMul`, `Gemm`, `QLinearMatMul`
- `GridSample`

## Phase 5: Representation Bridge — DONE (2026-06-04)

The GUI compiler edits scalar color tensors `[1,1,30,30]`; the competition
encoding is one-hot `[1,10,30,30]`.

- [x] Automatic scalar->one-hot bridge at export (`compile_competition_model`):
      ArgMax decode in, palette-Equal encode out, input-region padding mask.
- [x] `representation: "onehot"` graph mode for building directly on
      `[1,10,30,30]` (required for resizing tasks).
- [x] Validation runs the official semantics in both modes; `/api/run`
      previews both (one-hot outputs decoded via convert_from_numpy rules).
- [ ] Explicit bridge nodes in the palette (cheaper hand-tuned variants of the
      automatic bridge, e.g. maskless or Conv-based encodes).
