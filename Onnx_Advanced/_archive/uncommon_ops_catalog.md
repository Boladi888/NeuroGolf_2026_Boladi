# uncommon_ops_catalog.md — quickref of underused ONNX ops

One-line summaries of ops covered in this folder, plus a few more that exist but probably aren't worth a dedicated file. Use this to remember what's available; open the dedicated file when you want to actually use one.

## Tier 1 — high potential for NeuroGolf byte/MAC savings

| Op | Opset | One-line | Why care | Detail file |
|---|---:|---|---|---|
| `Trilu` | 14 | Upper/lower triangular mask from a tensor (or zero data) | **Zero-init triangular masks** — saves up to 900 BOOL bytes per 30×30 triangle | `zero_param_generators.md` |
| `EyeLike` | 9 | Identity matrix-like tensor (with optional k-offset) | **Zero-init diagonal masks** | `zero_param_generators.md` |
| `Range` | 11 | `[start, start+delta, ..., limit)` as a tensor | Replace `np.arange` initializers with 3 scalar inits (24 bytes vs N×bytes) | `zero_param_generators.md` |
| `ConstantOfShape` | 9 | Tensor of given shape filled with one value | Replace full-size scalar-fill inits | `zero_param_generators.md` |
| `Einsum` | 12 | Multi-axis contraction by Einstein notation | Collapse Transpose+MatMul+Reduce chains into 1 node | `einsum.md` |
| `GatherND` | 11 | Gather with multi-axis index tensor | Replace `flatten + Gather + reshape` patterns | `multi_axis_indexing.md` |
| `ScatterND` | 11 (16 for reduction) | Scatter with multi-axis index tensor, optional reduction | Already used (task118); reduction modes underexplored | `multi_axis_indexing.md` |
| `DepthToSpace` | 11 | `[N, C·r², H, W] → [N, C, H·r, W·r]` | Cheap channel→spatial reshape; param-free | `spatial_rearrangement.md` |
| `SpaceToDepth` | 1 | Reverse of DepthToSpace | Compact 30×30 into smaller spatial × more channels | `spatial_rearrangement.md` |
| `Mod` | 10 (int), 13 (float with fmod) | Element-wise modulo | Periodicity patterns; avoid arithmetic chains | `mod_and_bitwise.md` |
| `Resize` | 13 | Resize spatial dims with various interpolation | `mode="nearest"` for cheap stretching; can replace Tile patterns | `spatial_rearrangement.md` |
| `GridSample` | 16 | Coordinate-based remap (sampler) | Constant-grid form can collapse `Transpose+Slice` for fixed-shape dihedral transforms — ~20× cheaper on F32 | `symmetry_and_affine.md` |
| `Slice(steps=[-1])` | 10 (attr), 11 (`INT_MIN` sentinel) | Reverse along one or more axes | Cheapest H/V/180° rotation primitive; single node, no intermediate | `symmetry_and_affine.md` |
| `Transpose(perm=...)` | 1 | Swap axes via attribute (not counted as param) | Zero-cost main-diagonal reflection; foundation of 90°/270° rotations | `symmetry_and_affine.md` |

## Tier 2 — situational

| Op | Opset | One-line | Notes |
|---|---:|---|---|
| `GatherElements` | 11 | Per-position gather using an index tensor of same shape | Like advanced fancy indexing; differs from `Gather` and `GatherND` |
| `ScatterElements` | 11 (16 for reduction) | Per-position scatter, optional reduction | task387 used this (see `.update/`) |
| `ReverseSequence` | 10 | Reverse portions of an axis per-batch | Reverse-row / reverse-col patterns |
| `BitShift` | 11 | Left/right bit-shift on integer tensors | Already mentioned for bit-unpacking |
| `BitwiseAnd/Or/Xor/Not` | 18 | Element-wise bitwise on integer tensors | **NEEDS EMPIRICAL** — opset 18 may exceed validator support |
| `Bernoulli` | 15 | Random bool tensor | Probably not useful (non-deterministic) |
| `AffineGrid` | 18 (preview), 20 (stable) | Generate sampling grid from 2×3 affine matrix | **NOT AVAILABLE** under opset ≤18 ceiling — must hand-construct grids for GridSample |
| `CumSum` | 11 (14 for F16) | Cumulative sum along axis | Already documented in `Ort_Advanced/ort_opset_support.md` |
| `LpNormalization` | 1 | L1/L2 normalize along an axis | Rarely useful for grid problems |
| `MaxUnpool` | 9 | Reverse of MaxPool using indices | Niche; probably not useful |
| `If` | 1 | Conditional subgraph | **BANNED** (contains subgraph) |
| `Loop` | 1 | Loop subgraph | **BANNED** |
| `Scan` | 8 | Recurrent scan | **BANNED** |
| `Compress` | 9 | Select elements where mask is True | **BANNED** |
| `NonZero` | 9 | Indices where input ≠ 0 | **BANNED** |
| `Unique` | 11 | Deduplicate values | **BANNED** |

## Tier 3 — exists but probably not useful for NeuroGolf

Signal processing, sequence ops, training-only ops, distributions:
- `DFT`, `STFT`, `BlackmanWindow`, `HammingWindow`, `HannWindow`, `MelWeightMatrix` (signal processing, opset 17+)
- `NegativeLogLikelihoodLoss`, `SoftmaxCrossEntropyLoss` (training)
- `SequenceConstruct`, `SequenceAt`, `SequenceErase`, `SequenceInsert`, `SequenceLength`, `SplitToSequence`, `ConcatFromSequence` (sequences — possibly banned; uses subgraph semantics)
- `OptionalGetElement`, `OptionalHasElement`, `Optional` (opset 15+; almost never useful for static graphs)
- `Multinomial`, `RandomNormal`, `RandomUniform`, `Bernoulli` (random — non-deterministic, breaks `np.array_equal` validation)
- `RoiAlign`, `MaxRoiPool`, `NonMaxSuppression` (object detection)
- `Det` (matrix determinant)
- `LSTM`, `GRU`, `RNN` (recurrent — probably banned via subgraph or too expensive)
- `Resize` with `cubic`/`linear` modes (cubic_coeff_a etc. — overkill for color grids)
- `Hardmax` (one-hot of argmax — `ReduceMax` + `Equal` is usually cheaper)

## Tier 4 — Python library features

| Module | Function | One-line | Detail file |
|---|---|---|---|
| `onnx_graphsurgeon` | `import_onnx`, `Graph`, `Variable`, `Constant`, `Node` | Cleaner alternative to raw protobuf surgery | `graphsurgeon.md` |
| `onnx.compose` | `merge_models`, `merge_graphs`, `add_prefix` | Splice two models together | `onnx_utilities.md` |
| `onnx.utils` | `extract_model` | Carve out a subgraph from input names → output names | `onnx_utilities.md` |
| `onnx.version_converter` | `convert_version` | Upgrade/downgrade a model's opset programmatically | `onnx_utilities.md` |
| `onnx.helper` | `make_attribute`, `make_value_info`, `make_empty_tensor_value_info`, `printable_graph` | Beyond `make_node`/`make_graph`/`make_model` | covered inline in `onnx_utilities.md` |
| `onnx.external_data_helper` | `load_external_data_for_model`, `convert_model_to_external_data` | Large init storage outside the .onnx | Probably not useful — NeuroGolf has 1.44 MiB limit |
