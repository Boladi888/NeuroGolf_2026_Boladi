# op_cost_patterns.md — priced op patterns + opset-boundary insurance (Phase-2, 2026-06-05)

> Salvaged A/B content from the deleted op-reference shelf (originals in `_archive/`). ZERO spec
> reprints — semantics questions go to the ONNX docs; dtype questions go to `ort_reality.md`.

## Einsum (from einsum.md)
5. **F16 Einsum**: ORT MatMul supports F16; Einsum should inherit, but verify.
## Empirically verified (2026-05-30, task191, ORT 1.24.x under ORT_DISABLE_ALL)
1. **ORT supports `Einsum` at opset 13** including the 3-operand chain `"ij,jk,kl->il"` with F16 operands. Loads and runs.
2. **⚠️ F16 multi-operand Einsum is NOT byte-equal to the chained MatMul it replaces.**
   `Einsum("ij,jk,kl->il", A, B, C)` ≠ `MatMul(MatMul(A,B), C)` in F16:
   the chained form rounds the intermediate `A@B` to F16 *before* the second matmul,
   Measured: `np.array_equal == False`, max abs diff ≈ 0.042 on random [5,30]·[30,30]·[30,5].
   **Consequence for Path A (byte-equal) work**: do NOT use Einsum to "collapse" an existing
   F16 `MatMul→MatMul` chain — it silently changes values and can flip downstream
   `ArgMax`/threshold decisions on hidden inputs. Einsum collapse is a Path-B
   (rule-rebuild) tool, not a byte-equal graph-surgery tool. The task264 Einsum wins
   byte-equal substitutions — and task264 has a Kaggle-public-negative history, so that

## onnx utilities salvage
- Functions/FunctionProto are validator-BANNED (canonical: core.md).
- value_info COUNTS toward memory (canonical: core.md / scoring_internals).
- ## `onnx.version_converter` — upgrade/downgrade opset
- from onnx import version_converter
- upgraded = version_converter.convert_version(model, target_version=13)

## spatial rearrangement salvage (see techniques #44 for the canonical catalog)
- - Init cost: ~32 bytes for scales (4 F32) or sizes (4 INT64).
- - Init cost: 8 × batch_dim bytes for sequence_lens (INT64).
- - `exemplars/09_periodicity_detection/task110_periodicity.md` — existing periodicity exemplar; check if Resize or SpaceToDepth would simplify it.

## mod_and_bitwise.md salvage
- ## Mod **[opset 10 (integer), opset 13 added F32/F64 with `fmod=1`]**
- For period-3 col index from position: `pos % 3`. Stored as 30 INT64 = 240 bytes via init OR 1 Mod node + scalar `3` init = 9 bytes.
- **Savings: ~230 bytes per period-N indexing pattern.**
- - Init: scalar divisor (1 element of dtype bytes).
- - F32/F64 requires `fmod=1` AND opset 13+.
- bit_i_raw = BitShift(packed_bytes, shift_i, direction="RIGHT")
- ## BitwiseAnd / BitwiseOr / BitwiseXor / BitwiseNot **[opset 18]**
- Cost: 4 nodes + 3 scalar inits (~32 bytes total) for a 30-element periodicity mask. vs storing the explicit mask: 30 BOOL bytes. **Init savings only emerge for multiple mask outputs from the same Mod intermediate.**
- - BitwiseOps at opset 18 may not work in NeuroGolf's scorer environment — verify before committing to a rewrite.
- 1. ORT support for opset 18 BitwiseAnd/Or/Xor/Not under ORT_DISABLE_ALL.

## multi_axis_indexing.md salvage
- ## GatherND **[opset 11; batch_dims attr opset 12+]**
- Cost: indices `[5, 2]` INT64 = 80 bytes. Output `[10, 5]` = 50 F32 = 200 bytes. vs storing a 900-elem mask = 900 bytes.
- - Index tensor cost: P × K × 8 bytes (INT64).
- - `batch_dims` is opset 12+ — bump opset_import if you need it.
- ## ScatterND **[opset 11; reduction attr opset 16+]**
- reduction: "none" (default), "add", "mul" (opset 16+: also "min", "max")
- task118 used bool ScatterND (-38,341 bytes). The reduction modes (especially `"add"`) are underexplored.
- zero_bool = ConstantOfShape([1, 10, 30, 30], value=False)  # ~33 bytes
- true_scalar = ConstantOfShape([N], value=True)  # ~17 bytes
- # Total stored: coords (N×4×8 bytes) + ~50 bytes overhead

## uncommon_ops_catalog.md salvage
- | `Trilu` | 14 | Upper/lower triangular mask from a tensor (or zero data) | **Zero-init triangular masks** — saves up to 900 BOOL bytes per 30×30 triangle | `zero_param_generators.md` |
- | `Range` | 11 | `[start, start+delta, ..., limit)` as a tensor | Replace `np.arange` initializers with 3 scalar inits (24 bytes vs N×bytes) | `zero_param_generators.md` |
- | `ScatterND` | 11 (16 for reduction) | Scatter with multi-axis index tensor, optional reduction | Already used (task118); reduction modes underexplored | `multi_axis_indexing.md` |
- | `ScatterElements` | 11 (16 for reduction) | Per-position scatter, optional reduction | task387 used this (see `.update/`) |
- | `BitwiseAnd/Or/Xor/Not` | 18 | Element-wise bitwise on integer tensors | **NEEDS EMPIRICAL** — opset 18 may exceed validator support |
- - `DFT`, `STFT`, `BlackmanWindow`, `HammingWindow`, `HannWindow`, `MelWeightMatrix` (signal processing, opset 17+)
- - `OptionalGetElement`, `OptionalHasElement`, `Optional` (opset 15+; almost never useful for static graphs)


## Salvage restoration (VERIFY 2026-06-05 P9 — dropped in the Phase-2 merge)
- **And vs BitwiseAnd mnemonic**: logical `And/Or/Xor/Not` = BOOL-only kernels; integer planes need opset-18 `Bitwise*` — mixing them is a load-time reject.
- **ReverseSequence axis trap**: `batch_axis`/`time_axis` must differ and `sequence_lens` length must equal the batch-axis extent — a [1,1,H,W] grid needs reshaping first.
- **Mod sign rule**: integer `Mod` (fmod=0) follows the DIVISOR sign — folds differ from python only when operand signs mix.
- **Expand cannot repeat non-1 dims** (broadcast-only); `Tile` is the repeater.
- **CumSum: no F16, no U8** on ORT CPU — INT32/F32 lanes only.
- **graphsurgeon does NOT enforce the banned-op list** — it happily emits Loop/Scan/NonZero graphs; the validator rejects them at submit. Check core.md bans yourself.
- **DepthToSpace blocksize must divide the axis**: on 30-wide grids legal blocksizes are {1,2,3,5,6,10,15,30}.
- **GatherND/ScatterND win-thresholds (measured)**: sparse reads win under ~50 cells touched; ScatterND assembly wins for N < ~280 writes — above those, dense plane ops are cheaper.
