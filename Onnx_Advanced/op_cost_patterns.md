# op_cost_patterns.md — priced op patterns + opset-boundary insurance (Phase-2, 2026-06-05)

> Salvaged A/B content from the deleted op-reference shelf (originals in `_archive/`). ZERO spec
> reprints — semantics questions go to the ONNX docs; dtype questions go to `ort_reality.md`.

## §BitUnpackLane — Bit-unpack U8-lane dtype narrowing (task077, 2026-06-24; techniques #157)

When #45/#47 packed-row graphs spatialize via `BitwiseAnd(word[1,R,1], pow_mask[1,1,W])`, the AND output inherits the packed dtype (U32 = 4 B/cell). Split each packed word into `ceil(W/8)` **U8 lanes** before expanding:

```
lane_k = Cast(BitShift(word[1,R,1], shift=8*k, direction=RIGHT), UINT8)   # [1,R,1]
subplane_k = Greater(BitwiseAnd(lane_k, pow8[1,1,8]), zero)                # [1,R,8] BOOL
bool_plane = Concat(subplane_0, subplane_1, ..., axis=-1)                  # [1,R,W] BOOL
```

**Lane cost law:** `ceil(W/8)` U8 planes [R,1] + `ceil(W/8)` BOOL planes [R,8] + Concat — always cheaper than `R×W×B_word`.
- task077 W=21 → 3 lanes: 1,680 B U32 AND → ≈1,043 B total (3×R U8 + 3×8R BOOL + 420 B Concat) → **net -637 B; graph 7,161→6,524 (-8.9%).**
- General: saves `(B_word - 1) * R * W` bytes on the per-cell plane.

**ORT ops needed:** `BitShift` (UINT8/UINT32/UINT64 ✓), `Cast` (U32→U8 ✓), `BitwiseAnd` (UINT8 ✓ at opset 18). All confirmed working at opset 18, ORT 1.24.4.

**Re-check task338:** ort_reality.md §38 states "per-cell U32 unpack DEAD for task338 [25,25]". With U8 lanes (W=25 → 4 lanes), the exit plane drops 4×: verify before permanently closing that avenue.

## §LocalTopology2x2 — exact local topology reductions (O7, 2026-07-03)

For a binary mask, add a one-cell zero border, then `Conv` with `Kcode=[[8,4],[2,1]]` to encode each valid `2x2` window as an integer code `0..15` over the `31x31` window grid. `Equal(code,k) -> Cast -> ReduceSum` gives exact local pattern counts. If the mask starts as BOOL, cast before `Conv`; if it is a fixed F32 one-hot channel/union, keep the F32 binary mask.

Euler is exact only under a foreground/background dual-connectivity convention. Let `n1 = #(0001,0010,0100,1000)`, `n3 = #(0111,1011,1101,1110)`, and `nd = #(0110,1001)`. Then:
- `4*chi_fg4_bg8 = n1 - n3 + 2*nd`
- `4*chi_fg8_bg4 = n1 - n3 - 2*nd`

The numerator is bounded by the `2x2` window count, so F16 integer exactness is safe for the Euler accumulator on a `30x30` grid; use INT32/F32 when using area-derived formulas such as `4F - 2H - 2V` that can exceed 2048 before the final comparison. Perimeter is also local: `P = n(Q1)+n(Q2)+2*n(QD)+n(Q3)`. Convex/concave corner counts are `C+ = n(Q1)`, `C- = n(Q3)`, with the diagonal ambiguity handled by the Euler formulas above.

Hole count needs a component count: for a known single connected component, `holes = 1 - chi`; otherwise unrestricted components/reachability/largest-component aggregation are non-local and should use the existing flood/labeling family, not this local-reduction lane.

## §WindowMulticolorSumTest — `sum != max*count` beats complement-MinPool pairs (task158, 2026-07-10)

To test "at least two distinct non-zero values in a KxK window" on a small U8
value plane, do not default to the complement-MinPool trick. If
`max_value * window_area <= 255`, use:

```
window_sum = QLinearConv(Z, ones_KxK)
window_max = MaxPool(Z)
window_cnt = QLinearConv(Min(Z, 1), ones_KxK)
same_value = Equal(window_sum, Mul(window_max, window_cnt))
```

`same_value` is true exactly when every non-zero value equals the max; empty
windows also give `0 == 0`. So `Not(same_value)` is the multicolor predicate.
task158 replaced six complement/MaxPool support planes with this three-plane
test and saved 1,950 B, funding an OOB safety clamp while staying byte-equal.
Re-check the live graph opset: U8/INT8 `Mul` is legal at opset 14+.

## §FloodReachSafety — alternating single-axis flood needs 2K rounds (task243, 2026-07-12)

For gated dilation/flood tasks, a combined 4-connected round reaches path
distance K after K rounds. A cheaper single-axis alternating schedule (VHVH...)
advances a winding one-cell corridor only on matching-axis rounds, so N
alternating rounds guarantee only path-distance `floor(N/2)`. To be a superset
of a K-round combined LIVE flood, require `N >= 2K` and at least K rounds per
axis.

Visible, random truth, and public probes may miss the failure. Use a gated-comb
adversarial: vertical teeth joined by alternating top/bottom one-cell connectors
force a long serpentine path inside a small box. task243's max45 variant passed
normal gates but hidden-failed; the comb reproduced the under-reach, while the
64-round version cleared it.

## Einsum salvage (full original: `_archive/einsum.md`; live coverage: techniques #42/#46/#89)
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

## mod_and_bitwise salvage (full original: `_archive/mod_and_bitwise.md`)
- ## Mod **[opset 10 (integer), opset 13 added F32/F64 with `fmod=1`]**
- For period-3 col index from position: `pos % 3`. Stored as 30 INT64 = 240 bytes via init OR 1 Mod node + scalar `3` init = 9 bytes.
- **Savings: ~230 bytes per period-N indexing pattern.**
- - Init: scalar divisor (1 element of dtype bytes).
- - F32/F64 requires `fmod=1` AND opset 13+.
- bit_i_raw = BitShift(packed_bytes, shift_i, direction="RIGHT")
- ## BitwiseAnd / BitwiseOr / BitwiseXor / BitwiseNot **[opset 18]**
- Cost: 4 nodes + 3 scalar inits (~32 bytes total) for a 30-element periodicity mask. vs storing the explicit mask: 30 BOOL bytes. **Init savings only emerge for multiple mask outputs from the same Mod intermediate.**
- ~~BitwiseOps at opset 18 may not work in NeuroGolf's scorer environment~~ **RESOLVED 2026-06-07: `BitwiseAnd` is GRADER-VALIDATED** (task048 k6 submit; ort_reality grader-validation row) — opset-18 Bitwise* are live levers (#62b, #112b).

## multi_axis_indexing salvage (full original: `_archive/multi_axis_indexing.md`)
- ## GatherND **[opset 11; batch_dims attr opset 12+]**
- Cost: indices `[5, 2]` INT64 = 80 bytes. Output `[10, 5]` = 50 F32 = 200 bytes. vs storing a 900-elem mask = 900 bytes.
- - Index tensor cost: P × K × 8 bytes (INT64).
- - `batch_dims` is opset 12+ — bump opset_import if you need it.
- - Sparse F32 input reads (task033/task036 split): for N scattered cells from
  the canonical one-hot input, flat `GatherND(idx[N,3])` beats Slice+row/col
  Gather because any separable read births a full-width F32 plane. For contiguous
  fixed-size crops, dynamic `Slice` plus concrete value_info is cheaper than a
  dense `GatherND` index table.
- **Cast-before-Gather narrowing (task138):** moving `Cast` before `Gather` deletes
  a gather-result tensor only when the gathered data never overflows the narrow
  dtype. Measure corpus abs-max first; overflow to F16 `inf` can reorder a
  downstream `Einsum` into `inf*0 = nan` while truth still passes.
- **O8 F16 narrowing gate (2026-07-03):** moving `Cast` before `Gather`/`Slice`
  narrows values that the old graph may never have selected. Selected-result
  equality is not proof. Require (1) `absmax <= 65504` for every narrowed value
  that can reach arithmetic, unless overflow is an intentional compare-only
  `+inf` carrier; (2) representative dumps with no new `nan` and no
  undocumented `+inf`/`-inf`; and (3) no reachable invalid pattern before
  decode (`0*inf`, `inf+(-inf)`, `inf/inf`, `0/0`, or contraction/reduction
  routes where zeros/opposite signs can meet infinities). `+inf > 0` decodes
  true; `nan > 0` decodes false in ONNX-reference/IEEE behavior.
- ## ScatterND **[opset 11; reduction attr opset 16+]**
- reduction: "none" (default), "add", "mul" (opset 16; "min", "max" at opset 18)
- task118 used bool ScatterND (-38,341 bytes). **O2 correction (2026-06-19): duplicate-target
  `ScatterND` reductions are unsafe on ORT CPU 1.24.4** (parallel shared-output writes); use
  ScatterND for unique sparse patches / disjoint trailing-slice writes, not histograms or overlap reducers.
- zero_bool = ConstantOfShape([1, 10, 30, 30], value=False)  # ~33 bytes
- true_scalar = ConstantOfShape([N], value=True)  # ~17 bytes
- # Total stored: coords (N×4×8 bytes) + ~50 bytes overhead
- **ScatterElements input-overlay floor (task147, 2026-06-22):** reuse the F32 `input` as the scatter
  `data` so the graph `output` is FREE (only a small `updates` delta plane is charged). The floor is
  three OP-FORCED costs, provable not guess: (1) **`updates` dtype is LOCKED to the `data` dtype** — ORT
  shape-inference rejects `ScatterElements(data=F32, updates=F16)` ("updates has inconsistent type"); if
  `data` is the F32 input, the delta plane CANNOT be F16/U8/BOOL (the only escape, casting the whole input
  first, births a full `[1,10,30,30]` cast = fatal). (2) **`indices.shape == updates.shape` is enforced**
  (no index broadcast); to write K channels in one scatter pay a K×region INT64 index — but as an
  INITIALIZER it is param-counted not byte-counted, so INT64 vs INT32 is a tie and it is cheap. Do NOT
  runtime-generate the indices to "save params": a materialized I64 index is 8 B/elem in MEMORY and dwarfs
  the param save (task147: 100→32 params but cost 532→1616). (3) **One scatter, not a chain** — chaining
  scatters to touch more channels births a full-input `[1,10,30,30]` F32 intermediate between them (fatal),
  which is why all target channels must be written in the single call (forcing the multi-channel
  `updates`/`indices` shape). Corollaries both priced out: single-channel count-map + Sign/Mul tail is
  STRICTLY WORSE (532→1244, each single-channel plane adds its own 144 B and still needs a Concat back to
  the multi-channel `updates`); shrinking `data` to a cropped slice + Pad-after is WORSE (532→3428, both the
  crop and the scatter output are charged — the whole point is writing straight to the free `output`). Floor
  = slice-in producer (Conv) + dtype-locked F32 multi-channel `updates` + K-channel INT64 index init. A
  multi-output Conv emitting all delta channels at once is the cheapest producer — don't decompose it.

## uncommon_ops_catalog salvage (full original: `_archive/uncommon_ops_catalog.md`)
- | `Trilu` | 14 | Upper/lower triangular mask from a tensor (or zero data) | **Zero-init triangular masks** — saves up to 900 BOOL bytes per 30×30 triangle | `zero_param_generators.md` |
- | `Range` | 11 | `[start, start+delta, ..., limit)` as a tensor | Replace `np.arange` initializers with 3 scalar inits (24 bytes vs N×bytes) | `zero_param_generators.md` |
- | `ScatterND` | 11 (16 for `add`/`mul`, 18 for `min`/`max`) | Scatter with multi-axis index tensor, optional reduction | Already used (task118); unique sparse writes only unless a local duplicate-reduction probe proves determinism | salvage section above + `_archive/multi_axis_indexing.md` |
- | `ScatterElements` | 11 (16 for reduction) | Per-position scatter, optional reduction | task387 used this (techniques #8c/#34; provenance `_archive/update_DONE_2026-05-23T031500Z_task387_colorgrid_scatterelements.md`) |
- | `BitwiseAnd/Or/Xor/Not` | 18 | Element-wise bitwise on integer tensors | **GRADER-VALIDATED 2026-06-07** (BitwiseAnd, task048; see ort_reality) — live levers #62b/#112b |
- - `DFT`, `STFT`, `BlackmanWindow`, `HammingWindow`, `HannWindow`, `MelWeightMatrix` (signal processing, opset 17+)
- - `OptionalGetElement`, `OptionalHasElement`, `Optional` (opset 15+; almost never useful for static graphs)


## Salvage restoration (VERIFY 2026-06-05 P9 — dropped in the Phase-2 merge)
- **And vs BitwiseAnd mnemonic**: logical `And/Or/Xor/Not` = BOOL-only kernels; integer planes need opset-18 `Bitwise*` — mixing them is a load-time reject.
- **ReverseSequence is VALIDATOR-BANNED** (op type contains the substring "Sequence" — core.md / ort_reality.md; R6 scorer-printed "not permitted" 2026-06-19, and ORT rejects `time_axis>=2`). The old axis-trap note is moot — use `Slice` negstep (techniques #44) for every flip.
- **Mod sign rule**: integer `Mod` (fmod=0) follows the DIVISOR sign — folds differ from python only when operand signs mix.
- **Expand cannot repeat non-1 dims** (broadcast-only); `Tile` is the repeater.
- **CumSum dtype note:** F16 works at opset 14+; UINT8/INT8/BOOL do not. At
  opset <=13, F16 is rejected by schema before kernel lookup.
- **graphsurgeon does NOT enforce the banned-op list** — it happily emits Loop/Scan/NonZero graphs; the validator rejects them at submit. Check core.md bans yourself.
- **DepthToSpace blocksize must divide the axis**: on 30-wide grids legal blocksizes are {1,2,3,5,6,10,15,30}.
- **GatherND/ScatterND win-thresholds (measured)**: sparse reads win under ~50 cells touched; ScatterND assembly wins for N < ~280 writes — above those, dense plane ops are cheaper.
- **BitwiseAnd broadcast-shape (task047)**: `BitwiseAnd(A[1,C,30,1], B[1,1,1,30])` outputs the BROADCAST shape `[1,C,30,30]` — price the full broadcast result, not the smaller operands, when estimating its memory.
- **Fixed-unroll round floor (task251):** for flood/BFS/morphology graphs with K hardcoded rounds, K is the max convergence round over the full cached truth corpus, not a sample. Model the exact op sequence first; task251's `vertical | (vertical<<1) | (vertical>>1)` is 8-connected V-then-H dilation and converges faster than a 4-connected reference BFS.
- **Scalar-base boundary trap (task047)**: a scalar / `ConstantOfShape` base used for boundary masking silently breaks when the base value can collide with a real index at the edge — verify the sentinel is out of range AT the boundary, not just the interior.
