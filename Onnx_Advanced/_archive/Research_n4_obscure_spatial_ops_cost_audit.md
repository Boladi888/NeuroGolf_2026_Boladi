# N4 - Obscure Spatial Ops Cost Audit

Runtime target: ONNX Runtime 1.24.4 `CPUExecutionProvider`. Layout target: NCHW `[1,C,H,W]`, `C <= 10`, `H,W <= 30`. Cost model: `intermediate_memory_bytes + param_elements`; inputs and final outputs are free. Initializers and `Constant` tensors are charged by element count, not bytes. [PROMPT]

## Source Index

- [ONNX Col2Im](https://onnx.ai/onnx/operators/onnx__Col2Im.html) [PRIMARY]
- [ONNX DepthToSpace](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html) [PRIMARY]
- [ONNX SpaceToDepth](https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html) [PRIMARY]
- [ONNX Resize](https://onnx.ai/onnx/operators/onnx__Resize.html) [PRIMARY]
- [ONNX GridSample](https://onnx.ai/onnx/operators/onnx__GridSample.html) [PRIMARY]
- [ONNX RoiAlign](https://onnx.ai/onnx/operators/onnx__RoiAlign.html) [PRIMARY]
- [ONNX MaxRoiPool](https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html) [PRIMARY]
- [ONNX LpPool](https://onnx.ai/onnx/operators/onnx__LpPool.html) and [ONNX GlobalLpPool](https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html) [PRIMARY]
- [ONNX AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html) and [ONNX MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html) [PRIMARY]
- [ONNX MaxUnpool](https://onnx.ai/onnx/operators/onnx__MaxUnpool.html) [PRIMARY]
- [ONNX ReverseSequence](https://onnx.ai/onnx/operators/onnx__ReverseSequence.html) [PRIMARY]
- [ONNX Pad](https://onnx.ai/onnx/operators/onnx__Pad.html), [ONNX Tile](https://onnx.ai/onnx/operators/onnx__Tile.html), and [ONNX Trilu](https://onnx.ai/onnx/operators/onnx__Trilu.html) [PRIMARY]
- [ONNX DFT](https://onnx.ai/onnx/operators/onnx__DFT.html) [PRIMARY]
- [ONNX CenterCropPad](https://onnx.ai/onnx/operators/onnx__CenterCropPad.html) [PRIMARY]
- [ONNX NonMaxSuppression](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) [PRIMARY]
- [ORT 1.24.4 generated CPU kernel matrix](https://raw.githubusercontent.com/microsoft/onnxruntime/rel-1.24.4/docs/OperatorKernels.md) [PRIMARY]
- [ORT 1.24.4 CPU NonMaxSuppression source](https://github.com/microsoft/onnxruntime/blob/rel-1.24.4/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc) [PRIMARY]

Local smoke context: `onnx==1.21.0`, `onnxruntime==1.24.4`, available providers `['AzureExecutionProvider', 'CPUExecutionProvider']`. I ran in-memory CPU probes for `DepthToSpace`, `SpaceToDepth`, `Pad(mode=wrap)`, `CenterCropPad`, and `NonMaxSuppression`; results are listed near the end. [LOCAL]

## 1. Operator Shelf

The shelf is mostly mined. The only obscure spatial op I would still probe first is `DepthToSpace`, because ORT CPU supports `uint8` and it can replace reshape/transpose/tile style pixel-shuffle chains with zero charged intermediates when it is the final output. `CenterCropPad` is the other notable probe, but it is a function-backed ONNX op, not a dedicated ORT CPU kernel, so it needs validator testing before adoption.

`MaxPool`, `Pad`, `Resize(nearest)`, and final-output `Tile` remain practical, but they are already standard shelf items. `GridSample`, ROI ops, `Col2Im`, `MaxUnpool`, `DFT`, and `NonMaxSuppression` are mostly dead for exact small integer ARC grids because of float-only paths, coordinate params, dynamic output, or non-grid semantics. [DERIVED]

## 2. Table A - Operator Support and Cost Mechanism

ORT dtype cells are for ORT 1.24.4 CPU. ONNX schema dtypes can be wider than ORT kernels; the ORT cell is the one that matters for this environment. [PRIMARY]

| op | since-opset | ORT CPU dtypes | output shape formula | incumbent it may replace | cost mechanism | caveat | source tags |
|---|---:|---|---|---|---|---|---|
| `Col2Im` | 18 | 18+: `float` | `[N,C,image_shape...]`; input is `[N,C*prod(block_shape),L]` | `ScatterND`/overlap-add from column patches | Can delete large scatter index tables if a column tensor already exists | Float-only; no ONNX `Im2Col` producer; overlap semantics must match | [PRIMARY: ONNX Col2Im, ORT kernels] |
| `DepthToSpace` | 13 current; older 1,11 | 13+: `double,float,uint8`; 11-12 same; 1-10 `double,float` | `[N,C/(b*b),H*b,W*b]` | `Reshape -> Transpose -> Reshape`, pixel shuffle, phase-channel render | Attr-only op; if final output, deletes all shuffle intermediates | `C` divisible by `b*b`; `DCR` vs `CRD` order is a common off-by-phase trap | [PRIMARY: ONNX DepthToSpace, ORT kernels] |
| `SpaceToDepth` | 13 current; older 1 | 13+: `double,float`; 1-12 `double,float` | `[N,C*b*b,H/b,W/b]` | `Slice` every phase then `Concat`; patch-to-channel extraction | If input is already float and final output, deletes phase slice planes | ORT CPU has no `uint8` kernel; u8 grids require costly cast | [PRIMARY: ONNX SpaceToDepth, ORT kernels; LOCAL] |
| `Resize` | 19 current; older 10,11,13,18 | 19+,18,13-17,11-12,10: `float,int32,int8,uint8` | With `sizes`: selected axes become `sizes`; with `scales`: roughly `floor(input_dim*scale)` subject to mode/roi rules | `Tile`, `Gather`, `Slice`/`Pad` resize stencils | Static `sizes` costs rank params; nearest final output can delete tile/gather intermediates | Interpolation and coordinate mode are exactness traps; `scales` floats and `roi` add params | [PRIMARY: ONNX Resize, ORT kernels] |
| `GridSample` | 22 current; older 16,20 | 22+: `double,float`; 20-21 same; 16-19 `float` | 4-D: `X[N,C,Hin,Win]`, `grid[N,Hout,Wout,2]` -> `Y[N,C,Hout,Wout]` | GatherND/remap/grid-warp chains | One op can fuse a remap if grid is already cheap or final is float | Coordinate grid params cost `2*Hout*Wout`; u8 needs casts; only nearest with exact normalized coordinates is discrete-safe | [PRIMARY: ONNX GridSample, ORT kernels] |
| `RoiAlign` | 22 current; older 10,16 | 16+: `double,float`; 10-15 same | `[num_rois,C,output_height,output_width]` | crop+resize+average pooling over ROIs | Static ROIs can fuse crop and pooling | Float-only; bilinear/avg sampling is usually non-exact; ROI constants cost 4R plus batch indices | [PRIMARY: ONNX RoiAlign, ORT kernels] |
| `MaxRoiPool` | 22 current; older 1 | 1+: `float` | `[num_rois,C,pooled_h,pooled_w]` | ROI crop + `MaxPool` | Fuses ROI crop and max pooling | Float-only in ORT CPU; ROI constants cost; rarely better than static `Slice` + `MaxPool` | [PRIMARY: ONNX MaxRoiPool, ORT kernels] |
| `LpPool` | 22 current; older 2,11,18 | 22+,18-21,11-17,2-10: `float` | Pool formula: `floor((in + pads - dilation*(kernel-1) - 1)/stride + 1)` per spatial axis, with ceil-mode variant | Conv with all-ones or p-norm window stacks | `p=1` on nonnegative masks gives window sum without weight params | Float-only; absolute value semantics; not useful for signed/color labels without casts and care | [PRIMARY: ONNX LpPool, ORT kernels] |
| `GlobalLpPool` | 2 | 2+: `float` | `[N,C,1,1]` | full-plane `Abs -> ReduceSum` for p=1 | One final output for global nonnegative mask counts | Float-only; `ReduceSum` is often already as cheap | [PRIMARY: ONNX GlobalLpPool, ORT kernels] |
| `AveragePool` | 22 current; older 1,7,10,11,19 | 22+,19-21,11-18,10,7-9: `float` | Same pool formula as above | Conv+divide, local mean, window fraction | No kernel weights; final output can be free | Float-only; sums require multiply by window area and exact divisibility/rounding care | [PRIMARY: ONNX AveragePool, ORT kernels] |
| `MaxPool` | 22 current; older 1,8,10,11,12 | 22+ and 12-21: `double,float,int8,uint8`; 8-11 `double,float`; 1-7 `float` | Same pool formula; optional `Indices` output for newer opsets | ReduceMax over neighbor stack, morphology dilation, local OR for binary masks | Deletes KxK shifted planes; supports u8 directly in current opsets | If using indices, second output is an int64 tensor and can become expensive | [PRIMARY: ONNX MaxPool, ORT kernels] |
| `MaxUnpool` | 22 current; older 9,11 | 22+,11-21,9-10: `float`; `I` and optional `output_shape` are `int64` | If `output_shape` supplied, use it; otherwise inverse pool formula from `X`, `kernel`, `strides`, `pads` | scatter sparse maxima back to a canvas | Could avoid explicit scatter if indices are already available | Float-only; needs MaxPool indices; output-shape static safety requires int64 params | [PRIMARY: ONNX MaxUnpool, ORT kernels] |
| `ReverseSequence` | 10 | 10+: broad tensor set including `uint8`, `int*`, `float*`, `bool`, `string` | Same shape as input | full or prefix reversal along first two axes | For rank-2 tensors, one `sequence_lens` vector can replace reverse `Slice` params | ONNX only allows `batch_axis`/`time_axis` 0 or 1; NCHW spatial H/W reversal needs transpose/reshape | [PRIMARY: ONNX ReverseSequence, ORT kernels] |
| `Pad` modes: `constant`, `edge`, `reflect`, `wrap` | 24/25 current; wrap appears in later Pad schemas; core since 2 | 25,24,23,21-22,19-20,18,13-17: `bool,double,float,int32,int64,int8,uint32,uint64,uint8`; 11-12 lacks bool; 2-10 `double,float` | For 2-D spatial axes: `[N,C,H+top+bottom,W+left+right]`; axes subset allowed in newer opsets | `Concat` of edge strips, reflect/wrap slices, border tiles | Direct boundary extension with pads params only; can be final output | Reflect requires enough extent; wrap exactness needs opset/mode acceptance; padding as intermediate is still charged | [PRIMARY: ONNX Pad, ORT kernels; LOCAL] |
| `Tile` | 13 current; older 6 | 13+,6-12: `bool,double,float,int16,int32,int64,int8,string,uint16,uint32,uint64,uint8` | `[d0*r0,d1*r1,...]` | constant table expansion, repeated pattern render | Final output tile costs only repeats params | If not final, output size is charged; often causes the cost problem rather than solving it | [PRIMARY: ONNX Tile, ORT kernels] |
| `Trilu` | 14 | 14+: `bool,double,float,int64` | Same shape; operates over matrix/batched matrix last two dims | triangular coordinate mask chains | Can zero upper/lower triangle of existing bool/float/int64 tensor with one op | No u8/int32; creating an all-ones source can cost as much as a static mask | [PRIMARY: ONNX Trilu, ORT kernels] |
| `DFT` | 20 current; older 17 | 20+,17-19: `double,float`; `dft_length` `int32/int64` | Same rank; transformed axis may use `dft_length`; complex represented with real/imag component | transform-domain correlation/convolution tricks | None for exact grid work | Float/complex semantics, rounding, and transform params make it dead for discrete masks/colors | [PRIMARY: ONNX DFT, ORT kernels] |
| `CenterCropPad` | 18 | ONNX function schema, not a dedicated generated ORT CPU kernel row; local ORT 1.24.4 CPU executed `uint8` via function support | Target shape from `shape` input; unspecified axes unchanged | centered `Slice` plus zero `Pad` arithmetic | One op can replace center-crop/pad chains and static offset constants | Function-backed op may be validator-risk if function ops are banned; zero padding only | [PRIMARY: ONNX CenterCropPad; LOCAL] |
| `NonMaxSuppression` | 11 current; older 10 | ORT CPU source registers opset 10 and 11 kernels over float boxes/scores; output `int64` | `[num_selected_indices,3]` | detection-style selection, not normal grid spatial | None for static-grid cost | First dimension is data-dependent; fatal under static-output constraint | [PRIMARY: ONNX NMS, ORT NMS source; LOCAL] |

## 3. Table B - Cost Comparisons

Byte examples use `B(T)` for element byte width. For u8 `B=1`, f32/int32 `B=4`, int64/f64 `B=8`. `Hout,Wout` are output spatial sizes. Costs assume the obscure op is the final output unless noted; if it feeds another node, charge its output bytes too. [DERIVED]

| task pattern | incumbent chain and cost | obscure-op chain and cost | exactness conditions | verdict |
|---|---|---|---|---|
| 2x pixel shuffle from phase channels `[1,4C,H,W] -> [1,C,2H,2W]` | `Reshape -> Transpose -> Reshape`; two full-size shuffle intermediates: `2 * 4*C*H*W*B(T)` | `DepthToSpace(blocksize=2)`: `0` params, `0` intermediates if final | Channel order must be `DCR` or `CRD` matched; `C_in % 4 == 0`; ORT supports u8 | adopt/probe |
| 2x phase extraction `[1,C,H,W] -> [1,4C,H/2,W/2]` | Four strided `Slice`s then final `Concat`; slice intermediates total about `C*H*W*B(T)` | `SpaceToDepth(blocksize=2)`: `0` if input already float; for u8, `Cast` to f32 costs `4*C*H*W` bytes before op | H,W divisible by 2; ORT CPU lacks u8 kernel | task-specific |
| nearest integer resize/upscale | `Tile/Reshape/Transpose` or gather table; often one or two full output-size intermediates plus repeat/index params | `Resize(mode=nearest,sizes=...)`: `rank` size params; no intermediate if final | Set `coordinate_transformation_mode` and `nearest_mode` to reproduce incumbent exactly; avoid linear/cubic | adopt for nearest |
| arbitrary remap of u8 grid to same shape | `GatherND` with int index table: params `2*Hout*Wout`; output final free | `Cast u8->f32` + `GridSample(nearest)` + optional cast back: `4*C*H*W` bytes plus grid params `2*Hout*Wout`; optional f32 output intermediate `4*C*Hout*Wout` | Only exact if grid coordinates hit pixel centers under selected `align_corners` and `padding_mode`; nearest only | probe rarely |
| static affine/coordinate remap with generated grid | Range/Tile/math grid generation can create several `Hout*Wout` f32 intermediates before gather/sample | Static `GridSample` grid initializer: `2*Hout*Wout` params, no generated-grid intermediates | Params can dominate less than dynamic grid bytes, but casts remain | task-specific |
| wrap/reflect/edge border extension | `Slice` strips + `Concat` around spatial axes; roughly padded output plus strips as intermediates | `Pad(mode=edge/reflect/wrap)`: pads params `2*rank` or `2*axes`; final output free | `reflect` cannot pad more than available reflected extent; `wrap` needs later Pad mode accepted by target opset/runtime | adopt/probe |
| local binary dilation or color max | shifted neighbor stack + `ReduceMax`; stack costs `K*K*C*H*W*B(T)` | `MaxPool(kernel=K,pads=...)`: `0` params; final output free | Max semantics only; current ORT supports u8/int8 | adopt |
| local p=1 nonnegative window sum | `Conv` with all-ones weights: params `K*K*C` plus f32 cast/output if needed | `LpPool(p=1)`: no weights, final output free; cast u8->f32 costs `4*C*H*W` | Input values must be nonnegative if sum(abs(x)) is intended sum; float exactness only within small integer range | task-specific/probe |
| local average/fraction mask | `Conv`/`ReduceMean` over shifted stack, with possible `Div` | `AveragePool`: no weights, final output free | Float-only; use only when average itself is consumed or thresholding tolerates exact binary fractions | task-specific |
| whole-plane nonnegative count | `Abs -> ReduceSum` or `Cast -> ReduceSum`; final reduce often already cheap | `GlobalLpPool(p=1)`: final output free, no axes params | Float-only; output is per channel `[N,C,1,1]` | task-specific |
| center crop or center zero-pad to static shape | Center offsets built with `Shape/Sub/Div/Slice/Pad`, or manually constant `Slice`/`Pad`; several scalar and spatial intermediates if dynamic | `CenterCropPad` with shape initializer: shape params `rank`; final output free | Zero pad only; centered odd differences round as spec says; function-backed validator risk | probe |
| full spatial flip in NCHW | `Slice` with negative step: starts/ends/axes/steps constants, final output free | `ReverseSequence` cannot target axes 2/3 directly; transpose/reshape would add full tensor intermediates | Only useful for rank-2 or channel-axis reversal; not direct NCHW H/W | dead for spatial |
| triangular mask or triangular zeroing over H/W | coordinate `Range`/`Less`/`Where` mask chain or static triangle initializer `H*W` params | `Trilu` on existing bool/float/int64 tensor: no params; if creating all-ones input, pay `H*W*B(T)` intermediate | Last two dims are matrix dims; no u8/int32 kernel | task-specific |
| sparse unpool/scatter from maxima | `ScatterND` with indices params and output canvas | `MaxPool(..., Indices)` + `MaxUnpool`: avoids explicit scatter only if indices already exist | Float-only; indices output is int64 and expensive if materialized solely for unpool | dead/task-specific |
| fold columns back to image | `ScatterND`/adds using patch indices, often large index params | `Col2Im`: image shape params plus block shape params; no scatter indices if column tensor exists | Float-only; overlap accumulation semantics; usually no cheap column producer | dead/task-specific |
| ROI crop and pool | Static `Slice`/`Resize`/`MaxPool` with exact integer coordinates | `RoiAlign` or `MaxRoiPool` with ROI constants | Float-only; `RoiAlign` interpolation/averaging is not ARC-exact; `MaxRoiPool` only max | dead except detection-like tasks |
| transform-domain shortcut | Conv/correlation in spatial domain | `DFT` plus complex math | Not exact for integer masks/colors under ORT float kernels; extra real/imag dimensions | dead |
| detection selection | None in standard grid pipeline | `NonMaxSuppression` | Output shape `[num_selected_indices,3]` is dynamic; semantics are boxes/scores, not static grid | dead/fatal |

## 4. Exactness Traps

Interpolation: `Resize`, `GridSample`, and `RoiAlign` are usually non-exact for discrete ARC grids unless using nearest-neighbor sampling with coordinates that land exactly on source pixel centers. `linear`, `cubic`, bilinear ROI averaging, antialias filters, and ROI coordinate conventions are float workflows. [PRIMARY] [DERIVED]

Coordinate normalization: `GridSample` grid values are normalized coordinates, not integer indices. A static remap grid costs `2*Hout*Wout` parameter elements. A generated grid often costs more because `Range/Tile/Concat/Cast` intermediates are charged in bytes. [PRIMARY] [DERIVED]

Boundary modes: `Pad(edge)` is exact replication, `Pad(reflect)` excludes the edge value in the reflected run per ONNX semantics, and `Pad(wrap)` is periodic. `reflect` has extent limits; local ORT 1.24.4 accepted `Pad(mode='wrap')` for u8 opset 19. [PRIMARY] [LOCAL]

Dtype conversions: `GridSample`, `RoiAlign`, `MaxRoiPool`, `LpPool`, `GlobalLpPool`, `AveragePool`, `MaxUnpool`, `Col2Im`, and `DFT` are float-only or effectively float-only under ORT CPU for this shelf. On u8 grids, a required input cast creates a charged f32 tensor of `4*C*H*W` bytes. If final output must be u8, the float op output is also charged before the final cast. [PRIMARY] [DERIVED]

Multi-output ops: `MaxPool` indices are only useful if a later op consumes them. Otherwise do not request the second output. If requested and consumed, the int64 indices can dominate cost. [PRIMARY] [DERIVED]

Dynamic outputs: `NonMaxSuppression` outputs `[num_selected_indices,3]`; even though ORT CPU has a source kernel and the local smoke test ran, the first dimension is data-dependent and violates the static-output constraint. [PRIMARY] [LOCAL]

Function-backed schema: `CenterCropPad` is `function: True` in the ONNX schema. Local ORT 1.24.4 CPU executed it, but it does not appear as a dedicated row in the generated CPU kernel matrix. If the NeuroGolf validator bans function-bodied ops, it is fatal despite local execution. [PRIMARY] [LOCAL] [DERIVED]

## 5. Ranked Adoption List

| rank | op | class | reason |
|---:|---|---|---|
| 1 | `DepthToSpace` | adopt/probe | Best obscure candidate: u8 CPU support, exact layout permutation, zero params, deletes shuffle intermediates. |
| 2 | `Pad(edge/reflect/wrap)` | adopt/probe | Good for boundary construction, especially wrap/reflect cases that otherwise need slice/concat strips. |
| 3 | `Resize(nearest)` | adopt | Useful exact final-output scale/copy when coordinate attributes are pinned. |
| 4 | `MaxPool` | adopt | Already standard, but still the best exact u8 window reducer. |
| 5 | `CenterCropPad` | probe | Cost-attractive for centered crop/pad, but function-backed validator risk. |
| 6 | `Tile` | task-specific | Strong only when final output is a repeated pattern; dangerous as an intermediate. |
| 7 | `SpaceToDepth` | task-specific | Excellent pattern but ORT CPU lacks u8; good only in float subgraphs. |
| 8 | `LpPool(p=1)` | task-specific/probe | Can delete all-ones Conv params for nonnegative masks if already float. |
| 9 | `AveragePool` | task-specific | Useful for means/fractions, not exact label logic unless downstream stays float. |
| 10 | `Trilu` | task-specific | Good only when operating on existing bool/float/int64 H/W matrices. |
| 11 | `GlobalLpPool` | task-specific | Possible global p=1 counts, but `ReduceSum` is often equally cheap. |
| 12 | `GridSample` | probe rarely | Only beats incumbent if it deletes generated grid/remap machinery; static grid params and casts usually dominate. |
| 13 | `ReverseSequence` | dead for NCHW spatial | Cannot directly target H/W axes under ONNX axis limits; `Slice` is cheaper for flips. |
| 14 | `Col2Im` | dead/task-specific | Float-only and needs a column tensor; no general ARC-grid win. |
| 15 | `RoiAlign` | dead/task-specific | Float interpolation and ROI params; not exact for discrete grids. |
| 16 | `MaxRoiPool` | dead/task-specific | Float-only ROI max; static `Slice` + `MaxPool` usually wins. |
| 17 | `MaxUnpool` | dead/task-specific | Float-only and needs int64 indices from MaxPool. |
| 18 | `DFT` | dead | Float/complex transform, not exact or cost-friendly. |
| 19 | `NonMaxSuppression` | dead/fatal | Dynamic output and detection semantics. |

## 6. Minimal Local Probes

These are the probes I would keep as a regression shelf. The first five were run locally with ORT 1.24.4 CPU in this workspace. [LOCAL]

| probe | shape/dtype/opset | expected result |
|---|---|---|
| `DepthToSpace_u8_opset13` | X `[1,4,1,1]` `uint8`, `blocksize=2`, opset 13 | Runs; output `[1,1,2,2]` values `[0,1,2,3]` under `DCR` for the simple input. |
| `SpaceToDepth_u8_opset13` | X `[1,1,2,2]` `uint8`, `blocksize=2`, opset 13 | Fails `NOT_IMPLEMENTED`; confirms ORT CPU dtype gap despite ONNX schema allowing wider types. |
| `Pad_wrap_u8_opset19` | X `[1,1,1,3]` `uint8`, pads width pre/post 2, `mode=wrap`, opset 19 | Runs; `[1,2,3]` becomes `[2,3,1,2,3,1,2]`. |
| `CenterCropPad_u8_opset18` | X `[1,1,4,4]` `uint8`, shape `[1,1,2,2]`, opset 18 | Runs locally; center crop values `[5,6,9,10]`. |
| `NonMaxSuppression_opset11` | boxes `[1,1,4]` f32, scores `[1,1,1]` f32, max output 1 | Runs locally; output `[[0,0,0]]`, but shape is dynamic `[?,3]`. |
| `GridSample_nearest_identity_opset22` | X `[1,1,2,2]` f32, grid `[1,2,2,2]` f32 with exact pixel centers | Should return identity; use to pin `align_corners` and normalized coordinate construction. |
| `MaxPool_u8_3x3_pad1_opset22` | X `[1,1,3,3]` `uint8`, kernel 3, pads 1 | Should match explicit 3x3 border-padded max; use without indices output first. |

## Final Verdict

The spatial surface is mostly mined. The best obscure spatial op still worth probing is `DepthToSpace`, specifically for u8 pixel-shuffle/final-render patterns where current graphs use reshape/transpose/tile chains or carry phase channels. `CenterCropPad` is worth one validator probe because it can collapse centered crop/pad arithmetic, but its ONNX schema is function-backed. `GridSample` should not be a default remap replacement: for u8 ARC grids, its float casts and `2*H*W` grid params usually erase the gain unless it deletes a much larger generated-coordinate chain.
