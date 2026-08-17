# N3 - Shape and constant generators cost audit

## Scope and verdict

Runtime target: ONNX Runtime 1.24.4 `CPUExecutionProvider`. Score model from the card:

```
score = intermediate_memory_bytes + param_elements
```

The important result is blunt: a generator that materializes a full `[1,10,30,30]` tensor as an intermediate usually does **not** beat a stored `[1,10,30,30]` constant under this score. A stored constant costs `9000` params. A generated `UINT8` or `BOOL` full intermediate costs `9000` bytes plus the tiny generator params, so it loses by the generator overhead. `FLOAT16` and `FLOAT` generators lose harder.

The generator shelf is still valuable in three places:

1. Final graph outputs, because the final generated tensor has `0` intermediate-byte charge. [PROMPT][DERIVED]
2. Small axis/basis carriers such as `[30]`, `[1,1,30,1]`, `[1,1,1,30]`, `[30,30]`, or a tiny motif, broadcast into a later op without expanding to full `[1,10,30,30]`. [DERIVED]
3. Rewrites where the output tensor is already unavoidable and the generator deletes a larger existing intermediate chain, not just a dense initializer. [DERIVED]

## Source Tags

- `[PROMPT]` - given by the research card.
- `[PRIMARY]` - ONNX operator spec, ONNX Runtime 1.24.4 kernel docs, or ONNX Runtime 1.24.4 source.
- `[DERIVED]` - arithmetic from the prompt cost model plus cited operator semantics.
- `[LOCAL]` - local one-op probe against installed `onnxruntime==1.24.4`, `onnx==1.21.0`, `numpy==2.4.4`.

## Primary Sources

ONNX operator specs:

- [`ConstantOfShape`](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) - one-element `value`, int64 shape input, output dtype from `value`, default float32 zero. [PRIMARY]
- [`Range`](https://onnx.ai/onnx/operators/onnx__Range.html) - `start`, `limit`, `delta`, exclusive limit, output length formula. [PRIMARY]
- [`Shape`](https://onnx.ai/onnx/operators/onnx__Shape.html), [`Size`](https://onnx.ai/onnx/operators/onnx__Size.html), [`Gather`](https://onnx.ai/onnx/operators/onnx__Gather.html) - shape tensor extraction. [PRIMARY]
- [`EyeLike`](https://onnx.ai/onnx/operators/onnx__EyeLike.html) - 2D diagonal matrix with optional `dtype` and `k`. [PRIMARY]
- [`Trilu`](https://onnx.ai/onnx/operators/onnx__Trilu.html) - upper/lower triangular retention over rank >= 2 tensors. [PRIMARY]
- [`Expand`](https://onnx.ai/onnx/operators/onnx__Expand.html), [`Tile`](https://onnx.ai/onnx/operators/onnx__Tile.html), [`Concat`](https://onnx.ai/onnx/operators/onnx__Concat.html), [`Pad`](https://onnx.ai/onnx/operators/onnx__Pad.html), [`Resize`](https://onnx.ai/onnx/operators/onnx__Resize.html). [PRIMARY]
- [`DepthToSpace`](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html), [`SpaceToDepth`](https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html). [PRIMARY]
- [`Unsqueeze`](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html), [`Squeeze`](https://onnx.ai/onnx/operators/onnx__Squeeze.html), [`Reshape`](https://onnx.ai/onnx/operators/onnx__Reshape.html), [`Transpose`](https://onnx.ai/onnx/operators/onnx__Transpose.html). [PRIMARY]
- [`OneHot`](https://onnx.ai/onnx/operators/onnx__OneHot.html) - inserts one class dimension; output dtype equals `values`; out-of-range semantics. [PRIMARY]
- [`CumSum`](https://onnx.ai/onnx/operators/onnx__CumSum.html) - inclusive/exclusive and reverse prefix sums. [PRIMARY]
- [`Equal`](https://onnx.ai/onnx/operators/onnx__Equal.html), [`Less`](https://onnx.ai/onnx/operators/onnx__Less.html), [`Greater`](https://onnx.ai/onnx/operators/onnx__Greater.html), [`Where`](https://onnx.ai/onnx/operators/onnx__Where.html), [`Mod`](https://onnx.ai/onnx/operators/onnx__Mod.html), [`Add`](https://onnx.ai/onnx/operators/onnx__Add.html). [PRIMARY]

ONNX Runtime primary support references:

- [`docs/OperatorKernels.md` for ONNX Runtime v1.24.4](https://github.com/microsoft/onnxruntime/blob/v1.24.4/docs/OperatorKernels.md) - generated CPU kernel/opset/type table. [PRIMARY]
- [`constant_of_shape.cc`](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cpu/generator/constant_of_shape.cc) - CPU registrations and fill implementation. [PRIMARY]
- [`eye_like.cc`](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cpu/tensor/eye_like.cc) - CPU registrations list official output types as `float`, `double`, `uint64_t`, `int64_t`, `int32_t`. [PRIMARY]
- [`trilu.cc`](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cpu/tensor/trilu.cc) - CPU registration for `float`, `double`, `int64_t`, `bool`. [PRIMARY]
- [`onehot.cc`](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cpu/tensor/onehot.cc) - CPU typed registrations for selected index/depth/value combos; no `BOOL` or `UINT8` values. [PRIMARY]
- [`pad.cc`](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cpu/tensor/pad.cc) and [`padbase.h`](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cpu/tensor/padbase.h) - CPU type registrations and `constant`/`reflect`/`edge`/`wrap` mode parse. [PRIMARY]

Local check summary:

- `ConstantOfShape`: `BOOL`, `UINT8`, `FLOAT16`, `FLOAT`, `INT64` loaded and ran. [LOCAL]
- `EyeLike`: `BOOL` and `UINT8` failed; `FLOAT`, `INT32`, `INT64`, `UINT64` ran. Local wheel also ran `FLOAT16`, but official v1.24.4 source/kernel docs do not list `FLOAT16`; treat `FLOAT16 EyeLike` as local-probe-only. [LOCAL][PRIMARY]
- `Trilu`: `BOOL`, `FLOAT`, `INT64` ran; `UINT8` failed. Local wheel also ran `FLOAT16`, but official v1.24.4 source/kernel docs do not list `FLOAT16`; treat it as local-probe-only. [LOCAL][PRIMARY]
- `OneHot`: `BOOL`, `UINT8`, `INT32` output values failed; `FLOAT` and `INT64` ran. Local wheel also ran `FLOAT16`, but official v1.24.4 source/kernel docs do not list `FLOAT16`; treat it as local-probe-only. [LOCAL][PRIMARY]
- `Range`: `FLOAT`, `INT16`, `INT32`, `INT64` ran; `UINT8` and `FLOAT16` failed. [LOCAL][PRIMARY]
- `Pad(mode="wrap")`: small `UINT8` opset-19 smoke test ran. [LOCAL][PRIMARY]

## Symbols

- `E = product(output_shape)`.
- `b(T) = bytes per output element for dtype T`.
- `r = rank of a target shape`.
- `H=30`, `W=30`, `C=10` for the common `[1,10,30,30]` target.
- A stored dense constant of shape `S` costs `product(S)` params, dtype-blind. [PROMPT]
- A generated tensor costs its tiny initializer params plus `product(S) * b(T)` bytes if it is an intermediate; if it is the final graph output, its output bytes are `0`. [PROMPT]

## 1. Generator Inventory

### Fill and scalar generators

- `ConstantOfShape(shape, value)` generates a tensor of a requested shape. `shape` is an `INT64` 1D tensor; `value` is optional and must be a one-element tensor; omitted value gives float32 zero. Lowest usable opset is 9 for common numeric/bool types; newer opsets widen exotic dtype schema. ORT 1.24.4 CPU has broad support for `BOOL`, `UINT8`, `FLOAT16`, `FLOAT`, integer types, etc. [PRIMARY]
- `Constant` scalar or tiny tensor initializers remain the cheapest source for shape vectors, axes, small ramps, channel ids, and small lookup tables. This is not a runtime generator, but it is often the winning "generator shelf" component under dtype-blind params. [DERIVED]
- `Expand(scalar_or_small, shape)` broadcasts a small tensor to a target shape. It is only a cost win if the expanded tensor is final, or if the expanded tensor is avoided and the small input is broadcast directly by a later op. [PRIMARY][DERIVED]

### Shape and dimension generators

- `Shape(data)` emits an `INT64` vector of the input shape; newer schemas support slicing the shape vector by attrs. [PRIMARY]
- `Size(data)` emits an `INT64` scalar equal to element count. [PRIMARY]
- `Gather(shape, indices)`, `Slice(shape, ...)`, and `Concat(shape_parts)` extract and assemble shape vectors for `Reshape`, `Expand`, `ConstantOfShape`, `Pad`, `Resize`, and similar shape inputs. [PRIMARY]
- These are useful for model plumbing, but data-dependent shape chains are a profiling and static-inference hazard. Prefer literal shape initializers when the target is fixed. [PROMPT][DERIVED]

### Index and coordinate generators

- `Range(start, limit, delta)` emits a 1D arithmetic progression with length `max(ceil((limit-start)/delta),0)`. ONNX opset 11 supports `FLOAT`, `DOUBLE`, `INT16`, `INT32`, `INT64`; ONNX later schema widens, but ORT 1.24.4 CPU support remains `DOUBLE`, `FLOAT`, `INT16`, `INT32`, `INT64`. [PRIMARY]
- Stored iota vectors like `[0..29]` usually beat `Range` when the range is an intermediate: `[30] INT64 Range = 3 params + 240 bytes`, while stored iota = 30 params. [DERIVED]
- Axis vectors with `Unsqueeze`/`Reshape` are still excellent if kept small and broadcast by comparisons/arithmetic instead of expanded to full grids. [DERIVED]

### Structural masks

- `EyeLike(input, dtype, k)` generates a 2D diagonal matrix shaped like a rank-2 carrier. ONNX schema allows many dtypes; official ORT 1.24.4 CPU docs/source list `FLOAT`, `DOUBLE`, `INT32`, `INT64`, `UINT64`, not `BOOL` or `UINT8`. [PRIMARY]
- `Trilu(input, k, upper)` keeps upper or lower triangular parts of rank >= 2 tensors. Official ORT 1.24.4 CPU source lists `BOOL`, `FLOAT`, `DOUBLE`, `INT64`, not `UINT8`. [PRIMARY]
- `OneHot(indices, depth, values, axis)` inserts a class dimension. Official ORT 1.24.4 CPU source supports selected typed kernels, practically `FLOAT` and `INT64` output values for the common `INT64` index/depth path; it does not support `BOOL` or `UINT8` output values. [PRIMARY][LOCAL]
- `CumSum(x, axis, exclusive, reverse)` can generate prefix ramps, ordinal labels, and prefix masks from sparse start markers, but ORT 1.24.4 CPU kernel docs list `DOUBLE`, `FLOAT`, `INT32`, `INT64`; not `BOOL` or `UINT8`. [PRIMARY]

### Layout, repetition, padding, and small motifs

- `Tile(input, repeats)` repeats a seed tensor. Best for final outputs or tiny motifs consumed directly. Full intermediate tiles lose by at least the seed/repeat param overhead for `BOOL`/`UINT8`. [PRIMARY][DERIVED]
- `Resize(mode="nearest")` upsamples a small seed. ORT 1.24.4 CPU supports `FLOAT`, `INT32`, `INT8`, `UINT8`; not `BOOL` or `FLOAT16` per kernel docs. [PRIMARY]
- `DepthToSpace(blocksize, mode)` rearranges channel blocks into spatial blocks. ORT 1.24.4 CPU supports `DOUBLE`, `FLOAT`, and `UINT8` for opset 11+. `SpaceToDepth` supports `DOUBLE` and `FLOAT`, not `UINT8`, per kernel docs. [PRIMARY]
- `Pad` adds borders/frames or wraparound. ORT 1.24.4 CPU supports `BOOL` at opset 13+, `UINT8` at opset 11+, and `wrap` mode at opset 19+ by spec/source/probe. [PRIMARY][LOCAL]
- `Concat`, `Unsqueeze`, `Squeeze`, `Reshape`, `Transpose` move or assemble generated pieces. Treat their output as byte-priced if it is a named intermediate. [PRIMARY][DERIVED]

## 2. Cost Formulas

Core formulas:

- Stored dense constant `S`: `params = E`; `intermediate bytes = 0` if the constant is only an initializer. [PROMPT]
- Generated intermediate `Y`: `params = small initializer element count`; `intermediate bytes = E * b(T)`. [PROMPT]
- Generated final output `Y`: `params = small initializer element count`; `intermediate bytes = 0`. [PROMPT]
- Break-even against a stored dense constant of same shape and dtype `T`: generator wins only if `small_params + E*b(T) < E`, which is impossible for any dtype with `b(T) >= 1` unless the generated tensor is final, smaller than the stored artifact, or replaces other intermediates. [DERIVED]

### Table A

| op/idiom | opset | ORT 1.24.4 CPU dtypes | params needed | intermediate bytes | best use | caveat | source tags |
|---|---:|---|---:|---:|---|---|---|
| `ConstantOfShape(shape,value)` uniform fill | 9+; bool/numeric from spec, wider later | Broad: bool, u8, f16, f32, integers per ORT kernel docs/source | `r + 1`, or `r` for default f32 zero | `E*b(T)` unless final | Final uniform tensors; small broadcast carriers; shape plumbing | Full same-shape intermediate loses by `r+1` for bool/u8 and more for wider dtypes | [PRIMARY][DERIVED] |
| Scalar `Constant` + `Expand` | 8+ | `Expand`: bool, f16, f32, ints, string per ORT docs | `1 + r` | `E*b(T)` if expanded | Scalar broadcast when final, or when a later op can broadcast the scalar without explicit `Expand` | Explicit full `Expand` intermediate usually loses | [PRIMARY][DERIVED] |
| Tiny initializer vector/table | any | Any initializer dtype accepted by consumer | `S` | `0` as initializer | Iota `[30]`, channel ids `[10]`, LUTs, shape vectors | Not a runtime generator, but often beats `Range`/`Concat` under this score | [PROMPT][DERIVED] |
| `Range` ramp | 11+ | ORT docs/probe: f32, f64, i16, i32, i64; no u8/f16 | `3` | `L*b(T)` | Final ramp output; dynamic range if explicitly profiled | Stored `[0..29]` usually wins for static intermediate ramps | [PRIMARY][LOCAL][DERIVED] |
| `Shape` | 1+; attrs in newer opsets | Any input tensor type per ORT docs, output i64 | `0` | `8*r` | Runtime shape plumbing | Dynamic-shape profiling hazard if shape depends on data-dependent graph | [PRIMARY][DERIVED] |
| `Size` | 1+ | Broad input dtypes, output i64 | `0` | `8` | Scalar element count | Rarely useful for fixed NeuroGolf shapes | [PRIMARY][DERIVED] |
| `Gather` from shape/iota | 1+ | Broad data dtypes; indices i32/i64 | `m` index params if initializer | `m*b(data)` | Extract `H`, `W`, `C`, or channel ids | For fixed shapes, literal scalar initializer is cheaper | [PRIMARY][DERIVED] |
| `EyeLike` diagonal | 9+; 22 adds bf16 schema | Official ORT source/docs: f32, f64, i32, i64, u64. Local wheel accepted f16; no bool/u8 | carrier shape params, usually `2` or `3` | `N*M*b(T)` | Final or directly consumed 2D diagonal in numeric dtype | No official bool/u8 CPU output; 2D only | [PRIMARY][LOCAL] |
| `Trilu` triangular | 14+ | Official ORT source/docs: bool, f32, f64, i64. Local wheel accepted f16; no u8 | existing input: `0` or `1` for `k`; generated ones carrier: `rank+1(+k)` | Same as input/output `E*b(T)` | Apply triangular retention directly to existing data; final bool/f32 masks | Generated full mask intermediate barely loses vs stored bool mask unless it replaces larger repeated-channel artifact | [PRIMARY][LOCAL][DERIVED] |
| `OneHot` category planes | 9+; use 11+ | Official ORT source/docs: selected combos; common safe outputs f32 and i64. Local wheel accepted f16; no bool/u8 | `depth scalar 1 + values[2] + index source` | `prod(indices_shape)*depth*b(T3)` | Direct final one-hot output from compact index grid | Full one-hot intermediate is often the biggest trap | [PRIMARY][LOCAL][DERIVED] |
| `CumSum` prefix/rank | 11+; 14 schema change | ORT docs: f32, f64, i32, i64; no bool/u8 | axis scalar `1` plus input source | input bytes plus output `E*b(T)` if generated input is also intermediate | Prefix labels, ordinal masks, run ranks when replacing larger chains | Not a constant replacement; output full bytes count | [PRIMARY][DERIVED] |
| `Tile` motif repetition | 6+ | ORT docs: bool, f32/f64, ints, string; no f16 in table | `seed_elems + rank(repeats)` | `E*b(T)` | Final checkerboards/periodic outputs from tiny seed | Full intermediate tile loses by seed/repeats overhead for bool/u8 | [PRIMARY][DERIVED] |
| `Resize(nearest)` upsample | 10+; prefer 11/13/18+ depending inputs | ORT docs: f32, i32, i8, u8 | seed + scales/sizes/roi params | `E*b(T)` | Final nearest upsample from tiny seed | No bool/f16 CPU; attributes must pin nearest semantics | [PRIMARY][DERIVED] |
| `DepthToSpace` block stamps | 1+; u8 from 11+ | ORT docs: f64, f32, u8 for opset 11+ | seed elems only; attrs free | `E*b(T)` | Final or unavoidable u8 block-stamp layout | Channel count must be divisible by blocksize^2 | [PRIMARY][DERIVED] |
| `SpaceToDepth` block compact | 1+ | ORT docs: f64, f32 | `0` | `E*b(T)` | Numeric layout compaction before smaller op | No official u8/bool CPU support | [PRIMARY] |
| `Pad` frame/wrap | 11+ for input pads; bool from 13+; wrap from 19+ | ORT docs/source: f64/f32/i32/i64/i8/u8/u32/u64; bool from 13+ | pads `2r`, constant scalar `1`, axes optional | output `E*b(T)` plus input if generated | Final border/frame; final toroidal pad; crop-to-canvas output | Padded intermediate often outprices stored bool/u8 mask | [PRIMARY][LOCAL][DERIVED] |
| `Concat` assemble pieces | 1+ | Broad: bool, f16, f32, ints, string | sum piece params | output `E*b(T)` | Build shape vectors, small rows, or tiny seeds | Full data `Concat` is byte-priced | [PRIMARY][DERIVED] |
| `Reshape`/`Unsqueeze`/`Squeeze`/`Transpose` | varies; 13+ input axes for Unsqueeze/Squeeze | Broad tensor support per ORT docs | shape/axes params, often small | output `E*b(T)` if materialized | Make broadcastable row/col/channel carriers | Treat as connectors, not free generators | [PRIMARY][DERIVED] |

## 3. Idiom Catalog

### Uniform fill

Recipe:

```
shape = Constant initializer INT64 [1,10,30,30]     # 4 params
value = scalar initializer T [v]                    # 1 param
Y = ConstantOfShape(shape, value=value)             # shape [1,10,30,30]
```

Cost for `[1,10,30,30]`:

- Stored dense fill: `9000` params.
- Generated `BOOL`/`UINT8` intermediate: `5 params + 9000 bytes = 9005`; loses by `5`.
- Generated `FLOAT16` intermediate: `5 + 18000`; loses badly.
- Generated `FLOAT` intermediate: `5 + 36000`; loses badly.
- Generated final output: `5 + 0 = 5`; wins by `8995`.

Use it for final all-zero/all-one/all-color outputs, or for small carriers such as `[1,1,30,1]`, `[1,1,1,30]`, `[30,30]`, not for full same-shape intermediates. [PRIMARY][DERIVED]

### Scalar broadcast

Recipe:

```
scalar = Constant([v])                  # 1 param
shape  = INT64 target shape             # r params
Y = Expand(scalar, shape)
```

Cost is identical in spirit to `ConstantOfShape`: `1+r` params plus `E*b(T)` if `Y` is an intermediate. Prefer relying on ONNX multidirectional broadcasting directly:

```
Where(mask, scalar_color, input)
Equal(input, channel_ids[1,10,1,1])
```

This keeps the scalar or channel vector small and pushes the unavoidable full output into the actual consumer. [PRIMARY][DERIVED]

### Static row/column ramps

Best static recipe:

```
iota30 = Constant INT64 [0,1,...,29]       # 30 params
row = Reshape(iota30, [1,1,30,1])          # broadcast carrier
col = Reshape(iota30, [1,1,1,30])          # broadcast carrier
```

Cost:

- Stored full row-coordinate field `[1,10,30,30]`: `9000` params.
- Stored iota carrier: `30` params, plus at most small reshape output bytes if the scoring tool materializes the carrier. Even if `[1,1,30,1] INT64` is charged, it is `240` bytes, still far below `9000`.
- `Range(0,30,1)` carrier: `3 params + 240 bytes = 243`, which loses to a stored `iota30` initializer but still beats a stored full field.

Verdict: use stored iota for fixed 30; reserve `Range` for final ramp outputs or explicitly profiled dynamic lengths. [PRIMARY][DERIVED]

### One-hot and diagonal

Diagonal via `EyeLike`:

```
carrier = ConstantOfShape([30,30])     # f32 zero by default
diag_f32 = EyeLike(carrier, dtype=FLOAT, k=0)
```

Costs:

- Stored `[30,30]` diagonal mask: `900` params.
- Official ORT `EyeLike FLOAT` intermediate: `2 params + 3600 bytes`; loses.
- `EyeLike` final output: `2` params; wins.
- `EyeLike` cannot be relied on for `BOOL` or `UINT8` on official ORT CPU. [PRIMARY]

Bool diagonal alternative:

```
i = iota30 reshaped [30,1]
j = iota30 reshaped [1,30]
diag_bool = Equal(i, j)
```

This is useful when the `Equal` output is final, or when the `[30,30]` diagonal replaces a full repeated-channel `[1,10,30,30]` mask. It is not a win just to replace a stored `[30,30]` bool initializer with a `[30,30]` bool intermediate. [PRIMARY][DERIVED]

### Triangular masks

Direct data triangular retention:

```
Y = Trilu(X, upper=0)    # lower triangular over the last two axes
```

Cost:

- Params: `0` for default `k=0`, or `1` if `k` scalar is supplied.
- Intermediate bytes: output bytes for `Y`, which are usually the wanted result.
- Official ORT CPU supports `BOOL`, `FLOAT`, `DOUBLE`, `INT64`; no official `UINT8`. [PRIMARY]

Generated bool triangular mask:

```
ones = ConstantOfShape([30,30], value=True)    # 3 params
tri = Trilu(ones, upper=0)                     # [30,30] bool
```

Cost if `tri` is intermediate: `3 params + 900 bytes` plus the `ones` output if counted separately. Against a stored `[30,30]` bool mask (`900` params), this loses by at least `3`; against a stored repeated `[1,10,30,30]` mask (`9000` params), it wins if you keep it 2D and broadcast. [DERIVED]

### Periodic tiling

Checkerboard final output:

```
seed = Constant UINT8 [[0,1],[1,0]]        # 4 params
repeats = Constant INT64 [15,15]           # 2 params
Y = Tile(seed, repeats)                    # [30,30]
```

Cost:

- Stored checkerboard `[30,30]`: `900` params.
- Generated final output: `6` params; wins by `894`.
- Generated intermediate: `6 + 900 bytes = 906`; loses by `6` vs stored `UINT8`/`BOOL`, and loses more for wider dtypes.

For non-integer scale motifs, use `Resize(nearest)` with `UINT8`/`INT8`/`INT32`/`FLOAT` only. [PRIMARY][DERIVED]

### Border/frame masks

Coordinate comparison final frame:

```
row = iota30 -> [30,1]
col = iota30 -> [1,30]
top    = Equal(row, 0)
bottom = Equal(row, 29)
left   = Equal(col, 0)
right  = Equal(col, 29)
frame  = Or(Or(top,bottom), Or(left,right))
```

Cost:

- Stored `[30,30]` frame: `900` params.
- Final frame output: mostly iota params and small comparison carriers; can win.
- Intermediate full frame: output bytes count and often erase the win.

`Pad` frame alternative:

```
interior = ConstantOfShape([28,28], value=False)
frame = Pad(interior, pads=[1,1,1,1], constant_value=True)
```

If `frame` is final: `interior` costs `784` bool bytes plus about `8` params, so it can beat a stored `900` mask by roughly `108`. If `frame` is intermediate, add another `900` bytes and it loses. [PRIMARY][DERIVED]

### Checkerboards and parity

Prefer tiny seed tiling over arithmetic parity for fixed checkerboards. Arithmetic parity:

```
parity = Equal(Mod(Add(row, col), 2), 0)
```

This is expressive but can materialize full `INT32`/`INT64` intermediates (`Add`, `Mod`) before the final bool. For a fixed checkerboard, a `2x2` `Tile` final output is cheaper and simpler. Use arithmetic parity only when parity is combined with other dynamic row/col conditions and those full intermediates already exist. [PRIMARY][DERIVED]

### Coordinate grids

Avoid materializing `[H,W]` coordinate grids. Keep row and column coordinates factored:

```
row = iotaH -> [1,1,H,1]
col = iotaW -> [1,1,1,W]
mask = Less(row, threshold_row)     # broadcasts
```

For fixed `H=W=30`, stored iota vectors beat `Range`. For dynamic crop sizes, `Range` can be used but should be flagged for local dynamic-shape profiling and static-shape inference checks. [PRIMARY][PROMPT][DERIVED]

### Small lookup tables

Small LUTs are usually best as initializers:

```
lut = Constant UINT8 or INT64 [K]       # K params
Y = Gather(lut, small_index)
```

Do not generate a LUT with `Concat`, `Range`, or `OneHot` unless it is final or unless the LUT is reused enough to amortize its intermediate bytes. If a pattern needs a large initializer to define it, it is not a generator. [PROMPT][DERIVED]

### Shape-vector construction

Use literal shape vectors for fixed targets:

```
shape_full = Constant INT64 [1,10,30,30]      # 4 params
```

Use `Shape`/`Gather`/`Concat` only when a graph truly has a runtime dimension. For this project's fixed common shapes, `Shape` chains add dynamic-surface risk and usually do not reduce params meaningfully. [PRIMARY][PROMPT][DERIVED]

## 4. Dead and Trap List

1. **Full generated intermediate as a dense-constant replacement**. `ConstantOfShape([1,10,30,30], UINT8)` costs `9005` as an intermediate versus `9000` for a stored dense constant. The final-output exception is the whole trick. [PROMPT][DERIVED]

2. **`Range` for fixed `[0..29]` ramps**. `Range(INT64)` costs `3 + 30*8 = 243`; a stored iota initializer costs `30`. Even `Range(FLOAT)` costs `3 + 120 = 123`. Use stored iota for fixed axes. [PRIMARY][DERIVED]

3. **`EyeLike` as a bool mask generator**. ONNX schema allows bool/u8, but official ORT 1.24.4 CPU source/docs list only f32/f64/i32/i64/u64. Local `BOOL` and `UINT8` probes failed. Use `Trilu` for bool triangular masks or coordinate `Equal` for bool diagonals. [PRIMARY][LOCAL]

4. **`Trilu(UINT8)`**. Official ORT CPU source supports bool/f32/f64/i64, and local `UINT8` probe failed. Cast to bool/f32 only if the cast/output bytes are justified. [PRIMARY][LOCAL]

5. **`OneHot(BOOL/UINT8)`**. Official ORT CPU source has no bool/u8 values/output registrations; local probes failed. Use `FLOAT`/`INT64` only per official support, or use `Equal(label, channel_ids)` for bool channel masks. [PRIMARY][LOCAL]

6. **Full `OneHot` intermediate**. A `[1,30,30,10] FLOAT` one-hot costs `36,000` bytes plus params. A stored dense one-hot constant would be `9000` params. Direct final `OneHot` can win; intermediate `OneHot` is usually a trap. [PRIMARY][DERIVED]

7. **`Expand` or `Tile` to full shape before the real consumer**. If the consumer can broadcast, do not explicitly expand. If it cannot, price the full output bytes; for bool/u8 same-shape constants, explicit expansion loses by overhead. [PRIMARY][DERIVED]

8. **`Pad` frame as an intermediate**. `Pad` is good for final outputs and for replacing other spatial plumbing, but a padded bool `[30,30]` intermediate adds `900` bytes. Small-interior plus final `Pad` can win; small-interior plus intermediate `Pad` usually loses. [PRIMARY][DERIVED]

9. **`Pad(mode="wrap")` enthusiasm**. `wrap` is available in ONNX `Pad` and local ORT accepted it at opset 19, but wrapped padded output bytes still count. Use it for final toroidal outputs or when it deletes heavier Slice/Concat emulation, not as a free periodic mask. [PRIMARY][LOCAL][DERIVED]

10. **`Resize(nearest)` dtype gaps**. ORT CPU supports f32/i32/i8/u8 for `Resize`, not bool/f16 per generated kernel docs. For bool motifs, use u8 and compare/cast only if needed. [PRIMARY]

11. **`DepthToSpace`/`SpaceToDepth` dtype asymmetry**. `DepthToSpace` supports u8 at opset 11+ on CPU; `SpaceToDepth` is f32/f64 only per generated kernel docs. Do not assume the pair is symmetric for byte-tight u8 layouts. [PRIMARY]

12. **Shape-driven dynamic output surfaces**. `Shape`, `Size`, `Range`, `Resize(sizes)`, `Pad(pads)`, and `ConstantOfShape` driven by computed values can be semantically correct but interact badly with static shape inference and runtime profiling. Keep fixed NeuroGolf shapes as initializers unless the dynamic behavior is deliberate and locally profiled. [PROMPT][PRIMARY][DERIVED]

13. **Default `ConstantOfShape` dtype**. Omitting `value` produces float32 zero. That can silently create a 4-byte-per-element intermediate or a downstream dtype mismatch. Always set a one-element `value` when dtype matters. [PRIMARY]

14. **Large initializer seeds**. If the seed or LUT is already large enough to encode the pattern, it is not a generator. Price the seed params plus every materialized output. [PROMPT][DERIVED]

## 5. Table B

| target artifact | incumbent stored constant cost | generated construction | generated params | generated intermediates | net verdict |
|---|---:|---|---:|---:|---|
| Final `[1,10,30,30]` uniform `UINT8`/`BOOL` fill | `9000` params | `ConstantOfShape([1,10,30,30], value)` as graph output | `5` | `0` | Strong win: `5` vs `9000` |
| Intermediate `[1,10,30,30]` uniform `UINT8`/`BOOL` fill | `9000` params | same `ConstantOfShape`, consumed by another op | `5` | `9000 B` | Loses slightly: `9005` vs `9000` |
| Intermediate `[1,10,30,30]` uniform `FLOAT` fill | `9000` params | `ConstantOfShape(..., FLOAT)` | `5` | `36000 B` | Dead: `36005` vs `9000` |
| Static row coordinate full field `[1,10,30,30]` | `9000` params | stored `iota30` reshaped/broadcast as `[1,1,1,30]` | `30 + shape/axes` | `<=240 B` if carrier materialized; no full field | Strong win if no explicit full `Expand` |
| Static row ramp `[30] INT64` | `30` params | `Range(0,30,1)` | `3` | `240 B` | Loses: `243` vs `30`; use initializer |
| Final `[30,30]` checkerboard | `900` params | `Tile(seed[2,2], repeats[15,15])` as output | `6` | `0` | Strong win |
| Intermediate `[30,30]` checkerboard `UINT8` | `900` params | same `Tile`, consumed later | `6` | `900 B` | Loses slightly: `906` vs `900` |
| Final `[30,30]` frame mask | `900` params | coordinate comparisons, or `Pad(ConstantOfShape([28,28], false), true)` | about `8` plus iota if coordinate path | coordinate path small; Pad path `784 B` interior | Win if final; profile exact path |
| Intermediate `[30,30]` frame mask | `900` params | `Pad` interior or coordinate OR chain | `8+` | at least full `900 B`, often more | Usually loses unless it replaces a larger repeated-channel mask |
| Final one-hot `[1,10,30,30]` from index grid | `9000` params for dense one-hot | `OneHot(index_grid, depth=10, values=[0,1], axis=1)` as output | `3 + index source` | `0` for final output | Strong win if index grid is already compact |
| Intermediate one-hot `[1,10,30,30] FLOAT` | `9000` params | same `OneHot`, consumed later | `3 + index source` | `36000 B` | Dead unless it deletes even larger work |
| `[30,30]` diagonal bool mask | `900` params | `Equal(iota_row, iota_col)` or `EyeLike` numeric then compare | `60+` for two iotas if not shared | full bool output `900 B` if intermediate | Final output wins; intermediate loses vs stored `[30,30]`, but wins vs repeated `[1,10,30,30]` incumbent |
| Lower triangle applied to existing data | stored triangle mask plus `Where/Mul`: `900` or `9000` params | `Trilu(existing_tensor, upper=0)` | `0` or `1` for k | output bytes only | Strong when semantics are triangular retention, not mask materialization |
| Final toroidal padded tile | stored wrapped output params | `Pad(mode="wrap")` as final output | pads `2r` | source seed/crop only | Good if source seed is small and output final |

## 6. Top-10 Adoption Table

| rank | idiom | likely payoff | adoption guidance |
|---:|---|---|---|
| 1 | Final-output `ConstantOfShape` uniform fills | Very high | Add as a first-class pattern: full fill final output costs shape+scalar params only |
| 2 | Direct final `OneHot` from compact index grid | Very high | Use when an index canvas already exists; avoid intermediate one-hot |
| 3 | Fixed axis iota carriers instead of full coordinate fields | Very high | Store `[0..29]` once; reshape to row/col carriers; avoid `Range` for fixed axes |
| 4 | `Trilu` directly on existing data | High | Use for triangular retention without separate mask materialization |
| 5 | 2D structural masks broadcast across channels | High when incumbent is repeated-channel | Keep `[30,30]` or `[1,1,30,30]` mask factored; do not expand to `[1,10,30,30]` |
| 6 | Final checkerboards/periodic motifs via `Tile` | High | Tiny seed plus repeats wins only when output is final or full output already unavoidable |
| 7 | `Pad(mode="wrap")` for final toroidal/circular outputs | Medium-high | Opset 19+; use when it deletes Slice/Concat wrap emulation |
| 8 | Final/broadcast frame masks via coordinate comparisons | Medium | Prefer coordinate carriers if final; beware full OR intermediates |
| 9 | `DepthToSpace` u8 block-stamp layouts | Medium | Useful for block/interleave motifs; channel divisibility is the hard constraint |
| 10 | `CumSum` prefix/rank labels | Situational | Adopt for replacing chains, not for replacing constants; no bool/u8 CPU kernel |

## Verification Checklist

- [x] Every listed generator has ONNX spec and ORT CPU support citations in the source list.
- [x] Every idiom has param and intermediate-byte arithmetic or a formula.
- [x] Uniform fills, ramps, diagonal/triangular masks, tiling, frames, checkerboards, and coordinate grids are covered.
- [x] Dead/trap patterns are explicitly priced.
- [x] Three strongest additions for `op_cost_patterns.md` or `techniques.md` are named below.

## Final Blunt Verdict

Add these three to `Onnx_Advanced/op_cost_patterns.md` or `techniques.md`:

1. **Final-output generators are the real zero-param win**: `ConstantOfShape`, `Tile`, `Pad`, and `OneHot` can replace thousands of stored params only when the generated full tensor is the named graph output or otherwise not byte-charged. [PROMPT][DERIVED]
2. **Small broadcast carriers beat full constants**: fixed `[30]` iotas, `[1,1,30,1]`/`[1,1,1,30]` coordinate carriers, `[30,30]` structural masks, and channel id vectors are the safest param-side shelf. Do not `Range` or `Expand` them to full shape unless final. [DERIVED]
3. **Structural ops are for avoiding masks, not making full intermediates**: `Trilu(existing_tensor)` is strong; `EyeLike`/`Trilu` generated masks are strong only as final outputs or when replacing a larger repeated-channel artifact. `OneHot` is strong as direct final output and dangerous as a full intermediate. [PRIMARY][DERIVED]

