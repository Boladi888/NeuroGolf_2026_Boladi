# N1 - Opset 23-26 Delta Audit on ONNX Runtime 1.24.4 CPU

Assumptions: local target is `onnx==1.21.0`, `onnxruntime==1.24.4`, `CPUExecutionProvider`, graph optimizations disabled, and the card's validator/scorer constraints apply. `[PROMPT]`

Source key used below:

- ONNX changelog/spec: official ONNX operator docs and generated changelog for default-domain opset 23-26. [ONNX changelog](https://raw.githubusercontent.com/onnx/onnx/main/docs/Changelog.md), [ONNX operators index](https://onnx.ai/onnx/operators/). `[PRIMARY]`
- ORT kernels: generated ONNX Runtime 1.24.4 `OperatorKernels.md`, CPUExecutionProvider section. The file states it is generated from registered kernels. [ORT v1.24.4 OperatorKernels.md](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.24.4/docs/OperatorKernels.md). `[PRIMARY]`
- ORT opset gate: ORT validates imported opsets against the schema registry's released opset versions during model load. [model_load_utils.h](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/graph/model_load_utils.h) and [model.cc](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/graph/model.cc). `[PRIMARY]`
- Local sanity probes: in-memory `onnx==1.21.0` / `onnxruntime==1.24.4` CPU probes with optimizations disabled for `Constant23`, `CastLike23`, `Swish24`, `TensorScatter24`, `Attention23`, `RMSNormalization23`, and opset-26 load. `[DERIVED]`

## 1. Delta Map and ORT CPU Support

Dtype shorthand:

- `num-core`: `f64,f32,i32,i64`.
- `arc-core`: `bool,u8,i8,i32,f16,f32`.
- `common14`: `bf16,bool,f64,f32,f16,i8,i16,i32,i64,u8,u16,u32,u64,string`.
- `shape8`: `common14 + float8e4m3fn/fnuz/e5m2/e5m2fnuz`.
- `cast23`: ONNX schema permits `common14 + float8x4 + i4/u4 + float4e2m1` except complex; ORT direct Cast row exists.
- `cast24`: `cast23 + float8e8m0`.
- `cast25`: `cast24 + i2/u2`.
- `transpose23_24`: `shape8 + i4/u4`.
- `transpose25`: `transpose23_24 + i2/u2`.
- `pad-cpu`: ORT CPU Pad set is `bool,f64,f32,i8,i32,i64,u8,u32,u64`; no `f16`, `u16`, `i16`, strings, complex, or sub-byte dtypes.
- `qlinear23_24`: ORT CPU Q/DQ supports `x/y` around `f32,f16` with quantized `i8/u8/i16/u16/i4/u4,float8x4` as applicable; schema adds `float4e2m1`, but ORT CPU does not list it.
- `qlinear25`: `qlinear23_24 + i2/u2` for quantized tensors.

Table A:

| op | opset change | semantic delta | ORT 1.24.4 CPU kernel? | supported dtypes | scorer risk | source tags |
|---|---:|---|---|---|---|---|
| Attention | added in 23 | Fused scaled-dot-product attention over Q/K/V with optional mask, cache, and optional qk output. | Yes, direct rows for 23 and 24+. | Q/K/V `f32,f16`; mask `bool,f32,f16`. | Float-only; optional multi-output forms risky, but single `Y` output works in local probe. | `[PRIMARY][DERIVED]` ONNX Attention spec; ORT kernels. |
| Cast | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; Cast semantics unchanged. | Yes, direct row for 23. | Effective schema `cast23`; ORT row is broad but downstream kernels rarely consume new sub-byte/float4 types. | No param saving because params are dtype-blind; only useful if it moves a conversion to free final output or shrinks later intermediates. | `[PRIMARY][DERIVED]` ONNX Cast spec; ORT kernels; `[PROMPT]`. |
| CastLike | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; casts input to target input dtype. | No direct CPU kernel row; ONNX page marks it as a function/context-dependent function, and local ORT expands/loads it. | Schema `cast23`; not a registered direct CPU kernel. | Trap under the card's kernel rule; no advantage over `Cast` if target dtype is statically known. | `[PRIMARY][DERIVED]` ONNX CastLike spec; ORT kernels absence; local probe. |
| Constant | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; emits one attribute tensor/scalar. | Special graph constant handling; no generated CPU kernel row; local ORT loads `Constant23`. | Schema all tensor types incl. complex/sub-byte/string. | Constant tensor elements are charged as params; dtype does not reduce params. | `[PRIMARY][DERIVED][PROMPT]` ONNX Constant spec; ORT kernels absence; local probe. |
| ConstantOfShape | 21 -> 23 | Type-constraint refresh: adds `float4e2m1` to schema. | Yes, direct row for 23. | ORT CPU supports numeric/bool common types plus float8x4, but not `i4/u4/i2/u2/float4e2m1/float8e8m0`. | Old zero-param generator; v23 adds no practical new scorer benefit. | `[PRIMARY][DERIVED]` ONNX ConstantOfShape spec; ORT kernels. |
| DequantizeLinear | 21 -> 23 | Adds separate output type parameter/`output_dtype`; schema adds `float4e2m1` input. | Yes, direct row for 23. | `qlinear23_24`; output `f32/f16` in ORT row. | Scales/zero-points are params; dtype-blind params kill "narrow weights are cheaper" arguments. | `[PRIMARY][DERIVED][PROMPT]` ONNX DQ spec; ORT kernels. |
| Flatten | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; shape semantics unchanged. | Yes, direct row for 23. | `common14`; ORT CPU row does not include the new float4/sub-byte schema types. | Shape-only; adds a charged output unless final. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Identity | 21 -> 23 | Type-constraint refresh incl. sequence/optional/sub-byte schema types. | Yes, direct row for 23. | Broad tensor/sequence/optional set, but no cost reason to insert it. | Usually increases charged intermediates; sequence/optional forms scorer-risky. | `[PRIMARY][DERIVED]` ONNX Identity spec; ORT kernels. |
| If | 21 -> 23 | Type-constraint refresh for branch outputs. | Yes, direct row for 23. | Broad tensors/sequences/optionals. | Validator/subgraph banned in practice. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Loop | 21 -> 23 | Type-constraint refresh for loop-carried values. | Yes, direct row for 23. | Broad tensors/sequences/optionals. | Explicitly banned. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Pad | 21 -> 23 | Type-constraint refresh; Pad modes/axes behavior unchanged from recent versions. | Yes, direct row for 23. | `pad-cpu`. | Old op; no new dtype that helps ARC scoring. | `[PRIMARY][DERIVED]` ONNX Pad spec; ORT kernels. |
| QuantizeLinear | 21 -> 23 | Type split around scale/zero-point; schema includes 4-bit/float8/float4-era quantized types. | Yes, direct row for 23. | `qlinear23_24`. | Adds scale/zero-point params; dtype-blind params; useful only if downstream quantized op already exists. | `[PRIMARY][DERIVED][PROMPT]` ONNX Q spec; ORT kernels. |
| RMSNormalization | added in 23 | Root-mean-square normalization over an axis with a scale tensor. | Yes, direct row 23+. | `f64,f32,f16` for X/scale/output. | Float-only, same-shape output, scale param; no ARC exact-grid mechanism. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels; local probe. |
| Reshape | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; shape semantics unchanged. | Yes, direct row for 23. | `shape8`; shape input `i64`. | Shape-only; old op; charged output unless final. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| RotaryEmbedding | added in 23 | Applies rotary embedding using cosine/sine caches and optional position ids. | Yes, direct row 23+. | ORT CPU `f32,f16`; position ids `i64`. | Needs cos/sin cache params, float-only, LLM-specific. | `[PRIMARY][DERIVED]` ONNX RotaryEmbedding spec; ORT kernels; local probe. |
| Scan | 21 -> 23 | Type-constraint refresh for subgraph scan values. | Yes, direct row for 23. | Broad tensor set incl. float8x4. | Explicitly banned and subgraph-based. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Shape | 21 -> 23 | Type-constraint refresh: input may be newer tensor types; output remains `i64`. | Yes, direct row for 23. | `shape8` input; output `i64`. | Usually shape plumbing; dynamic/symbolic shape risk if not constant-folded. | `[PRIMARY][DERIVED][PROMPT]` ONNX changelog; ORT kernels. |
| Size | 21 -> 23 | Type-constraint refresh; output remains scalar `i64`. | Yes, direct row for 23. | CPU input `bool,f64,f32,i8/i16/i32/i64,u8/u16/u32/u64,string`; output `i64`. | Shape scalar intermediate, normally not cost-reducing. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Squeeze | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; axes input unchanged. | Yes, direct row for 23. | `common14`. | Shape-only; old op. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Transpose | 21 -> 23 | Type-constraint refresh: adds newer low-bit/float8-ish schema types. | Yes, direct row for 23. | `transpose23_24`. | Shape-only; old op; low-bit tensors not broadly consumable. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Unsqueeze | 21 -> 23 | Type-constraint refresh: adds `float4e2m1`; axes input unchanged. | Yes, direct row for 23. | `common14`. | Shape-only; old op. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Attention | 23 -> 24 | Adds optional `nonpad_kv_seqlen` input for external KV-cache update / skipping padded tokens. | Yes, direct row 24+. | Q/K/V `f32,f16`; mask `bool,f32,f16`; `nonpad_kv_seqlen` `i64`. | Same float-only attention caveats; extra input is LLM-cache specific. | `[PRIMARY][DERIVED]` ONNX Attention spec; ORT kernels. |
| Cast | 23 -> 24 | Adds `round_mode` for `float8e8m0`; schema adds `float8e8m0`. | Yes, direct row for 24. | `cast24`. | `float8e8m0` is not useful unless later kernels consume it; params remain dtype-blind. | `[PRIMARY][DERIVED][PROMPT]` ONNX Cast spec; ORT kernels. |
| CastLike | 23 -> 24 | Adds `round_mode`; schema adds `float8e8m0`. | No direct CPU kernel row; function/context-dependent expansion only. | Schema `cast24`. | Direct-kernel trap; no static advantage over `Cast`. | `[PRIMARY][DERIVED]` ONNX CastLike spec; ORT kernels absence. |
| Constant | 23 -> 24 | Type-constraint refresh: adds `float8e8m0`. | Special graph constant handling; no generated CPU kernel row. | Schema all tensor types through IRv12. | Params charged by element count regardless of dtype. | `[PRIMARY][DERIVED][PROMPT]` ONNX Constant spec; ORT kernels absence. |
| ConstantOfShape | 23 -> 24 | Type-constraint refresh: adds `float8e8m0`. | Yes, direct row for 24. | ORT row still omits `float8e8m0`, `float4e2m1`, and sub-byte ints. | Existing generator; no new ORT-backed dtype win. | `[PRIMARY][DERIVED]` ONNX ConstantOfShape spec; ORT kernels. |
| DequantizeLinear | 23 -> 24 | Scale type adds `float8e8m0` in schema. | Yes, direct row for 24. | ORT scale/output `f32/f16`; quantized inputs as `qlinear23_24`. | No param shrink; scale dtype extension not useful on CPU row. | `[PRIMARY][DERIVED][PROMPT]` ONNX DQ spec; ORT kernels. |
| Flatten | 23 -> 24 | Type-constraint refresh: adds `float8e8m0`. | Yes, direct row for 24. | `common14`; no ORT `float8e8m0` row. | Shape-only; no new saving. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Identity | 23 -> 24 | Type-constraint refresh incl. newer tensor/sequence types. | Yes, direct row for 24. | Broad tensor/sequence/optional set. | No-op output is usually extra memory. | `[PRIMARY][DERIVED]` ONNX Identity spec; ORT kernels. |
| If | 23 -> 24 | Type-constraint refresh for outputs. | Yes, direct row for 24. | Broad tensors/sequences/optionals. | Subgraph/control-flow rejected. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Loop | 23 -> 24 | Type-constraint refresh for loop-carried values. | Yes, direct row for 24. | Broad tensors/sequences/optionals. | Explicitly banned. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Pad | 23 -> 24 | Type-constraint refresh; Pad semantics unchanged. | Yes, direct row for 24. | `pad-cpu`. | No new practical dtype or shape behavior. | `[PRIMARY][DERIVED]` ONNX Pad spec; ORT kernels. |
| QuantizeLinear | 23 -> 24 | `y_scale` may differ from input type; schema includes `float8e8m0` scale/related typing. | Yes, direct row for 24. | `qlinear23_24`; ORT row does not make params cheaper. | Scale/zero-point overhead; downstream quantized kernels still decide value. | `[PRIMARY][DERIVED][PROMPT]` ONNX Q spec; ORT kernels. |
| Reshape | 23 -> 24 | Type-constraint refresh: adds `float8e8m0`. | Yes, direct row for 24. | `shape8`; no `float8e8m0` in ORT row. | Shape-only; old op. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Scan | 23 -> 24 | Type-constraint refresh for scan values. | Yes, direct row for 24. | Broad tensor set incl. float8x4. | Explicitly banned and subgraph-based. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Shape | 23 -> 24 | Type-constraint refresh: input may be `float8e8m0` in schema. | Yes, direct row for 24. | `shape8`; ORT row omits `float8e8m0`. | Shape scalar/vector output; no direct cost reduction. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Size | 23 -> 24 | Type-constraint refresh only. | Yes, direct row for 24. | Same as Size-23. | Scalar intermediate, not reducer. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| SplitToSequence | 11 -> 24 | Sequence op type refresh; output remains a sequence. | Yes, direct row for 24+. | Input CPU row mainly `bool,f64,f32,f16,i32,i64,string`; split `i32/i64`; output `seq(common14)`. | Sequence output is scorer/validator suspect. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Squeeze | 23 -> 24 | Type-constraint refresh: adds `float8e8m0`. | Yes, direct row for 24. | `common14`; no ORT `float8e8m0`. | Shape-only. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Swish | added in 24 | Function op: `x * sigmoid(alpha*x)`. | No direct CPU kernel row; ONNX marks function; local ORT loads expansion. | Schema `bf16,f16,f32,f64`; direct kernel none. | Not fused under registered-kernel rule; expansion materializes more ops with optimizations disabled. | `[PRIMARY][DERIVED]` ONNX Swish spec; ORT kernels absence; local probe. |
| TensorScatter | added in 24 | Writes an update tensor into a cache along an axis; `linear`/`circular` modes. | No CPU row; local ORT `NotImplemented` for opset 24. | Schema broad tensor set, but no ORT CPU kernel. | Looks useful for stamping/cache updates, but unusable on target stack. | `[PRIMARY][DERIVED]` ONNX TensorScatter spec; ORT kernels absence; local probe. |
| TopK | 11 -> 24 | Type-constraint refresh: schema adds unsigned/low precision numeric dtypes; semantics and two outputs unchanged. | Yes, direct row 24+. | CPU row only `f64,f32,i32,i64`; indices `i64`. | Multi-output suspect; no `u8/bool`; old op already known. | `[PRIMARY][DERIVED][PROMPT]` ONNX TopK spec; ORT kernels. |
| Transpose | 23 -> 24 | Type-constraint refresh: adds `float8e8m0` in schema. | Yes, direct row for 24. | `transpose23_24`; ORT row omits `float8e8m0`. | Shape-only; old op. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Unsqueeze | 23 -> 24 | Type-constraint refresh: adds `float8e8m0`. | Yes, direct row for 24. | `common14`; no ORT `float8e8m0`. | Shape-only. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Cast | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | `cast25`. | Low-bit tensors have poor downstream CPU coverage; params still dtype-blind. | `[PRIMARY][DERIVED][PROMPT]` ONNX Cast spec; ORT kernels. |
| CastLike | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | No direct CPU kernel row; function/context-dependent expansion only. | Schema `cast25`; direct kernel none. | Direct-kernel trap. | `[PRIMARY][DERIVED]` ONNX CastLike spec; ORT kernels absence. |
| Constant | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Special graph constant handling; no generated CPU row. | Schema all tensor types through IRv13. | Param count unchanged by dtype. | `[PRIMARY][DERIVED][PROMPT]` ONNX Constant spec; ORT kernels absence. |
| ConstantOfShape | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | ORT row omits `i2/u2`, `i4/u4`, `float4e2m1`, `float8e8m0`. | No new generator win on CPU. | `[PRIMARY][DERIVED]` ONNX ConstantOfShape spec; ORT kernels. |
| DequantizeLinear | 24 -> 25 | Type-constraint refresh: adds `i2/u2` quantized inputs. | Yes, direct row 25+. | `qlinear25`; output `f32/f16`. | Only helps if a downstream low-bit path exists; scale params remain. | `[PRIMARY][DERIVED][PROMPT]` ONNX DQ spec; ORT kernels. |
| Flatten | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | `common14`; no ORT `i2/u2`. | Shape-only; no new saving. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Identity | 24 -> 25 | Type-constraint refresh incl. `i2/u2` tensors. | Yes, direct row 25+. | Broad tensor/sequence/optional set, but practical tensor row omits many schema-only types. | Usually extra memory; optional/sequence forms risky. | `[PRIMARY][DERIVED]` ONNX Identity spec; ORT kernels. |
| If | 24 -> 25 | Type-constraint refresh incl. `i2/u2` in branch outputs. | Yes, direct row 25+. | Broad tensors/sequences/optionals. | Subgraph/control-flow rejected. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Loop | 24 -> 25 | Type-constraint refresh incl. `i2/u2`. | Yes, direct row 25+. | Broad tensors/sequences/optionals. | Explicitly banned. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Pad | 24 -> 25 | Type-constraint refresh; semantics unchanged. | Yes, direct row 25+. | `pad-cpu`. | No new practical dtype. | `[PRIMARY][DERIVED]` ONNX Pad spec; ORT kernels. |
| QuantizeLinear | 24 -> 25 | Adds `i2/u2` quantized outputs in schema/ORT row. | Yes, direct row 25+. | `qlinear25`. | New low-bit output alone does not shrink params and has weak downstream CPU support. | `[PRIMARY][DERIVED][PROMPT]` ONNX Q spec; ORT kernels. |
| Reshape | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | `shape8`; no ORT `i2/u2`. | Shape-only. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Scan | 24 -> 25 | Type-constraint refresh incl. low-bit schema types. | Yes, direct row 25+. | Broad tensor set. | Explicitly banned/subgraph. | `[PRIMARY][PROMPT]` ONNX changelog; ORT kernels. |
| Shape | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | `shape8`; output `i64`; no ORT `i2/u2`. | Shape-only; no cost win. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Size | 24 -> 25 | Type-constraint refresh only. | Yes, direct row 25+. | Same as Size-23/24. | Scalar output, not reducer. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Squeeze | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | `common14`; no ORT `i2/u2`. | Shape-only. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Transpose | 24 -> 25 | ORT/schema add `i2/u2` to transpose support. | Yes, direct row 25+. | `transpose25`. | Low-bit transpose alone is not useful without low-bit consumer kernels. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| Unsqueeze | 24 -> 25 | Type-constraint refresh: adds `i2/u2`. | Yes, direct row 25+. | `common14`; no ORT `i2/u2`. | Shape-only. | `[PRIMARY][DERIVED]` ONNX changelog; ORT kernels. |
| BitCast | added in 26 | Reinterprets bits without value conversion; same bit width, same shape. | No for target stack: ORT 1.24.4 rejects default-domain opset 26 before kernel lookup; no CPU row. | Schema broad non-string tensors, but unusable in ORT 1.24.4 default settings. | Would be interesting for no-copy reinterpretation, but unavailable. | `[PRIMARY][DERIVED]` ONNX BitCast spec; ORT opset gate; local probe. |
| CumProd | added in 26 | Cumulative product along an axis with `exclusive`/`reverse`. | No for target stack: ORT 1.24.4 rejects default-domain opset 26 before kernel lookup; no CPU row. | Schema `u32,u64,i32,i64,f16,f32,f64,bf16`, axis `i32/i64`; unusable. | Unavailable; even if available, product scan is niche and output same size. | `[PRIMARY][DERIVED]` ONNX CumProd spec; ORT opset gate; local probe. |

## 2. Cost-Reducer Shortlist

Only one genuinely new opset-23+ operator has a plausible NeuroGolf mechanism: `Attention`. `ConstantOfShape` and dtype conversion remain relevant carry-forward tools, but their opset 23-25 deltas do not materially improve the old toolkit.

Table B:

| op | cost mechanism | expected saving on [1,10,30,30] or smaller | caveat | local probe to run | source tags |
|---|---|---|---|---|---|
| Attention-23/24 | `fuse`, `delete-intermediate`, sometimes `free-output` | If a graph already does global attention over `S=H*W=900`, `E=10`: explicit `MatMul(Q,K^T)` emits `[1,900,900] f32` = 3,240,000 B, `Softmax` emits another 3,240,000 B, and optional mask-add emits another 3,240,000 B. A single-output `Attention` can hide those as kernel-internal temporaries, saving 6.48-9.72 MB of charged intermediates if `Y` is final. | Float-only; exact ARC semantics are not obvious; Q/K/V preparation may cost more than it saves unless that chain already exists. | Build opset-23 model: `Q,K,V:[1,900,10] f32`, `q_num_heads=1`, `kv_num_heads=1`, only output `Y:[1,900,10]`. Compare numerically to explicit `Transpose+MatMul+Softmax+MatMul`; pass if outputs match tolerance and scorer sees no qk/softmax value_info. | `[PRIMARY][DERIVED][SPECULATIVE]` ONNX Attention spec; ORT kernels; cost model. |
| ConstantOfShape-23/24/25 | `zero-param-generator`, `param-shrink` | Replacing a dense `[1,10,30,30]` constant tensor costs 9000 params. `ConstantOfShape` needs shape `[1,10,30,30]` = 4 initializer elements plus one scalar value = 5 params, saving 8995 params. Intermediate bytes are unchanged if the generated tensor is not final. | Not a new post-22 capability; ORT CPU does not support the new schema-only low-bit/float4 outputs. | Build opset-25 `ConstantOfShape` for `uint8` and `f32`; compare param count to dense `Constant`; pass if exact values and scorer charges only shape+scalar params plus normal output bytes. | `[PRIMARY][DERIVED]` ONNX ConstantOfShape spec; ORT kernels; cost model. |
| Cast-23/24/25 or QuantizeLinear-23/25 | `free-output`, limited `delete-intermediate` via dtype shrink | A full `[1,10,30,30]` f32 intermediate is 36,000 B; uint8/bool is 9,000 B. If one charged Cast/Quantize output enables two later full-tensor intermediates to be uint8 instead of f32, net memory can save about `2*(36,000-9,000) - 9,000 = 45,000 B`. If Cast is final output, the Cast output is free. | This is not new to opset 23-25 except low-bit schema refreshes. Params are dtype-blind, and low-bit tensors have poor downstream CPU kernel coverage. | Build f32 mask path versus `Cast(to=UINT8)` followed by two uint8-capable ops such as `Where`/bitwise or simple shape ops; pass only if all downstream kernels accept uint8 and scorer counts uint8 bytes. | `[PRIMARY][DERIVED][PROMPT]` ONNX Cast/Q specs; ORT kernels; cost model. |

## 3. Dead / Trap List

- `BitCast-26`: schema is attractive because it promises reinterpretation without conversion, but ORT 1.24.4 rejects default-domain opset 26 models before node assignment. Blocker: ORT opset gate, then no CPU row. `[PRIMARY][DERIVED]`
- `CumProd-26`: same opset-26 load blocker. Even if future ORT supports it, cumulative product is same-size and niche for small color grids. Blocker: ORT opset gate. `[PRIMARY][DERIVED]`
- `TensorScatter-24`: semantically useful for in-place-looking cache/stamp updates, but ORT 1.24.4 CPU has no generated kernel row; local probe failed `NOT_IMPLEMENTED`. Blocker: no CPU kernel. `[PRIMARY][DERIVED]`
- `Swish-24`: ONNX marks it as a function and the function body is `Constant -> CastLike -> Mul -> Sigmoid -> Mul`; no direct CPU kernel row. With optimizations disabled, this is not a fused cost reducer. Blocker: no registered direct kernel and float-only semantics. `[PRIMARY][DERIVED]`
- `CastLike-23/24/25`: no direct CPU kernel row; local ORT can expand it, but a static `Cast` is cheaper and clearer when target dtype is known. Blocker: direct-kernel trap. `[PRIMARY][DERIVED]`
- `If-23/24/25`, `Loop-23/24/25`, `Scan-23/24/25`: ORT rows exist, but the card bans Loop/Scan and rejects subgraph/control-flow attributes in practice. Blocker: validator/subgraph. `[PRIMARY][PROMPT]`
- `SplitToSequence-24`: CPU row exists, but output is `seq(tensor(...))`; sequence outputs are scorer/validator suspect. Blocker: sequence output. `[PRIMARY][PROMPT]`
- `TopK-24`: CPU kernel exists only for `f64,f32,i32,i64` plus `i64` indices, and it always has two outputs. Blockers: multi-output scorer risk and no uint8/bool support. `[PRIMARY][DERIVED][PROMPT]`
- `RMSNormalization-23`: direct CPU kernel exists, but it is float-only, needs a scale parameter, and emits a same-shape tensor. Blocker: cost and dtype impractical for exact ARC grids. `[PRIMARY][DERIVED]`
- `RotaryEmbedding-23`: direct CPU kernel exists for `f32/f16`, but requires cos/sin caches and is LLM-position encoding, not a compact discrete-grid primitive. Blocker: parameter burden and dtype/semantics. `[PRIMARY][DERIVED]`
- `Constant-23/24/25`: local ORT accepts it through graph constant handling, but no generated CPU kernel row. It is also explicitly charged as params by the scoring model. Blocker: not a reducer; dtype-blind params. `[PRIMARY][DERIVED][PROMPT]`
- `Identity-23/24/25`: direct kernel exists, but it copies/adds an output and can introduce optional/sequence traps. Blocker: cost-negative no-op. `[PRIMARY][DERIVED]`
- `Flatten`, `Reshape`, `Shape`, `Size`, `Squeeze`, `Unsqueeze`, `Transpose`, `Pad`: all are old structural tools. The opset 23-25 deltas are almost entirely type-constraint refreshes, and ORT CPU often omits the new schema-only low-bit/float4 types. Blocker: no new cost mechanism. `[PRIMARY][DERIVED]`
- `QuantizeLinear` / `DequantizeLinear` changes: CPU kernels exist, including some 4-bit and 2-bit Q/DQ rows by opset 25, but parameter elements are dtype-blind and scale/zero-point tensors add params. Blocker: no param-side saving unless a downstream quantized compute op deletes larger intermediates. `[PRIMARY][DERIVED][PROMPT]`

## 4. Top Probes

1. `Attention` fused global attention probe.

   - Shape/dtype/opset: `Q,K,V:[1,900,10] f32`, opset 23, attrs `q_num_heads=1`, `kv_num_heads=1`, output `Y:[1,900,10]`.
   - Before graph: `Transpose(K) -> MatMul(Q,Kt) -> Softmax -> MatMul(...,V)`. Charged intermediates include at least two `[1,900,900] f32` tensors = 6,480,000 B; if mask/add is present, 9,720,000 B.
   - After graph: one `Attention` node with only `Y` as graph output; internal qk/softmax tensors are not node outputs and should be uncharged by the prompt's cost model.
   - Pass/fail: pass if ORT CPU output matches explicit chain within float tolerance and scorer/profiler charges no qk/softmax intermediate. Fail if validator rejects optional-input gaps or if Q/K/V construction dominates cost.
   - Sources: `[PRIMARY][DERIVED][SPECULATIVE]` ONNX Attention spec; ORT kernels; cost model.

2. `ConstantOfShape` dense-constant replacement probe.

   - Shape/dtype/opset: shape initializer `int64[4]=[1,10,30,30]`, scalar value attribute `uint8{0}` or `float{0}`, opset 23 or 25.
   - Before graph: dense `Constant` or initializer `[1,10,30,30]` = 9000 charged params.
   - After graph: 4 shape params + 1 scalar value param = 5 params; expected `param-shrink` = 8995. Output memory is unchanged if intermediate and free if final.
   - Pass/fail: pass if output exact and scorer charges 5 params, not 9000. Fail if validator/scorer treats shape/value attributes differently.
   - Sources: `[PRIMARY][DERIVED]` ONNX ConstantOfShape spec; ORT kernels; cost model.

3. `Cast`/`QuantizeLinear` dtype-shrink path probe.

   - Shape/dtype/opset: input `[1,10,30,30] f32`; cast or quantize to `uint8`; use only downstream CPU kernels proven to accept `uint8`.
   - Before graph: two full f32 intermediates = 72,000 B.
   - After graph: one charged `uint8` conversion output plus two `uint8` intermediates = 27,000 B if all are materialized; expected saving 45,000 B. If conversion is final output, conversion output is free.
   - Pass/fail: pass only if downstream kernels run as `uint8` and scorer uses 1 byte/element. Fail if an immediate `DequantizeLinear` or Cast back to f32 is needed.
   - Sources: `[PRIMARY][DERIVED][PROMPT]` ONNX Cast/Q/DQ specs; ORT kernels; cost model.

4. `TensorScatter` negative probe.

   - Shape/dtype/opset: `past_cache:[1,10,30,30] uint8`, `update:[1,10,1,30] uint8`, `write_indices:[1] int64`, opset 24.
   - Expected delta if it existed: could replace slice/concat/stamp chains with one same-shape output, saving charged slice/concat intermediates.
   - Pass/fail: expected fail on ORT 1.24.4 CPU with no implementation. If a future ORT starts passing, rerun cost arithmetic.
   - Sources: `[PRIMARY][DERIVED]` ONNX TensorScatter spec; ORT kernels absence; local probe.

5. `BitCast` future-only probe.

   - Shape/dtype/opset: `[1,10,30,30] f32 -> i32`, opset 26.
   - Expected delta if it existed: could avoid a value-converting `Cast` and preserve bytes/shape, but does not reduce bytes by itself.
   - Pass/fail: expected fail on ORT 1.24.4 because opset 26 is rejected. Revisit only when ORT officially supports opset 26 and the CPU kernel matrix lists `BitCast`.
   - Sources: `[PRIMARY][DERIVED]` ONNX BitCast spec; ORT opset gate; local probe.

## Verification Checklist

- [x] Opset 23, 24, 25, and 26 deltas covered.
- [x] Opset <=22 not re-audited except where a newer version changed an old op.
- [x] Every candidate has ONNX spec/changelog citation and ORT 1.24.4 CPU support status.
- [x] Every cost-reducer claim names a scoring mechanism and includes byte/param arithmetic.
- [x] Every dead/trap claim states the blocker: validator, schema/function expansion, CPU kernel absence, dtype, dynamic/control shape, sequence/multi-output, or cost.

## Final Verdict

The opset 23-26 surface is barren for normal NeuroGolf work. The only new direct-kernel candidate worth a real probe is `Attention-23/24`, and only for a graph that already expresses a large `MatMul -> Softmax -> MatMul` attention-like chain where hidden kernel temporaries can delete megabytes of charged qk/softmax intermediates. For ordinary small exact ARC color-grid programs, the practical answer is still the opset <=22 toolkit plus old `ConstantOfShape`/dtype-cast tricks; opset 26 is unavailable in ORT 1.24.4, `TensorScatter` has no CPU kernel, and the low-bit dtype churn does not beat dtype-blind parameter scoring.
