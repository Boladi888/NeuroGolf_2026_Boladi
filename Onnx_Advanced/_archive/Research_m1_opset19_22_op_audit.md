# Opset Nineteen Through Twenty Two CPU Audit for ONNX Runtime One Point Twenty Four Point Four

## Scope and assumptions

This note covers only **ai.onnx opset 19, 20, 21, and 22** on **ONNX Runtime 1.24.4, CPUExecutionProvider**. I explicitly exclude **opset ≤18** because your prior **K3** audit already owns that surface. I also treat the following as prompt facts: your grader runs ORT 1.24.4 CPU EP; opset-18 `ScatterElements(reduction='max')` and opset-19 `QLinearConv` already held; your validator rejects `Loop`, `Scan`, `NonZero`, `Unique`, `Compress`, and `If`/FunctionProto-style control-flow subgraphs; and your score is `intermediate_memory_bytes + param_elements`, with **params counted by element count only** and **input/final-output tensors free**. `[PROMPT]` citeturn25view0turn19view0turn20view1turn21view1

Two practical caveats drive almost every conclusion below. First, **ORT support is per `(op, opset-range, dtype)`**, so a spec update can exist while the CPU kernel is absent or narrower than the ONNX schema. Second, because your `param_elements` are **dtype-blind**, any “savings” based only on using narrower weights are fake in your metric; only deleting charged intermediates, fusing charged nodes, exploiting the free final output, or replacing a large constant tensor with a tiny structural generator can actually win. `[PROMPT]` citeturn25view0

I use the following dtype shorthands in Table A to keep it readable. `F = tensor(float)`. `DF = tensor(double), tensor(float)`. `DFH = tensor(double), tensor(float), tensor(float16)`. `Tgen19 = tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)`. `Tgen21 = Tgen19 + tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)`. `CastLike12 = tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)`. `PadCPU = tensor(bool), tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(int8), tensor(uint32), tensor(uint64), tensor(uint8)`. `RedBool20 = tensor(bool), tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(int8), tensor(uint8)`. `Grid20 = X/grid both {tensor(double), tensor(float)}`. `QMM21 = a/b/y all {tensor(int8), tensor(uint8)}; scales tensor(float)`. `Q21 = input {tensor(float), tensor(float16)}; output {tensor(float8e4m3fn), tensor(float8e5m2), tensor(int4), tensor(int8), tensor(uint4), tensor(uint8)}`. `DQ19 = x {tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int32), tensor(int8), tensor(uint8)}; scale/output {tensor(float), tensor(float16)}`. `DQ21 = x {tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int8), tensor(uint16), tensor(uint4), tensor(uint8)}; scale/output {tensor(float), tensor(float16)}`. `[PRIMARY]` ORT dtypes are from **ORT v1.24.4 `docs/OperatorKernels.md`**, and opset additions/updates are from the official ONNX release notes and operator pages. citeturn25view0turn19view0turn21view0turn20view1turn21view1turn30view0turn30view1turn30view2turn31view1turn34view0turn34view1turn34view2turn34view3turn34view4turn34view5

## Support map

For **opset 19**, I combined the official ONNX v1.14.0 release notes for the publicly called-out changes with the individual ONNX operator pages for the schema-version bumps that are present in the operator docs but not spelled out in the release summary. ORT support/dtypes come from the **CPU section** of `docs/OperatorKernels.md`. `[PRIMARY]` citeturn19view0turn30view0turn30view1turn30view2turn31view1turn34view0turn34view1turn34view2turn34view3turn34view4turn34view5turn25view0turn41view0turn42view2turn43view0turn36view3

### Table A

| op | opset added/changed | ORT 1.24.4 CPU kernel? (Y/N + version) | dtypes supported | source-tag |
|---|---|---:|---|---|
| AveragePool | 19 changed | Y (`[19,21]`, and also `22+`) | `F` | `[PRIMARY] ONNX v1.14.0 release notes + ORT docs/OperatorKernels.md` |
| Cast | 19 changed | Y (`[19,20]`) | wide cast kernel; includes bool/string/double/float/float16/bfloat16/int8/16/32/64/uint8/16/32/64 plus float8/int2/int4/uint2/uint4 in ORT’s current row | `[PRIMARY] ONNX Cast operator version history + ORT docs/OperatorKernels.md` |
| CastLike | 19 changed | Y (`19+`) | `CastLike12` | `[PRIMARY] ONNX CastLike operator page + ORT docs/OperatorKernels.md` |
| Constant | 19 changed | **N as CPU-kernel row** | — | `[PRIMARY] ONNX Constant operator page + ORT docs/OperatorKernels.md` |
| DeformConv | 19 added | **N** | — | `[PRIMARY] ONNX v1.14.0 release notes + ORT docs/OperatorKernels.md` |
| DequantizeLinear | 19 changed | Y (`[19,20]`) | `DQ19` | `[PRIMARY] ONNX DequantizeLinear operator page + ORT docs/OperatorKernels.md` |
| Equal | 19 changed | Y (`19+`) | bool output; inputs `tensor(bool), tensor(double), tensor(float), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)` | `[PRIMARY] ONNX v1.14.0 release notes + ORT docs/OperatorKernels.md` |
| Identity | 19 changed | Y (`19+`) | heterogeneous generic tensor/sequence/optional set widened at IRv9 | `[PRIMARY] ONNX Identity operator page + ORT docs/OperatorKernels.md` |
| If | 19 changed | Y (`19+`) but **validator-banned for you** | heterogeneous generic tensor/sequence/optional set widened at IRv9 | `[PRIMARY] ONNX If operator page + ORT docs/OperatorKernels.md` |
| Loop | 19 changed | Y (`19+`) but **validator-banned for you** | heterogeneous generic tensor/sequence set widened at IRv9 | `[PRIMARY] ONNX Loop operator page + ORT docs/OperatorKernels.md` |
| Pad | 19 changed | Y (`[19,20]`) | `PadCPU` | `[PRIMARY] ONNX v1.14.0 release notes + ORT docs/OperatorKernels.md` |
| Reshape | 19 changed | Y (`[19,20]`) | `Tgen19` | `[PRIMARY] ONNX Reshape operator page + ORT docs/OperatorKernels.md` |
| Resize | 19 changed | Y (`19+`) | `tensor(float), tensor(int32), tensor(int8), tensor(uint8)` | `[PRIMARY] ONNX v1.14.0 release notes + ORT docs/OperatorKernels.md` |
| Scan | 19 changed | Y (`19+`) but **validator-banned for you** | heterogeneous generic tensor/sequence set widened at IRv9 | `[PRIMARY] ONNX Scan operator page + ORT docs/OperatorKernels.md` |
| Shape | 19 changed | Y (`[19,20]`) | input `Tgen19`, output `tensor(int64)` | `[PRIMARY] ONNX Shape operator page + ORT docs/OperatorKernels.md` |
| Size | 19 changed | Y (`19+`) | input wide generic tensor set, output `tensor(int64)` scalar | `[PRIMARY] ONNX Size operator page + ORT docs/OperatorKernels.md` |

For **opset 20**, the ONNX v1.15.0 release notes are already a clean complete list of the new/updated ai.onnx operators, and the ORT side is mostly straightforward from `docs/OperatorKernels.md`. `[PRIMARY]` citeturn21view0turn25view0turn27view0turn27view1turn27view2turn27view3turn27view4turn27view5turn42view0turn42view1turn36view1

| op | opset added/changed | ORT 1.24.4 CPU kernel? (Y/N + version) | dtypes supported | source-tag |
|---|---|---:|---|---|
| AffineGrid | 20 added | Y (`20+`) | theta `DF`, size `tensor(int64)` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| ConstantOfShape | 20 changed | Y (`20`, also `[9,19]` older and `[21,22]` newer) | output wide numeric/bool set; input `tensor(int64)` shape | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| DFT | 20 changed | Y (`20+`) | input/output `DF`; `dft_length` int32/int64 | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| Gelu | 20 added | Y (`20+`) | `F` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| GridSample | 20 changed | Y (`[20,21]`) | `Grid20` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| ImageDecoder | 20 added | **N** | — | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| IsInf | 20 changed | Y (pre-20 CPU kernel) | pre-20 CPU float-family only; no evidence of the new float8 schema widening on CPU | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| IsNan | 20 changed | Y (pre-20 CPU kernel) | pre-20 CPU float-family only; no evidence of the new float8 schema widening on CPU | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| ReduceMax | 20 changed | Y (`20+`) | `RedBool20` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| ReduceMin | 20 changed | Y (`20+`) | `RedBool20` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| RegexFullMatch | 20 added | Y (`20+`) | input `tensor(string)`, output `tensor(bool)` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| StringConcat | 20 added | Y (`20+`) | `tensor(string)` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |
| StringSplit | 20 added | Y (`20+`) | input `tensor(string)`; outputs `tensor(string)` and `tensor(int64)` | `[PRIMARY] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md` |

For **opset 21**, the ONNX v1.16.0 release notes again enumerate the added/updated ai.onnx operators. The ORT support picture is mixed: some ops have true 21+ CPU rows; some are only covered by older open-ended registrations; and `GroupNormalization` is the big negative surprise, because the main-domain op is **not** in the CPU matrix even though ORT has contrib-domain `GroupNorm`. `[PRIMARY]` citeturn20view1turn25view0turn28view0turn28view1turn28view2turn28view3turn28view4turn36view0turn36view1turn36view2turn37view2turn37view3turn37view5turn39view2

| op | opset added/changed | ORT 1.24.4 CPU kernel? (Y/N + version) | dtypes supported | source-tag |
|---|---|---:|---|---|
| Cast | 21 changed | Y (`[21,22]`) | wide cast kernel | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| CastLike | 21 changed | Y (`21+`) | `CastLike12` | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Constant | 21 changed | **N as CPU-kernel row** | — | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| ConstantOfShape | 21 changed | Y (`[21,22]`) | output wide numeric/bool set including float8 family in ORT’s 21/22 row | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| DequantizeLinear | 21 changed | Y (`21+`) | `DQ21` | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Flatten | 21 changed | Y (older open-ended CPU kernel) | pre-21 generic tensor set; no evidence of 21’s new int4/uint4/float8 widening on CPU | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| GroupNormalization | 21 changed | **N** | — | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Identity | 21 changed | Y (`19+`, older open-ended row) | heterogeneous generic tensor/sequence/optional set | `[PRIMARY] ONNX v1.16.0 release notes + ONNX Identity page + ORT docs/OperatorKernels.md` |
| If | 21 changed | Y (`19+`) but **validator-banned for you** | heterogeneous generic tensor/sequence/optional set | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Loop | 21 changed | Y (`19+`) but **validator-banned for you** | heterogeneous generic tensor/sequence set | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Pad | 21 changed | Y (`[21,22]`) | `PadCPU` | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| QLinearMatMul | 21 changed | Y (`21+`) | `QMM21` | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| QuantizeLinear | 21 changed | Y (`21+`) | `Q21` | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Reshape | 21 changed | Y (`[21,22]`) | `Tgen21` | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Scan | 21 changed | Y (`19+`) but **validator-banned for you** | heterogeneous generic tensor/sequence set | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Shape | 21 changed | Y (`[21,22]`) | input `Tgen21`, output `tensor(int64)` | `[PRIMARY] ONNX v1.16.0 release notes + ONNX Shape page + ORT docs/OperatorKernels.md` |
| Size | 21 changed | Y (`[21,22]`) | wide generic tensor input; output `tensor(int64)` | `[PRIMARY] ONNX v1.16.0 release notes + ONNX Size page + ORT docs/OperatorKernels.md` |
| Squeeze | 21 changed | Y (`[21,22]`) | older generic tensor set; no isolated evidence of 21’s full int4/float8 widening on CPU | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Transpose | 21 changed | Y (older open-ended CPU kernel) | older generic tensor set; no isolated evidence of 21’s full int4/float8 widening on CPU | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |
| Unsqueeze | 21 changed | Y (older open-ended CPU kernel) | older generic tensor set; no isolated evidence of 21’s full int4/float8 widening on CPU | `[PRIMARY] ONNX v1.16.0 release notes + ORT docs/OperatorKernels.md` |

For **opset 22**, the single important pattern is this: ONNX v1.17.0 mostly added **bfloat16** support, but **ORT 1.24.4 CPU usually does not pick up that bfloat16 widening**. In many cases the CPU kernel exists only because an older open-ended registration already covers the op-version, and the actual CPU dtype set stays **float-only** or **float-family only**. That makes opset 22 much less interesting for your score model than the raw changelog suggests. `[PRIMARY]` citeturn21view1turn25view0turn35view0turn35view2turn35view3turn35view4turn35view5turn42view1turn43view0turn43view1turn43view2turn43view3

| op | opset added/changed | ORT 1.24.4 CPU kernel? (Y/N + version) | dtypes supported | source-tag |
|---|---|---:|---|---|
| Acos | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Acosh | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Asin | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Asinh | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Atan | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Atanh | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| AveragePool | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Bernoulli | 22 changed | **N** | — | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Conv | 22 changed | Y (`22+`) | `F` on CPU, **not** the new schema bf16 set | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| ConvTranspose | 22 changed | Y (`22+`) | `F` on CPU, **not** the new schema bf16 set | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Cos | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Cosh | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| DeformConv | 22 changed | **N** | — | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Det | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Dropout | 22 changed | Y (`22+`) | data/output `DF`, mask `tensor(bool)` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Elu | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| EyeLike | 22 changed | Y (older open-ended row) | bool/double/float/float16/int8/16/32/64/string/uint8/16/32/64; **no isolated CPU bf16 evidence** | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| GRU | 22 changed | Y (`22+`) | `DF`; `sequence_lens` is int32 | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| GlobalAveragePool | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| GlobalLpPool | 22 changed | Y (older open-ended row) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| GlobalMaxPool | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| GridSample | 22 changed | Y (`22+`) | `Grid20`; **no CPU bf16** | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| HardSigmoid | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| HardSwish | 22 changed | Y (pre-22 CPU kernel) | pre-22 CPU float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| InstanceNormalization | 22 changed | Y (pre-22 CPU kernel) | pre-22 CPU float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| LSTM | 22 changed | Y (pre-22 CPU kernel) | pre-22 CPU float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| LpNormalization | 22 changed | Y (older open-ended row) | float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| LpPool | 22 changed | Y (older open-ended row) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| MaxPool | 22 changed | Y (pre-22 CPU kernel) | pre-22 CPU set, typically float/float16/int8/uint8; **not** schema bf16 | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| MaxRoiPool | 22 changed | Y (older open-ended row) | float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| MaxUnpool | 22 changed | Y (older open-ended row) | float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Mish | 22 changed | Y (pre-22 CPU kernel) | float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Multinomial | 22 changed | Y (older open-ended row) | float-family input; integer output; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| NegativeLogLikelihoodLoss | 22 changed | Y (pre-22 CPU kernel) | float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| RNN | 22 changed | Y (pre-22 CPU kernel) | pre-22 CPU float-family only; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| RandomNormal | 22 changed | Y (older open-ended row) | CPU float-family generator; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| RandomNormalLike | 22 changed | Y (older open-ended row) | CPU float-family generator; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| RandomUniform | 22 changed | Y (older open-ended row) | CPU float-family generator; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| RandomUniformLike | 22 changed | Y (older open-ended row) | CPU float-family generator; no isolated CPU bf16 evidence | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| RoiAlign | 22 changed | Y (older open-ended row) | `DF` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Round | 22 changed | Y (older open-ended row) | `DFH` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Selu | 22 changed | Y (pre-22 CPU kernel) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Sin | 22 changed | Y (`22+`) | `DF` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Sinh | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Softplus | 22 changed | Y (pre-22 CPU kernel) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Softsign | 22 changed | Y (pre-22 CPU kernel) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| Tan | 22 changed | Y (`22+`) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |
| ThresholdedRelu | 22 changed | Y (pre-22 CPU kernel) | `F` | `[PRIMARY] ONNX v1.17.0 release notes + ORT docs/OperatorKernels.md` |

## Cost-reducer shortlist

Only a small subset of the kernel-backed post-18 surface looks genuinely useful under your scoring rule. The best single lever is **`ConstantOfShape`**, because it can replace a large counted `Constant` tensor with a tiny counted shape tensor and scalar value. The next real win is **bool-capable `ReduceMax`/`ReduceMin` at opset 20**, because they let you delete cast-to-numeric scaffolding around boolean masks. The most domain-specific but still real lever is **`Pad(mode="wrap")`**, which can collapse slice/concat emulation of circular padding into one free-output node. `Gelu` is also a real fuse, but in your ARC-style integer-grid setting it is less likely to arise than the first three. `[DERIVED]` from the prompt cost model + ONNX/ORT support map. citeturn21view0turn42view0turn41view0turn40view0turn42view1turn25view0

### Table B

| op | mechanism (delete/fuse/free-out/param-shrink/zero-param) | est. saving on [1,10,30,30] | dtype/shape caveat | source-tag |
|---|---|---:|---|---|
| ConstantOfShape | zero-param-generator / param-shrink | **−8995 param_elements** when replacing a `[1,10,30,30]` `Constant` tensor (9000 elems) with shape `[4]` + scalar fill (about 5 elems total) | Works only for fill-pattern tensors; output bytes do **not** shrink, only params do | `[PRIMARY]+[DERIVED] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md + prompt cost model` |
| ReduceMax | delete-intermediate / fuse | **about −9001 bytes** on a bool mask if pre-20 you used `Cast(bool→uint8) -> ReduceMax -> Cast` and the final cast is the graph output; higher if you had to cast to f32 | This is the opset-20 **bool** enablement; inputs supported only by `RedBool20` on CPU | `[PRIMARY]+[DERIVED] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md + prompt cost model` |
| ReduceMin | delete-intermediate / fuse | **about −9001 bytes** under the same bool-mask pattern, for “all/none” style reductions | Same caveat as `ReduceMax` | `[PRIMARY]+[DERIVED] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md + prompt cost model` |
| Pad mode wrap | fuse / delete-intermediate / free-out | **about −2400 bytes** for one-axis 1-pixel circular pad emulated by two `Slice` outputs of shape `[1,10,30,1]` in f32; larger for 2-D toroidal padding | CPU opset-19/21 `Pad` is dtype-narrower than the full ONNX schema (`PadCPU`) | `[PRIMARY]+[DERIVED] ONNX v1.14.0 release notes + ORT docs/OperatorKernels.md + prompt cost model` |
| Gelu | fuse / free-out | **about −144000 bytes** if it replaces a standard float32 GELU chain with four charged full-size intermediates before the free final output | Float-only on CPU in ORT 1.24.4; relevant only if you truly need GELU semantics | `[PRIMARY]+[DERIVED] ONNX v1.15.0 release notes + ORT docs/OperatorKernels.md + prompt cost model` |

## Dead and trap list

The **no-kernel traps** are the most important negative findings. `ImageDecoder` is not in the CPU matrix. `DeformConv` is not in the CPU matrix for either its opset-19 introduction or opset-22 bf16 schema update. `Bernoulli` is likewise absent from the CPU matrix. `GroupNormalization` is the nastiest one conceptually: ONNX added/changed the **main-domain** op in opset 21, but ORT 1.24.4 CPU exposes **contrib-domain `GroupNorm`**, not a main-domain `GroupNormalization` CPU kernel, so the ONNX op is not something I would target without a direct probe. `[PRIMARY]` citeturn19view0turn21view0turn21view1turn27view1turn35view1turn35view4turn28view0

The second trap class is **checker-passes but dtype-narrow on CPU**. Opset 22 looks rich because ONNX added bf16 to many ops, but ORT CPU often stays float-only or float-family only. Concrete examples from the matrix: `Conv` and `ConvTranspose` have `22+` CPU rows that are still **float-only**; `AveragePool`, `GlobalAveragePool`, and `GlobalMaxPool` stay **float-only**; `GridSample` gets a `22+` CPU row but only for `double`/`float`; `Dropout` at `22+` is only `double`/`float`; `Elu` remains `float`; and `Det` remains `float`. So “ONNX 22 says bf16” is usually **not** “ORT 1.24.4 CPU will run bf16 for this op.” `[PRIMARY]` citeturn40view1turn42view1turn35view2turn43view1turn43view2turn43view0turn25view0

The third trap class is **score-model mirages**. Quantization-family ops such as `QuantizeLinear`, `DequantizeLinear`, `QLinearMatMul`, and even your confirmed-working `QLinearConv` do **not** automatically save score, because your `param_elements` ignore dtype width. An `int8` weight tensor and an `f32` weight tensor of the same shape cost the same in your metric. These ops only help if they also let you delete charged intermediates or keep a charged intermediate in a narrower runtime dtype; simply “using smaller weights” is **not** a saver in your setup. `[PROMPT]+[DERIVED]` citeturn36view0turn36view1turn36view2turn41view1

`AffineGrid` is the cleanest **param-shrink-but-still-loses** example. If you replace a stored affine sampling grid `[1,30,30,2]` with `AffineGrid`, you cut the parameter from about `1800` elements down to a tiny `theta + size`, but you introduce a charged float intermediate grid of `1800 * 4 = 7200` bytes before `GridSample`. Under your score, that usually **worsens** the total instead of improving it. `[DERIVED]` from `AffineGrid`/`GridSample` support and your cost model. citeturn27view2turn42view1turn25view0

Finally, several ops are simply **pipeline-banned or adjacent to bans**. `If`, `Loop`, and `Scan` do have ORT CPU support in the relevant schema generations, but your validator rejects them, so they are dead on arrival for you. `Compress` has a CPU kernel but you explicitly banned it. `NonZero` and `Unique` are also banned. `[PROMPT]+[PRIMARY]` citeturn31view1turn34view0turn34view1turn36view3turn25view0

## Worked micro-examples

`ConstantOfShape` is the cleanest param win.

```text
Before:  Constant(value=zeros[1,10,30,30]) -> Add -> ...
After:   Constant(shape=[1,10,30,30]) + Constant(fill=0) -> ConstantOfShape -> Add -> ...
Delta:   param_elements 9000 -> 5  =>  -8995
```

The output tensor is the same size either way, so the win is entirely on the **param side**. If the filled tensor is the graph output, you also exploit the “final output is free” rule and pay no intermediate bytes either way. `[DERIVED]` from your cost model and `ConstantOfShape` support. citeturn40view0turn25view0

`ReduceMax` on bool becomes a real fusion at opset 20.

```text
Before:  mask(bool [1,10,30,30]) -> Cast(to=uint8) -> ReduceMax -> Cast(to=bool out)
After:   mask(bool [1,10,30,30]) -> ReduceMax(out)
Delta:   remove Cast output (9000 B) + remove charged scalar from pre-final ReduceMax (~1 B)
         => about -9001 intermediate_memory_bytes
```

If your pre-20 workaround cast to `float` rather than `uint8`, the saving is larger: `36000 B + 4 B`. The key point is that opset-20 bool reductions let you delete the numeric detour entirely. `[DERIVED]` from ORT’s opset-20 bool support and your scoring rule. citeturn42view0turn25view0

`Pad(mode="wrap")` can delete slice/concat emulation.

```text
Before:  X[f32 [1,10,30,30]] -> Slice(last_col) + Slice(first_col) -> Concat(out)
After:   X -> Pad(mode="wrap", pads=[...,1,1])(out)
Delta:   two Slice intermediates of shape [1,10,30,1] => 2 * 300 * 4 = -2400 B
```

That is the conservative one-axis example. If you emulate 2-D toroidal padding, you usually pay more slices and at least one extra concat stage, so the true saving can be larger. `[DERIVED]` from the opset-19 `wrap` addition, ORT CPU `Pad` support, and your byte-count rule. citeturn19view0turn41view0turn25view0

## Verdict and checklist

The blunt verdict is: **yes, there is something worth adopting after opset 18, but only barely**. The post-18 surface is **mostly barren for your metric**. The **single best find is `ConstantOfShape`**: it is the only post-18 kernel-backed change here that reliably attacks the part of your score that matters most for many ARC-like constructions, namely counted `Constant` tensors. After that, the only broadly plausible wins are **bool-capable `ReduceMax`/`ReduceMin` at opset 20** and **`Pad(mode="wrap")` at opset 19** if you actually emulate those patterns today. Everything else is either a no-kernel trap, validator-banned, dtype-widening that ORT CPU does not really honor, or a mirage because parameter dtype does not affect `param_elements`. `[DERIVED]` from the full support map + your cost model. citeturn40view0turn42view0turn41view0turn21view1turn25view0

Open questions / limitations: I have **high confidence** in the opset-19/20/21 findings and in the opset-22 pattern that “spec bf16 usually does not become CPU bf16 in ORT 1.24.4.” For some **less relevant opset-22 rows** outside the shortlist, I did not isolate every single CPU dtype line individually before synthesis; those rows are therefore best read as “kernel exists, but no evidence of the new bf16 schema widening on CPU,” rather than as a recommendation to adopt them. The important take-away for your workflow is still solid: probe only `ConstantOfShape`, bool `ReduceMax`/`ReduceMin`, and `Pad(mode="wrap")` first. `[PRIMARY]+[DERIVED]` citeturn21view1turn25view0

- [x] Opset 19–22 covered; opset ≤18 explicitly excluded (K3 owns it).
- [x] Every listed op tagged Y/N for an actual ORT-1.24.4 CPU kernel entry, or explicitly called out as a special non-kernel/runtime case (`Constant`).
- [x] Cost-reducer claims tied to the exact cost-model mechanism, with byte/param deltas.
- [x] dtype-blind-params trap explicitly applied; no “narrow dtype = cheaper params” claim is used.
- [x] One-paragraph blunt verdict included: **adopt `ConstantOfShape` first; otherwise the post-18 surface is mostly barren, with only bool `ReduceMax`/`ReduceMin` and `Pad(mode="wrap")` as situational wins.**