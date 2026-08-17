# Research Card N1 - Opset 23-26 delta audit on ONNX Runtime 1.24.4 CPU

**Deliverable**: a single copyable Markdown research note. Suggested save name: `n1_opset23_26_delta_audit.md`.

## Researcher contract

You are an internet-research assistant. You have web access but NO access to my repository or files. Produce ONE self-contained Markdown research note. Do not ask follow-up questions; state assumptions inline. Every operator claim must carry a source tag and a human-readable citation.

Use official ONNX operator specs/changelogs and ONNX Runtime kernel registration docs/source as primary sources. Do not rely on model-zoo examples or blog posts for kernel support.

## Situation

I build tiny ONNX graphs for a cost-golf competition. The local stack is:

```
onnx==1.21.0
onnxruntime==1.24.4
CPUExecutionProvider
Graph optimizations disabled
```

Prior completed research already covered:

- K3: default-domain residual ops through opset 18.
- M1: opset 19-22 audit.

Do NOT re-cover those broad surfaces except when an opset 23-26 change modifies an old op in a way that matters.

## Cost model

```
Score = intermediate_memory_bytes + param_elements

param_elements = sum of element counts across initializer and Constant-node tensors.
Params are dtype-blind: int8 and float32 initializers of the same shape cost the same.

intermediate_memory_bytes = sum of byte sizes of value_info tensors materialized between
input and final output. The graph input and final graph output are free.

An op is useful only if it deletes a charged intermediate, fuses charged nodes, moves work
into a free final output node, or replaces a large parameter tensor with a smaller structural generator.
```

Typical tensor: `[1, C, H, W]`, where `C <= 10` and `H,W <= 30`. Values are small ARC colors `0..9`, masks, or coordinates carried as bool/uint8/int32/f16/f32.

## Hard constraints

```
Banned by validator: Loop, Scan, NonZero, Unique, Script, Function, Compress.
Rejected in practice: subgraph attributes / If-style control flow, custom domains, dynamic symbolic shapes.
Prefer single-output tensor ops. Multi-output or sequence/string/optional outputs are suspect unless proven scorer-safe.
```

`check_model` passing is not enough. The op must have an ONNX Runtime 1.24.4 CPU kernel for the target opset and dtype.

## Required answer order

1. **Delta map**: list every default-domain ONNX op added or version-changed in opsets 23, 24, 25, and 26.
2. **ORT CPU support**: for each row, state whether ORT 1.24.4 CPU registers a kernel for that op/version and the supported dtypes.
3. **Cost-reducer shortlist**: identify any ops that could plausibly reduce NeuroGolf cost. Name the mechanism: `delete-intermediate`, `fuse`, `free-output`, `param-shrink`, or `zero-param-generator`.
4. **Dead/trap list**: ops that are schema-only, no CPU kernel, sequence/string/optional-only, output dynamic, dtype-impractical, or validator-banned-adjacent.
5. **Top probes**: propose the smallest local probe for each promising candidate, including input/output shapes, dtype, opset, expected byte/param delta, and pass/fail criteria.

## Required tables

Table A:

`| op | opset change | semantic delta | ORT 1.24.4 CPU kernel? | supported dtypes | scorer risk | source tags |`

Table B:

`| op | cost mechanism | expected saving on [1,10,30,30] or smaller | caveat | local probe to run | source tags |`

## Candidate areas to scrutinize

- `Attention` and related neural-network ops only if they can replace explicit MatMul/Einsum/Gather chains at smaller intermediate cost. Dtype and parameter burden must be priced honestly.
- `BitCast` or cast/version changes, if present, only if they can reinterpret without charged conversion or avoid a materialized Cast.
- Later `ConstantOfShape`, `Pad`, `Reshape`, `Squeeze`, `Unsqueeze`, `Transpose`, `TopK`, `QuantizeLinear`, `DequantizeLinear`, or `Cast` changes that affect dtypes, axes/pads inputs, output shape, or param-side shrink.
- Normalization/LLM helper ops such as `RMSNormalization` or `RotaryEmbedding` are probably dead, but verify kernel support and cost shape before dismissing.

## Source-tag standard

`[PROMPT]` fact given here.
`[PRIMARY]` ONNX spec/changelog, ORT kernel docs/source, or peer-reviewed source.
`[SECONDARY]` explanatory docs.
`[DERIVED]` deduction from cited sources plus the cost model.
`[SPECULATIVE]` unproven construction.

## Verification checklist

- [ ] Opset 23, 24, 25, and 26 deltas covered.
- [ ] Opset <=22 not re-audited except where changed by a newer version.
- [ ] Every candidate has ONNX spec citation and ORT 1.24.4 CPU kernel support status.
- [ ] Every cost-reducer claim names the exact scoring mechanism and byte/param arithmetic.
- [ ] Every dead/trap claim states whether the blocker is validator, schema, CPU kernel, dtype, dynamic shape, or cost.
- [ ] Final blunt verdict: name the best candidate, or say the opset 23-26 surface is barren for NeuroGolf.
