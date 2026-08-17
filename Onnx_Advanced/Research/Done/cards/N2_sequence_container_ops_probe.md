# Research Card N2 - Sequence and container ops as possible cost reducers

**Deliverable**: a single copyable Markdown research note. Suggested save name: `n2_sequence_container_ops_probe.md`.

## Researcher contract

You are an internet-research assistant. You have web access but NO access to my repository or files. Produce ONE self-contained Markdown note. Do not ask follow-up questions. Source every ONNX/ORT claim with a human-readable citation.

## Situation

The project normally avoids ONNX sequence, optional, sparse, and string types because the NeuroGolf validator is tensor-centric. That avoidance may be correct, but it has not been isolated as a dedicated research card. This card asks whether any sequence/container operator can safely lower cost under the project scorer.

Prior broad audits found most non-tensor surfaces unpromising. Do not re-explain standard tensor ops. Focus only on default-domain ONNX container/sequence/optional/string-ish operators and their bridges to normal tensors.

## Environment and constraints

```
Runtime: ONNX Runtime 1.24.4 CPUExecutionProvider
Common opset range: 10-26, but any proposed op must be accepted by ORT 1.24.4 CPU.
Typical tensors: [1,C,H,W], C <= 10, H,W <= 30.
Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.
Validator risk: custom domains, subgraphs, dynamic symbolic shapes, and non-tensor graph outputs are likely fatal unless proven otherwise.
```

## Cost model

```
Score = intermediate_memory_bytes + param_elements
Input tensor and final output tensor are free.
Intermediate tensor outputs are charged by byte size.
Initializers and Constant-node tensors are charged by element count, not bytes.
```

A sequence/container op is useful only if it hides a large charged tensor, deletes a materialized tensor, or gives a cheaper bridge back to a normal tensor output. If it merely replaces a tensor with an unpriced sequence that cannot become the final required tensor output, mark it dead.

## Required answer order

1. **Operator inventory**: enumerate relevant ONNX default-domain sequence/container/optional/string ops and version ranges.
2. **ORT CPU support**: state ONNX Runtime 1.24.4 CPU kernel support for each, including output type.
3. **Validator/scorer risk analysis**: for each operator family, explain whether it produces tensor, sequence, optional, string, or map-like values and whether a tensor-only scorer is likely to reject it.
4. **Bridge patterns**: identify legal ways back to ordinary tensors, such as `ConcatFromSequence`, `SplitToSequence`, sequence length, or identity-like transfers. Price every bridge.
5. **Ranked verdict**: classify each family as `probe-now`, `only-if-needed`, or `dead`.

## Required tables

Table A:

`| op | since-opset | input/output type | ORT 1.24.4 CPU kernel? | tensor bridge available? | validator risk | source tags |`

Table B:

`| candidate pattern | before tensor-only chain | after container chain | bytes/params saved or lost | fatal caveat | verdict |`

## Specific ops/families to include

- `SequenceEmpty`, `SequenceInsert`, `SequenceErase`, `SequenceLength`, `SplitToSequence`, `ConcatFromSequence`, and `SequenceAt` if present in the supported schema/runtime surface.
- `Optional*` operators, if available in the schema/runtime.
- String ops with CPU kernels such as `StringNormalizer`, `RegexFullMatch`, `StringConcat`, `StringSplit`, and `TfIdfVectorizer`, but only to determine whether they are categorically irrelevant/fatal for numeric grid scoring.
- Sparse-tensor related paths only if they can be used as normal tensor values without validator rejection.

## Mandatory caveats

- If an op output is not a normal tensor, mark that clearly and explain the exact bridge required.
- If a bridge materializes the same tensor later, it likely saves nothing under the scorer.
- If support exists in ONNX spec but ORT CPU lacks a kernel, mark it dead.
- If runtime support exists but the final graph output type cannot be the required tensor, mark it dead unless the bridge is cheap and exact.

## Source-tag standard

`[PROMPT]` fact given here.
`[PRIMARY]` ONNX spec/changelog or ORT kernel docs/source.
`[SECONDARY]` explanatory docs.
`[DERIVED]` deduction from cited sources plus the cost model.
`[SPECULATIVE]` unproven construction.

## Verification checklist

- [ ] All default-domain sequence/container/optional/string families covered.
- [ ] Every row distinguishes schema support from ORT CPU kernel support.
- [ ] Every non-tensor output names its tensor bridge and prices it.
- [ ] Every `probe-now` candidate includes a minimal graph shape/dtype/opset to test locally.
- [ ] Final blunt verdict: can sequence/container ops ever lower NeuroGolf cost, or should they stay declined?
