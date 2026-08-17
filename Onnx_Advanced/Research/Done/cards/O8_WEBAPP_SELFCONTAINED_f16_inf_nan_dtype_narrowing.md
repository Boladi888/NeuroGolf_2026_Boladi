# Research Card O8 — F16 `inf`/`nan` safety and dtype-narrowing rules for static ONNX graphs

**Deliverable**: one copyable Markdown research note (no file writes, no code execution). Save under `Research/Done` with basename `o8_f16_inf_nan_dtype_narrowing`. **Priority: MEDIUM-HIGH** — this targets a confirmed failure mode where a cheaper dtype-narrowing rewrite passed visible/truth checks but produced `nan` on real inputs.

## Researcher contract

You are using a web app / deep-research chat with internet access and **NO access to my repository or computer**. Produce one self-contained Markdown note. Do not ask follow-up questions. Use human-readable citations such as "ONNX Cast operator spec", "ONNX Runtime einsum implementation file/function", or "IEEE 754-2019 floating-point standard". Tag every nontrivial claim `[PROMPT]`, `[PRIMARY]`, `[SECONDARY]`, `[DERIVED]`, or `[SPECULATIVE]`.

## Project facts (all `[PROMPT]`)

- We build minimal static ONNX graphs for small grid tasks.
- Runtime target: ONNX Runtime 1.24.x CPU, optimization disabled where possible.
- ONNX opsets: normally <=18, but source discussion may include <=25 if relevant.
- Input tensor: `[1,10,30,30]` F32 one-hot; output is decoded per channel by `output > 0.0`.
- Cost model: `cost = intermediate_memory_bytes + param_elements`.
- `param_elements` is the sum of element counts of initializer and Constant tensors; dtype does not reduce param cost.
- `intermediate_memory_bytes` is the sum of byte sizes of materialized tensors between input and final output. The graph input and final graph output are not charged.
- A common optimization is to move a `Cast` before a `Gather`, `Slice`, or other selector so the selected result is narrower and cheaper.
- Confirmed failure to explain: in one free-output graph, moving `Cast(F32->F16)` before `Gather` was algebraically equivalent on selected scalar values and passed visible plus small truth gates, but the graph produced `nan` on real inputs. The likely cause was F16 overflow to `inf`, followed by downstream `Einsum`/multiplication that created `inf * 0 = nan`. Since output decode is `>0.0`, `inf > 0` can be acceptable but `nan > 0` decodes false.

## The question this card answers

When is dtype narrowing, especially `Cast` before `Gather`/`Slice` and F32-to-F16 conversion, safe in ONNX Runtime CPU graphs whose final output is decoded by `>0.0`? Give source-cited rules for overflow, `inf`, `nan`, comparison semantics, ORT implementation order where available, and a small verification checklist that builders can apply before accepting a narrowing rewrite.

## Required answer order

1. **Bottom-line verdict.** One blunt paragraph: when F32-to-F16 narrowing is safe, when it is unsafe, and why truth percentage can miss `nan`.
2. **ONNX operator semantics.** Source-cite `Cast`, `Gather`, `Slice`, `Einsum`, `Mul`, `Add`, `Where`, `Greater`, and any comparison behavior relevant to `nan` and `inf`. State what the ONNX spec guarantees versus what it leaves to numeric/runtime behavior.
3. **Floating-point facts.** Source-cite IEEE 754 or an equally authoritative floating-point reference for F16 max finite value, overflow to infinity, NaN propagation, `inf * 0`, comparisons involving NaN, and sign behavior.
4. **ONNX Runtime CPU behavior.** Source-cite ORT 1.24.x implementation docs/source when possible: Cast conversion behavior to MLFloat16, Einsum contraction order or lowering, and whether intermediate arithmetic is F16 or promoted. If exact ORT 1.24 source is hard to cite, say so and use nearest public source with a clear caveat.
5. **Safe narrowing conditions.** Provide a decision table for `Cast-before-Gather`, `Cast-before-Slice`, narrowing values that feed `Einsum`/`MatMul`/`Mul`, and narrowing values that only feed comparisons. Include an abs-max test, a finite-output test, and a "no `inf*0` downstream" dataflow test.
6. **Instrumentation recipe.** Give a no-repo pseudo-procedure for checking a candidate: run on representative real inputs, dump all node outputs, count finite/inf/nan per tensor, compare decoded outputs, and reject any `nan` in tensors that can affect positive decode.
7. **Final builder checklist.** A concise checklist with pass/fail criteria.

## Required table schemas

### Operator/numeric behavior table

| topic | spec/source claim | source tag | citation | implication for F16 narrowing |
|---|---|---|---|---|

### Rewrite safety table

| rewrite | potential saving | required proof | unsafe pattern | suggested instrumentation | verdict |
|---|---:|---|---|---|---|

## Demands

- Do not hand-wave with "floating point is hard"; give specific ONNX/ORT/IEEE rules.
- Distinguish three cases: finite F16 values, intentional `inf` carriers that still decode correctly, and `nan` poison that can silently decode false.
- Explain why checking selected scalar equality is not sufficient if moving the cast changes the dtype/range of unselected values or the downstream contraction order.
- State whether `Greater(nan, 0)` is guaranteed false by the relevant runtime/spec behavior, or mark it `[DERIVED]` with source evidence.
- Keep code snippets to pseudocode only. The deliverable is a research note, not an implementation.

## Source policy

Primary sources: official ONNX operator specs; ONNX Runtime source/docs for CPU kernel behavior; IEEE 754 or authoritative floating-point references. Secondary sources may explain F16/NaN behavior but cannot replace primary citations for final verdicts.

## Final checklist

- [ ] Bottom-line verdict explicitly says whether `Cast-before-Gather` is safe only after abs-max and finite-output checks.
- [ ] Table covers `Cast`, `Gather`, `Slice`, `Einsum`, `Mul`, and `Greater`.
- [ ] Note explains `inf > 0` versus `nan > 0` under the project decode.
- [ ] Note gives a concrete instrumentation recipe for counting `nan`/`inf` per node output.
- [ ] Note distinguishes ONNX spec guarantees from ORT implementation details and `[DERIVED]` deductions.
