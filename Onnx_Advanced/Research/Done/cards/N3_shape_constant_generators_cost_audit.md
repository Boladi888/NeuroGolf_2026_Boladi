# Research Card N3 - Shape and constant generators for param-side cost reduction

**Deliverable**: a single copyable Markdown research note. Suggested save name: `n3_shape_constant_generators_cost_audit.md`.

## Researcher contract

You are an internet-research assistant. You have web access but NO access to my repository or files. Produce ONE self-contained Markdown note. Cite official ONNX specs and ONNX Runtime CPU kernel support for every operator claim.

## Situation

In this project, params are charged by element count. Replacing a large stored constant tensor with a small structural generator can be a huge win, even when the generated tensor has the same runtime size. The current docs already know about `ConstantOfShape`, `Range`, `EyeLike`, `Shape`, `Size`, `Expand`, `Tile`, `Pad`, `Trilu`, and related patterns, but they have not been audited together as a param-side generator shelf.

This card should find reusable generator idioms for large uniform fills, ramps, coordinate fields, masks, triangular/diagonal structure, repeated tiles, padding frames, and small lookup tables.

## Environment and constraints

```
Runtime: ONNX Runtime 1.24.4 CPUExecutionProvider
Opset: use the lowest opset that supports the needed dtype/semantics; higher opsets allowed if ORT CPU supports them.
Typical target shape: [1,10,30,30] or smaller cropped variants.
Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.
All output shapes must be statically inferable unless explicitly flagged for local dynamic-shape profiling.
```

## Cost model

```
Score = intermediate_memory_bytes + param_elements

param_elements are dtype-blind. A stored [1,10,30,30] Constant costs 9000 params no matter the dtype.
Intermediate outputs are byte-priced. A generated [1,10,30,30] UINT8 tensor costs 9000 B if intermediate,
but 0 B if it is the final graph output.
```

An idiom is useful if it reduces param count more than it adds intermediate bytes, or if the generator node can be the final free output.

## Required answer order

1. **Generator inventory**: list ONNX ops that can generate or reshape constants/indices/masks with small initializers.
2. **Cost formulas**: for each generator, give param cost, intermediate byte cost, and output shape formula.
3. **Idiom catalog**: produce reusable recipes for uniform fill, scalar broadcast, arange row/col ramps, one-hot/diagonal, triangular masks, periodic tiling, border/frame masks, checkerboards/parity, and coordinate grids.
4. **Dead/trap list**: patterns that look like param savings but lose under intermediate memory, dtype restrictions, dynamic-shape profiling, or ORT kernel gaps.
5. **Top-10 adoption table**: rank the generator idioms by likely NeuroGolf payoff.

## Required tables

Table A:

`| op/idiom | opset | ORT 1.24.4 CPU dtypes | params needed | intermediate bytes | best use | caveat | source tags |`

Table B:

`| target artifact | incumbent stored constant cost | generated construction | generated params | generated intermediates | net verdict |`

## Operators/idioms to include

- `ConstantOfShape`, including scalar `value` caveats and dtype support.
- `Range`, `Shape`, `Size`, `Gather` from shape tensors, and static-shape ramp generation.
- `EyeLike`, `Trilu`, diagonal/triangular masks, and whether their output dtype can be bool/uint8/f16/f32 on ORT CPU.
- `Expand`, `Tile`, `Concat`, `Pad`, `Resize(nearest)`, `DepthToSpace`, `SpaceToDepth`, `Unsqueeze`, `Squeeze`, `Reshape`, `Transpose`.
- `OneHot` for small category planes and coordinate bases, including index/depth dtype matching.
- `CumSum` as a generator of prefix ramps/masks where exact and cheap.
- Later-opset generator changes such as `Pad(mode="wrap")` only if ORT 1.24.4 CPU supports them.

## Mandatory caveats

- Generated tensors still cost intermediate bytes unless they are the final output.
- Dtype-blind params mean narrower initializers do not help; fewer elements help.
- `Range` and shape-driven output can interact badly with static shape inference and runtime profiling. Separate static-shape constructions from data-dependent constructions.
- If an idiom needs a large initializer to define the pattern, it is not a param generator.

## Source-tag standard

`[PROMPT]` fact given here.
`[PRIMARY]` ONNX spec/changelog or ORT kernel docs/source.
`[SECONDARY]` explanatory docs.
`[DERIVED]` deduction from cited sources plus the cost model.
`[SPECULATIVE]` unproven construction.

## Verification checklist

- [ ] Every generator has ONNX spec and ORT CPU support citations.
- [ ] Every idiom has params and intermediate byte arithmetic.
- [ ] Uniform fills, ramps, diagonal/triangular masks, tiling, frames, and coordinate grids are covered.
- [ ] Dead/trap patterns are explicitly priced.
- [ ] Final blunt verdict: name the three generator idioms most worth adding to `Onnx_Advanced/op_cost_patterns.md` or `techniques.md`.
