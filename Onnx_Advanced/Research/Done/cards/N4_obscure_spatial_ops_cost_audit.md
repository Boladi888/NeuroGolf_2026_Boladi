# Research Card N4 - Obscure spatial/window ops as NeuroGolf cost reducers

**Deliverable**: a single copyable Markdown research note. Suggested save name: `n4_obscure_spatial_ops_cost_audit.md`.

## Researcher contract

You are an internet-research assistant. You have web access but NO access to my repository or files. Produce ONE self-contained Markdown note. Cite official ONNX operator specs and ONNX Runtime CPU kernel docs/source for every op support claim.

## Situation

Most NeuroGolf graphs use Conv, QLinearConv, Pad, Slice, Gather, Scatter, Reduce, Resize, Tile, and GridSample-like remaps. There may still be obscure spatial/window operators that replace larger hand-built chains. Prior broad audits touched some of them, but this card asks for a cost-priced spatial shelf.

Do not re-explain standard Conv/Gather/Slice basics. Focus on lesser-used spatial/window operators and exact cost comparisons against incumbent ONNX patterns.

## Environment and constraints

```
Runtime: ONNX Runtime 1.24.4 CPUExecutionProvider
Tensor layout: NCHW
Typical shape: [1,C,H,W], C <= 10, H,W <= 30
Data: small integer color grids, masks, coordinate fields
Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress
Output shapes must be static unless the op is explicitly marked fatal/dynamic-risk.
```

## Cost model

```
Score = intermediate_memory_bytes + param_elements
Input and final output are free.
Intermediate tensors are charged by byte size.
Initializers and Constant-node tensors are charged by element count, not bytes.
```

An op is useful only if it deletes charged intermediates, fuses a spatial/window operation, avoids a large parameter table, or can be the final output node.

## Required answer order

1. **Operator shelf**: enumerate obscure spatial/window/layout ops relevant to 2-D grids.
2. **ORT CPU support**: for each op, state version range and dtype support under ORT 1.24.4 CPU.
3. **Cost comparison**: compare each viable op to the incumbent construction: Conv/MaxPool/AveragePool/Slice/Gather/Scatter/GridSample/Resize/Tile/Pad.
4. **Exactness traps**: interpolation, boundary behavior, coordinate normalization, padding modes, multi-output results, dynamic shapes, and dtype conversions.
5. **Ranked adoption list**: classify as `adopt`, `probe`, `task-specific`, or `dead`.

## Required tables

Table A:

`| op | since-opset | ORT CPU dtypes | output shape formula | incumbent it may replace | cost mechanism | caveat | source tags |`

Table B:

`| task pattern | incumbent chain and cost | obscure-op chain and cost | exactness conditions | verdict |`

## Operators to include

- `Col2Im`, `DepthToSpace`, `SpaceToDepth`, `Resize`, `GridSample`, `RoiAlign`, `MaxRoiPool`.
- `LpPool`, `GlobalLpPool`, `AveragePool`, `MaxPool`, including p=1/window-sum alternatives and dtype support.
- `MaxUnpool`, if CPU-supported and shape-safe.
- `ReverseSequence`, `Pad` modes (`constant`, `edge`, `reflect`, and later `wrap` if supported), `Tile`, and `Trilu`.
- `DFT` or other transform-like ops only if ORT CPU supports them and exact integer/mask use is plausible.
- `CenterCropPad`, if present in ONNX schema, with explicit kernel/no-kernel verdict.
- `NonMaxSuppression` only as a dynamic-output/fatal check; do not spend time on detection semantics if output shape is unsuitable.

## Mandatory caveats

- Interpolation ops are usually non-exact for discrete ARC grids unless `nearest` or exact coordinate construction is used.
- Coordinate-grid params can dominate: price the grid initializer or generated grid.
- Pooling/window ops help only if their window output is needed directly or replaces multiple charged planes.
- Float-only spatial ops require input/grid casts; price both casts.
- Static ROI constants count as params; dynamic ROI extraction may be validator-fatal.

## Source-tag standard

`[PROMPT]` fact given here.
`[PRIMARY]` ONNX spec/changelog or ORT kernel docs/source.
`[SECONDARY]` explanatory docs.
`[DERIVED]` deduction from cited sources plus the cost model.
`[SPECULATIVE]` unproven construction.

## Verification checklist

- [ ] Every listed op has ONNX spec and ORT 1.24.4 CPU support status.
- [ ] Every viable op is costed against a concrete incumbent chain.
- [ ] Boundary/interpolation/exactness traps are explicit.
- [ ] At least five minimal local probes are proposed with shape, dtype, opset, and expected result.
- [ ] Final blunt verdict: name the best obscure spatial op still worth probing, or say the spatial surface is already effectively mined.
