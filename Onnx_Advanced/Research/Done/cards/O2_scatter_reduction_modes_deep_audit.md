# Research Card O2 - Scatter reduction modes deep audit

**Deliverable**: one copyable Markdown research note. Save the note under `Research/Done` with basename `o2_scatter_reduction_modes_deep_audit`.

## Researcher contract

You are an internet-research assistant. You have web access but NO repo access. Produce one self-contained Markdown note. Cite official ONNX specs and ONNX Runtime 1.24.4 CPU kernel docs/source for every op/version/dtype claim.

## Why this card exists

The current library says `ScatterND` reduction modes are underexplored. `ScatterElements(reduction='add'/'min'/'max')` has known histogram and per-category uses, but `ScatterND` reduction modes may still contain score wins for sparse writes, histograms, dynamic patches, and compact assembly.

This is not a broad Scatter tutorial. It is a reduction-mode pricing audit.

## Project facts

```
Runtime: ONNX Runtime 1.24.4 CPUExecutionProvider.
Typical tensor: [1,C,H,W], C<=10, H,W<=30.
Cost = intermediate_memory_bytes + param_elements.
Graph input and final graph output are free.
Params are dtype-blind.
Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.
```

## Required answer order

1. **Spec matrix**: `ScatterElements` and `ScatterND` by opset, available `reduction` values, index dtype, update/data dtype, duplicate-index semantics, and shape rules.
2. **ORT CPU matrix**: ONNX Runtime 1.24.4 CPU support by op, opset, reduction mode, and dtype. Distinguish schema support from kernel support.
3. **Cost idioms**: enumerate possible NeuroGolf idioms for `add`, `mul`, `min`, `max`, and default overwrite.
4. **Collision semantics**: explain when duplicate writes are deterministic and exact enough for histograms, min/max position aggregation, multi-write stamps, or priority routing.
5. **Probe plan**: define small local probes that should be run by a repo-access worker if any mode looks promising.

## Required tables

Table A:

`| op | opset | reduction | ORT CPU dtypes | duplicate-index semantics | output shape | source tags |`

Table B:

`| idiom | incumbent chain | scatter-reduction chain | bytes/params saved | exactness conditions | verdict |`

## Idioms to price

- Count-per-value histograms.
- Per-category min/max row or column position.
- Multi-cell dynamic patch assembly with overlapping writes.
- Priority routing using `max` or `min` instead of a one-hot slot stack.
- Sparse accumulation into a compact `[K]`, `[K,C]`, `[K,H]`, or `[K,W]` tensor.
- Using `mul` reduction as an AND / all-present aggregator.
- Replacing a `Where(mask, update, base)` overlay with sparse scatter updates.

## Mandatory caveats

- Duplicate write order must be deterministic for the target mode, or the idiom is dead.
- `add` on small integer-like values must stay exact under the chosen dtype.
- If the base tensor is full `[1,10,30,30]` and charged as an intermediate, sparse scatter may lose.
- A scatter final-output node can be free; a scatter intermediate is byte-priced.
- Indices tensors often dominate params; price them explicitly.

## Source-tag standard

`[PROMPT]`, `[PRIMARY]`, `[SECONDARY]`, `[DERIVED]`, `[SPECULATIVE]`.

## Final checklist

- [ ] `ScatterElements` and `ScatterND` both covered.
- [ ] Every reduction mode covered.
- [ ] ORT 1.24.4 CPU kernel support separated from ONNX schema.
- [ ] At least 6 idioms priced.
- [ ] Final blunt verdict: name the best untried scatter-reduction probe, or say current scatter shelf is complete.
