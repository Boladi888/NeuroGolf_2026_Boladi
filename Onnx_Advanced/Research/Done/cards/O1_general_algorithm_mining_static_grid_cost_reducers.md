# Research Card O1 - General algorithm mining for static-grid cost reducers

**Deliverable**: one copyable Markdown research note. Save the note under `Research/Done` with basename `o1_general_algorithm_mining_static_grid_cost_reducers`.

## Researcher contract

You are an internet-research assistant. You have web access but NO repo access. Produce one self-contained Markdown note. Do not ask follow-up questions. Every nontrivial claim must use source tags and human-readable citations.

Prefer primary sources: peer-reviewed papers, official algorithm descriptions, canonical implementation repos, ONNX specs, and ONNX Runtime docs/source. Use secondary sources only to find primary sources.

## Why this card exists

The current ONNX operator audit is mostly complete: opset 11-26, sequence/container ops, generator ops, and obscure spatial ops were already swept. This card should look outside ONNX operator lists for *algorithmic identities* that might lower NeuroGolf cost when encoded as static ONNX DAGs.

The goal is to surface ideas the library has not framed yet, not to repeat standard ONNX usage.

## Project facts

```
Runtime: ONNX Runtime 1.24.4 CPUExecutionProvider.
Typical tensor: [1,C,H,W], C<=10, H,W<=30.
Values: small ARC colors 0..9, masks, coordinates, counts.
Cost = intermediate_memory_bytes + param_elements.
Graph input and final graph output are free.
Params are dtype-blind element counts.
Loops/control flow are unavailable: Loop, Scan, NonZero, Unique, Script, Function, Compress are banned.
Sequence-family ops are scorer-rejected. ReverseSequence is also banned by substring.
```

## Already-covered or declined topics

Do not spend time re-proposing these unless a primary source gives a genuinely new static-DAG construction with byte/param arithmetic:

- Broad ONNX unexploited-operator audits.
- Connected-component labeling and pathfinding/wavefront as generic topics.
- ARC-AGI winner surveys.
- Bit-packing as a general topic.
- Equality saturation/superoptimizer ports.
- FunctionProto/If/subgraph tricks.
- Sequence/container/string/sparse values.

## Required answer order

1. **Survey frame**: name the algorithm families you searched and why they might transfer to fixed 30x30 discrete grids.
2. **Candidate table**: list every nontrivial transfer candidate, with source, core identity, required ONNX primitives, expected score mechanism, and reason it may or may not beat existing library techniques.
3. **Deep dives**: fully price the top 5 candidates against current NeuroGolf-style incumbents.
4. **Dead ends**: list attractive but non-transferable ideas and the exact blocker.
5. **Recommended follow-up cards**: if any candidate deserves repo-access probing, write the exact proposed card title and probe scope.

## Algorithm families to include

At minimum, search primary sources for:

- Summed-area / integral-image identities for rectangle sums, box tests, variance, and moment features.
- Fast morphology beyond naive dilation/erosion, including separable rectangular morphology and van Herk/Gil-Werman-style ideas.
- Distance-transform and medial-axis ideas, but only if they become bounded static reductions rather than generic pathfinding.
- Rank/order filters, min/max filters, and sliding-window extrema algorithms.
- Small-grid assignment, matching, correspondence, and collision-free layout identities.
- Tensor-network / low-rank / separability tricks outside the existing CP-rank cards, especially for structured integer maps.
- Cellular automata or finite-state transducer identities that compile into short bounded-depth static DAGs.

## Required tables

Table A:

`| family | primary source | core identity | possible ONNX lowering | score mechanism | current incumbent | verdict |`

Table B:

`| candidate | before cost model | after cost model | exactness conditions | repo probe needed? | source tags |`

## Source-tag standard

`[PROMPT]` fact given here.
`[PRIMARY]` peer-reviewed paper, official docs/spec, or canonical source repo.
`[SECONDARY]` explanatory source.
`[DERIVED]` deduction from cited sources and the cost model.
`[SPECULATIVE]` unproven construction.

## Final checklist

- [ ] At least 8 distinct algorithm families searched.
- [ ] At least 15 primary-source-backed candidate/negative rows.
- [ ] Top 5 candidates have byte/param arithmetic.
- [ ] Declined topics were not reintroduced as broad cards.
- [ ] Final blunt verdict: name the single most promising new research direction, or say the outside-algorithm scan found nothing above current library.
