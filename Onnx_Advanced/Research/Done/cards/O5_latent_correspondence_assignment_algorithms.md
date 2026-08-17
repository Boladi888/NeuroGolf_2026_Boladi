# Research Card O5 - Latent correspondence and assignment algorithms for static ONNX

**Deliverable**: one copyable Markdown research note. Save the note under `Research/Done` with basename `o5_latent_correspondence_assignment_algorithms`.

## Researcher contract

You are an internet-research assistant. You have web access but NO repo access. Produce one self-contained Markdown note. Use primary sources where possible: papers, canonical algorithm references, official docs, and source repos.

## Why this card exists

The live taxonomy names latent correspondence / analogy transfer and general packing / assignment / collision-free layout as open-research families. These tasks often need matching input objects to output slots, resolving collisions, or transferring a transformation from one object pair to another.

We need to know whether any assignment/correspondence algorithms have small-grid static-DAG forms that can beat existing NeuroGolf idioms such as pairwise comparisons, TopK, ScatterElements routing, or hardcoded slot banks.

## Project facts

```
Runtime: ONNX Runtime 1.24.4 CPUExecutionProvider.
Typical object count: small, often <=10.
Grid size: <=30x30.
Cost = intermediate_memory_bytes + param_elements.
Loops/control flow and data-dependent dynamic output shapes are unavailable.
Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.
TopK is float-only for values; indices are int64.
ScatterElements/ScatterND and small pairwise comparison matrices are available.
```

## Required answer order

1. **Algorithm survey**: assignment, bipartite matching, stable matching, auction algorithms, Hungarian variants, bottleneck assignment, graph matching, and small-N correspondence heuristics.
2. **Static-DAG transfer test**: for each family, explain whether it can be unrolled for N<=10 without loops and whether the tensor shapes stay static.
3. **Cost comparison**: compare each transferable idea to current incumbents: pairwise comparison rankers, TopK, ScatterElements routing, `(K+1)` noop bins, and slot-wise one-hot banks.
4. **Exactness and tie rules**: state how ties/collisions are resolved and whether the method gives deterministic exact outputs.
5. **Probe shortlist**: identify any repo-access probes worth running.

## Required tables

Table A:

`| algorithm family | primary source | static unroll possible for N<=10? | ONNX primitives | tie/collision behavior | likely cost | verdict |`

Table B:

`| NeuroGolf pattern | incumbent | assignment/correspondence alternative | byte/param delta estimate | exactness condition | probe? |`

## Mandatory caveats

- Generic polynomial-time algorithms are not useful if they require data-dependent loops.
- Dense `N x N x features` pair matrices can be cheap for N<=10 but expensive if expanded over H,W.
- Matching can usually be replaced by sorting/ranking if the generator gives a scalar key; identify when that is true.
- If all objects have unique colors/sizes/positions, simpler histograms or TopK may dominate.

## Source-tag standard

`[PROMPT]`, `[PRIMARY]`, `[SECONDARY]`, `[DERIVED]`, `[SPECULATIVE]`.

## Final checklist

- [ ] At least 6 algorithm families surveyed.
- [ ] Every plausible transfer has a static tensor shape plan.
- [ ] Every top candidate is priced against a current incumbent.
- [ ] Tie/collision behavior is explicit.
- [ ] Final blunt verdict: name the best correspondence/assignment probe, or say current slot-routing techniques dominate.
