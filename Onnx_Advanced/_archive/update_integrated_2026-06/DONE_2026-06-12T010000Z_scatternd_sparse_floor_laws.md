# Sparse ScatterND-delta floor laws (task154 Aditya 3857 decode, rework-worker-154b)

## What I discovered (cost-model floor laws for the §38/§39 sparse-ScatterND "free Where-from-input" pattern)

Decoded the team-best for task154 (a fixed-15x15 reflection-MOVE task): a 28-node opset-16 graph,
`GatherND×7 ScatterND×2 Mul×14 Sub×2 Concat×2 Split×1`, cost **3857 = memory 2184 + params 1673**,
static==profile. Output = `ScatterND(input, final_indices[270,4], delta_updates[270],
reduction='add')` — input passes through FREE, only the changed cells get F32 deltas. This is the
cleanest exemplar of §38/§48 I've seen and it exposes several reusable floor laws the library
doesn't state explicitly:

1. **The scorer's params = Σ initializer ELEMENT counts, dtype-AGNOSTIC** (read from
   `neurogolf_utils.calculate_params`: `params += math.prod(init.dims)`, no itemsize). So
   **I64→I32 index narrowing saves ZERO params.** This contradicts the intuition in several worker
   prompts ("halve index dtype width to cut params"). The ONLY way to cut index params is fewer
   index ELEMENTS (fewer scatter rows, or smaller index last-dim). Worth a one-liner in
   scoring_internals.md / op_cost_patterns.md.

2. **ScatterND updates dtype is HARD-BOUND to data dtype == output dtype** in ORT. Probed at
   opset 16: F16 updates into F32 data → `Type Error: Type parameter (T) of Optype (ScatterND)
   bound to different types (tensor(float) and tensor(float16))`. So for a free F32-output
   ScatterND, `delta_updates` is F32-floored; F16 cannot shrink it (a Cast just re-materialises the
   F32 tensor). ort_reality.md ScatterND row should note "updates/data/output dtype must match;
   F16-into-F32 rejected".

3. **The batch coordinate cannot be dropped from a rank-4 scalar ScatterND index** even when it's
   always 0. To use [N,3] indices you must operate on a rank-3 tensor, but reshaping/slicing the
   free `input [1,10,30,30]` charges a full intermediate (e.g. [10,30,30] F32 = 36000B). The only
   free large tensor is the LITERAL `input` node — any reshape/slice/transpose between input and the
   final scatter breaks the free chain. So index last-dim = data rank is a hard floor.

4. **ScatterND reduction='add' with DUPLICATE indices in a single call is nondeterministic** (ONNX
   spec). For many-to-one writes (multiple sources → one destination, e.g. reflections that
   collide), you must PRE-COMPACT via a separate intermediate `ScatterND-add` into a unique-key
   buffer, THEN scatter unique final indices. Aditya's "dup path" does exactly this. This is a
   correctness requirement, not removable scaffolding — worth noting next to §38/§139.

5. **A constant ±1 update block (here source_delta[160], pure function of channel: ch0→+1, ch5→−1)
   is cheapest as an INITIALIZER (160 params), NOT as ConstantOfShape/Expand-generated intermediates
   (640B memory for [80]+[80] F32).** Because params and memory weigh equally in cost = memory+params,
   a runtime-generated vector only wins when it would otherwise be an initializer LARGER than the
   intermediate bytes it produces. For small constant update vectors, keep them as initializers.

## Where
- Team-best: `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task154.onnx`
  (SHA 598f6d83…), profiled static==profile==3857 over 266 real inputs.
- Decode + floor proof: `The_400_Tasks/Task_by_Task/task154.md` (Cost analysis section).

## Why it matters
Saved a rework session from chasing dead levers (F16 delta, I32 indices, coverage trimming) that
LOOK promising but are all blocked by the cost-model facts above. Verdict on task154: 3857 is the
true sparse floor; DROP. These five floor laws generalise to every §38/§39 sparse-ScatterND ship.

## What's missing from the library
- op_cost_patterns.md / scoring_internals.md: the "params = element count, dtype-agnostic" law is
  implied but not stated as a worker-facing rule. Several prompts still suggest I32-index params
  savings.
- ort_reality.md ScatterND row: add the F16-into-F32 type-binding rejection + the duplicate-index
  nondeterminism note.
- A by_symptom route for "reflection/translation MOVE of one color on a FIXED grid" → §38/§48
  sparse ScatterND-add deltas (the cheapest known family), with the 5 floor laws as the build checklist.
