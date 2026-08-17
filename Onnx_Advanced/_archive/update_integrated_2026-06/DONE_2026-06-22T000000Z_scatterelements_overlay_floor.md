# ScatterElements input-overlay family — the three irreducible costs (task147)

## What I discovered

The "scatter-delta overlay" pattern (reuse the F32 `input` as ScatterElements `data` so the
graph output is FREE; only materialize a small `updates` delta plane) has THREE structural
costs, and each is locked by an op-level constraint — so the family floor is provable, not
guess-and-check:

1. **`updates` dtype is locked to the `data` dtype.** ORT shape-inference rejects
   `ScatterElements(data=F32, updates=F16)` with *"updates has inconsistent type
   tensor(float16)"*. If `data` is the F32 `input`, the delta plane CANNOT be F16/U8/BOOL —
   it is F32-locked by the scatter, regardless of how cheap the producer (Conv etc.) could
   emit it. The only escape is casting the whole input to the smaller dtype first, which
   births a full `[1,10,30,30]` cast intermediate (fatal).

2. **`indices.shape == updates.shape` is enforced.** No broadcasting of indices. To write K
   channels in one scatter you pay a K×(region) INT64 **initializer** index block — but as an
   initializer it's element-counted (params), NOT byte-counted, so INT64 vs INT32 is a tie and
   it's cheap. Do NOT runtime-generate indices via ConstantOfShape/Concat to "save params":
   a materialized I64 index tensor costs 8 B/element in MEMORY and dwarfs the param saving
   (measured task147: 100→32 params but cost 532→1616).

3. **One scatter, not a chain.** Chaining scatters to touch multiple channels births a
   full-input-shaped F32 intermediate between them (`[1,10,30,30]` = 36000 B) — fatal. All
   target channels must be written in a single ScatterElements call, which is what forces the
   multi-channel `updates`/`indices` shape in the first place.

## Corollary dead-ends (measured on task147, green-neighbor recolor)

- **Single-channel count map + Sign/Mul tail** to avoid a 2-channel Conv output: STRICTLY WORSE
  (532→1244). Each single-channel intermediate adds its own 144 B AND you still must Concat
  them back to the multi-channel `updates` shape for the one scatter. A multi-output Conv that
  emits all delta channels at once is the cheapest producer — don't decompose it.
- **Shrinking scatter `data` to a cropped `[1,10,h,w]` slice + Pad-after**: WORSE (532→3428).
  The cropped data AND the scatter output are both charged intermediates; using the full input
  as `data` writes straight to the free `output`, which is the whole point.

## Why it matters

Saves a future worker from re-deriving the floor of any input-overlay scatter task. The floor =
slice-in (Conv input) + multi-channel F32 `updates` (dtype-locked) + K-channel INT64 index
initializer. All three are op-forced. If a prompt asks to "shrink to single-channel count map"
or "F16 the delta," these are already falsified with measured costs.

## Where

- Task: task147 (green-3 with 4-neighbor green-3 → cyan-8). Team-best 532 (Aditya scatter-overlay).
- Build probes: `/tmp/build_lever_a.py`, `/tmp/build_lever_d.py`, `/tmp/build_lever_idx.py`,
  `/tmp/probe_scatter_dtype.py` (transient; logic recorded in `Task_by_Task/task147.md`
  Sub-532 Audit 2026-06-22).
- Verdicts logged in `The_400_Tasks/Task_by_Task/task147.md` don't-do log.

## What's missing from the library

`op_cost_patterns.md` ScatterND/ScatterElements rows note win-thresholds but NOT the
`updates`-dtype-lock and `indices.shape==updates.shape` constraints, nor the
"full-input-as-data → free output" overlay floor law. Worth a short row under the
ScatterElements entry: "overlay floor = slice-in + dtype-locked F32 updates + K-channel index
init; single-channel decomposition and F16 updates both price out."
