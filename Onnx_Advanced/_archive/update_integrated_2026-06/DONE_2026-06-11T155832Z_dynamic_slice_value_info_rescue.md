# Dynamic-Slice crop: bake value_info to rescue the scorer (task036, 1690→1016)

## What I discovered

A `Slice` (or any op) whose output shape ONNX shape-inference cannot resolve —
e.g. a Slice with **runtime (data-dependent) `starts`/`ends`** — leaves a
`[?,?,?,?]` value_info. The scorer's `calculate_memory` (neurogolf_utils.py:228-233)
iterates EVERY node output and does `return None` the moment it hits a dim with
no concrete `dim_value`. So a model that runs perfectly in ORT scores
`memory=None` → `score_network` → None → "FAIL: score_network returned None".

**The fix (verified, ships):** bake a concrete value_info for the dynamic
output, then re-run `shape_inference(strict_mode=True)` so the concrete shape
propagates downstream. `infer_shapes` *merges with* — does not clear — existing
value_info, so a hand-set `[1,1,5,5]` survives and the rest of the graph
(ReduceMax/Cast/Where/Equal on it) re-resolves. Save the merged model.

```python
def _bake_value_info(model):
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model, strict_mode=True)
    for vi in model.graph.value_info:        # pin the dynamic Slice output
        if vi.name == "the_dynamic_crop":
            dims = vi.type.tensor_type.shape.dim
            del dims[:]
            for d in (1, 1, 5, 5): dims.add().dim_value = d
            break
    return shape_inference.infer_shapes(model, strict_mode=True)  # propagate
```

This is exactly how the team-best aditya task036 net (cost 1625) ships its own
dynamic `Slice` crop: its `obj_crop5` value_info is saved as `[1,1,5,5]` even
though a from-scratch `infer_shapes` returns `[?,?,?,?]` for it.

**Important nuance vs traps.md #3** (which says "clear stale value_info after
shape surgery, or ORT fails"): that is the OPPOSITE situation. #3 is about
*removing* a now-wrong baked shape. This is about *adding* a correct baked shape
that inference can't derive. Both are real; the library only documents the first.
Rule of thumb: clear value_info when your edit changed a shape inference CAN
recompute; bake value_info when inference CANNOT recompute it (runtime indices).

Only valid when the dynamic shape is provably constant across all real inputs
(here the crop window is a fixed 5×5, proven: 0 ArgMax/shape errors over 2000+
real officials + 30k fresh ARC-GEN). The profiler then observes the same shape
→ `max(static, runtime)` is exact, not an understated credit-probe.

## Where

- Builder: `task036/prototypes/build_task036_v4_slice.py`
- Staged: `staged_submits/task036_sliceswap_cost1016/` (manifest + onnx)
- Reference: `task036/prototypes/team_aditya/task036.onnx` (the 1625 net, same trick)

## Why it matters

task036: GatherND→Slice for a data-dependent fixed-size crop dropped the cost
from 1690 to **1016** (mem 1515→915), the single biggest cut being the
elimination of v1's `[1,1,5,5,3]` INT64 GatherND index intermediate (600 B =
39.6% of memory). The Slice geometry is two `INT64[3]` = 48 B. Beats the verified
team-best 1625 by +0.47 LB (ln 1625/1016). **This same value_info-bake unblocks
ANY dynamic-crop task** where Slice/GatherND reads a fixed-size window at a
runtime offset and the local scorer returns None.

## Secondary: GatherND-vs-Slice cost law for fixed-window dynamic reads

For a fixed-size window read at a data-dependent offset:
- **Slice** geometry = 2× `INT64[k]` (starts+ends over k axes) = `16k` bytes.
- **GatherND** index = a full `[1,1,h,w,rank]` INT64 tensor = `h*w*rank*8` bytes
  (task036: 5*5*3*8 = 600 B) PLUS the Add/Reshape offset chain to build it.

Slice dominates whenever the window has >~2 cells. Reach for Slice first; only
use GatherND when the gathered cells are genuinely scattered (non-contiguous).
This is the #39 "GatherND sparse reads only where needed" guidance made
quantitative: a contiguous crop is NOT a sparse read.

## Dead-end recorded (so nobody re-runs it)

Trying to make task036 100%-truth (the canonical base-16 row-span selector
mis-picks a span-6 noise color over a span-4 target on ~2/200k fresh inputs) by
swapping to a robust integer ReduceMax/ArgMax per-color span selector
(`build_task036_v5_robust.py`) costs **9216** — the six `[1,10,30,1]`/`[1,10,1,30]`
per-color mask reductions are 1200 B each — AND still misses 1/200k (smallest-area
heuristic ≠ true connected-component rule). Robust selection is intrinsically
~8× the memory of the base-16 trick at this task. The 100%-correct niche belongs
to a moment-based rule (aditya @1625); the cheap niche (@1016, byte-equal to the
shipped net) belongs to v4. Don't try to have both at sub-1625 cost.

## Sorter note: traps.md #3 — extend the "stale value_info" row with the INVERSE: a Slice/Gather with runtime starts/ends leaves `[?,...]` that calculate_memory rejects (returns None); BAKE a concrete value_info for it + re-run infer_shapes (merges, doesn't clear) to rescue the score. Confirmed on the team-best aditya task036 net and on the new task036 v4 (1690→1016).

## Sorter note: op_cost_patterns.md — add the GatherND-vs-Slice quantitative law for fixed-window dynamic reads: Slice geom = 16k bytes vs GatherND index = h*w*rank*8 bytes; Slice wins for any contiguous crop >~2 cells (task036 GatherND 600 B index → Slice 48 B, -552 B). Make #39 quantitative: a contiguous crop is NOT a sparse read.

## Sorter note: task_techniques_index.md — task036: GatherND→Slice + value_info-bake, 1690→1016, beats team-best 1625; family Object_Based/Cropping.
