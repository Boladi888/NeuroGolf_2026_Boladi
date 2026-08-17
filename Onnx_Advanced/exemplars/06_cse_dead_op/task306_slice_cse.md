# task306 — Slice common-subexpression elimination

| Field | Value |
|---|---|
| Task | task306 |
| Pattern | `slice_cse` (techniques.md §13) |
| Cost reduction | -6,504 |
| Detector | `tools/spread_patterns/pass_slice_cse.py` |

## Before

4 Slice nodes producing tensors `src_0_1`, `src_0_2`, `src_1_1`, `src_1_2` — BUT with identical params to existing Slice nodes that produced different output names. Identical inputs, identical outputs.

```
Slice(input, starts=[0,0,0,1], ends=[1,10,30,29], axes=[0,1,2,3]) → tensor_X    # original
Slice(input, starts=[0,0,0,1], ends=[1,10,30,29], axes=[0,1,2,3]) → tensor_Y    # duplicate!
... tensor_X and tensor_Y both consumed by downstream Where ops as fallbacks ...
```

`onnxsim` is SUPPOSED to catch this (common-subexpression elimination across nodes), but it missed on task306. Manual inspection found the duplicates.

## After

```
Slice(input, starts=[0,0,0,1], ends=[1,10,30,29], axes=[0,1,2,3]) → tensor_X    # original kept
... downstream Where ops rewired from tensor_Y → tensor_X ...
# tensor_Y Slice node and its starts/ends/axes inits dropped (no remaining consumers)
```

## Why this works

Slice with identical `(input, starts, ends, axes, steps)` produces byte-identical output. ONNX semantics guarantee this. Replacing all references to `tensor_Y` with `tensor_X` and dropping the duplicate Slice is exact.

## Why onnxsim missed it

`onnxsim` does graph-level CSE in theory, but the implementation may have heuristics that skip Slices with multi-input parameter tensors (`starts`, `ends`, etc. as separate initializers). When parameter inits are themselves duplicated (e.g., `starts_0` and `starts_1` are both `[0,0,0,1]`), onnxsim sees them as distinct and doesn't dedupe.

The custom `pass_slice_cse.py` hashes by the VALUE of the params, not the init name. This catches what onnxsim misses.

## Generalizes to

`techniques.md` §13 (slice_cse). Apply to ANY task where multiple Slice nodes have identical params.

Detector signature:
- Hash each Slice by `(input_name, tuple(starts_values), tuple(ends_values), tuple(axes_values), tuple(steps_values))`.
- Group by key.
- For each group with > 1 member, keep the first; rewire others' consumers to use the first's output; drop the duplicates.

## Reproducibility

Run `tools/spread_scan.py --tasks taskNNN`. Detector automated. Manual application: inventory Slice nodes, hash by value not name, find duplicates, rewire.

## Cross-ref

- `06_cse_dead_op/task093_dead_relu_post_maxpool.md` — different kind of dead-op (provably-identity Relu).
- `06_cse_dead_op/task240_dead_end.md` — documented dead-end (Pad mode merge fails).
- `techniques.md` §13.
