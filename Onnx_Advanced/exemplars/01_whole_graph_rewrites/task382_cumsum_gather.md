# task382 — CumSum + Gather geometric encoding (from-scratch)

| Field | Value |
|---|---|
| Task | task382 |
| Family | Spatial_and_Geometric |
| Cost reduction | -73,443 (from-scratch) |
| Pattern | Whole-graph rewrite using opset-11 CumSum |
| Build script | `The_400_Tasks/Task_by_Task/task382/prototypes/` |

## Rule

task382's algorithm involves geometric position computation — distances, cumulative offsets, or similar position-dependent values. The original ONNX implemented this via chains of Add and arithmetic.

## Key insight: CumSum collapses chained Adds

A chain like:
```
sum0 = Add(a0, a1)
sum1 = Add(sum0, a2)
sum2 = Add(sum1, a3)
...
sumN = Add(sumN-1, aN)
```

is equivalent to a single `CumSum` op (opset 11+):

```
all_inputs = Concat([a0, a1, ..., aN], axis=K)
cumulative = CumSum(all_inputs, axis=K)
```

One op, one intermediate (the cumulative tensor). The N-1 intermediate sums are eliminated.

For geometric tasks where the algorithm computes "for each position, the cumulative offset/distance/etc.", CumSum is the cheapest representation.

## Implementation

```python
# Bump opset to 11 for CumSum
model.opset_import[0].version = 11

# Replace Add-chain with CumSum
ones_init = init [W] INT32 of all-1s
cumsum_x = CumSum(ones_init, axis=0)   # [W] INT32 = [1, 2, 3, ..., W]
# Use cumsum_x as positional indices for downstream Gather
output = Gather(values, cumsum_x)
```

## Why it works

- CumSum is 1 op vs N Add ops. Saves N-1 intermediates.
- The math is identical (associative + commutative).
- opset 11+ supports CumSum natively in ORT.

## Generalizes to

`techniques.md` §6 (Add-chain → Sum) handles the simple case. CumSum is the broader version for "cumulative" patterns where you need ALL the partial sums, not just the final.

Apply to:
- Position-dependent computations (each output cell needs `f(position)`).
- Distance computations (cumulative offset along an axis).
- Geometric transforms requiring per-cell offsets.
- Histogram / prefix-sum-based reductions.

## Reproducibility

Read `The_400_Tasks/Task_by_Task/task382/prototypes/build_task382_*.py`.

The CumSum trick requires opset 11+. If your model is opset 10, the bump is safe — see `ort_reality.md`. Other opset-11+ ops worth knowing for geometric algorithms:
- `ScatterND` — scatter values into a tensor at index locations (replaces certain Gather-with-modify patterns).
- `Round` — for nearest-integer rounding.
- `BitShift` — for bit-level manipulation.

For any geometric task with N consecutive Adds, CumSum is the first optimization to try.

## Cross-ref

- `techniques.md` §6 (add_chain_to_sum — simpler Add → Sum).
- `ort_reality.md` — list of opset 11+ ops verified working.
- `Algorithm_Library/algorithm_patterns.md` §9 (whole-grid Conv summary — another geometric encoding).
