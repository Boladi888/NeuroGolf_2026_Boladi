# task349 — 10-site Equal-fold

| Field | Value |
|---|---|
| Task | task349 |
| Pattern | 10-site Equal-fold (techniques.md §5) |
| Cost reduction | -18,001 |
| Estimated LB | +0.094 |

## Pattern

task349 had 10 parallel instances of the Equal-fold pattern — one per color channel — each comparing a different basis vector against a fixed reference.

The cumulative savings across 10 simultaneous folds (each ~-1,800 bytes per channel) yielded the -18,001 total.

## Generalizes to

`techniques.md` §5 (Equal-fold). When a task has N parallel Sub-Mul-Less or Sub-Abs-Less chains, ALL N can be folded simultaneously.

For N > 1 with a shared variable input, also see `task368_shared_cast_equal_fold.md` for the opset-10 shared-Cast optimization.

## Reproducibility

The detector handles each chain independently. Running `tools/spread_scan.py` will report N applications of `equal_fold_distance_threshold` (or `shared_cast_equal_fold` if applicable). The cumulative savings stack additively.

## Cross-ref

- `task233_equal_fold.md` — pointwise version.
- `task133_equal_fold_reducesum.md` — ReduceSum variant.
- `task368_shared_cast_equal_fold.md` — shared-Cast for opset 10.
