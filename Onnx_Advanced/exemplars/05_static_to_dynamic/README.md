# 05_static_to_dynamic — worked case studies

Replace static lookup tables with dynamic compute at minimum rank.

## Files

- `task071_ring_mask.md` — ★ THE rank-2-vs-rank-4 insight. Static `[5, 1, 1, 9, 9]` ring-mask table replaced by 23 rank-2 `[9, 9]` compute nodes. Net -2,709 bytes + -2,007 params.
- `task201_reduce_reorder.md` — Reorder ReduceSum earlier in the chain to shrink downstream intermediates. -86,326 bytes.

## Key principle

**Rank promotion is the dominant cost factor, not node count.**

A rank-2 `[9, 9]` intermediate is ~50× cheaper than a rank-4 `[1, 1, 9, 9]` intermediate when ORT's profiler tracks the shape through downstream broadcasting. Drop unused dims.
