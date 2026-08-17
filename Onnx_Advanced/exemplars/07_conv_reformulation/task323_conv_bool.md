# task323 — Conv → bool reformulation

| Field | Value |
|---|---|
| Task | task323 |
| Cost reduction | -62,838 |
| Pattern | Conv → bool predicate reformulation |

## Rule

task323's algorithm used `Conv(input, kernel) → threshold → mask` pattern. The Conv computes a per-cell linear combination, then a threshold decides whether each cell is in the output.

When the Conv weights are 0/1 (just counting neighbors) and the threshold is integer (e.g., "at least 3 neighbors with the property"), the chain can be reformulated as pure bool ops:

```
# Before: Conv + threshold
neighbor_count = Conv(active_mask, [[1,1,1],[1,0,1],[1,1,1]])   # 8-conn count, F32 [1,1,H,W]
has_3_neighbors = Greater(neighbor_count, 2)                    # BOOL

# After: per-neighbor And/Or chains
# Build per-neighbor masks via Slice/Pad/Roll
neighbor_N  = Slice(active_mask, [+1, axis=2])    # shifted up by 1
neighbor_S  = Slice(active_mask, [-1, axis=2])    # shifted down by 1
neighbor_E  = Slice(active_mask, [-1, axis=3])    # shifted right by 1
neighbor_W  = Slice(active_mask, [+1, axis=3])    # shifted left by 1
# ... 4 diagonal neighbors too for 8-conn ...

# "At least 3 neighbors" = at least 3 of 8 bool inputs are True
# Express via counting BOOL adds (ReduceSum of stacked bools)
all_neighbors = Concat([N, S, E, W, NE, NW, SE, SW], axis=1)   # [1, 8, H, W] BOOL
count = ReduceSum(Cast(all_neighbors, INT32), axis=1)           # [1, 1, H, W] INT32 (NOT F32!)
has_3 = Greater(count, 2)                                       # BOOL
```

## Why it saves bytes

- Conv with F32 weights creates an F32 intermediate (the counting result).
- The bool reformulation keeps everything in BOOL/INT32 (1 byte for BOOL, 4 for INT32) instead of F32 (4 bytes for the Conv output).
- Specifically, the Conv output `[1, 1, H, W] F32 = 3,600 bytes` is replaced by `[1, 8, H, W] BOOL = 7,200 bytes` (the stacked neighbors) PLUS `[1, 1, H, W] INT32 = 3,600 bytes` (the count). Stacked neighbors larger, but the rest of the downstream chain stays smaller.

The actual savings depend on what the downstream consumer chain looks like. task323's chain stayed bool-first after the Conv→bool refactor, so cumulative -62,838 bytes.

## When to apply

When:
- Conv weights are 0/1 (counting/neighborhood ops).
- Threshold is integer.
- Downstream is bool-heavy.

When NOT to apply:
- Conv weights are non-trivial floats (real linear filtering).
- Threshold is at the boundary of integer precision (e.g., 2.5 — can't distinguish from 2 in INT counting).

## Generalizes to

`Algorithm_Library/algorithm_patterns.md` §7 (Connected components via 4-conn or 8-conn Conv). When the Conv is "count neighbors", the reformulation is straightforward.

`techniques.md` rule 8 (input → BOOL whole-graph) — task323 is a partial case where the input is bool-eligible and propagates through a custom counting Conv equivalent.

## Reproducibility

Identify Conv ops with integer weights. Replace with stack-and-count via Slice/Concat/ReduceSum chains. Keep BOOL through Threshold. Verify byte-exact.
