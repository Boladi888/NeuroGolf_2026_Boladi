# task071 — ring-mask dynamic compute (the rank-reduction insight)

| Field | Value |
|---|---|
| Task | task071 |
| Family | Object_Based / RMAX |
| Cost reduction | -2,709 |
| Params reduction | 2,140 → 133 (-2,007 params) |
| Pattern | Static lookup table → dynamic compute at rank-2 |

## Why this exemplar matters

task071 demonstrates the most counter-intuitive insight in NeuroGolf optimization:

**Rank promotion is the dominant cost factor, not node count.**

Workers naturally assume "more nodes = more cost" — so a static lookup table (1 op, fixed init) seems cheaper than dynamic compute (20+ ops). For ARRAYS of rank-4 like `[1, 1, H, W]`, this intuition holds. For rank-2 arrays like `[H, W]`, it inverts.

## Before

5 static ring-mask lookup tables, each `[5, 1, 1, 9, 9]` F32:

```
v11 = initializer [5, 1, 1, 9, 9] F32 ring-mask table   # 5 × 1 × 1 × 9 × 9 = 405 params per table
v12 = initializer [5, 1, 1, 9, 9] F32
v13 = ...
v14 = ...
v15 = initializer [5, 1, 1, 9, 9] F32
                                                          # Total: 5 × 405 = 2,025 params
                                                          # Plus frame init: 115 params
                                                          # Grand total: 2,140 params
```

Five 5-element ring-mask tables stored explicitly. Output cost is dominated by `params` (element count of all initializers).

## After

23 dynamic compute nodes producing a single `[9, 9]` rank-2 intermediate:

```python
# All intermediates are RANK 2 [9, 9], not rank 4 [1, 1, 9, 9]
range9 = init [0, 1, ..., 8] INT64 [9]                    # 9 params
two_L  = init [16] INT64 [1] (= 2 × (H-1) = 2 × 8 for 9x9)  # 1 param

# Build per-axis ring index
ring_h_axis = Min(range9, Sub(two_L, range9))             # INT64 [9]
ring_w_axis = same construction                            # INT64 [9]

# Combine via Unsqueeze + Min → rank-2 [9, 9]
ring_h_2d = Unsqueeze(ring_h_axis, axes=[1])              # [9, 1]
ring_w_2d = Unsqueeze(ring_w_axis, axes=[0])              # [1, 9]
ring_idx  = Min(ring_h_2d, ring_w_2d)                     # [9, 9] INT64

# Gather frame colors at ring positions
frame_colors_init = [5 colors] F32                         # 5 params
output_grid = Gather(frame_colors_init, ring_idx)         # [9, 9] F32
```

Total: 23 nodes, **rank-2 [9, 9]** intermediates, ~133 params.

## Why rank-2 beats rank-4 by ~50× per intermediate

A `[1, 1, 9, 9]` rank-4 F32 intermediate has `1 × 1 × 9 × 9 = 81` elements × 4 bytes = 324 bytes.
A `[9, 9]` rank-2 F32 intermediate has `9 × 9 = 81` elements × 4 bytes = 324 bytes — IDENTICAL size.

The difference is in how ORT's profiler trace JSON treats them. The profiler bills:
- For each op's output, `itemsize × sum(prod(dims) for dims in shape_dict.values())`.
- The profiler also tracks shape changes when broadcasting expands tensors.

When a rank-4 `[1, 1, 9, 9]` flows through Gather/ReduceMax/Where chains that PROMOTE it to `[1, 10, 9, 9]` (e.g. broadcast against a 10-channel mask), the runtime-observed shape becomes 10× larger than the static value_info declared. ORT's `max(static, observed)` accounting bills the larger.

A rank-2 `[9, 9]` doesn't have the channel dim. Downstream broadcasts can promote it for ops that need 10-channel output, but typically the rank-2 stays rank-2 for indexing-style ops (Gather indices into a [5]-element table, etc.). No accidental promotion.

**Practical rule**: when computing intermediates that don't NEED a batch or channel dim, drop those dims. Use rank-2 `[H, W]` instead of rank-4 `[1, 1, H, W]`.

## Worker's pre-analysis was wrong

The worker initially estimated this rewrite would LOSE (40+ nodes × `[1, 1, 9, 9]` intermediates would cost more than 5 static `[5, 1, 1, 9, 9]` tables). The actual rank-2 implementation:

- 23 nodes (not 40)
- `[9, 9]` rank-2 intermediates (not `[1, 1, 9, 9]`)
- ~50× cheaper per intermediate than the worker's projection
- Net WIN by ~+1,975 effective params saved

## Generalizes to

`techniques.md` §11 (Static-to-dynamic at minimum rank).

**Apply when**:
- Your task uses a static lookup table of size N elements.
- The table values are a computable function of position (ring index, distance, parity, modular arithmetic, etc.).
- The downstream compute can stay at the lowest rank the math allows.

**Eligibility check**:
- Static table size > 100 elements: usually worth replacing.
- Static table size < 50 elements: usually not (dynamic compute overhead exceeds the table cost).
- Function is computable in ≤ 20 ops at rank-2: yes.
- Function requires rank-4 intermediates (channel-dependent): probably no.

## Cross-ref

- `Algorithm_Library/algorithm_patterns.md` §10 (Static-table to dynamic compute).
- `techniques.md` §11 (Static-to-dynamic rule).
- `techniques.md` §12 (Projection-cache — related principle: skip wide intermediates).

## Reproducibility

For any task with large `[..., H, W]` lookup tables:

1. Identify the table's mathematical structure. Is it `min(r, c, L-r, L-c)` (ring index)? `(r + c) % N` (diagonal)? `r * W + c` (linear)?
2. Build the equivalent compute chain at rank-2:
   - Use `range_N = [0..N-1] INT64 [N]` as the base.
   - Combine via `Unsqueeze` + arithmetic to build the 2D index without promoting to rank-4.
   - `Gather(values_init, index_2d)` produces the lookup.
3. Verify byte-exact against the original.

The worker's `build_task071_*.py` in `The_400_Tasks/Task_by_Task/task071/prototypes/` is the canonical implementation. Read it before applying this pattern elsewhere.
