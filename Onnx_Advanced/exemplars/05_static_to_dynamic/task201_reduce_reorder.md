# task201 — reduce reorder (rank reduction via op reordering)

| Field | Value |
|---|---|
| Task | task201 |
| Cost reduction | -86,326 (171,221 → 84,895) |
| Pattern | Reorder reduction earlier to shrink downstream intermediates |

## The principle

When a wide tensor flows through a chain that ENDS in a reduction (ReduceSum / ReduceMax / etc.), see if the reduction can be MOVED EARLIER. The intermediates BETWEEN the reduction and the original position become smaller.

```
# Before
input [1, 10, 30, 30] F32
  → expensive_op_1 → [1, 10, 30, 30] F32 (36,000 bytes intermediate)
  → expensive_op_2 → [1, 10, 30, 30] F32 (36,000 bytes intermediate)
  → ReduceSum(axes=[1]) → [1, 1, 30, 30] F32

# After: move ReduceSum earlier if it commutes with the downstream ops
input [1, 10, 30, 30] F32
  → ReduceSum(axes=[1]) → [1, 1, 30, 30] F32 (3,600 bytes)
  → expensive_op_1 → [1, 1, 30, 30] F32 (3,600 bytes)
  → expensive_op_2 → [1, 1, 30, 30] F32 (3,600 bytes)
```

The intermediates went from 36,000 bytes each to 3,600 — 10× smaller.

## When it's safe

The reordering works ONLY when the downstream ops COMMUTE with the reduction:
- **Linear ops** (Add, Sub, Mul-by-scalar, MatMul-with-constant) commute with ReduceSum.
- **Max-monotonic ops** (Identity, Cast, Pad) commute with ReduceMax.
- **Per-element nonlinear ops** (Sigmoid, Tanh, Exp) do NOT commute. Reduce(f(x)) != f(Reduce(x)).

For task201's specific algorithm, the downstream ops were ReduceSum-commutable.

## Why it saves so much

The wide-tensor intermediate is the biggest single contributor to memory. Moving ReduceSum from "end of chain" to "after one op" eliminates ALL the wide intermediates in between.

## Generalizes to

Whenever you see a chain ending in ReduceSum / ReduceMax, ask: "can the reduction be applied earlier?" Check commutativity of each intermediate op.

This is related to but distinct from `techniques.md` §12 (projection-cache — compute the reduction directly from input, skipping the wide intermediate entirely).

## Reproducibility

For your task:
1. Identify the chain ending in reduction.
2. List the ops between input and reduction.
3. For each op, verify commutativity with the reduction:
   - ReduceSum commutes with: Add, Sub, Mul-by-scalar/per-channel-scalar.
   - ReduceMax commutes with: Add-by-scalar, Identity, Cast (preserving order), Pad (with 0 fill if needed).
4. Move the reduction to the earliest commuting position.
5. Verify byte-exact against the original.

## Cross-ref

- `techniques.md` §12 (projection-cache — extreme version: skip the wide intermediate entirely).
- `05_static_to_dynamic/task071_ring_mask.md` — different rank-reduction insight (rank-2 vs rank-4 intermediates).
