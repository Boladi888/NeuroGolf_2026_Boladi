# task359 — bool-first Where rewrite (ORT bool-bool-bool workaround)

| Field | Value |
|---|---|
| Task | task359 |
| Pattern | Where(bool, bool, bool) workaround (techniques.md §1 rule 8 + general bool-first) |
| Cost measurement | 9,000 bytes BOOL vs 36,000 bytes F32 equivalent (4× ratio confirmed) |

## The problem

ORT does NOT support `Where(bool, bool, bool)` — the data inputs (x, y) must be numeric. See `ort_reality.md §2`.

But task359's algorithm naturally wants to select between two bool tensors based on a bool condition. Casting up to F32 just to use Where is wasteful.

## The rewrite

Replace `Where(c, x_bool, y_bool)` with explicit boolean composition:

```
# Want: result_bool = Where(c, x_bool, y_bool)
# But ORT rejects.

# Solution 1: general And/Or chain
result_bool = Or(And(c, x_bool), And(Not(c), y_bool))

# Solution 2: when x or y is exploitable (e.g., x = one-hot, y = row-mask)
# Skip the Not by using the structure
result_bool = And(one_hot_bool, pick_row_b)   # captures the "where(c, ...)" semantics
or_with_col = Or(result_bool, col_mask)
```

## Measured byte savings

task359 measurement: the bool composition was 9,000 bytes total (BOOL intermediates) vs 36,000 bytes for the F32 equivalent (`Where(c, Cast(x, F32), Cast(y, F32))`).

This is the 4× ratio bool:F32 promises. The Cast nodes alone (going from bool to F32 just to use Where) cost 18,000 bytes — eliminating them saves more than the 4× from the data tensors.

## Generalizes to

`techniques.md` §1 (bool-first family). Any time you need to select between two bool tensors:

- General: `Or(And(c, a), And(Not(c), b))`.
- With one-hot structure: `And(one_hot, pick_row) + Or(row_mask, col_mask)` — exploit the structure to skip the Not.

## Cross-ref

- `ort_reality.md §2` — full ORT Where(bool,bool,bool) discussion.
- `task110_periodicity.md` — task110 worker hit this mid-rebuild; rewrote the period-selection Where chain accordingly.

## Reproducibility

When designing a graph that naturally wants `Where(bool, bool, bool)`:
1. STOP. Don't write that node — ORT will reject at load.
2. Compose with And/Or/Not explicitly OR exploit task-specific structure.
3. Verify ORT loads the model: `ort.InferenceSession(path)` should succeed.
