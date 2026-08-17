# Scalar row-moment dissolves presence plane (task188 v6)

## What was discovered

A `[1,1,30,1]` presence plane (ReduceMax + ReduceSum) that costs 120B can be replaced
by a single 4B intermediate using a weighted row-sum moment formula:

```
# REPLACE THIS (120B row_pres + 4B H scalar = 124B):
row_pres = ReduceMax(input, axes=[1,3])     # [1,1,30,1] F32 = 120B
H        = ReduceSum(row_pres)              # scalar [] = 4B

# WITH THIS (~20B of [1] scalars):
area     = ReduceSum(input)                 # [] F32 = 4B  (= H*W via one-hot property)
H_moment = Einsum('bchw,h->b', input, idx30)  # [1] F32 = 4B  (= W*H*(H-1)/2)
H        = 2*H_moment/area + 1             # exact integer for H in {2,3,4,6,8}
W        = area / H                        # no col_pres needed
```

**Key mathematical fact**: For one-hot encoded grids (each cell = exactly 1 in one channel),
`ReduceSum(input) = H_in * W_in` (total active cells). The idx30 = [0,1,...,29] weighted
sum `Einsum('bchw,h->b', input, idx30) = W_in * sum_{h=0}^{H_in-1} h = W_in * H_in*(H_in-1)/2`.
So `H_in = 2 * moment / area + 1`, which is exact for all valid H_in ∈ {1,2,3,...}.

**Precondition**: grid is a solid rectangle of one-hot cells (no background color-0 inside);
all cells have non-zero color. Background (ch0=1) would contribute to area — this formula
requires colors 1-9 only within the active grid (task188 satisfies this).

The `idx30` initializer is usually already present for the Less comparison in keep-vector
generation. No new large initializer needed; only `one_f` (+1 scalar param) is new.

## Where encountered

- Task: 188 (double-axis deduplication)
- Builder: `The_400_Tasks/Task_by_Task/task188/prototypes/build_task188_v6.py`
- Staged: `staged_submits/task188_scalar_invariant_cost451/`

## Why it matters

- Saves **107B** on task188 (558 → 451), +0.21 LB expected
- Dissolves the `row_pres` [1,1,30,1] plane that was previously claimed as irreducible
- Generalizes: any task where H is derived from a presence plane over a solid rectangular
  grid with non-zero colors can apply this technique
- The Einsum `'bchw,h->b'` is valid in ORT opset 12 and produces correct scalar output

## What's missing from the library

Searched `techniques_catalog.md` for "moment", "row_pres", "presence plane dissolve" — no
existing §. The closest is #115 (variance-identity) but that doesn't cover this exact pattern.

## Sorter note: techniques.md — add new § "scalar row-moment dissolves presence plane"

Tag: one-hot, presence, moment, row_sum, scalar, rectangle
Applicable when: grid is solid rectangle, colors 1-9 only (no ch0 inside), `idx30` init
already present.
Formula: `H = 2 * Einsum('bchw,h->b', input, idx30) / ReduceSum(input) + 1`
Savings: ~100B (replaces [1,1,30,1] 120B presence plane with [1] 4B moment)
Verified: task188 v6, opset 12, ORT 1.x, 200k corpus exact.
