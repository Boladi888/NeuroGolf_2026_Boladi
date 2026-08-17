# task115 — axis-factor reformulation

| Field | Value |
|---|---|
| Task | task115 |
| Family | Pattern_Recognition |
| Cost reduction | -135,600 |
| Pattern | Whole-graph rewrite — axis factorization |

## Rule

task115's algorithm originally computed per-cell values using a wide intermediate that depended on BOTH row and column information. The rewrite factors the computation into per-axis components that combine via outer product, dramatically reducing intermediate size.

## Before (single wide intermediate)

```
input → various ops → wide_intermediate [1, 10, 30, 30] F32 = 36,000 bytes
        → some condition computed per-cell as f(row, col)
        → output
```

## After (axis-factored)

```
input → row_factor [1, 10, 30, 1] F32 = 1,200 bytes (varies only with row)
input → col_factor [1, 10, 1, 30] F32 = 1,200 bytes (varies only with col)
combined = Mul(row_factor, col_factor)   # broadcasts to [1, 10, 30, 30] but counted at runtime
                                         # but per-axis intermediates are 1,200 each
output = ... using combined
```

The two per-axis tensors are 1,200 bytes each (30× smaller than 36,000). Their broadcast product equals the original wide tensor, but the intermediates that ORT counts are the small per-axis ones, not the broadcast result.

## Why it works

When a per-cell value `v(r, c)` can be decomposed as `f(r) × g(c)` (separable), you don't need the full `[1, C, 30, 30]` tensor. Store the two 1D factors and let broadcasting combine them at use sites.

This works when the cell value is **multiplicatively separable**:
- `v(r, c) = f(r) × g(c)` → outer-product reconstructible.
- `v(r, c) = f(r) + g(c)` → outer-sum reconstructible (Add instead of Mul).
- `v(r, c) = max(f(r), g(c))` → outer-max reconstructible.
- `v(r, c) = f(r) AND g(c)` → outer-And reconstructible (for bool).

## Generalizes to

`techniques.md` §15 (`separable_mask_collapse`) handles the static-init version of this — when a 2D init is separable along an axis, collapse to 1D.

axis-factor is the DYNAMIC version — when a runtime-computed 2D tensor is separable, compute the two 1D factors and combine via broadcast.

Apply when:
- Per-cell rule has explicit row + column structure.
- Row and column components are independent (don't depend on each other).
- Final output is a separable combination (Mul / Add / Max / And).

## Reproducibility

For your task:
1. Inspect the per-cell rule. Can it be decomposed as `f(row) × g(col)` (or similar)?
2. Compute `f(row)` as a `[1, C, H, 1]` tensor.
3. Compute `g(col)` as a `[1, C, 1, W]` tensor.
4. Combine via broadcast: `Mul(f_row, g_col)` produces `[1, C, H, W]` semantically but the intermediates are tiny.

Cross-ref `separable_mask_collapse` exemplar for the static-init case (task040 -21,960 bytes).

## Cross-ref

- `techniques.md` §15 (separable_mask_collapse).
- `Algorithm_Library/algorithm_patterns.md` §10 (rank reduction insights).
