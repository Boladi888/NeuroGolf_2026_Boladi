# task281 — crop13

| Field | Value |
|---|---|
| Task | task281 |
| Pattern | Crop-to-active-grid (techniques.md §2) |
| Active grid | 13×13 |

Same as `task117_crop15_f16.md` template with H=13. Smaller active grid = larger relative savings on every intermediate.

## When active grid is small (≤ 10)

The crop savings scale with `(30² - H²) / 30² = 1 - (H/30)²`:
- H=15: 75% savings on every intermediate.
- H=13: 81% savings.
- H=10: 89% savings.
- H=5: 97% savings.

For tasks with very small active grids (5-10), crop is the highest-leverage technique.
