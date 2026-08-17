# task128 — crop15 (Pattern_Recognition variant)

| Field | Value |
|---|---|
| Task | task128 |
| Pattern | Crop-to-active-grid (techniques.md §2) |
| Active grid | 15×15 |

Same template as task117 with 15×15 active grid. Read `task117_crop15_f16.md` for the full 5-line surgery and dimension-tied init rewrite.

task128 doesn't stack with the F16-output trick (no Concat→F16→Cast→F32 ending), so crop alone is the saving.
