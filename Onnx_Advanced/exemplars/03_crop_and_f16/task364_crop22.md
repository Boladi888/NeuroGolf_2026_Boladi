# task364 — crop22 (larger active grid)

| Field | Value |
|---|---|
| Task | task364 |
| Pattern | Crop-to-active-grid (techniques.md §2) |
| Active grid | 22×22 |
| Cost reduction | -22,532 |

When the active grid is closer to 30 (e.g., 22), crop savings are smaller proportionally:
- (30² - 22²) / 30² = 46% savings per intermediate.

Still meaningful for tasks with many intermediates. -22,532 bytes on task364 confirms.

## When NOT to crop

If active grid is 25+ and the model has few `[...,30,30]` intermediates, the Slice + Pad overhead may exceed the savings.

Rule of thumb: don't crop unless `(1 - (H/30)²)` × number_of_30x30_intermediates × 4 (F32) > 1,000 bytes.

For H=22 with even a few intermediates, crop wins. For H=25 with only 1-2 intermediates, probably not worth it.

## Cross-ref

- `task117_crop15_f16.md` — the canonical example.
- `techniques.md` §2.
- `traps.md §21` — pre-crop overhead trap for already-cropped models.
