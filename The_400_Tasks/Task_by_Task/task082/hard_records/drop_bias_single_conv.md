# hard_record: task082 — drop-bias single-Conv route is DEAD (WALL @ 190)

- **Date:** 2026-07-03 (batch48, slot05, Claude worker `bo_ng_cld_worker2`)
- **Live bar:** 190 (ng_current_bar, manifest-authoritative), sha256 `d48db5f200e4a784e10b492e2ae610af5874b7214fed9652cf371c0f1b187252`, owner Aditya.
- **LIVE graph:** single grouped `Conv`, kernel `[10,1,6,3]` (180 params) + `bias[10]` (10), group=10, pads `[5,1,0,1]`, 0 intermediates → memory=0, cost=190.

## Attempted route
Reduce params below 190 by **dropping the Conv bias** (190→180). Only reducible piece — the 180-element kernel is provably forced (10 ch × 6 rows to fill 6 output rows from row0 × 3 cols for the −1/0/+1 diamond taps).

## Exact failed gate
Official-grader harness (`prototypes/official_verify.py`, mirrors `neurogolf_utils.verify_subset`: `(out>0)` vs `convert_to_numpy(expected)` full 30×30 array-equal):
- `task082_nobias` (LIVE kernel, no bias): **0/265** — ch0 lights diamond/foreground cells (bias −5 was suppressing them).
- `task082_nobias_redesign` (bias-free ch0, rowsums 0 except ky5=6, odd taps a=c=7): **0/265** — lights padding col c=W at rows 6–10, plus last grid col.
- `task082_a0zero` (a₀=c₀=0): **0/265** — row5 foreground cells (c0,c2,c4,c6) lit as background.

## Why it's dead (forced-component proof)
Bottom output row 5 (odd) carves its diamond **only** via kernel row `ky0` (only row reaching input row0 for output row 5); its ±1 taps `a₀,c₀` must be ≥ background value T>0. But padding cell **output(row=10, col=W)** reads only input(row5, col W−1)=1 → `conv = a₀`, which must be ≤0 (padding, expected empty). Same weight `a₀` required both `>0` and `≤0` → contradiction. A constant negative offset resolves it; the Conv **bias is the only free way** to inject it (an `Add` would cost a 36,000-byte `[1,10,30,30]` intermediate; bias length must be M=10, shorter = UB per traps §8h). ⇒ **bias forced; 190 = 180+10 is the floor.**

## Remaining hypothesis (only one, low-confidence)
None viable for <190 in the single-Conv (0-memory) family. The single-Conv is memory-optimal; any multi-op graph pays ≥1,200 B intermediate > 10 B saved. **Verdict: WALL @ 190.** Do not re-explore the drop-bias route. If a future generator variant ever guarantees the grid is taller/differently padded such that output row 10 no longer reads a grid cell, revisit — but current data is fixed 6-row grids.

## Note on gate gap
`ng_gate` local `decode(arr,h,w)` only inspects the h×w region and would **falsely pass** an unclean-padding candidate; the real Kaggle grader (`verify_subset`) array-compares the full 30×30. Always validate padding with `official_verify.py`, not just ng_gate visible.
