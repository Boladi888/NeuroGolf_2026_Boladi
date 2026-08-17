# task163 — long add_chain_to_sum

| Field | Value |
|---|---|
| Task | task163 |
| Pattern | `add_chain_to_sum` (techniques.md §6) |
| Cost reduction | -4,840 |

A longer Add-chain than task327. Same mechanical rewrite, more intermediates eliminated.

## Reproducibility

Detector handles automatically. Manual: trace any sequence of Adds where the chain is strictly left-folded; collapse to one Sum with all leaf inputs.

## Cross-ref

- `task327_add_chain_to_sum.md` — the origin/canonical example.
- task013 — longest single chain documented (-17,280 bytes); no exemplar written, see `The_400_Tasks/Task_by_Task/task013.md` + techniques #55.
- `techniques.md` §6.
