# task055 carrier-narrow / Equal-tail route — DROP

- Live bar: **1,113** (static/profile), SHA-256
  `b7bab0c46528aece35efdce0ae4db78dfad986e2eb685b3103fba98a07cb7853`.
  Source: `python tools/ng_current_bar.py --task 55 --json`; the team manifest and
  workbook agree. Team_Best metadata's 1,106 and local task-state's 1,115 are stale.
- Attempted route: refresh and price the prompt's one-channel carrier plus
  Equal-tail fast probe before constructing a full graph.
- Exact failed gate: **charged-component existence / fast-price gate = FAIL**.
  The authoritative graph has no Conv, Reduce, Equal, repeated mask, full spatial
  carrier, canvas, or template. It already uses two 30-element F32 marginals,
  30-element U8 axis codes, and a free BitwiseAnd output. A conventional full-size
  U8 Equal-tail alone must materialize a `[1,1,30,30]` canvas (900 B), already above
  the prompt's 720 kill threshold before producer/intermediate/parameter costs.
- Why dead: the named charged structure was removed by the Kaggle-confirmed v14
  parent, and reinstating an Equal-tail is a regression. Per the prompt's 2026-07-10
  refresh rule, stop rather than build or spend 3,000/full truth pairs on a stale
  formulation.
- Remaining hypothesis: only a genuinely different source-read/free-output fusion
  that deletes the residual `xk` U8 `[1,10,30,1]` (300 B) while preserving the ch8
  rank-2 union and padding semantics could break the current formulation floor.
  It must not introduce a 30x30 canvas/Equal-tail or a full-input F32-to-U8 cast.

Machine-readable packet: `carrier_narrow_equal_tail_drop.json`.

