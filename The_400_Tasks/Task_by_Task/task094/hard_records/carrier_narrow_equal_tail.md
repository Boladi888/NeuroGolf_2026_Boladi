# CarrierNarrowEqualTail priced wall — 2026-07-10

- Live bar: **1141** (`memory=558`, `params=583`, real profile **1141**), SHA-256 `cc3ff9e00af763b89c9c4ba1dc5a5be7fd5290d5b64b243f7591ac9516e6eaa9`, from the authoritative team manifest (`Sho`, 2026-07-09T01:09:17Z).
- Attempted route: the existing task-local one-channel U8 canvas plus final Equal probe, `prototypes/task094_v1.onnx`, SHA-256 `ba8eb63a5f1982ae4e2da3642d493b74b4de615062843e64045d26e35f60dbfe`.
- Exact failed gate: visible `265/265 (100.00%) PASS`; then score/profile `static=3806 profile=3806 (memory 3750 + params 56; 265 real inputs)`. This is **+2665** versus live and **+3086** versus the prompt's ~720 target, so truth/AB/full were correctly not run.
- Why dead: the current live graph already removed the named carrier/canvas/template/final-Equal chain and ends in a free-output Einsum. The probe adds three dominant `+900 B` tensors (`ch1`, `corner_score`, `canvas30`), while the fixed 30x30 U8 Equal-tail canvas alone costs 900 B before detection. QLinearConv has no existing U8/I8 feed; casting the F32 Slice double-charges it, and casting the full input costs 9000 B.
- Remaining hypothesis: a future, separately scoped rewrite could target the live graph's `det` F32 `[1,9,9]` (324 B) plus `ReduceMax`/selector chain or further factor its `A7`/`B7f` parameters. Do not revisit carrier-narrow/Equal-tail for this live SHA.

