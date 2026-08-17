# task074 label-first router floor probe

Live/team bar used: 202, profile 202, SHA-256
`fdf348b26c90ffa1ee70ba94f20d7b329a571788f881c6a6780e68e3a35f5018` from
`python tools/ng_current_bar.py --task 74 --json` (`authoritative_source=manifest`,
owner Boladi, updated 2026-07-09T04:14:47Z).

Reviewer-locked route: preserve the one-output free-output Einsum and reduce input
coordinate axes through the existing compact labels before any relation touches output
coordinates. Built exactly one variant:
`The_400_Tasks/Task_by_Task/task074/prototypes/task074_label_router_worker1.onnx`
SHA `3513298e8eb2a9c4712d4a026b170fa785ce34155d5f55a7014a9992b2bedca1`.

Exact stop result:
- `score_network_v2`: `(0, 0, 206)`
- `python tools/agent_helpers/profile_real_cost.py --task 74 --cand The_400_Tasks/Task_by_Task/task074/prototypes/task074_label_router_worker1.onnx --max-n 1`
- `task074 task074_label_router_worker1.onnx: static=206 profile=206 (memory 0 + params 206; 1 real inputs)`

Named charged-structure delta:
- Added dense router `G[2,2,2,2]`: +16 params.
- Removed parent `H2[2,2,2]`: -8 params.
- Removed parent `switch[2,2]`: -4 params.
- Net: +4 cost, so the candidate is more expensive than the 202 parent.

Why this forces the local floor for the locked hypothesis: the label-first relation is
`G[t, normal_source_bit, transpose_source_bit, output_bit]`, selecting `normal_source`
when `t=0` and `transpose_source` when `t=1`. Dense `G` is 16 params. The current 202
parent already factorizes this exact router as
`switch[t,s] * H2[t, normal_source_bit, output_bit] * H2[s, transpose_source_bit, output_bit]`
with shared `H2` plus shared `switch`, totaling 12 params. That is the cheaper dense
label-space factorization.

Do not retry sparse `G`/`H2`/`Pcoord`: traps.md §8e says sparse initializer op inputs
are scorer-dead under full_check/strict shape inference. Do not retry the 198 sign-dot
route unless it preserves this label-first contraction shape; the direct endpoint-coupled
version already timed out in `hard_records/sign_relation_timeout.md`.

One remaining hypothesis: a real sub-202 win would need a new dense tensor identity that
encodes the same bit-router in fewer than 12 params without duplicating coordinate-feature
projections or touching output coordinates early.
