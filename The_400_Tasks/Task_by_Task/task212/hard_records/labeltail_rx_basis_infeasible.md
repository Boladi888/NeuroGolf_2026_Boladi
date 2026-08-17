# task212 hard record - rx-basis label tail infeasible

Worker: `bo_ng_cdx2_worker1`
Date: 2026-07-09

## Live bar used

- Source: `python tools/ng_current_bar.py --task 212 --json`
- Authoritative source: `Team_Submission/Backend/manifest.csv`
- Team-best owner: Sho
- Cost/profile cost: 514/514
- SHA-256: `01a998567afdf0a1a3db922f0c767faec147d7d4738eddb19c0505137d047019`

## Reviewer-locked route

Replace the full-output one-hot `Einsum` with a true active-10x10 one-channel
label/mask source, then make `Pad(sentinel) -> Equal(channel_ids)` the graph
output. First variant must reuse the 10-row `rxHI10`/`rxLO10` basis; build a
second generated/factored selector basis only if the first prices close.

Task-local probe:

- Builder: `The_400_Tasks/Task_by_Task/task212/prototypes/build_task212_rx_labeltail_probe.py`
- ONNX: `The_400_Tasks/Task_by_Task/task212/prototypes/task212_rx_labeltail_probe.onnx`
- Manifest: `The_400_Tasks/Task_by_Task/task212/prototypes/rx_labeltail_probe_manifest.json`
- SHA-256: `aad68a0281cfb92c2b375d15c73d2788c6575302bc6644b8c6a4cd9a211aec49`

The probe creates an active 10x10 source from the full one-hot input, decodes
channel labels through `rxHI10`/`rxLO10`, casts the one-channel label to U8, pads
with sentinel 10, and uses `Equal(channel_ids)` as graph output.

## Exact failed gate

- `score_v2`: `(0, 6600, 113)` => cost 6713
- `profile_real_cost`:
  `task212 task212_rx_labeltail_probe.onnx: static=6713 profile=6713 (memory 6600 + params 113; 265 real inputs)`
- `largest_intermediate.py --top 20`:
  - `active10_f32` F32 `[1, 10, 10, 10]` from `Slice` = 4,000 B
  - `label30_u8` UINT8 `[1, 1, 30, 30]` from `Pad` = 900 B
  - total static intermediate bytes = 6,600

No visible/truth/AB gates were run. The reviewer stop condition fired before
correctness: profile cost is far above 231 and the charged `[1,10,H,W]`
intermediate appears.

## Why this route is dead

The locked `Pad -> Equal` tail necessarily materializes the padded one-channel
label canvas as a non-output tensor. For this task shape that tensor is
`label30_u8` `[1,1,30,30]` = 900 B, already above the 231 target before any
real label-source logic.

The requested `rxHI10`/`rxLO10` active-source form also needs an active source
from the full `[1,10,30,30]` input. The direct 10x10 source materializes
`active10_f32` `[1,10,10,10]` = 4,000 B, which is explicitly disallowed by the
reviewer stop rule.

Because the first requested variant priced 6,713 and triggered both stop
conditions, the second generated/factored selector variant was not built.

## One remaining hypothesis

Further useful work should stay in the zero-memory full-output `Einsum` family
or find a new direct-output factorization that reduces the shared row/comparison
basis. The active-label `Pad -> Equal` formulation is not a sub-231 route for
task212.
