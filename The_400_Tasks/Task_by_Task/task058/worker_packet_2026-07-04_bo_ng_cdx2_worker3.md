# task058 worker packet - bo_ng_cdx2_worker3 - 2026-07-04

## Bar used

Authoritative bar from `python tools/ng_current_bar.py --task 58 --json`: live/team-best cost 128, owner Vamshi, source `manifest`, updated 2026-07-02T02:35:15Z, sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`.

Stale sources observed: local task_state cost 980, legacy live_costs/library cost 19,826. They were not used as the bar.

## Candidate/probe paths

No staged candidate and no manifest were produced.

Probe: `The_400_Tasks/Task_by_Task/task058/prototypes/task058_prefix_einsum127.onnx`
- static score: `(0, 0, 127)`, cost 127
- profile cost: not run; visible failed first
- sha256: `a4bca924c72a9eeb7d9a1c144050e05e93a164b10205ea10565d481e05671d91`
- delta vs live static cost: `127 - 128 = -1`
- reviewer-ready: no

Sibling not gated after family failure: `The_400_Tasks/Task_by_Task/task058/prototypes/task058_prefix_diag127.onnx`
- static score: `(0, 0, 127)`, cost 127
- sha256: `d4a925ee6946072d545f7f7ce5c7c00294213eb5e8f4f111f1fa762d6dae14bb`
- reason skipped: same single-Einsum prefix surrogate family as the visible-zero probe

## Gates and exact result lines

Static score:
`Team_Best_Onnx/task058.onnx (0, 16, 112) 128`
`The_400_Tasks/Task_by_Task/task058/prototypes/task058_prefix_einsum127.onnx (0, 0, 127) 127`
`The_400_Tasks/Task_by_Task/task058/prototypes/task058_prefix_diag127.onnx (0, 0, 127) 127`

Visible:
`[visible] 0/22 (0.00%)  FAIL`

No `profile_real_cost`, `truth`, `ab`, `battery`, or `--full` gates were run on the 127-byte family because the first visible gate failed.

Cost diff for the failed 127-byte probe:
`live memory=16 params=112 cost=128`
`candidate memory=0 params=127 cost=127`
`delta memory=-16 params=+15 cost=-1`
Named charged deltas: `coeff` table remap `+72 B`, removed `bases` `-12 B`, removed `phase` `-12 B`, removed `n_f` `-4 B`.

## What was tried

Read the prompt/protocol, refreshed the bar with `ng_current_bar.py`, checked BoBrain consistency, read the preview/task JSON/generator `external/ARC-GEN/tasks/task_28e73c20.py`, inspected the 128-byte team-best graph, and audited prior task-local 127-byte probes.

The decoded rule is a blank square `n x n`, `n in [5,20]`, mapped to a color-3 clockwise spiral. The live graph computes `n` with `ReduceL2(input)`, builds phase `[1, (-0.75)^n, 1]`, and renders the signed spiral with one final `Einsum`.

The below-bar route tried was a one-node `Einsum` surrogate that contracts two copies of the input into prefix/global size features and avoids the live `n_f`/`phase` intermediates. It scored 127 but failed visible 0/22, so the family is dead for staging.

I also checked whether the live formula had an obvious legal deletion:
- sparse `Einsum` initializers were already recorded dead in `hard_records/sparse_einsum_initializers.md`
- the live coefficient matrix is rank 2, but legal dense factorization costs 12 params versus the current 9
- replacing the duplicated phase constants with a 2-vector phase forces a 4D coefficient tensor and costs more
- generating tiny initializers such as `channel_sign`, `bases`, or `coeff` would save params but add charged intermediates, net losing

## Hard record / lesson packet

Created: `The_400_Tasks/Task_by_Task/task058/hard_records/single_einsum_prefix127.md`

No global `.update` lesson packet was created.

## Remaining hypothesis

A real sub-128 win needs an exact algebraic single-output formulation that removes `n_f`/`phase` without using the failed learned prefix surrogate, or a legal way to encode the sparse row/column coefficient structure below 9 dense params. The current live 128-byte graph is already close to the analytic floor: 90 coordinate-basis params + 9 coeff params + 10 channel-sign params + 3 phase-base params + 16 bytes scalar/vector memory.
