# DEAD ROUTE - task155 free-output selector-bank sub-138 probe (2026-07-13, bo_ng_cdx2_worker3)

**Live/team-best bar used:** cost/profile **138**, manifest owner Vamshi, sha
`0faf6dbc5525fca74bf5f2e0335d3a0691b1edcd44f0463c4c8148d20e1fd979`,
updated `2026-07-12T15:15:54Z` via:

`python tools/ng_current_bar.py --task 155 --json`

Team_Best static-cost metadata showed a stale 120 for the same sha, but
`profile_real_cost.py` on `Team_Best_Onnx/task155.onnx` returned:

`task155 task155.onnx: static=138 profile=138 (memory 136 + params 2; 266 real inputs)`

## Attempted route

Tested the prompt's free-output/separable write idea for FLIP-V:

1. Read `N` with the square-grid moment identity `ReduceL2(input)=N`.
2. Convert `N` to a slot in `{0..4}` for sizes 4..8.
3. `Gather` a static selector matrix `S_N[r,s] = 1 iff s=N-1-r`.
4. Make the final full one-hot output a free `Einsum("bcsw,krs->bcrw")`.

Prototype/script:

- `The_400_Tasks/Task_by_Task/task155/prototypes/probe_worker3_sub138_routes.py`
- `The_400_Tasks/Task_by_Task/task155/prototypes/task155_W3_selector_bank.onnx`
- `The_400_Tasks/Task_by_Task/task155/prototypes/task155_worker3_sub138_probe_results.json`

Fresh ARC-GEN moment check from the script:

`num=200, flip_v_matches=200, nonsquare=0, l2_size_misses=0, n_hist={4:42,5:46,6:38,7:41,8:33}`

## Exact failed gate

Visible passed, but the route fails the cost/profile gate before any truth/ab work:

- `python tools/ng_gate.py visible --task 155 --cand The_400_Tasks/Task_by_Task/task155/prototypes/task155_W3_selector_bank.onnx`
  - `[visible] 266/266 (100.00%)  PASS`
- `python The_400_Tasks/Task_by_Task/task155/prototypes/probe_worker3_sub138_routes.py`
  - `W3_selector_bank: static=8118 profile=8118 (memory 3616 + params 4502; 266 real inputs)`
  - `sha256=5d143040617b1ffbe1adee639e8dd60bd572a4ae941254d2f5f91121781e0e02`

Reference rebuild of the current Range tail:

- `task155_W3_range_parent.onnx`
  - `visible 266/266 PASS`
  - `static=138 profile=138 (memory 136 + params 2; 266 real inputs)`
  - `sha256=179fed7316331f72cad003af4f36d9d388f0c4c5e7fac91239d41312f47a7b95`

## Why this route is dead

The full canvas final `Einsum` is free, but the data-dependent row coupling is not.
Selecting `S_N` materializes a charged `[1,30,30]` F32 selector = 3,600 B and the
five-size selector bank costs 4,500 params. This is worse than the current-best
`Range` tail, whose only non-scalar payload is the `[30]` INT32 Gather index = 120 B.

Low-rank factorization does not approach 138 either: each `S_N` has rank `N`, so
the max-size case needs at least 8 row/source factors. A banked factor form is
already at least `5 * 8 * 30 * 2 = 2400` params before selected-basis intermediates.

## Remaining hypothesis

Only a legal scalar-length row-reversal op could plausibly beat 138 by deleting the
`idx[30]` payload. The known candidate, `ReverseSequence`, is scorer-illegal and
axis-locked per the prior hard record. A micro-shave would require ORT/scorer fusion
that removes one of the four scalar Range-prep tensors from profile cost; no ONNX
construction found here does that without adding a larger charged tensor.

**Status:** REJECT_LOCAL / no staged candidate.
