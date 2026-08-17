# task212 hard record - K<=5 moment/sign-polynomial separator

Worker: `bo_ng_cdx_worker2`
Date: 2026-07-14

## Live bar used

- Source: `python tools/ng_current_bar.py --task 212 --json`
- Authoritative source: `Team_Submission/Backend/manifest.csv`
- Team-best owner: Vamshi
- Cost/profile cost: 412/412
- SHA-256: `f5303effda475011070aa0ce6c0db01a4b934d2d788cc012c7b520a0069a6529`

## Locked reviewer hypothesis

Replace the live one-node free-output `Einsum` separator
`row_basis[7,30] + side_coeff[3,4,7]` with an exact `K<=5`
sign-coded/polynomial moment core. Do not build output tails or dense planes.
Run a numpy probe on visible plus at least 3000 fresh ARC-GEN rows before ONNX.

Task-local probe:

- Script: `The_400_Tasks/Task_by_Task/task212/prototypes/probe_task212_moment_core_k3_k5.py`
- Result JSON: `The_400_Tasks/Task_by_Task/task212/prototypes/moment_core_k3_k5_probe.json`
- Corpus: 265 visible + 3000 fresh `external/ARC-GEN/tasks/task_8d510a79.py` rows with seed `21220260714`

## Probe structure

For K in `{3,4,5}`:

1. Compute per-column/color row moments over colors 1 and 2 plus the horizon row.
2. Check moment-key collisions against output columns over all 3265 examples.
3. Replace the row basis with active monomial rows `[r^0..r^(K-1)]` on rows 0..9, zero outside.
4. Least-squares refit `side_coeff[...,K]` and `feat[...,K]` against the live separator functions.
5. Price the hypothetical one-node `Einsum` as `104 + 44*K` params:
   - K=3 -> 236 params (`-176` vs 412)
   - K=4 -> 280 params (`-132` vs 412)
   - K=5 -> 324 params (`-88` vs 412)

Moment collisions were not the blocker: K=3/4/5 each checked `32650` output columns with `0` collisions.

## Exact failed gate

The numpy-visible precheck failed before any ONNX was written:

- K=3: visible prefix `0/265`; first example had `48` tensor-cell mismatches; `side_coeff` residual `1.6090909090909262`, `feat` residual `29.0`
- K=4: visible prefix `0/265`; first example had `45` tensor-cell mismatches; `side_coeff` residual `1.517948717948703`, `feat` residual `29.0`
- K=5: visible prefix `0/265`; first example had `41` tensor-cell mismatches; `side_coeff` residual `1.3787878787881525`, `feat` residual `29.0`

All fitted K<=5 variants also required dense F32 coefficient tensors in the refit. Per reviewer stop
rule and worker protocol, a visible miss kills the family. No ONNX, profile, truth, AB, safety, or
staging gate was run.

## Why this route is dead

Low-order row/color moments are sufficient to identify the column examples on this sampled corpus, but
they are not sufficient to replace the live separator in the existing single-Einsum factor family at
K<=5. Rendering the separator still needs either K>=6 row modes or a different dense coefficient family,
both outside the locked hypothesis.

## One remaining hypothesis

Only a different tensor-network family, not a K<=5 polynomial replacement of the live
`row_basis/side_coeff` separator, can plausibly beat 412. Any next attempt should start from a new
analytic comparator construction and must price its coefficient tensors before ONNX.
