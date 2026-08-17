# task323 biasless mask classifier hard record

- Live/promoted bar: 1020 from `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Task-local reviewer bar: 1018 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_recenter_zero_reuse_cost1018/manifest.json`, sha256 `cbc8f9235f224c3c95c913be628ab67af8df375de0e18ad98f12c50f425f72b7`.
- Attempted route: remove the 2-param `mask_bias` by adding at most one scalar mask-stage zero-point, keeping the current 2x3 `mask_kernel` shape and final `mask_zero_i8` decode.
- Failed gate: pre-build separability audit over all 169 anchor positions found 17 unique 2x3 code patches; `scipy.optimize.linprog` found no feasible origin-separating 6-weight gray classifier for mask input zero-points `-12..-3` (cyan was separable for many values, gray was not).
- Why dead: the gray mask needs an affine threshold under the current code representation; deleting `mask_bias` without changing the code basis cannot preserve the visible rule.
- One remaining hypothesis: a different code tensor basis might make the mask stage biasless, but the current `code`/2x3-mask factorization requires `mask_bias`.
