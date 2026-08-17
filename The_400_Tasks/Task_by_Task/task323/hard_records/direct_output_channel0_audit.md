# task323 direct output channel0 audit hard record

- Live/promoted bar: 1020 from `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Task-local reviewer bar: 1018 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_recenter_zero_reuse_cost1018/manifest.json`, sha256 `cbc8f9235f224c3c95c913be628ab67af8df375de0e18ad98f12c50f425f72b7`.
- Attempted route: extend the prior all-U8 direct-output QLinearConv route by adding a legal channel-0 active-background classifier over the current `code` tensor, while keeping bottom/right padding nonpositive.
- Failed gate: pre-build separability audit over all 169 anchors and all 30x30 output positions. For direct kernels with `kh=2..6`, `kw=3..6`, and area <= 36 (the byte-winning limit), channel 0 had conflicting identical code patches with both background and non-background labels. The grouped-final variant with replacement mask kernels of area <= 10 also failed LP feasibility for both background and signed gray/cyan encodings.
- Why dead: under the current `code` basis, active background is not a local linear predicate that a single direct output filter can distinguish from gray/cyan while keeping padded output zero; adding channel 0 to the dead direct-output QLinearConv family is not viable within the cost-winning kernel area.
- One remaining hypothesis: a different code basis or a procedural replacement for the 156-param `code_kernel` may still lower cost; do not retry direct channel-0 classification on the current `code` patches.
