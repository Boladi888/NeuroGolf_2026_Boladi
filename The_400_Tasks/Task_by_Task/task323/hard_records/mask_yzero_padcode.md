# task323 mask y-zero padcode hard record

- Live/promoted bar: 1020 from `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Task-local reviewer bar: 1018 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_recenter_zero_reuse_cost1018/manifest.json`, sha256 `cbc8f9235f224c3c95c913be628ab67af8df375de0e18ad98f12c50f425f72b7`.
- Attempted route: `prototypes/build_worker3_mask_yzero_padcode.py` built `prototypes/task323_mask_yzero_padcode.onnx`; delete `mask_zero_i8` by reusing `pad_code_i8` as the `saturating_exact_masks` `y_zero_point` and the final `ConvInteger` input zero point.
- Failed gate: `python tools/ng_gate.py visible --task 323 --cand The_400_Tasks/Task_by_Task/task323/prototypes/task323_mask_yzero_padcode.onnx` -> `[visible] 0/172 (0.00%)  FAIL`.
- Why dead: the current mask stage relies on saturation near INT8 min with `y_zero_point=-127`; moving the output zero point to `pad_code_i8=-7` makes false scores vary (`centered around -7` values included `-46,-38,-14,-10,-8,-6,-4,-2,2`), so the final sign decode turns channel 0 positive across the active grid.
- One remaining hypothesis: scalar deletion needs a different mask scoring basis; do not retry `pad_code_i8` as the mask-stage output zero point on the current kernels.
