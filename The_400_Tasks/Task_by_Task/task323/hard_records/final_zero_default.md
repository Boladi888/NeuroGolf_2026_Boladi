# task323 final-zero-default hard record

- Live/promoted bar: 1020 from `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Task-local reviewer bar: 1018 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_recenter_zero_reuse_cost1018/manifest.json`, sha256 `cbc8f9235f224c3c95c913be628ab67af8df375de0e18ad98f12c50f425f72b7`.
- Attempted route: `prototypes/build_worker3_final_zero_default.py` built `prototypes/task323_final_zero_default.onnx`, sha256 `cb060557ad2eb949bfb1e2828e04cea0036e7584dfa50fe468157bc7dbb5d495`; recentered `class_masks` around ConvInteger default `x_zero=0` and deleted `mask_zero_i8`.
- Cost probe: `score_network_v2` returned `(0, 781, 236) 1017`.
- Failed gate: `python tools/ng_gate.py visible --task 323 --cand The_400_Tasks/Task_by_Task/task323/prototypes/task323_final_zero_default.onnx` -> `[visible] 0/172 (0.00%)  FAIL`.
- Why dead: the current two-mask final decode depends on a nonzero `ConvInteger` input zero-point; shifting masks to small signed values breaks the visible one-hot sign pattern even though it saves one param.
- One remaining hypothesis: a different two-mask coding or final decoder may still delete a scalar, but do not retry this final-default-zero recentering family.
