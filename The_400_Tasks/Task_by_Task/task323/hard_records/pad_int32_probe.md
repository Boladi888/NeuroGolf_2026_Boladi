# task323 Pad INT32 hard record

- Live/promoted bar: 1020 from `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Task-local reviewer bar: 1018 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_recenter_zero_reuse_cost1018/manifest.json`, sha256 `cbc8f9235f224c3c95c913be628ab67af8df375de0e18ad98f12c50f425f72b7`.
- Attempted route: `prototypes/build_worker3_pad_int32_probe.py` built `prototypes/task323_pad_int32_probe.onnx`, sha256 `05346f2c6aba40556dc76ed83b7de989623af39fa4fd242fd7a3a39d3828cdfc`; changed `pads_i64` Cast output from INT64 to INT32 to try to halve the charged dynamic Pad pads tensor.
- Failed gate: `score_network_v2` returned `None None`; visible load failed with `InvalidGraph: Type 'tensor(int32)' of input parameter (safe_name_17) of operator (Pad) in node (safe_name_18) is invalid.`
- Why dead: ORT requires Pad `pads` to be INT64 in this schema path, so the 32-byte `marker_pads` tensor cannot be narrowed.
- One remaining hypothesis: only a different marker-placement primitive can remove this INT64 pads cost; do not retry Pad pads as INT32.
