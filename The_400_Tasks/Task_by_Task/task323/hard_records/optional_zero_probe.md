# task323 optional_zero_probe hard record

- Live/team bar: 1020 from `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Pending task-local bar at time of probe: 1019 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_shared_mask_zero_cost1019/manifest.json`, sha256 `25034c68173197e9bd4b3eb6526f038086bc8a54a99186cde5df4fb16fa6b676`.
- Attempted route: delete additional zero-point initializers by leaving empty optional inputs in QLinearConv/ConvInteger (`prototypes/task323_optional_zero_probe.onnx`).
- Failed gate: `visible` load failed with `InvalidGraph: QLinearConv input 2 is marked single but has an empty string in the graph`; `_score_v2` returned `None None`.
- Why dead: ORT does not accept the empty-string optional input form for the required QLinearConv zero-point slots in this graph, so this representation is invalid before correctness can be tested.
- One remaining hypothesis: scalar savings must come from legal defaulting such as ConvInteger's omitted weight zero-point, or from a different mask/decode formulation; do not retry empty QLinearConv zero-point slots.
