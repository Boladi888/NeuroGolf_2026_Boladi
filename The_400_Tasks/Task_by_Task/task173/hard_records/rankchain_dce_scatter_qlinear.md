# task173 hard record - rank-chain DCE on ScatterElements/QLinear carriers

- **bar used**: team-best/profile cost `8710`, sha256 `27f31830cf75f60bdcf0273b403dcb804b8f4b6950817091f50e761be3a57940`.
- **bar source**: refreshed workbook `NG_OnnxlyFans_TaskStatus.xlsx` Tasks row and team manifest row `task173,Boladi,...,8710,...,2026-07-02T13:58:33Z`; task-local staged manifest agrees.
- **attempted route**: prompt route "rank-chain DCE on ScatterElements/QLinear carriers"; inspected current staged best `staged_candidates/task173_regionfillrewrite_cost8710/task173.onnx`.
- **failed gate**: graph-inspection gate, current op histogram is `Conv:2 | ScatterND:1 | TopK:1 | ...`; `ScatterElements=0`, `QLinearConv=0`, `ReduceMax=0`. The charged surface named by the prompt exists only in the obsolete v21/local-live graph, not the current 8710 team-best.
- **why dead**: deleting or narrowing the old `ScatterElements/QLinearConv` family cannot undercut the current team-best because those ops are already gone. The current cost is dominated by `dynamic_w` F32 `[10,10,3,3]` = 3600 B and `detect_map` F32 `[1,1,23,23]` = 2116 B.
- **nearby DCE probe**: `prototypes/task173_topk_blank_index_probe.onnx` tried to delete unused `proto_pos` (24 B). Visible/load gate failed exactly:
  `[visible] ERROR: InvalidGraph: ... Error Node (safe_name_21)'s output 1 is marked single but has an empty string in the graph`
- **one remaining hypothesis**: a different one-op render formulation that avoids materializing data-dependent F32 `[10,10,3,3]` weights entirely; not a rank-chain/QLinear carrier route.
