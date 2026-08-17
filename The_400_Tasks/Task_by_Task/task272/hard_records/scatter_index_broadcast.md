# task272 hard record: ScatterElements index broadcast

Live/team-best bar: cost 378, sha256 `f0302ce51976334b74f18b95e7cb9e455aa91d56d97a77bc47b1ee604000d526`, source `Team_Submission/Backend/manifest.csv` and `NG_OnnxlyFans_TaskStatus.xlsx` row `task272`.

Attempted route: collapse the team-best `[1,2,5,5]` ScatterElements channel-index initializer to `[1,2,1,1]` and rely on runtime broadcast over the update planes. Probe path: `The_400_Tasks/Task_by_Task/task272/prototypes/task272_worker3_idx_broadcast.onnx`, sha256 `d3ec0839e44933926472337414fe1e250e3ee1103f35a0bd6a7532cc8188f345`.

Exact failed gate: `python tools/ng_gate.py visible --task 272 --cand The_400_Tasks/Task_by_Task/task272/prototypes/task272_worker3_idx_broadcast.onnx` returned `ERROR: InvalidArgument ... ScatterElements ... Indices vs updates dimensions differs at position=2 1 vs 5`. Static scorer returned `None None` with the same ORT error.

Why dead: `ScatterElements` enforces `indices.shape == updates.shape`; the index table cannot broadcast over H/W. Runtime-generating or expanding the index table is also cost-worse because the materialized `[1,2,5,5]` INT64/INT32 index becomes charged memory instead of cheap initializer params.

One remaining hypothesis: only a final op that does not require explicit per-channel scatter indices can delete these 50 params, but the priced alternatives here (`Where`, generated small one-hot + Pad, chained scatters) all exceed the 378 bar.
