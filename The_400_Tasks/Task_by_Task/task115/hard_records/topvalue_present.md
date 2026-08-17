# topvalue_present dead route

- Live/team-best bar: 389, SHA-256 e959b06e6782df52dbb55e802bb0613d0825fb4f71bad262712975dcf8022357, source `Team_Submission/Backend/manifest.csv` and workbook row updated 2026-07-02T02:35:15Z.
- Attempted route: remove `top_present` by scaling `row_decay` 2000x and feeding `Cast(top_values)->present_h` into the current 389 graph.
- Candidate: `The_400_Tasks/Task_by_Task/task115/prototypes/bo_ng_cdx2_worker2_topvalue_present/task115_topvalue_present_s2000.onnx`, static cost 385, SHA-256 eec27a1d7bdd53962ccf798e8797cf88c7040d5fd04d1a0778043502953ff17b.
- Failed gate: `python tools/ng_gate.py visible --task 115 --cand ...task115_topvalue_present_s2000.onnx` -> `[visible] 221/266 (83.08%)  FAIL`.
- Why dead: score scaling keeps slots present but changes the F16 polynomial render enough to create visible output divergences; a visible miss is a certain hidden zero, so no truth/profile/full gates were run.
- Remaining hypothesis: a sub-389 win must remove or shrink a different charged item, likely `seq_h`/`OneHot` or `top_indices`, without scaling the final polynomial render.
