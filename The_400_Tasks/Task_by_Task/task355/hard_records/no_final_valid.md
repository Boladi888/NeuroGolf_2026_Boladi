# no_final_valid dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (OnnxlyFans workbook + manifest).
- Attempted route: delete final `scores = Where(valid, mod_score, 0)` and feed `mod_score` directly to `ReduceMax`/`Equal`; static estimate 407.
- Failed gate: `visible` on `task355_bo_ng_worker1_no_final_valid.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: invalid marker colors can carry positive `mod_score`; the final valid mask is load-bearing.
- Remaining hypothesis: only a different invalid-color suppression that costs less than one U8 `[1,10,1,1]` tensor could reopen this.
