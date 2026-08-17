# task332 tiled-column-basis route dead

- **Live bar:** cost 80, sha256 `ea1e530cde02e3cfc8aa793b77e574cf2e3b91a9c66bb809f44213dd4b8fba6f`, source `python tools/ng_current_bar.py --task 332 --json` (manifest, owner Vamshi, updated 2026-07-02T02:35:15Z).
- **Attempted route:** replace the 60-element dense `column_basis[2,30]` initializer with `column_seed[2,2]` + `tile_repeats[2]` + `Tile`, then feed the same final `Einsum`.
- **Prototype:** `The_400_Tasks/Task_by_Task/task332/prototypes/tiled_column_basis_probe.onnx`, sha256 `9912dda2fb25f5d6e128d0df9c765e36d2735ad934845f8f538880f1980bc9d8`.
- **Exact failed gate:** score/profile cost gate. Visible passed: `[visible] 267/267 (100.00%)  PASS`, but `score_v2: static=266 memory=240 params=26 macs=0` and `profile_real_cost: static=266 profile=266 (memory 240 + params 26; 267 real inputs)`.
- **Why dead:** the route saves 54 params (`column_basis` removed, `column_seed` + `tile_repeats` added) but materializes `column_basis_tiled[2,30]` as a 240-byte F32 intermediate, so net cost is +186 versus live. This confirms that generating the periodic basis outside the free-output `Einsum` is worse than storing it.
- **One remaining hypothesis:** only a different single final op that encodes both the constant column row and alternating right-anchor row without a materialized `[2,30]` tensor could break 80; no known ONNX op exposes that relation below the existing tied `Einsum` cost.
