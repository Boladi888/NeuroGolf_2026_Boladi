# no_p9_col_fallback

- live bar: 5,145 from `python tools/ng_current_bar.py --task 110 --json`, sha `d84c238027170cd8bfa1c7e6a56c7eb87a352aba8a5d3c8aa92f5c127811fc6d`; current task-local reviewer bar before this pass is 3,931 from `staged_candidates/task110_raggedbank_f16det_cost3931/manifest.json`, sha `6aaee80304bd51b28118a56cd9eaddfba439972abdb6359cdb6a394ad9148a55`.
- attempted route: remove the dynamic period-9 column fallback from the 3,931 linear-vote graph, replacing `v_Bcol[38,30] + v_colmask[38]` with identity-only `v_Bid[29,30]` in the final `Einsum`.
- priced cost: `python -c/import score_network_v2` on `prototypes/bo_ng_cdx_worker2/task110_no_p9_col.onnx` -> `(0, 1030, 2409) COST 3439`; sha256 `83fbe0204f6a60c544bb5e59caa43f9b5c0a3b41db5100f5b0d1901943bbefde`.
- exact failed gate: `python tools/ng_gate.py visible --task 110 --cand The_400_Tasks/Task_by_Task/task110/prototypes/bo_ng_cdx_worker2/task110_no_p9_col.onnx` -> `[visible] 265/266 (99.62%)  FAIL`.
- why dead: the period-9 column fallback is required by at least one official ARC-GEN example (`idx=225`, mismatches at columns 3/4 across rows 2/11/20). Under §VISIBLE-FAIL, the identity-column-only representation is certain hidden-zero; no profile/truth/AB/full gates were run.
- one remaining hypothesis: sub-3,931 would need a cheaper full period-9 fallback than the current rank-9 column bank plus scalar-expanded mask, not a partial or deleted fallback.
