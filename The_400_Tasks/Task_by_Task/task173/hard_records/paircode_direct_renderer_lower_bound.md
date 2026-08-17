# task173 hard record - pair-code direct renderer lower bound

- **bar used**: team-best/profile cost `8710`, sha256 `27f31830cf75f60bdcf0273b403dcb804b8f4b6950817091f50e761be3a57940`.
- **bar source**: `python tools/ng_current_bar.py --task 173 --json`, authoritative source `manifest`; staged `task173_regionfillrewrite_cost8710` and workbook agree on cost `8710`.
- **attempted route**: prompt route "template completion without dynamic F32 weight"; price a pair-code detector plus direct stamp renderer that avoids both `dynamic_w` F32 `[10,10,3,3]` and the current full F32 `detect_map` `[1,1,23,23]`.
- **exact failed gate**: pricing/graph-inspection gate on the closest no-23x23-detect-map family, `staged_candidates/task173_umax_paircodes_cost18463/task173.onnx`.
  - `score_v2`: `(0, 17159, 1304)`, cost `18463`.
  - `visible`: `[visible] 266/266 (100.00%)  PASS`.
  - `largest_intermediate`: `dyn_w` 3600, `label25_f` 2304, `weight_updates` 936, and many U8 `[1,1,24,24]` detector planes at 576 each.
- **why dead**: even deleting every clearly render-exclusive tensor from the visible-passing pair-code graph leaves a detector/triple-extraction lower bound of `12237`:
  `18463 - (dyn_w 3600 + weight_updates 936 + weight_indices 416 + weight_indices_u8 52 + weight_indices_flat_u8 52 + update_cond_bool 26) - (base_w 900 + update_const 234 + conv_bias 10) = 12237`.
  This is already `+3527` above the `8710` live bar before adding any direct label/stamp renderer. The current 8710 detector is cheaper only because it keeps the forbidden full F32 `detect_map`, so it is not a valid route for this prompt.
- **one remaining hypothesis**: find a new detector/triple emitter that uses neither the 23x23 F32 Conv map nor the 24x24 U8 pair-code plane stack, and leaves enough budget for rendering below `8710`; no existing task-local family provides that surface.
