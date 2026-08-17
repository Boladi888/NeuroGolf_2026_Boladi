# task045 hard record: reviewer locked label-tail i8

- Live bar: 512 bytes, sha256 `4230502505d6e36e8aab07c4039c8af5a6aaf9b53c6217b209b9bc677aad7fcf`, source `python tools/ng_current_bar.py --task 45 --json` / manifest owner Sho.
- Source graph: `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task045.onnx`.
- Attempted route: preserve live `left_i8/right_i8/same_color/nonmatch_i8`; remove `left2/right2`, the 2-plane `embed_pair`, and final `q_cls_w/q_cls_b`; use current second `q_shift_w` group as `q_label_w` over `nonmatch_i8,left_i8,right_i8`; output `Equal(label_i8, channel_ids_i8)`.
- Profile gate: `task045 task045_label_tail_i8.onnx: static=312 profile=312 (memory 260 + params 52; 265 real inputs)`.
- Named cost delta: `ng_graph_cost_diff` reported net `-200` predictor bytes from removing `embed_pair`, `shift_row_feat`, `left2`, `right2`, `q_cls_w`, `q_cls_b`, and half of `q_shift_w`, offset by `label_i8`, `label_row_feat`, `q_label_w`, and `channel_ids_i8`.
- Exact failed gate: `python tools/ng_gate.py visible --task 45 --cand prototypes/task045_label_tail_i8.onnx` returned `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: variant A loads and executes, so the reviewer's UINT8 fallback condition did not trigger. The locked Equal-tail representation is visible-zero and must stop here.
- Remaining hypothesis: only a different proof of final decoder semantics could revive this route; do not retry diff2/Where or UINT8 unless INT8 Equal/load typing, not visibility, is the blocker.
