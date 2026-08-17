# task119 — low-rank selector route: DEAD

- **Live bar**: cost **1131** (score_v2 `(0, 1005, 126)`), profile cost
  1131 refreshed in this pass, creator Aditya, sha256
  `38381961b34e7bd6868048abbea87fbae2a20afe34414248bf7bc11f3aaddae0`.
  Refreshed for this pass with `python tools/ng_current_bar.py --task 119 --json`;
  authoritative source is the team manifest with workbook agreement.
- **Attempted route**: replace the terminal ScatterND patch with a final
  low-rank identity-plus-sparse-correction selector:
  `input + row_t[h] * col_t[w] * input_ch0 * (ch3 - ch0)`.
  Prototype:
  `The_400_Tasks/Task_by_Task/task119/prototypes/task119_worker2_low_rank_selector_probe.onnx`
  built by
  `The_400_Tasks/Task_by_Task/task119/prototypes/worker2_low_rank_selector_probe.py`.
- **Exact failed gate**:
  `python tools/ng_gate.py visible --task 119 --cand ...task119_worker2_low_rank_selector_probe.onnx`
  returned `[visible] 265/266 (99.62%)  FAIL`.
- **Price kill**: `score_v2` returned `(0, 5885, 1346) -> 7231`, which is
  **+6100 bytes** versus the 1131 bar. `ng_graph_cost_diff` reports the intended
  removal of `scatter_indices` (-640 B) and `scatter_indices_u8` (-80 B), but the
  route adds F32 selector materialization and a term bank:
  `worker2_term_bank_f32` +4400 B, `worker2_row_terms` +1320 B,
  `worker2_col_terms` +1320 B, `worker2_row_sel_f32` +1200 B,
  `worker2_col_sel_f32` +1200 B.
- **Why dead**: a final free-output Einsum can avoid ScatterND only by carrying
  numeric row/col selector operands compatible with the F32 input and by encoding
  identity plus correction terms. Those charged selectors and channel terms are
  much larger than the 720 B index pack they delete. The first visible miss also
  kills this representation family under the worker protocol.
- **One remaining hypothesis**: a legal final op that consumes the existing
  `vec_r_u8`/`vec_c_u8` coordinate vectors directly and updates both ch0 and ch3
  per cell, without F32 one-hot selector casts, a full 30x30 mask, a layout
  transpose, or ScatterND INT64 indices. No known ONNX op in the allowed set does
  that on NCHW input.
