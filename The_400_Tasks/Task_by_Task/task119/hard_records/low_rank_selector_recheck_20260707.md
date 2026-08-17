# task119 - low-rank selector recheck (2026-07-07)

- Live bar: cost 1131 from `python tools/ng_current_bar.py --task 119 --json`; authoritative source `manifest`, owner Aditya, sha256 `38381961b34e7bd6868048abbea87fbae2a20afe34414248bf7bc11f3aaddae0`.
- Attempted route: reuse the task-local no-ScatterND low-rank selector probe at `The_400_Tasks/Task_by_Task/task119/prototypes/task119_worker2_low_rank_selector_probe.onnx`.
- Exact failed gate: `python tools/ng_gate.py visible --task 119 --cand ...task119_worker2_low_rank_selector_probe.onnx` -> `[visible] 265/266 (99.62%)  FAIL`.
- Price failure: `score_v2` -> `(0, 5885, 1346) 7231`, which is +6100 bytes vs the 1131 live bar. `ng_graph_cost_diff` names the deleted `scatter_indices` (-640 B) and `scatter_indices_u8` (-80 B), but adds F32 selector/term materialization: `worker2_term_bank_f32` +4400 B, row/col terms +1320 B each, row/col selectors +1200 B each.
- Why dead: this representation deletes the INT64 ScatterND index bank only by materializing F32 row/col selectors and an identity-plus-correction term bank; it is both visibly wrong and far above bar. Per visible-fail protocol, do not build variants in this family.
- One remaining hypothesis: find a legal final ONNX op that consumes the existing `vec_r_u8`/`vec_c_u8` coordinate vectors and updates both ch0 and ch3 per cell without full-rank ScatterND indices, a 30x30 mask, a full-layout transpose/reshape, or F32 selector materialization.
