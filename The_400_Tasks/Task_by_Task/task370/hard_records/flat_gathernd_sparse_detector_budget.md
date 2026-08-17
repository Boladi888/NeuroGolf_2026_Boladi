# task370 hard record — flat_gathernd_sparse_detector_budget

- live bar: 6204, SHA `d9643e949d751f6d3c5266908a35b400ce280ed185d96db71c1b3037ce9c5a94` (`python tools/ng_current_bar.py --task 370 --json`, manifest source, 2026-07-04T05:54:11Z).
- attempted route: reviewer-locked flat `GatherND` sparse source-read detector before rendering; no projection-kernel trim, active-mask deletion, full-input pre-cast, or current-graph shave.
- pricing stub: `The_400_Tasks/Task_by_Task/task370/prototypes/price_task370_flat_gathernd_sparse_detector.py`.
- exact pricing result:
  - `hidden_lengths_3_4_lower_bound: reads=100 index=2400 f32=400 subtotal=2800`
  - `visible_clean_lengths_2_3_4_5_lower_bound: reads=216 index=5184 f32=864 subtotal=6048`
  - `uniform_5x5_all_visible_candidates: reads=400 index=9600 f32=1600 subtotal=11200`
  - `target=2800 verdict=STOP reason=pre-render lower bound is not strictly below target`
- why dead: ORT `GatherND` indices are INT64-only and `GatherND` output inherits F32 from the graph input. Even the best-case hidden-generator-only sparse reads (four directions, lengths 3 and 4, batch_dims=1 indices `[1,N,3]`) cost exactly 2800 before dynamic index assembly, detector/select logic, projection/render, active masking, or output overlay. The reviewer target is strictly below 2800, so this route fails before ONNX build. Visible-clean support including fixed length-2 and length-5 examples is already 6048 before detector/render.
- one remaining hypothesis: a non-GatherND representation would need to avoid materializing all direction/length candidate reads, likely by returning to a small contiguous Slice/window or a fused Conv-style detector; that is outside this locked pass.
