live bar: 3481 bytes, sha f727187c16c9d3caad99e7cbb71827701f4d9573e2378adc325f354951148c36 (ng_current_bar manifest, 2026-07-02)

attempted route: reviewer-locked fused signed QLinearConv/Conv detector map for the generator's W vs W-1 grouping. Built an intentionally optimistic price-only skeleton at `prototypes/task277_fused_detector_skeleton.onnx`: one signed 6x6 QLinearConv detector map is used directly as the color-1 mask, with no anchor-to-shape stamping, no foreground re-gating, no width4-vs-width3 global disambiguation, and no exact edge-validity repair.

failed gate: skeleton score/profile gate before exact logic. `_score_v2` = `(0, 1300, 57)`, cost `1357`; `profile_real_cost --arcgen 266` = `static=1357 profile=1357 (memory 1300 + params 57; 266 real inputs)`. This is already `>=1200`, so the reviewer stop condition triggers.

why dead: the lower-bound charged tensors are forced before the real detector logic: `mask10` F32 400B, `mask_bool` 100B, `mask_u8` 100B, one `det_u8` map 100B, `det_bool` 100B, `color2_bool` 100B, `bg_bool` 100B, and `rgb3` pre-Pad 300B = 1300B memory, plus 57 params. Exact logic would only add cost: foreground-gating detector hits, stamping anchors across component cells, global W/W-1 selection, and edge/boundary validity.

one remaining hypothesis: only a radically different output assembly that deletes the 300B `rgb3` pre-Pad or avoids either `mask_bool`/`mask_u8` could reopen a sub-1200 path; the reviewed single fused detector-map formulation itself is over budget before correctness gates.
