# task258 hard record: non-10-channel free-output mask

- live bar: 160, sha256 `9cf85f5ef87760197eaab665af872adc9c2a38a6b9c2133f506deb1e51402d09` (`python tools/ng_current_bar.py --task 258 --json`, source `manifest`, updated 2026-07-02T02:35:15Z).
- attempted route: one-node mask-only Conv output `[1,1,30,30]`, cost 31 (`W[1,10,1,3]` + `B[1]`), probe `../prototypes/task258_freeoutput_mask_probe.onnx`, sha256 `d50f412143bca46b0a6babd0b0418a9ab326aa28c33d68d09a6e909540ed85fc`.
- exact failed gate: `python tools/ng_gate.py visible --task 258 --cand The_400_Tasks/Task_by_Task/task258/prototypes/task258_freeoutput_mask_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- why dead: today's `ng_gate` compares raw thresholded output tensors against the canonical encoded expected tensor `[1,10,30,30]`; a one-channel/cropped graph output cannot be reviewer-ready even if its local static cost is below 64.
- remaining hypothesis: only a single-node full `[1,10,30,30]` output can stay under the 160 bar; improving below 160 would need a legal way to express the full 10-channel horizontal 3-tap rule with fewer than `W[10,5,1,3]+B[10]`, not another non-10-channel tail.
