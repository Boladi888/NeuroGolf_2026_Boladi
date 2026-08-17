# task356 pad-only writer dead route

- Live/team-best bar: 1220 profile_cost, SHA `e81463c0cf23b583f80f00c8224100f082a46eeab21f8ea5460ef26e28c3aedc` from OnnxlyFans manifest/workbook.
- Attempted route: replace the current ConvInteger final writer with direct Pad expansion of the U8 fill mask to save 4 params (`task356_u8_pad_expand_4B786273.onnx`, SHA `4b786273d70c0d4afb26d95ee67d8e8b262dffe2ef0de7e4a19e8dd568a991cd`).
- Cost gate: `(0, 1200, 16) 1216`; `profile_real_cost`: `task356 task356_u8_pad_expand_4B786273.onnx: static=1216 profile=1216 (memory 1200 + params 16; 266 real inputs)`.
- Failed gate: `python tools/ng_gate.py visible --task 356 --cand The_400_Tasks/Task_by_Task/task356/prototypes/task356_u8_pad_expand_4B786273.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: Pad can place the foreground channel but cannot write active-grid background channel 0 as positive while clearing it on filled cells; this is a certain hidden zero, not a candidate family.
- One remaining hypothesis: beat 1220 only by deleting a required 100 B span/union tensor or by finding a legal final writer with fewer than the current `quant_scale + conv_weights + conv_x_zero_point` 12 params while preserving channel-0 background completeness.
