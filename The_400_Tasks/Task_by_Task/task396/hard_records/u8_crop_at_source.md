# task396 hard record: U8 crop at source

Live bar:
- Authoritative bar: cost/profile_cost 3694 from `python tools/ng_current_bar.py --task 396 --json`.
- Live SHA-256: `511a52eee6dea62a3f1d7f53160b2e2d51e594da50b15d55d8bb645d6cd9054a`.
- Source: `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`.

Attempted route:
- Literal prompt route: add `Cast(input -> UINT8)` before the frame-channel Slice, make the Slice produce `frame_work_u8`, and remove the post-Slice Cast.
- Prototype: `The_400_Tasks/Task_by_Task/task396/prototypes/task396_u8_cast_at_source_probe.onnx`.
- Builder: `The_400_Tasks/Task_by_Task/task396/prototypes/build_task396_u8_cast_at_source_probe.py`.

Exact failed gate:
- Static score: `(0, 11073, 325) 11398`.
- Visible: `[visible] 266/266 (100.00%)  PASS`.
- Profile: `task396 task396_u8_cast_at_source_probe.onnx: static=11398 profile=11398 (memory 11073 + params 325; 266 real inputs)`.

Why dead:
- The graph input is F32 under the NeuroGolf harness. `QLinearConv` cannot consume F32, so the only legal source-cast version must materialize `input_u8_source` at `[1,10,30,30] = 9000` bytes.
- That 9000-byte full-source cast dwarfs the intended `frame_work_pre_f32` 1296-byte deletion; cost rises from 3694 to 11398.

One remaining hypothesis:
- Do not retry full-source U8 casting. The viable sibling is a fused read that keeps the input F32 and avoids materializing the crop; worker1 staged `task396_fused_f32_prefix_cost3378` using a negative-pad dynamic F32 Conv prefix read.
