# task259 blue_u8_minmax dead route

- Live bar: 907 from `python tools/ng_current_bar.py --task 259 --json`, sha256 `6f3f1ec1a5937f38712575db8f21f6cdbc08d3df0cb2701c221bc986d39d925f`.
- Effective staged bar: 898 from `staged_candidates/task259_bool_tail_pad_axes_cost898/manifest.json`, sha256 `d5f8ce990629e58a08c5a7379dac297debaf864afc14e73a6006a100943c127e`.
- Attempted route: `prototypes/task259_v12_blue_u8_minmax.onnx`, built by `prototypes/build_task259_v12_blue_u8_minmax.py`; keep F32 `blue7_f`, cast to U8, replace F32 row/col `ReduceSum` + `Less` with U8 `ReduceMin`/`ReduceMax` and `ArgMin`.
- Price: `score (0, 842, 39) 881`, sha256 `1ba566b65f91ec6d9e83dcb1b813a1ff10b331f22c8ca7b3843a90802f39b3f6`.
- Failed gate: `[visible] 40/266 (15.04%)  FAIL`.
- Why dead: one visible miss is a certain hidden zero; the U8 min/max representation conflates non-blue object cells with zero padding/trim cells often enough to mis-locate the crop.
- One remaining hypothesis: a different representation must avoid post-slice min/padding ambiguity; likely target the forced F32 dynamic color crop or final 3x3 crop instead of recoding blue row/col sums after the F32 slice.
