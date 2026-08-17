# task259 drop_size_floor_max dead route

- Live bar: 469 from `python tools/ng_current_bar.py --task 259 --json`, sha256 `c6b61efc5effdb5774fcdc06d89542a86ae1b25ac1e4b2bf66a638c6556f1846`.
- Attempted route: `prototypes/task259_v16_drop_size_floor_max.onnx`, built by `prototypes/build_task259_v16_drop_size_floor_max.py`; delete Sho's two `Max(..., [1,1,0])` floor-mask repairs and feed the gathered row/col activity vectors directly to the final mask product.
- Price: `score (0, 414, 46) 460`; profile line `task259 task259_v16_drop_size_floor_max.onnx: static=460 profile=460 (memory 414 + params 46; 266 real inputs)`; sha256 `9025debcf6a03e6fd2df3b8dec70dbae91d694ad60ebcb2e51381803886ac405`.
- Failed gate: `python tools/ng_gate.py visible --task 259 --cand The_400_Tasks/Task_by_Task/task259/prototypes/task259_v16_drop_size_floor_max.onnx` -> `[visible] 253/266 (95.11%)  FAIL`.
- Why dead: the first two rows/cols in the 3-slot output frame are not guaranteed active in visible officials; the `[1,1,0]` floor repair is semantic, not redundant.
- One remaining hypothesis: look for byte-equal shaves that preserve the floor-mask repair, such as replacing a different singleton/index subchain or changing the final output basis without deleting bbox-size guards.
