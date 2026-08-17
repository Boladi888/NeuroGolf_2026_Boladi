# task390 hard record: orientselect final-Einsum gray map floor

- live bar used: 2568 from `python tools/ng_current_bar.py --task 390 --json`, sha `9f41e653a7fdfe6a14acd059af7c75faf0d77fcd0c3c6b3b77054931c84d07ff`; stricter task-local team-best bar is 2567 from `staged_candidates/task390_w09_hflag_r0_cost2567/manifest.json`.
- attempted route: scalar-before-assembly orientation selector with #93-style case selection, then a single final-Einsum gray renderer using an 8x8 gray crop, shared `[4,8,15]` source-map bank, and shared `[15,30]` active-to-output embed.
- prototype: `The_400_Tasks/Task_by_Task/task390/prototypes/probe_orientselect_einsum_floor.py`
- lower-bound ONNX: `The_400_Tasks/Task_by_Task/task390/prototypes/task390_orientselect_einsum_gray_floor.onnx`, sha256 `69bc209f9e0caad489b19fdc6b6e1753853c0f9ee80a0e67fde91c52a7463a4a`
- exact failed gate: fast probe scored `score_v2=(0, 256, 1003), cost=1259` after `numpy_full_generator_probe=3000/3000`; real-input profile confirmed `static=1259 profile=1259 (memory 256 + params 1003; 266 real inputs)`.
- why dead: this is gray-only and omits red/background rendering plus dynamic selector construction, yet already exceeds the prompt's `<900` kill line. The forced pieces are `fives8` F32 slice 256B, `map_bank[4,8,15]` 480 params, shared `active_embed[15,30]` 450 params, and small mode/channel selectors 73 params. Full assembly can only add cost.
- one remaining hypothesis: a non-Einsum scoreable surface that emits the reflected gray plane as U8 without dense source/output maps, while still avoiding the rejected two full axisconv branches.
