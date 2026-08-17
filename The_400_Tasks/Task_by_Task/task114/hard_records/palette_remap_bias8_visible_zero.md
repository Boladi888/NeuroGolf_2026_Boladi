# task114 palette-remap bias8 visible-zero

- Live bar used: 232, sha256 `434f570e6befaabfaac56fe0234f67d8cc4f9eb68cfdb517f89e2c84652e7f12`, source `python tools/ng_current_bar.py --task 114 --json` / team manifest.
- Attempted route: remap the compact task palette so output channel 8 could use a no-bias QLinearConv decoder slot, trimming `v_w7` from 9 to 8 INT32 bias elements (`task114_palette_remap_bias8.onnx`, static/profile 231).
- Failed gate: `python tools/ng_gate.py visible --task 114 --cand The_400_Tasks/Task_by_Task/task114/prototypes/task114_palette_remap_bias8.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the parent graph's scalar code `0` is not freely remappable; all-zero padding reads also produce code `0`, and the existing assembly/shape tests rely on that false/background behavior. Moving color 8 to code `0` makes padding decode as color 8 and breaks every visible sample.
- One remaining hypothesis: a sub-232 win must remove a charged tensor in the 5x5 assembly/final decode tail, not just remap palette codes or trim the QLinearConv bias.
