# task162 smaller_anchor_directconv

- Live/team bar used: 4066, SHA `d7425dac1b9a067bbbb80234467f28718a0e4ac2ed2d2ff3a5765699e1c9084d` from `python tools/ng_current_bar.py --task 162 --json`.
- Attempted route: replace the exact 3x3 direct Conv anchor read with a 2x3 direct Conv read (`W_direct` 90 params -> 60 params), then keep the same QLinearConv dedupe, MaxPool dilation, Pad, and Where tail. Prototype: `prototypes/task162_directconv_2x3_probe.onnx`, static cost 4050.
- Failed gate: `python tools/ng_gate.py visible --task 162 --cand The_400_Tasks/Task_by_Task/task162/prototypes/task162_directconv_2x3_probe.onnx` -> `[visible] 229/266 (86.09%)  FAIL`.
- Why dead: a 2x3 all-black window is not equivalent to a true 3x3 all-black anchor on officials; it overfires strips/partial black regions. Per the visible-fail axiom this family is hidden-zero and should not get truth/ab/battery.
- One remaining hypothesis: only a representation that preserves exact 3x3 anchor evidence while removing dense zero Conv params could beat 4066; sparse initializers are documented scorer-dead, so the credible open lever is a genuinely different nonlinear fusion, not another smaller anchor kernel.
