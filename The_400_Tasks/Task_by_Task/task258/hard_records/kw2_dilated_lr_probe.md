# task258 hard record: kw2 dilated left/right Conv probe

- live bar: 150, sha256 `66f01abc7c629250bbfcdec2238603f4e6fa651d9cc9bfd06b9930e692b326ba` (`python tools/ng_current_bar.py --task 258 --json`, source `manifest`, owner Vamshi, updated 2026-07-12T15:15:54Z).
- attempted route: one-node grouped Conv with `W[10,5,1,2]`, `group=2`, `dilations=[1,2]`, `pads=[0,1,0,1]`; nominal params 100, no bias. Probe builder `../prototypes/build_task258_kw2_dilated_lr_probe.py`; candidate `../prototypes/task258_kw2_dilated_lr_probe.onnx`; sha256 `660830a37fa9a6311c7a868cb3e54cbad525ea6d293518bd0b0f4924ea6692b6`.
- exact failed gate: `python tools/ng_gate.py visible --task 258 --cand The_400_Tasks/Task_by_Task/task258/prototypes/task258_kw2_dilated_lr_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- why dead: the two-tap left/right representation can test neighboring markers for ch2, but it cannot also preserve/suppress the center background channel; task258 needs the center tap for the full 10-channel output, not just for boundary safety.
- remaining hypothesis: only a legal way to get output channel 2 into the same group as input channels 0/1 with fewer than `W[10,5,1,3]` params could beat 150; group=5/depthwise and non-10-channel outputs are already dead.
