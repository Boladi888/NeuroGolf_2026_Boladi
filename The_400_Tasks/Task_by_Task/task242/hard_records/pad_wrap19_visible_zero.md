# task242 pad_wrap19 visible-zero

- live bar: 422, SHA-256 `18963c9bb81868c1787bee15757387cd2e370259f6bff9f5a9f844b7eb4c48d6` (`Team_Best_Onnx/task242.onnx`, manifest owner Vamshi, updated 2026-07-12T15:15:54Z).
- attempted route: change the 422 graph's final Pad to opset-19 `mode="wrap"` as a direct Pad-wrap probe.
- prototype: `The_400_Tasks/Task_by_Task/task242/prototypes/task242_vamshi422_pad_wrap19.onnx`, SHA-256 `fa7016e7bf6c252ec559248cd02dc3ffa83aabeab230877b61c11733b3e92144`.
- failed gate: `python tools/ng_gate.py visible --task 242 --cand ...pad_wrap19.onnx --json --out ...pad_wrap19_visible.json` returned `correct=0 total=266 pct=0.0 pass=false`.
- why dead: task output semantics require zero padding outside the 3x3 copied patch; wrap padding repeats the patch through the padded canvas and changes every visible output.
- one remaining hypothesis: Pad-wrap is only relevant if the rule is a toroidal tiling, not for this final zero-padded extraction. Any real sub-422 route must delete or avoid the charged `patch9` crop or one of the required params without changing final constant padding.
