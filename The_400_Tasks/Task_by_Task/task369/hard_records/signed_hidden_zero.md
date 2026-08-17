# task369 - signed hidden zero-point scalar shave is dead

- Live bar: cost 1111, sha256 `3583d772e665e5eb9bfad145de708f8e7f916af80642d92ef96da6e1cc3d8d85`, source `python tools/ng_current_bar.py --task 369 --json` / manifest.
- Attempted route: replace the live U8 hidden feature stack with signed INT8 hidden features and reuse `i8_zero` for all QLinearConv/ConvInteger zero-points, deleting `u8_zero` for an expected 1-param shave.
- Prototype: `The_400_Tasks/Task_by_Task/task369/prototypes/signed_hidden_probe/task369_signed_hidden_zero.onnx`, sha256 `02bce63e73d138e8e4b24451ea50618c8f8d0f4e832b2470b75386058a234aa8`.
- Failed gate: `python tools/ng_gate.py visible --task 369 --cand The_400_Tasks/Task_by_Task/task369/prototypes/signed_hidden_probe/task369_signed_hidden_zero.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: the live first QLinearConv relies on U8 lower clipping; signed-zero INT8 preserves negative feature values, so the existing two-conv classifier is not even visible-correct. Per visible-fail axiom, do not run truth/AB/battery or build minor variants in this representation family.
- One remaining hypothesis: only an exact encoded INT8 hidden zero point such as `-128` can emulate U8 clipping, but it needs a second scalar zero point and therefore ties the 1111 bar rather than beating it.
