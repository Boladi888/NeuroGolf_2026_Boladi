# source3_extrap_resize

Live bar: 468 bytes, manifest sha256 8bc8c0fc9401965b9a1889dab5aaba275f3c5e9996e1805f7b2c559863731a0f.

Attempted route: remove the 4x4 zero-sentinel Pad and feed the 3x3 U8 source directly to final `Resize(tf_crop_and_resize)` with ROI numerator 14.51/N, relying on `extrapolation_value=0` for cells beyond 3N.

Failed gate: `python tools/ng_gate.py visible --task 269 --cand The_400_Tasks/Task_by_Task/task269/prototypes/task269_source3_extrap_u8.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.

Why dead: visible-zero means this representation family is hidden-zero; no truth/AB/battery gates run and no source3-extrap variants should be built.

Remaining hypothesis: only a source-backed sentinel or a different final op family can zero-fill the post-3N region safely; source3 direct Resize is not it.
