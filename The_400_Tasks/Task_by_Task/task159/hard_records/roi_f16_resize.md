live bar: 472 profile cost, sha256 e0241fe189209d009711a368f3e782f0afab6435aedac5b3199da11233e6c277 (`ng_current_bar.py --task 159 --json`, manifest authority).

attempted route: `prototypes/build_task159_roi_f16.py` narrows only the final `Resize` ROI path to F16 (`roi6`, `roi_end`, `ch_roi_end_v`) while keeping QLinearConv scales F32; candidate `prototypes/task159_roi_f16.onnx`, sha256 49aa0f91225bb84920d1dd4a5b708e31aec9010ea5a607b74e85bbf02978b2ad.

failed gate: `python tools/ng_gate.py visible --task 159 --cand The_400_Tasks/Task_by_Task/task159/prototypes/task159_roi_f16.onnx` errors in ORT `Resize` with `Tensor type mismatch. T != ... MLFloat16`; `profile_real_cost.py --arcgen 266` fails with the same runtime exception.

why dead: ONNX shape inference accepts F16 ROI, but ORT 1.24.4's `Resize` kernel does not run with F16 ROI on this graph, so the apparent cost-diff saving is invalid.

one remaining hypothesis: only a formulation that removes the ROI vector entirely, or fuses spatial/channel placement into a different final op, can save this 24-byte F32 `roi6` cost; dtype narrowing the `Resize` ROI is not viable.
