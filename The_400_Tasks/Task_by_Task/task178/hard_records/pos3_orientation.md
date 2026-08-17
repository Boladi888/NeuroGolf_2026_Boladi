live bar: team manifest task178 cost/profile_cost 639, sha f4b51ab75e24041161fb42a1667a079089ebe35575d202234eca0f6125f8a756.
attempted route: replace the three-cell row orientation detector with a single position-3 active-and-different test in `prototypes/task178_v33_pos3_orientation.onnx`.
failed gate: `python tools/ng_gate.py visible --task 178 --cand The_400_Tasks/Task_by_Task/task178/prototypes/task178_v33_pos3_orientation.onnx` -> `[visible] 259/268 (96.64%)  FAIL`.
why dead: horizontal examples can have total active width 3 or repeat the first color at position 3, so position 3 alone collapses the output to the first color.
one remaining hypothesis: keep the required positions 1/2/3 but replace the current color-sum detector with a cheaper bool OR/ReduceMax form if ORT scoring accepts it.
