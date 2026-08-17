# bool_maxpool_path

- live bar: 3637, sha `0c5b087e1708ee1e348809c46c3bc1e5dcb94234fc4c11444875f9696cc4dcf3` (`ng_current_bar.py --task 265 --json`, manifest source).
- attempted route: keep the repair and dilation path entirely BOOL (`build_task265_bool_path_probe.py`) to remove the UINT8 anchor/MaxPool lane.
- exact failed gate: `python tools/ng_gate.py visible --task 265 --cand The_400_Tasks/Task_by_Task/task265/prototypes/build_task265_bool_path_probe.onnx` -> `InvalidGraph ... Type 'tensor(bool)' ... MaxPool ... is invalid.`
- why dead: ORT 1.24.4 does not support BOOL `MaxPool`; the route cannot load, so no visible/truth gate can be meaningful.
- one remaining hypothesis: BOOL `ScatterND` is valid if the graph casts to UINT8 before `MaxPool`; v8 uses that narrower repair-only variant and reaches 3636.
