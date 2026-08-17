# bool_argmin_toprow — invalid checker route

- Live bar used: manifest/team-best cost **301**, sha `e47521f8db71c7b3adebc1ed79182d29bedefc18739e55338f9d038cd3d1030e` (`python tools/ng_current_bar.py --task 121 --json`).
- Attempted route: remove the 3-byte `Cast(top_bg_bool -> top_bg_u8)` from the 295 booltop candidate and feed `top_bg_bool` directly to `ArgMin`.
- Builder: `The_400_Tasks/Task_by_Task/task121/prototypes/build_booltop_argmin_bool.py`.
- Exact failed gate: `python The_400_Tasks/Task_by_Task/task121/prototypes/build_booltop_argmin_bool.py` -> `ShapeInferenceError (op_type:ArgMin): data typestr: T, has unsupported type: tensor(bool)`.
- Why dead: pinned ONNX checker/shape inference rejects BOOL `ArgMin`; no scoreable artifact is produced.
- One remaining hypothesis: the accepted floor for this selector is the staged `booltop_gather` route with a 3-byte U8 cast before `ArgMin`, unless a non-ArgMin arithmetic first-zero selector can beat its extra tensors.
