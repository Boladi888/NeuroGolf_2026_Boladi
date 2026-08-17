# task244 hard record - low-level selector legality shaves

Live/team bar: cost 85, sha256 db55e34c9cd94f6da4514281bdd2a377d1b4783c59a83e8d62f09353a60f175e (workbook + team manifest, updated 2026-07-02T02:35:15Z).

Attempted route: shave the current selector without changing the algorithm by removing the Bool->U8 cast before `ArgMin`, or by storing the ROI table as F16.

Prototype 1: `The_400_Tasks/Task_by_Task/task244/prototypes/task244_cdx2_argmin_bool.onnx`, sha256 cab3f5fc92875eaaaefe2475a79db5a07f31b9cb9c9cc4ea9de93626416f5a9f, static score `None`.

Exact failed gate 1: checker/load -> `ShapeInferenceError (op_type:ArgMin): data typestr: T, has unsupported type: tensor(bool)` and ORT `Type 'tensor(bool)' of input parameter (passed) of operator (ArgMin) ... is invalid`.

Prototype 2: `The_400_Tasks/Task_by_Task/task244/prototypes/task244_cdx2_roi_f16_mixed.onnx`, sha256 7c6718ebbbf23f7e95949eb37f3e939cd3f11a7a54cf54a6b8be25af33804150, static score `None`.

Exact failed gate 2: checker/load -> `ShapeInferenceError (op_type:RoiAlign): rois has inconsistent type tensor(float16)` and ORT `Type parameter (T1) of Optype (RoiAlign) bound to different types (tensor(float) and tensor(float16)`.

Why dead: `ArgMin` does not accept Bool in this scorer/ORT path, and `RoiAlign` requires `X` and `rois` to share the same float type; casting either side adds more memory than it saves.

One remaining hypothesis: only a legal exact selector with fewer params and no extra vector intermediates can beat 85; changing the Bool cast to INT8 is legal and visible-clean but ties at 85.
