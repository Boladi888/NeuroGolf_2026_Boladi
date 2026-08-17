# tail_i8_zpmerge — invalid ORT/scorer route

- Live bar used: manifest/team-best cost **301**, sha `e47521f8db71c7b3adebc1ed79182d29bedefc18739e55338f9d038cd3d1030e` (`python tools/ng_current_bar.py --task 121 --json`).
- Attempted route: convert the 3x3 color-code tail from UINT8 to INT8 so the final `QLinearConv` can share a single INT8 zero-point for x/w/y, deleting one scalar initializer.
- Artifact: `The_400_Tasks/Task_by_Task/task121/prototypes/task121_tail_i8_zpmerge.onnx`, sha `281287bfe152044882d9f8598df335a316be640e52f30d8a82abcab17e5758b0`.
- Exact failed gate: `python -c "from tools._score_v2 import score_network_v2; p=r'The_400_Tasks/Task_by_Task/task121/prototypes/task121_tail_i8_zpmerge.onnx'; r=score_network_v2(p); print(r, None if r is None else r[1]+r[2])"` -> `None None`.
- Failure detail: direct sanitized ORT load raises `NotImplemented: Could not find an implementation for Where(16)` for INT8 data branches.
- Why dead: `Where(INT8, INT8)` is not supported by pinned ORT even though checker/shape inference accept the graph; adding casts would erase the 1-param zero-point saving.
- One remaining hypothesis: a nonnegative UINT8-only QLinearConv decoder might remove the signed weight zero-point, but it would need a free `100 - id^2`-style second feature; the obvious construction adds a counted 3x3 tensor and loses.
