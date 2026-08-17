# task244 hard record - continuous log-affine ROI

Live/team bar: cost 85, sha256 db55e34c9cd94f6da4514281bdd2a377d1b4783c59a83e8d62f09353a60f175e (workbook + team manifest, updated 2026-07-02T02:35:15Z).

Attempted route: delete the threshold/ArgMin/Gather selector and generate the RoiAlign `rois` vector as an affine function of `Log(r11)`, where `r11` is the existing high-degree row/color fingerprint.

Prototype: `The_400_Tasks/Task_by_Task/task244/prototypes/task244_cdx2_log_affine_roi.onnx`, sha256 4ca6df3b09377610140eeb951f7d98071788925638401424b3e8c23872ec273d, static score `(0, 40, 9)` cost 49.

Exact failed gate: `python tools/ng_gate.py visible --task 244 --cand The_400_Tasks/Task_by_Task/task244/prototypes/task244_cdx2_log_affine_roi.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.

Why dead: the continuous ROI approximation visibly mis-samples every official/arc-gen pair; per task protocol, one visible miss kills this representation family and no truth/ab/battery gate is valid.

One remaining hypothesis: a discrete selector could still win only if it produces the exact 8 ROI buckets with less than the current 24-byte threshold-vector overhead; observed scalar quadratic and two-feature affine bucketizers were infeasible.
