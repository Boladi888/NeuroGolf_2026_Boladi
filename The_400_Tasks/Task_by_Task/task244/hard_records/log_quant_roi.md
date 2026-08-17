# task244 hard record - log-quantized ROI selector

Live/team bar: cost 85, profile_cost 85, sha256 db55e34c9cd94f6da4514281bdd2a377d1b4783c59a83e8d62f09353a60f175e (`ng_current_bar.py --task 244 --json`, authoritative source `manifest`, updated 2026-07-02T02:35:15Z).

Attempted route: replace the live `Greater(thresholds[8]) -> Cast(U8) -> ArgMin -> Gather(roi_table[8,1,4])` selector with `Max/Min -> Log -> QuantizeLinear -> U8 Sub(17) -> Gather(roi_table9[9,1,4])`, keeping `RoiAlign` as the free output node.

Prototype: `The_400_Tasks/Task_by_Task/task244/prototypes/task244_cdx2_log_quant_roi.onnx`, sha256 faa5cbecf5192f349afe8897a9a52bd5a692be197b4bcadfcd06c6ba2dd1b605, static score `(0, 42, 42)` cost 84, real profile cost 84.

Exact failed gate: `python tools/ng_gate.py truth --task 244 --cand The_400_Tasks/Task_by_Task/task244/prototypes/task244_cdx2_log_quant_roi.onnx --full` -> `[truth] vs ARC-GEN ground truth: 199716/200000 (99.858%)`.

Why dead: visible passed, but full-corpus bucket equivalence against the live threshold selector found hidden boundary collisions. First mismatch: corpus index 3506, `r11=853052882944.0`, live bucket 1, quantized code 3, quant bucket 2. Full 200k bucket ranges force either a bigger quantized table (`sqrt(log)` needs 12 code rows; `log(log)` needs 13) or two explicit boundary repairs; both price above the 85-byte live bar.

One remaining hypothesis: an exact selector can still win only if it deletes at least one current charged scalar/vector without adding threshold repair logic; direct quantization is not that route.
