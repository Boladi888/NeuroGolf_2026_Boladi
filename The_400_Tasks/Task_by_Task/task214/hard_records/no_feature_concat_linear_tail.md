# task214 no-feature-concat linear tail

- Live bar: manifest/workbook cost 584, sha256 `ca0416ec520278f1daa49a47cd96a75bfc1205d9be8128dad99ae0df9a86de5c`, owner Vamshi, updated 2026-07-02T02:35:15Z.
- Effective local bar for this pass: task-local staged candidate `task214_xz18_wz64_nobias_cost578`, profile_cost 578, sha256 `6c1297c3b5031e436c33494f57b515161389a54b316f9f8d363674cd5fe85f77`.
- Attempted route: delete `Mul(label,label)` and `Concat([label,label_square])`, feeding single-channel `label_canvas` directly to a one-feature affine `QLinearConv` tail with bias.
- Prototype: `The_400_Tasks/Task_by_Task/task214/prototypes/task214_no_feature_concat_linear_probe.onnx`.
- Price: score_v2 `(0, 429, 46)`, cost 475; saves the 33-byte square surface, 66-byte feature concat, and 10 feature weights versus the 578 staged rung.
- Failed gate: `python tools/ng_gate.py visible --task 214 --cand The_400_Tasks/Task_by_Task/task214/prototypes/task214_no_feature_concat_linear_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: a one-feature affine tail can only make each channel positive on a half-line of scalar labels; task214 needs singleton interior label tests for colors 1..9. The quadratic feature is the compact separator that makes those singleton tests possible.
- Remaining hypothesis: any sub-578 route must avoid or shrink the 324-byte F32 source crop, or create the quadratic/equality decision inside one final op without materializing `sq`/`feat`; do not build more single-feature affine tails.
