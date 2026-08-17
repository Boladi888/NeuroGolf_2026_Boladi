# task249 compact output shape dead route

Live/team bar: cost 246, sha256 1e0ed5ec19fb8d0caefc1f3c3425cf88d8f8a242a11667806c4caae1782cb611, source `Team_Submission/Backend/manifest.csv` and workbook row.

Attempted route: remove the team-best index Pad and feed `gather_idx10` directly to final Gather, emitting compact `[1,10,30,10]` output. Candidate `prototypes/task249_worker3_compact_output_probe.onnx`, static score `(0, 92, 31)`, cost 123.

Exact failed gate: `python tools/ng_gate.py visible --task 249 --cand The_400_Tasks/Task_by_Task/task249/prototypes/task249_worker3_compact_output_probe.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.

Why dead: the gate target is fixed `[1,10,30,30]` one-hot; compact output shapes do not compare equal even when active columns are correct. One visible miss means hidden 0, so no truth/profile/ab gates were run.

One remaining hypothesis: only a fundamentally different full-canvas graph that still emits `[1,10,30,30]` can beat 246; do not retry compact-width output for task249 under the current harness.
