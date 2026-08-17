live bar: manifest cost 324, sha256 6e22e8e4ff9933bb0e94b59a68ad84be640c2ed3e6419bc14ccfa8a59b803467; task-local staged bar before this probe: task343_scalar_splitless_cost319, profile_cost 319.
attempted route: `prototypes/task343_narrow15_probe.onnx` removed the zero-source Pad and emitted only the 15 active output columns as `output` shape [1,10,30,15].
failed gate: `python tools/ng_gate.py visible --task 343 --cand The_400_Tasks/Task_by_Task/task343/prototypes/task343_narrow15_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
why dead: NeuroGolf visible comparison requires the full [1,10,30,30] tensor; shortening the active-width output is not accepted even though the logical ARC grid is 5x15.
remaining hypothesis: a legal sub-319 route must keep full-width output semantics and save bytes inside period detection or the INT32 gather-index bank, not by changing output rank/width.
