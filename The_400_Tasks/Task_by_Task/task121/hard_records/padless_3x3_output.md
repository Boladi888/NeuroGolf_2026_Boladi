# padless_3x3_output dead route

- Live/team bar: 327, SHA-256 b3a7297c064902e13b5bd86fc46aeb686ab4f1d476015a977193772c19a4a0be, source workbook + manifest.
- Attempted route: remove final `Pad`, rename `Where` output to graph `output`, and declare output shape [1,10,3,3].
- Probe: `prototypes/task121_padless_3x3_named_output.onnx`, static score 229 = memory 169 + params 60.
- Failed gate: `python tools/ng_gate.py visible --task 121 --cand The_400_Tasks/Task_by_Task/task121/prototypes/task121_padless_3x3_named_output.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: NeuroGolf `convert_to_numpy` encodes expected outputs as [1,10,30,30], and `run_network` always requests output name `output`; a 3x3 graph output cannot match visible examples.
- One remaining hypothesis: keep [1,10,30,30] output but reduce params or selector memory before the final Pad.

