# task263 no-pad tile3 output route

Live/team bar: 472 cost, SHA-256 `8ddcdb4887dc6ad862198b673ec921675d830a239844bb40edf1b96a08ac7ea7`, source `Team_Submission/Backend/manifest.csv` and workbook row `Tasks!265`.

Attempted route: remove the accepted graph's final Pad and emit the rendered 3x3 U8 tile directly as graph output `output`, expecting the task's logical 3x3 output crop might be accepted without 30x30 top-left padding.

Exact failed gate: `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2_no_pad_tile3.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.

Why dead: NeuroGolf verification compares the raw one-hot tensor against the 30x30 encoded expected output, so the model output must remain `[1,10,30,30]` named `output`; logical 3x3 crop tensors are not accepted as final outputs.

One remaining hypothesis: a real win must keep the final Pad/full-output contract and delete cost before the Pad, most likely by avoiding either the 40-byte `color_vec_f` slice or the 90-byte `tile3_u8` render without materializing a F32 selected tile.
