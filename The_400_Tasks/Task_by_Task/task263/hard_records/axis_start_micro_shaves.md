# task263 axis/start micro-shaves

Live/team bar: 472 cost, SHA-256 `8ddcdb4887dc6ad862198b673ec921675d830a239844bb40edf1b96a08ac7ea7`, source `Team_Submission/Backend/manifest.csv` and workbook row `Tasks!265`.

Attempted route 1: reuse the existing INT32 `[2,3]` Slice axes initializer as the Pad axes input, deleting `pad_axes_hw_i64`.

Failed gate 1: `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2_pad_i32_axes.onnx` -> `ERROR: ... Pad ... ParseData type mismatch ... Expected:int64 Actual:int32`.

Attempted route 2: skip the two 1-byte Reshape nodes that flatten row/column start scalars before Concat, feeding the singleton-shaped U8 tensors directly to Slice starts.

Failed gate 2: `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2_skip_start_reshapes.onnx` -> `ERROR: ... Slice ... Starts must be a 1-D array`.

Why dead: Pad axes are hard-typed INT64 in ORT, and Slice starts must be rank-1 at runtime. The accepted graph's duplicate axes initializer and two start Reshapes are load-bearing, not removable annotation noise.

One remaining hypothesis: any sub-472 candidate needs a semantic rewrite that avoids a larger tensor, not scalar/axis metadata shaving.
