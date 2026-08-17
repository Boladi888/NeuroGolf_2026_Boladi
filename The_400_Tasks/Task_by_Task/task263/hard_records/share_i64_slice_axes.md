# task263 share INT64 Slice axes route

Live/team bar: 472 cost/profile cost, SHA-256 `8ddcdb4887dc6ad862198b673ec921675d830a239844bb40edf1b96a08ac7ea7`, source `Team_Submission/Backend/manifest.csv` via `python tools/ng_current_bar.py --task 263 --json`.

Attempted route: reuse existing `pad_axes_hw_i64=[2,3]` as the `Slice` axes input and delete duplicate `hw_axes_i32=[2,3]`, targeting a 2-param no-memory shave.

Exact failed gate: `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2b_share_i64_hw_axes.onnx` -> `[visible] ERROR: Fail: [ONNXRuntimeError] : 1 : FAIL : Type Error: Type parameter (Tind) of Optype (Slice) bound to different types (tensor(int32) and tensor(int64) in node (safe_name_57).`

Why dead: ORT binds all `Slice` index inputs (`starts`, `ends`, `axes`) to one `Tind`; the current dynamic starts/ends are INT32, so the INT64 Pad axes cannot be shared. Widening starts/ends to INT64 would add 16 runtime bytes to save 2 params, a net loss.

One remaining hypothesis: a valid axes shave would need to make Pad accept INT32 axes, but that opposite direction is already dead in `axis_start_micro_shaves.md`.
