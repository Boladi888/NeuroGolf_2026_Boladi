# task102 FreeOutputPadTail stale-premise record (2026-07-10)

- Live bar: **1,751** (`memory=1,552`, `params=199`), SHA-256 `5bf47a937a2707dfd0e6d1a6ad772a36be4e8d997e1a7447698b5c2f6e84998a`, from the authoritative backend manifest via `python tools/ng_current_bar.py --task 102 --json`.
- Attempted route: prompt-directed deletion of the charged FLOAT16 `[1,10,12,12]` 2,880-byte prepad/carrier, replacing it with a one-channel predicate/index and free Equal/Where output tail.
- Exact failed gate: **pre-build charged-component presence check** on `Team_Best_Onnx/task102.onnx`. The refreshed live graph is `Slice -> QuantizeLinear -> QLinearConv -> QLinearConv -> Concat -> ConvInteger`; it has no FLOAT16 tensor, no 2,880-byte carrier, and no `Pad`, `Where`, or `Equal` node. The prompt's own workflow says to DROP when the charged component is already gone.
- Why dead: the named saving is already incorporated. Current exact score/profile is `static=1751 profile=1751 (memory 1552 + params 199; 267 real inputs)`, so deleting the absent carrier saves **0 bytes**. The task-local 1,810 rung is also stale/worse by 59 bytes.
- Gate discipline: no new ONNX was built, so no candidate truth/AB/full gate was run. The refreshed live baseline passed visible `267/267`; its safety scan is `WARN` only for the x86 quantized-op family (`QLinearConv`, `ConvInteger`).
- One remaining hypothesis: only a genuinely new single-output rewrite that fuses the four square-size predicates and interior painting into the terminal output-writing op could materially move the floor; it must eliminate the charged F32 crop (576 B), U8 hit stack (400 B), feature stack (288 B), and two U8 planes (144 B each). Prior direct centered-stencil and one-channel coded-hit/LP probes already rule out the simple linear versions.

Worker packet: `The_400_Tasks/Task_by_Task/task102/worker_packet_task102_freeoutput_drop_20260710.json`.
