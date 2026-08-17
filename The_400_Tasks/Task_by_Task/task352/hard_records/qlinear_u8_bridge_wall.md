# task352 - QLinearTinyCarrier / U8 bridge wall

- **Live bar used:** 460, authoritative manifest via `python tools/ng_current_bar.py --task 352 --json`, owner Sho, updated `2026-07-09T01:09:17Z`, sha256 `19a43537e8137b3c8b1a9f18246e81a9b448b5f7446b3f79f510b7bce5dec9e2`.
- **Attempted route:** replace the one-node F32 grouped `Conv` with `Cast(input -> UINT8) -> QLinearConv -> output`, encoding the existing thresholds with integer weights/bias. Prototype: `The_400_Tasks/Task_by_Task/task352/prototypes/cand_qlinear_u8.onnx`.
- **Candidate sha256:** `8af705671e9926747a6231d800392e2bbc2aeca10f74a49683c350d7aa9bbd40`.
- **Exact failed gate:** price/profile regression after visible pass:
  - `python tools/ng_gate.py visible --task 352 --cand The_400_Tasks/Task_by_Task/task352/prototypes/cand_qlinear_u8.onnx` -> `[visible] 266/266 (100.00%)  PASS`
  - `score_network_v2(...)` -> `(0, 9000, 466)` = cost `9466`
  - `python tools/agent_helpers/profile_real_cost.py --task 352 --cand The_400_Tasks/Task_by_Task/task352/prototypes/cand_qlinear_u8.onnx --arcgen 266` -> `static=9466 profile=9466 (memory 9000 + params 466; 266 real inputs)`
- **Why dead:** `QLinearConv` cannot consume the official F32 NeuroGolf input. The required U8 bridge materializes `input_u8 [1,10,30,30]` for 9000 charged bytes. That single tensor is already `+8540` above the entire live 460-byte one-node Conv floor, before counting QLinear scalar initializers. The probe does remove the F32 carrier, but only by replacing a zero-memory output Conv with a charged full-frame U8 cast.
- **Do not re-run:** any QLinear/ConvInteger route that starts by casting the full F32 input frame to U8/INT8 for this same grouped-conv rule.
- **One remaining hypothesis:** only a single output-producing op that accepts the raw F32 input and needs fewer than 460 initializer elements, or a future zero-copy quantized input contract, could beat the current graph. No pinned ORT 1.24.4 QLinear path provides that.
