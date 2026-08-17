# task076 source-red ArgMax U8 cast route

- Live bar used: 19,811, SHA `A23CDED50F406EC61DD27B094876FF28199C0EA578424FB11B48B366C0A71E02` from `python tools/ng_current_bar.py --task 76 --json`.
- Attempted route: narrow the 7-element `source_red_mask` cast before `ArgMax` from INT32 to UINT8.
- Failed gate: capped A/B against `Team_Best_Onnx/task076.onnx` after the cast produced divergences despite `visible 266/266`; the exact failing build was reverted from `build_task076_w3_u8_label_bool_reducemax.py`.
- Why dead: ORT accepted `ArgMax(UINT8)`, but tie/ordering behavior was not byte-equivalent on task-shaped ARC-GEN inputs.
- One remaining hypothesis: if revisited, isolate this cast on the byte-equal `worker076_w3_u8_board_and_patch.onnx` base and compare first-divergence source-red ties before using it.
