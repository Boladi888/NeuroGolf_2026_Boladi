# task076 label-output decoder route

- Live bar used: 19,811, SHA `A23CDED50F406EC61DD27B094876FF28199C0EA578424FB11B48B366C0A71E02` from `python tools/ng_current_bar.py --task 76 --json`.
- Attempted route: replace final `out5_bool -> Pad(output)` with a U8 label plane, U8 spatial Pad, and final output-exempt `Equal`.
- Failed gate: `python tools/ng_gate.py ab --task 76 --ref Team_Best_Onnx/task076.onnx --cand The_400_Tasks/Task_by_Task/task076/prototypes/worker076_w3_u8_board_label_output.onnx --n 2000` -> `[ab] vs explicit: match 2254/2266 | divergences=12 | byte_equal=False (first_div=796)`.
- Why dead: visible and profile passed (`266/266`, cost/profile `17369`), but this task's current bar is ambiguous and byte-equality to `Team_Best_Onnx` is the gold gate; the label decoder is not byte-equal on ARC-GEN.
- One remaining hypothesis: inspect first divergence 796 and emulate the current graph's exact channel/default behavior in a label decoder; otherwise keep the byte-equal U8 board/patch candidate.
