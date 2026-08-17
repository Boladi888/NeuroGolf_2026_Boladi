# task076 opset20 bool-ReduceMax route

- Live bar used: 19,811, SHA `A23CDED50F406EC61DD27B094876FF28199C0EA578424FB11B48B366C0A71E02` from `python tools/ng_current_bar.py --task 76 --json`.
- Attempted route: bump the label-output graph to opset 20 and reduce `candidate_bad` directly as BOOL, deleting the bool->U8->bool roundtrip.
- Failed gate: `python tools/ng_gate.py ab --task 76 --ref Team_Best_Onnx/task076.onnx --cand The_400_Tasks/Task_by_Task/task076/prototypes/worker076_w3_u8_label_bool_reducemax.onnx --n 2000` -> `[ab] vs explicit: match 2254/2266 | divergences=12 | byte_equal=False (first_div=796)`.
- Why dead: visible and profile passed (`266/266`, cost/profile `16976`), but it inherited the non-byte-equal label-output behavior and is not a safe compression of the current ambiguous bar.
- One remaining hypothesis: retry bool ReduceMax only on a byte-equal decoder base; do not pair it with the current label-output route.
