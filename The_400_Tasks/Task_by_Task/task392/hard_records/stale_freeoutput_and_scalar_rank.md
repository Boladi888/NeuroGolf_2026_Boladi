# task392 dead route — stale FreeOutputPadTail / scalar-rank tie

- Live bar: **1001**, SHA-256 `4dfb666721285b099201df853ccf962b8598bf6cacc66a2c1b6c282358d8f283` (`python tools/ng_current_bar.py --task 392 --json`, authoritative manifest source).
- Attempted route: refresh the prompt's Pad/Where/Equal-tail premise, then test rank-0 F16 broadcast constants as a formulation-local shave.
- Exact failed gate: `python -c "from tools._score_v2 import score_network_v2; print(score_network_v2(r'The_400_Tasks/Task_by_Task/task392/prototypes/task392_scalar_constants.onnx'))"` returned `[score_v2] (0, 770, 231)` after visible `[visible] 266/266 (100.00%) PASS`.
- Why dead: the live 1001 graph already deleted the prompt's charged Pad/prepad carrier and ends in a free output `ConvTranspose`; an Equal/Where canvas starts around 900–1000 bytes before runtime decoding. Rank-0 tensors also save nothing because the current scorer bills every scalar as one element for both memory and parameters.
- Remaining hypothesis: jointly optimize the now-proven 11-probe signed hash below table length 67; bounded length-61 searches reached two collisions but did not solve.
