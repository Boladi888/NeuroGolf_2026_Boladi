# task272 hard record: U8 QLinearConv red-slice scatter overlay

Live/team-best bar: cost 378, sha256 `f0302ce51976334b74f18b95e7cb9e455aa91d56d97a77bc47b1ee604000d526`, source `python tools/ng_current_bar.py --task 272 --json` / manifest, owner Vamshi, updated 2026-07-02T02:35:15Z.

Attempted route: replace the team-best `Slice(ch2 5x5) -> Conv(two F32 update planes) -> ScatterElements(output)` detector with a legal U8 `QLinearConv`: `Slice(red5 F32) -> Cast(U8) -> QLinearConv(two U8 singleton/non-singleton update planes) -> Cast(F32) -> ScatterElements`. Probe path: `The_400_Tasks/Task_by_Task/task272/prototypes/task272_bo_ng_worker1_u8_qlinearconv.onnx`, sha256 `e9ff17d2f54d88ac4405e4624c19238201ece790dbd2c6688e946f57b8b2dfae`.

Exact gate results: visible passed with `[visible] 267/267 (100.00%)  PASS`; static score was `(0, 375, 81) 456`; real-input profile was `static=456 profile=456 (memory 375 + params 81; 267 real inputs)`. Truth/AB were not run because the profile cost is above the 378 live bar.

Why dead: `QLinearConv` itself can express the singleton detector correctly, but it cannot consume the F32 NeuroGolf input. The required `red5 -> U8` cast adds 25B, the U8 update tensor adds 50B, and `ScatterElements` still writes into the F32 input so the updates must be cast back to a 200B F32 tensor. Net official profile delta is +78 vs live.

One remaining hypothesis: a final-output formulation that avoids F32 scatter cast-back and the 50-element channel-index table could still undercut 378, but the obvious `Where`, full-canvas condition, and small one-hot Pad tails price above the current scatter overlay.
