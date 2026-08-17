# task227 hard record — channel-pruned final render is visible-zero

- Live bar used: cost **94**, sha `7c6310929b0992919558cd9670192b312e738d602f5f84586de92acb482b9df0`, owner Sho, manifest authoritative via `python tools/ng_current_bar.py --task 227 --json` on 2026-07-14.
- Attempted route: keep the live signed `[1,1,4,4]` score plane, but shrink the free final `ConvTranspose` render from 10 output channels to 3 because task227 only emits colors 0 and 2.
- Prototype: `The_400_Tasks/Task_by_Task/task227/prototypes/channel_pruned_render3.onnx`, built by `build_channel_pruned_render_probe.py`.
- Local cost if the contract accepted it: `score_v2 (0, 64, 23) cost 87` versus live `score_v2 (0, 64, 30) cost 94`.
- Exact failed gate: `python tools/ng_gate.py visible --task 227 --cand The_400_Tasks/Task_by_Task/task227/prototypes/channel_pruned_render3.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: NeuroGolf visible verification compares the thresholded output tensor directly to the 10-channel encoded target. A `[1,3,30,30]` output is a shape mismatch even when the only semantic colors are 0 and 2.
- Verdict: `REJECT_LOCAL`; do not stage, truth/ab/profile/safety are intentionally skipped after the visible-zero.
- One remaining hypothesis: only a scorer/decoder change that accepts reduced output channel axes would make this 87-byte family viable; under the current hard 10-channel contract, the 94-byte live graph remains the floor.
