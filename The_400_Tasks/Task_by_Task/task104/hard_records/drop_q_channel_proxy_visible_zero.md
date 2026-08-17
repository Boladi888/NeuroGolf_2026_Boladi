# drop_q_channel_proxy_visible_zero

- Live bar used: cost 118, SHA-256 `34571a7d5ea37c04e7562fe7ab21d293de3197cf9746aa02dec81bacda25d84c`, source `python tools/ng_current_bar.py --task 104 --json` / team manifest.
- Attempted route: remove `q` entirely and let raw input channel axes carry output channel `c`; prototype `../prototypes/task104_probe_drop_outq_bothc.onnx`, static cost 108, SHA-256 `f96d3f95e68e507f2b2a4f8419fb54690a6caf3d6c4b137cf521fcade1476813`.
- Exact failed gate: `python tools/ng_gate.py visible --task 104 --cand The_400_Tasks/Task_by_Task/task104/prototypes/task104_probe_drop_outq_bothc.onnx` -> `[visible] 0/7 (0.00%)  FAIL`.
- Why dead: channel 2 and the raw color-count channel logits cannot replace the shared `q` sign vector; every visible case misses.
- One remaining hypothesis: only a different channel-elimination formulation that preserves channel-2 suppression without a length-10 vector would be worth revisiting.
