# task342 drop_output_selector dead route

- Live/team bar at probe time: manifest 126 bytes, SHA `9a749314b24fdbf582eee9c05b47dc9361a441c5baf00c18afac8f7c6d499d27`; task-local staged bar 116 bytes, SHA `4F5147C0802213ACF5F076D7A4EC3F56AEF55C9C076FEDAC823E38F8C7B04D55`.
- Attempted route: delete the 20-element `output_selector` initializer from the one-op Einsum, producing static cost 106.
- Failed gate: `python tools/ng_gate.py visible --task 342 --cand The_400_Tasks/Task_by_Task/task342/prototypes/task342_einsum_drop_output_selector.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: `output_selector` carries the output-channel/color gating; removing it makes the one-op renderer visible-zero despite the lower param count.
- One remaining hypothesis: a different equation-level factorization that preserves color gating while sharing selector storage with another tensor, not a raw deletion of `output_selector`.
