# task030 one-feature QLinear dead route

- Live bar used: 1054 profile cost from `python tools/ng_current_bar.py --task 30 --json`; live SHA-256 `88cbb94de8c671d33eef5aefee30ea2224755632b487d15dc5f68528598835e8`.
- Attempted route: remove `code_sq` and `code_features`, feed scalar `code_grid` directly to a one-feature `QLinearConv` output placer.
- Probe artifact: `The_400_Tasks/Task_by_Task/task030/prototypes/worker2_probe_onefeature_qlinear.onnx`.
- Price: static/profile 744 (`memory 647 + params 97`) versus live profile 1054 (`memory 947 + params 107`).
- Failed gate: `python tools/ng_gate.py visible --task 30 --cand The_400_Tasks/Task_by_Task/task030/prototypes/worker2_probe_onefeature_qlinear.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: one scalar code with a bias-free/constant-free `QLinearConv` can only threshold a half-line per output channel, so it cannot isolate the interior code classes required by the canonical 10-channel one-hot output. The removed 300 B quadratic decoder is load-bearing.
- Remaining hypothesis: a sub-1054 win needs a different final decoder that separates the five code classes without materializing both `code_sq` and `code_features`, or a cheaper way to generate a two-feature convex code plane.
