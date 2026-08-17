# Color-Basis Nonzero Absorb 243 Probe

- Live/staged bars used: authoritative live cost 262, sha256 `0394d8786c55b024ef6b7188817b8267b0c4d5c834cd35744a8e4ed83454a978`; task-local staged reviewer bar cost 251, sha256 `a8052987d7b160721429ddd03f552ffff29d848603e1dab6626ff029f493e23b`.
- Attempted route: change `color_basis[0]` from all-ones to `nonzero_f`, then replace every input-side `nonzero_f` operand with `color_basis * first_pick` and delete the 10-param `nonzero_f` initializer.
- Exact failed gate: `task051_worker1_v09_colorbasis_row0_nonzero_probe.onnx` scored 251 but `ng_gate.py visible` returned `[visible] 0/265 (0.00%)  FAIL`; `task051_worker1_v10_colorbasis_absorb_nonzero.onnx` scored 243 but also returned `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: `color_basis` row 0 is not just a non-background mask; it is load-bearing for the final free-output background-channel interpolation.
- Remaining hypothesis: any `nonzero_f` deletion must preserve a true all-channel constant row for the terminal output, likely requiring an additional delta/background feature that cancels the param saving.
