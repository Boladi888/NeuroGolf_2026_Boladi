# Count-Basis Constant 245 Probe

- Live/staged bars used: authoritative live cost 262, sha256 `0394d8786c55b024ef6b7188817b8267b0c4d5c834cd35744a8e4ed83454a978`; task-local staged reviewer bar cost 251, sha256 `a8052987d7b160721429ddd03f552ffff29d848603e1dab6626ff029f493e23b`.
- Attempted route: replace dynamic `count_basis = Concat(one_m, colored_count)` with a fitted constant `[1, 12.478608]`, deleting the 8-byte Concat and adding a 2-param initializer.
- Exact failed gate: `task051_worker1_v14_count_const_fit.onnx` scored `(0, 120, 125) cost 245`, but `ng_gate.py visible` returned `[visible] 134/265 (50.57%)  FAIL`; capped A/B returned `[ab] vs submission.zip: match 378/765 | divergences=387 | byte_equal=False (first_div=1)`.
- Why dead: the terminal output tail uses the 9-vs-16 foreground count dynamically; an average or constant count breaks both object sizes.
- Remaining hypothesis: count-basis savings require an alternate in-kernel representation of `[1,N]`, not a constant replacement. The existing 8-byte Concat is likely the cheapest exact carrier.
