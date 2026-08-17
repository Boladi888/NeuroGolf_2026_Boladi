# Raw Orientation 243 Probe

- Live/staged bars used: authoritative live cost 262, sha256 `0394d8786c55b024ef6b7188817b8267b0c4d5c834cd35744a8e4ed83454a978`; task-local staged reviewer bar cost 251, sha256 `a8052987d7b160721429ddd03f552ffff29d848603e1dab6626ff029f493e23b`.
- Attempted route: delete `Sign(row_delta)`/`Sign(col_delta)` and feed raw `row_delta`/`col_delta` directly into `orientation_basis`, keeping the terminal Einsum unchanged.
- Exact failed gate: `task051_worker1_v12_raw_orientation.onnx` scored `(0, 120, 123) cost 243`, but `ng_gate.py visible` returned `[visible] 0/265 (0.00%)  FAIL`. Follow-up `task051_worker1_v13_raw_orientation_fit.onnx` refit the four `component_scale` coefficients against the 251 graph's official-bundle raw output, still scored `(0, 120, 123) cost 243`, and failed `[visible] 0/265 (0.00%)  FAIL` plus `[ab] vs submission.zip: match 0/765 | divergences=765 | byte_equal=False (first_div=0)`.
- Why dead: the terminal free-output polynomial is calibrated to orientation signs `{-1,0,1}`, not just sign-equivalent nonzero deltas with magnitudes `5`/`14`; a four-coefficient count rescale is not expressive enough.
- Remaining hypothesis: a valid no-Sign route would need a deeper terminal coefficient solve that separates first- and second-order orientation terms, likely adding params and erasing the 8-byte memory saving.
