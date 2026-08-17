live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route: delete the scalar payload fallback
`Less(payload_fullmax_u8, u8_4) -> Where(..., u8_1_zp, payload_fullmax_u8)`
and feed `payload_fullmax_u8` directly into `Equal(desc_u8, payload)`.

Build: `prototypes/bo_ng_worker1_20260713/task089_no_payload_lt4_probe.onnx`
(sha256 not staged; local profile cost 4069, a nominal -5 vs live). Named static
delta from `ng_graph_cost_diff`: removes `payload_lt4_b` 2 B,
`payload_color_u8` 2 B, and `u8_4` 1 param.

Failed gate: `ng_gate visible --task 89 --cand ...task089_no_payload_lt4_probe.onnx --json`
-> 207/267 (77.5281%) FAIL. Truth/ab/full gates correctly skipped.

Why dead: the `<4 -> 1` remap is semantic, not cleanup. Visible cases need marker-color
or empty center maxima (0/2/3) to resolve to payload color 1 before descriptor matching.
Feeding the raw center max leaves marker colors in the payload selector and drops copied
cells.

Remaining hypothesis: a cheaper equivalent for the piecewise map `0..3 -> 1, 4..9 -> x`
would need to preserve the same per-family scalar behavior. The current Less+Where costs
only four scalar bytes plus one threshold param, so arithmetic or lookup alternatives are
unlikely to beat it without reusing an already-materialized predicate.
