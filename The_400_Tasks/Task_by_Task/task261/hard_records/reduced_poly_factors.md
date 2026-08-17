# Reduced polynomial factors dead route

Live/team-best bar: cost 122, SHA-256 `61c58f8fbb942e5a13ed755b3fdc0f7e68c58c405f091240a235d4b0a85359be`.

Attempted route: remove row-relation factors from the single-output `Einsum` polynomial while keeping the same `coord_basis`, `feature_core`, `input_term`, and `output_sign` structure. Probes:
- `prototypes/task261_poly2_gain8.onnx`: cost 106.
- `prototypes/task261_poly5_gain1.onnx`: cost 118.

Exact failed gate:
- `task261_poly2_gain8.onnx`: `visible 152/265 (57.36%) FAIL`.
- `task261_poly5_gain1.onnx`: `visible 64/265 (24.15%) FAIL`.

Why dead: the row selector needs a degree-6 response over same-column source rows. The current graph's six roots suppress nearby vertical 8s and provide enough positive margin over the active-column background count; dropping factors creates visible misses, so no truth/profile/full gates are valid.

One remaining hypothesis: a different non-polynomial row-local baseline that avoids summing the whole active column could lower the factor count, but a quick arbitrary-polynomial LP found no degree <=5 separator for the bundled pairs in the tested column-sum and row-local formulations.
