# combined_hash_original_bad_order

- Live bar at time of failed probe: 578 profile cost, SHA `2212b6d4cf203924615d5ba522a1e53b8d9d19b333e0283adcf2d8a4632ba1a5` from workbook + team manifest.
- Attempted route: algebraically replace `n2Q`, `termB`, `combo`, and `Sum -> H` with one term-axis `Einsum`.
- Exact failed gate: original prototype `task143_worker3_combined_hash.onnx` failed `visible` with ORT allocation error: `Failed to allocate memory for requested buffer of size 29160000000`.
- Why this specific route is dead: the original operand order `input,input,input,hpows,...` makes ORT's left-deep `Einsum` path materialize a huge cross-product before small factors can reduce each input.
- Superseded note: this is NOT a dead record for the algebraic fold. Reviewer-requested reordered variant `task143_worker2_combined_hash_order1.onnx` passed visible/full truth/full A/B/battery and profiles at 554.
