# final_remap_rank_shrink

- Live bar: 578 profile cost, SHA `2212b6d4cf203924615d5ba522a1e53b8d9d19b333e0283adcf2d8a4632ba1a5` from workbook + team manifest (Aditya, updated 2026-06-15T19:08:38Z).
- Attempted route: shrink the final free-output color remap below the current 4-lane CP factorization (`remap_lane10/remap_in8/remap_out8`, 120 params).
- Exact failed gate: no ONNX gate was run because the algebraic pre-gate failed. Dropping any existing lane broke 53-71 of the 80 required `(match color, input color)` sign cases. Bounded searches for CP rank 3 and Tucker ranks with params `<120` found no zero-bad sign table; best observed misses were still 6-8 cases in the promising low-rank families.
- Why dead: the current 4-lane remap appears to be at or near the sign-rank floor for the required threshold-decoded color routing; sub-120-param low-rank variants cannot even satisfy the exact finite sign table.
- One remaining hypothesis: a non-low-rank formula using color-index polynomials plus an output-channel bias could be different enough to test, but it needs an algebraic sign-table proof before any visible gate.
