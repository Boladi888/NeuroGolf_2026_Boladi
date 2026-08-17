# Single-polynomial weighted-channel route dead

Live/team-best bar: cost 122, SHA-256 `61c58f8fbb942e5a13ed755b3fdc0f7e68c58c405f091240a235d4b0a85359be`, from `python tools/ng_current_bar.py --task 261 --json` / manifest.

Attempted route: replace the current two-state `input_term[10,2]` plus degree-6 target polynomial with one proportional row-difference polynomial shared by color-0 and color-8 channel weights. Target cost would be about 100 params for degree 6 (`coord_basis 60 + feature_core 8 + input_weight 10 + output_sign 10 + 6 coeff vectors 12`), below the 122 bar.

Fast numpy gate: `python The_400_Tasks/Task_by_Task/task261/prototypes/search_single_poly_weighted_20260714_bo_ng_cdx2_worker1.py`

Exact failed result:
- `rows_with_fresh 365 cases 11073`
- `K 1 searched 13 proportional_hits 0 baseline_hits 0`
- `K 2 searched 78 proportional_hits 0 baseline_hits 0`
- `K 3 searched 286 proportional_hits 0 baseline_hits 0`
- `K 4 searched 715 proportional_hits 0 baseline_hits 0`
- `K 5 searched 1287 proportional_hits 0 baseline_hits 0`
- `K 6 searched 1716 proportional_hits 0`
- `current_K6_baseline (-inf, -0.19444444444444445)`

Why dead: no integer-root product over row differences `-6..6` with degree <=6 can separate visible plus 100 fresh ARC-GEN rows when color-0 and color-8 contributions are constrained to be proportional through one shared polynomial. The current family still needs the non-proportional split: constant negative active-column baseline plus a separate degree-6 color-8 polynomial. Degree <=5 in the current baseline family also has zero integer-root hits on the same corpus, matching the prior reduced-factor visible failures.

One remaining hypothesis: only a non-integer/complex-root separator or a different final-op family could beat 122. If revisited, solve a real-coefficient feasibility problem before building ONNX; do not repeat integer-root or factor-removal probes.
