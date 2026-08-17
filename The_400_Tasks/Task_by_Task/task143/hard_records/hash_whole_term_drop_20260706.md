# hash_whole_term_drop_20260706

- Live bar: manifest cost 578, SHA `2212b6d4cf203924615d5ba522a1e53b8d9d19b333e0283adcf2d8a4632ba1a5`, owner Aditya, refreshed with `python tools/ng_current_bar.py --task 143 --json` on 2026-07-06.
- Effective local bar during this worker pass: staged `task143_combined_hash_order1_cost554`, profile cost 554, SHA `b906347e6787c2199af41018a881e109f1e208f7e01d46c47100678c7f355994`.
- Attempted route: shorten the combined cubic shape hash by dropping whole term-axis slices from `coefA3/coefB3/coefC3` in `task143_worker2_combined_hash_order1.onnx`.
- Exact failed gate: visible gate. `drop_hash_t0_n2q`, `drop_hash_t1_termB`, `drop_hash_t2_combo`, `hash_t0_only`, `hash_t1_only`, and `hash_t2_only` all returned `[visible] 0/266 (0.00%) FAIL`.
- Why dead: every whole hash term is load-bearing under the current equality-to-template hash. The fixed L-anchor does not make any complete cubic term redundant.
- One remaining hypothesis: preserve all three terms but factor duplicated coefficient slices; `worker3_build_hash_factor.py` does this exactly via shared `one/q1` basis and profiles at 502 pending full gates.
