# einsum_label_saturation dead route

- Live/team bar used: cost 278, profile 278, sha256 `cc1f7c0947f4951eb199dfa07fa2c52d82c7b81da34717aa98a7f736a3a19f06` (`python tools/ng_current_bar.py --task 305 --json`, manifest authority, Vamshi, 2026-07-02T02:35:15Z).
- Attempted route: factor the dense small bases inside the terminal polynomial `Einsum`, especially `hbasis[10,5] -> hfeat[10,2] x hmap[2,5]`, `qvec[1,5] -> qfeat[1,2] x qmap[2,5]`, and `coef_t[5] -> coef3_t[3] x coefmap[3,5]`.
- Exact failed gate: pre-build feasibility. The current `Einsum` equation already uses all 52 legal single-letter labels (`A-Z` and `a-z`). A tiny checker/ORT probe with digit/underscore labels failed shape inference (`Rank of input 0 (2) does not match the equation indices (1)`), confirming non-letter labels are not available.
- Why dead: each factored basis needs one fresh contracted label per polynomial factor to keep factor sums independent. Reusing an existing label couples factor choices and changes the product-of-sums math; splitting the `Einsum` would materialize a full `[1,10,30,30]` intermediate and lose by orders of magnitude.
- One remaining hypothesis: revisit only if a different polynomial formulation reduces the 16-factor product or frees labels first; do not retry direct small-basis factoring on the current saturated one-node equation.
