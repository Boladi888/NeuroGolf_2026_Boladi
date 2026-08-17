# qvec_elision dead route

- Live/team bar used: cost 278, profile 278, sha256 `cc1f7c0947f4951eb199dfa07fa2c52d82c7b81da34717aa98a7f736a3a19f06` (`python tools/ng_current_bar.py --task 305 --json`, manifest authority, Vamshi, 2026-07-02T02:35:15Z).
- Attempted route: remove `q_sum -> Concat -> qvec` and make the terminal free `Einsum` derive `q` by summing the already-computed `present` vector inside each polynomial factor.
- Exact failed gate: pre-build algebra/pricing gate. The terminal op is a product of 16 independent summed factors, not a sum of 16 terms. Splitting `A + qB` into separate additive groups changes the product-of-sums math. The faithful single-`Einsum` replacement needs a `present x qsrc[10,5]` basis selector to produce `[1,1,1,1,q]`, which adds 50 params while saving only qvec memory 24 + `qvec_ones` params 4, pricing at least 300 before verification.
- Why dead: a correct no-`qvec` version either exceeds the 278 bar or requires an extra contracted label per factor; the current equation already consumes all 52 legal `Einsum` labels, as recorded in `einsum_label_saturation.md`.
- One remaining hypothesis: only revisit qvec if a different polynomial with fewer independent factors frees labels or avoids the dynamic `[1,1,1,1,q]` basis entirely.
