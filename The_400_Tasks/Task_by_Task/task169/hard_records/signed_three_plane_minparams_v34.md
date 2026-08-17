# task169 v34 signed-three-plane minparams — dead micro-route

- Live bar: 1,483, SHA-256 `7cc4f8bc47857933c1951c0968eb1b7530042328def7d180ea53af3443753d69` (`python tools/ng_current_bar.py --task 169 --json`, authoritative manifest).
- Attempted route: v33-style three-plane signed final assembly with shared scale 21, bias-free size-4 core/dilation, and shared final zero point; candidate cost/profile cost 1,254, SHA-256 `fa97ca87cd2036c21507b31081f123abc354ee5404b60e2b294c7658c6cc4d67`.
- Exact failed gate: `[truth] vs ARC-GEN ground truth: 199974/200000 (99.987%)`.
- Attribution: `[ab] vs submission.zip: match 200240/200266 | divergences=26 | byte_equal=False (first_div=5070)`; the 26 divergences equal the 26 truth misses, so the min-parameter quantizer introduces a deterministic rare-case regression.
- Why dead: full truth is not 100% and the candidate is not byte-equal; status `REJECT_LOCAL`. Do not tune further variants of the scale-21/bias-deletion formulation.
- One remaining hypothesis: preserve v33's explicit scale/bias/offset encoding (verified full truth and byte-equal at cost 1,258); any lower rewrite must analytically separate the 26 rare carrier states before deleting parameters.
