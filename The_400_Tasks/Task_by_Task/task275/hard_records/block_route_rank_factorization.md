# block_route_rank_factorization dead route

Live bar: cost 584, SHA 296c3ce83287cfab6b5305b44f45de2b6ede867b58f10851278c55af3bd73859 (`python tools/ng_current_bar.py --task 275 --json`, manifest-authoritative).

Attempted route: apply the same one-node Einsum initializer factorization used for `mod_route` to the dense `block_route [2,30,4]` initializer.

Exact failed gate: pre-gate rank/cost pricing on the correctly ordered flattening `(z,p) x i`:
`block_route rank 7`, so an exact `block_pos [30,7] + block_coeff [2,4,7]` factorization costs `30*7 + 2*4*7 = 266` params versus the existing dense 240 params.

Why dead: the factorization adds 26 params and would not beat the 584 live bar or the 572 `mod_route` factorized candidate. No ONNX gate was run because the priced candidate is strictly worse before validation.

One remaining hypothesis: a different formulation must avoid or generate the 30-wide route map rather than exact-factorizing it inside the current one-node Einsum.
