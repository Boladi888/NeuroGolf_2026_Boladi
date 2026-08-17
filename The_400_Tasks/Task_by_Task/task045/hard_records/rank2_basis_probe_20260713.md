# task045 hard record: rank-2 free-output basis probe

- Live bar: 90 bytes, sha256 `151314a926f7a39e4ab3bef82481b135f2e5382cef065996d7d59c1a6115db84`, source `python tools/ng_current_bar.py --task 45 --json` / manifest owner Vamshi, updated 2026-07-12T15:15:54Z.
- Attempted route: keep Vamshi's one-node free-output `Einsum` architecture but reduce the shared column basis from rank 3 (`[3,30]`, 90 params) to rank 2 (`[2,30]`, 60 params).
- Local solvability test: simulated the exact `Einsum` algebra over the generator-domain row table: blank `(0,0)` plus all nonzero/nonzero endpoint pairs. Correct decoder condition was target channel `>0` and non-target channels `<=0`.
- Result: tied rank-2 search over left/interior/right vectors failed; best decoder-faithful run was `6814/8200` signs correct with non-target max `4.256`. Relaxed rank-2 search with an independent symmetric `2x2` hidden `H` matrix also failed; best run was `7406/8200` signs correct with target min `-0.1206` and non-target max `2.1272`.
- Exact failed gate: no ONNX was built or visible-gated because the simulated signs were not exact; running a visible gate on a known-bad rank-2 ONNX would only burn the route.
- Why dead: rank 2 did not solve even the relaxed algebra; the task needs three column roles (left/interior/right) in the terminal kernel, matching the current `[3,30]` basis cost of 90.
- Remaining hypothesis: a different one-node equation that carries the output-width axis from a free input operand, rather than from a learned column basis, is the only plausible sub-90 rewrite.
