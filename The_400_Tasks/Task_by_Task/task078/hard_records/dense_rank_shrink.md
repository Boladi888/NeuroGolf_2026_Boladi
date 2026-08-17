# task078 dense rank-shrink probes

- Live bar used: 82, sha256 `54db7a9ffcc7777a96dc05b7d3a9d2f0dc81c7eb53dd57eeba37688fd03b287f` from `python tools/ng_current_bar.py --task 78 --json` (manifest, Vamshi, 2026-07-12T15:15:54Z).
- Attempted route: keep the one-node final `Einsum` class but reduce dense params by dropping `H[2,1]` (80 params) or collapsing `Q[2,30]` to broadcast `Q[1,30]` (52 params).
- Exact failed gates:
  - `task078_probe_noH_sameQ_cost80.onnx`: `[visible] 0/266 (0.00%) FAIL`.
  - `task078_probe_noH_lstsqQ_cost80.onnx`: `[visible] 0/266 (0.00%) FAIL`.
  - `task078_probe_noH_optimized_cost80.onnx`: optimizer best loss still had 63 bad unique visible-feature cases; gate `[visible] 0/266 (0.00%) FAIL`.
  - `task078_probe_rank1Q_optimized_cost52.onnx`: optimizer best loss still had 63 bad unique visible-feature cases; gate `[visible] 0/266 (0.00%) FAIL`.
- Why dead: the row/count decision is not separable enough for these rank-reduced dense bases; the existing `H[2,1]` plus two-row `Q[2,30]` interaction is load-bearing.
- One remaining hypothesis: a different dense terminal-Einsum equation that reuses the graph input channel axis as part of the output channel selector could reduce the dense `W[2,10]` zero tail, but it must still preserve a full `[1,10,30,30]` output and avoid any charged pre-output node.
