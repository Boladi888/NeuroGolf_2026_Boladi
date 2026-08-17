# task078 sparse-initializer tail probe

- Live bar used: 82, sha256 `54db7a9ffcc7777a96dc05b7d3a9d2f0dc81c7eb53dd57eeba37688fd03b287f` from `python tools/ng_current_bar.py --task 78 --json` (manifest, Vamshi, 2026-07-12T15:15:54Z).
- Attempted route: convert the 82-byte one-Einsum parent `Q[2,30]` dense initializer to `SparseTensorProto`; theoretical params drop 82 -> 42 if the scorer accepted sparse.
- Exact failed gates:
  - `task078_probe_sparseQ_cost42_if_accepted.onnx`: `ng_gate visible` load error, `Node input 'safe_name_2' is not a graph input, initializer, or output`.
  - `task078_probe_sparseQ_prenamed_cost42_if_accepted.onnx`: `[visible] 266/266 (100.00%) PASS`, but `profile_real_cost.py` fails in `onnx.checker.check_model(... full_check=True)` with `Rank of input 8 (0) does not match the equation indices (2)`, and `score_network_v2` returns `None`.
- Why dead: sparse initializers are not scoreable for this Einsum path; pre-naming bypasses sanitizer enough for visible, but official scorer shape inference still sees the sparse-fed input as rank-0/unknown.
- One remaining hypothesis: none in sparse form; only a dense final-Einsum factorization with fewer initializer elements can beat 82.
