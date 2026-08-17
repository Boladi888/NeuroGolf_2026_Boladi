# task171 hard record: sparse initializers on final Einsum

- Live/team-best bar: cost 78, sha256 `f81e8726f351bb2df475f70b40f788e2941b5fef7d9f8e00a0bb88d02e7db3de`, source `python tools/ng_current_bar.py --task 171 --json` / manifest owner Vamshi updated 2026-07-05T04:54:56Z.
- Attempted route: convert dense final-Einsum initializers (`V`, `R`, `B`) to ONNX `sparse_initializer` to reduce param count below 78.
- Probe artifacts:
  - `prototypes/task171_sparse_v.onnx`, sha256 `841630d772903a832b8b285257514eccf1ceb9565ee82e5009cbed35e45e6300`
  - `prototypes/task171_sparse_rv.onnx`, sha256 `112e74f3b81541c389db17a7702119836b77683288603e10f657d036c07c6eaa`
  - `prototypes/task171_sparse_brv.onnx`, sha256 `5640cee0dea913f11fa290f602a26494e364c9693f9f4e16d24c3dd52a68e239`
- Exact failed gate:
  - `score_network_v2(...)`: `None` for all sparse variants.
  - `ng_gate visible --task 171 --cand prototypes/task171_sparse_v.onnx --json`: `Invalid model. Node input 'safe_name_2' is not a graph input, initializer, or output of a previous node.`
  - `onnx.checker.check_model(full_check=True)` on sparse `V`: `Rank of input 12 (0) does not match the equation indices (1).`
- Why dead: scorer sanitization and strict shape inference do not preserve sparse initializer tensor shape/name for `Einsum` inputs; raw ORT loading before sanitization is not enough for NeuroGolf scoring.
- One remaining hypothesis: a dense-only sub-78 win would need a new algebra that removes at least one dense initializer element from `B[2,30]`, `R[2,2,2]`, or `V[10]` without replacing it by a charged vector/scalar pipeline; the obvious predecessor shave (`T[2]` folded into `R`) is already present in the live 78-byte graph.
