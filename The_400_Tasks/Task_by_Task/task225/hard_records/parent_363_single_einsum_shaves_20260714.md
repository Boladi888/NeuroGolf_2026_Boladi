# task225 hard record — parent 363 single-Einsum shaves

- Live/team-best bar used: cost 363, sha256 `682e08ff319c04ab4d5b8b7e89294895559525a24f8a29f4167afc3e4754e610`, source `python tools/ng_current_bar.py --task 225 --json` / team manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.
- Parent graph: one terminal `Einsum`, profile/static `363 = memory 0 + params 363`; initializers `G[6,30]`, `N[3,6,6]`, `V[3,10]`, `U[10,3]`, `R[3,3]`, `S[3,2]`.
- Attempted route: sub-bar parent-equivalent shaves by deleting/reusing operands or factorizing dense params while preserving the single free-output Einsum.
- Failed gates:
  - `task225_probe_direct_source_channel.onnx`: static 333; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_reuse_r_for_s_gate.onnx`: static 357; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_reuse_v_for_u_decoder.onnx`: static 333; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_reuse_u_for_v_codebook.onnx`: static 333; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_reuse_v_for_s_gate.onnx`: static 357; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_reuse_u_for_s_gate.onnx`: static 357; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_factor_r_through_s.onnx`: static 360; `ng_gate visible` = `0/265 (0.00%) FAIL`.
  - `task225_probe_sparse_g.onnx`, `task225_probe_sparse_n.onnx`, `task225_probe_sparse_gn.onnx`: `score_network_v2` returned `None`; sparse initializer route invalid before visible.
  - `task225_probe_cyclic_n_dense.onnx`: static 363; `ng_gate visible` = `22/265 (8.30%) FAIL`, so boundary-clipped `N` is load-bearing.
  - `task225_probe_n_cp7.onnx`: static 360; rank-7 CP fit for clipped `N` had L2 residual `0.0728049374631`; `ng_gate visible` = `0/265 (0.00%) FAIL`. Bounded ALS found rank-8 exact-ish (`8.83e-09`) but rank 8 costs 120 params for `N` vs dense 108, so it cannot beat 363.
- Why dead: the 363 parent uses sign-rank/channel-code and boundary-clipped shift factors that are semantically load-bearing; every sub-bar deletion/reuse/factor approximation caused visible-zero. Sparse initializers are rejected by the scorer path.
- One remaining hypothesis: a genuinely new formulation below 363 would need to reduce the active-grid basis `G[6,30]` or replace the whole sign-rank/shift construction, not shave operands from the current single-Einsum parent.
