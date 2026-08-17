# sub400_micro_golf_audit_20260709

- Live/team-best bar used: cost 400, static/profile 400 (`181` memory + `219` params), sha256 `418eee6e3107fe8c2c28ab6ecf0464fc47e7b07f007446ba5d25c1d83fcabc89`, source `tools/ng_current_bar.py --task 336 --json` -> authoritative manifest row, owner Vamshi, updated `2026-07-02T02:35:15Z`.
- Attempted route: audit the current 400-byte `task336_rank2_quadratic_final_einsum` for an exact sub-400 micro-golf after the prompt's pad-tail/canvas route was found stale.
- Exact gates/checks run this pass:
  - `python tools/ng_gate.py visible --task 336 --cand Team_Best_Onnx/task336.onnx` -> `[visible] 31/31 (100.00%)  PASS`
  - `score_network_v2('Team_Best_Onnx/task336.onnx')` -> `(0, 181, 219)`
  - `python tools/agent_helpers/profile_real_cost.py --task 336 --cand Team_Best_Onnx/task336.onnx --arcgen 266` -> `task336 task336.onnx: static=400 profile=400 (memory 181 + params 219; 31 real inputs)`
- Why the audited route is dead: the current graph already renders directly into the free graph output with one terminal `Einsum`; output materialization is a 36,000-byte loss. The remaining charged surface is small and tightly coupled: `B[3,30]` is shape-forced by the full 30-wide output axes, `features[5,3]` and the rank-5 `U/Vfac/Q` coupling survived prior strict-below visible probes, and F16/sparse/dynamic-generation variants do not reduce credited params without adding larger charged memory.
- One remaining hypothesis: an exact algebraic refactor that shares or factors the duplicated `U`/`Vfac` column structure inside the single terminal `Einsum` without a dense permutation/coupling tensor. Naive rank reduction and feature-row truncation are already visible-zero; do not repeat them.
