# reducemaxcarrier_absent_in_live

- Live/team-best bar: cost 400, static/profile 400 (`181` memory + `219` params), sha256 `418eee6e3107fe8c2c28ab6ecf0464fc47e7b07f007446ba5d25c1d83fcabc89`, source `Team_Submission/Backend/manifest.csv` and workbook row updated `2026-07-02T02:35:15Z`.
- Attempted route: follow the prompt's `ReduceMax/Mul carrier region-fill fold` against the current team-best graph.
- Exact failed gate: pre-build graph inspection, not a correctness gate. `inspect_graph.py` on the authoritative team-best shows opset 16 graph `task336_rank2_quadratic_final_einsum` with `Einsum:4`, `Mul:6`, and no `ReduceMax`; the old `Cast:11|Mul:10|ReduceMax:10` surface belongs to stale local/library context, not the live 400-byte graph.
- Why dead: there is no current `ReduceMax` carrier tensor/op to delete, and the live graph has already folded the region-fill into scalar descriptors plus one final free-output `Einsum`.
- One remaining hypothesis: only micro-golfing the 400-byte rank-2 quadratic final-Einsum graph can beat the bar; the likely target is exact selector/constant fusion, not revisiting the stale carrier representation.
