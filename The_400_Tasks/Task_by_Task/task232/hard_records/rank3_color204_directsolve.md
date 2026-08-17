# task232 hard record - direct-solved rank3 color embeddings

- Live/team-best bar: cost 205, profile_cost 205, sha256 `f54c7425eeb6a1da3977db56be71af2cadeafdc729064e47ce402f3cc5cf80e5`.
- Bar source: `python tools/ng_current_bar.py --task 232 --json` -> authoritative source `manifest`, owner Vamshi, updated `2026-07-02T02:35:15Z`.
- Attempted route: replace the live shared `R[10,4] + M[3,4,4]` color side with separate rank-3 output/input embeddings `Ro[10,3]`, `Rc[10,3]`, `M[3,3,3]`, keeping the proven `Q[30,3] + S[3,3,3]` spatial side. Static target cost is 204.
- Exact failed gate: prebuild feasibility gate, not staged. LP over decoded generator row states with widths 7..14 and seed colors `{1,2,3,4,6,7,8,9}` found max strict margin `gamma = 0` for fixed current/wide/full-circle 3D embeddings. A direct NumPy/Adam nonconvex sign solve over arbitrary `Ro/Rc/M` reached only `5823/7476` sign-complete row-position constraints, best argmax accuracy `0.9202782236490101`, min sign margin `-2.5782452072131985`.
- Concrete sibling already gated: `The_400_Tasks/Task_by_Task/task232/prototypes/probe_svd_color_rank3_cost204.onnx` scored `(0, 0, 204) 204` and failed `ng_gate visible` with `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: one byte is available only by removing one color degree. The decoded rule needs separate background, gray/nonzero bias, and two seed-hue degrees under the `>0` channel decoder; the rank-3 direct solves could not satisfy even the generator row family.
- One remaining hypothesis: only revisit if an exact nonconvex certificate finds a strict-margin arbitrary `Ro/Rc/M`, or a sign-rank proof shows the 4D color basis is forced.
