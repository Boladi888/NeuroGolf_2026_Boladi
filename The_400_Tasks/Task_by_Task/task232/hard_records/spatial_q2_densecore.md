# task232 hard record - two-feature spatial dense core

- Live/team-best bar: cost 205, profile_cost 205, sha256 `f54c7425eeb6a1da3977db56be71af2cadeafdc729064e47ce402f3cc5cf80e5`.
- Bar source: `python tools/ng_current_bar.py --task 232 --json` -> authoritative source `manifest`, owner Vamshi, updated `2026-07-02T02:35:15Z`.
- Attempted route: keep the live color basis `R[10,4]`, collapse the spatial basis from `Q[30,3]` to `Q[30,2]`, and replace `M/S` with one dense `H[4,4,2,2]` core. Static target cost is `40 + 60 + 64 = 164`.
- Exact failed gate: prebuild feasibility gate, not staged. LP over decoded generator row states with widths 7..14 found max strict margin `gamma = 0` for fixed two-feature bases `one/x`, `one/parity`, `x/parity`, `one/(x + lambda*parity)`, and `x/(parity + lambda)` for lambda in `{-10,-5,-2,-1,-0.5,0.5,1,2,5,10}`.
- Why dead: the rule simultaneously needs a right-of-seed inequality and parity-of-offset alternation. A two-feature fixed spatial embedding could not linearly separate the generator row family even with a dense 64-element color/spatial core.
- One remaining hypothesis: only revisit with a true arbitrary `Q[30,2]` nonconvex solve or a sign-rank lower-bound proof; do not build hand `Q2` variants around the listed bases.
