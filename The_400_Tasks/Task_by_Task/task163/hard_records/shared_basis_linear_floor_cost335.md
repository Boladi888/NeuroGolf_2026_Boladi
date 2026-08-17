# task163 — shared-basis linear route: LOCAL_GOLD at 335, residual wall below

## Bar and winning rung

- Authoritative live bar: **392** (`memory=0`, `params=392`), refreshed before and after the long gates with `python tools/ng_current_bar.py --task 163 --json`.
- Live source/SHA: team backend manifest, owner Aditya; `10fc7b7adca638eb17f7ad03f990a9be34a43af217a27423a3d6435af917df80`.
- Winning candidate: `staged_candidates/task163_joint_basis_diagonal_yellow_cost335/task163.onnx`.
- Candidate SHA: `93f43991ee07c436ed0186fa158930576b194a3d050ec12500f664cf3b280663`.
- Exact price: `static=335 profile=335 (memory 0 + params 335; 267 real inputs)`, delta **-57**.
- Proof: visible `267/267`; truth capped/full `2000/2000`, `200000/200000`; A/B capped/full `2267/2267`, `200267/200267`, zero divergences and byte-equal; safety PASS with zero warnings/failures.

## Why the old 392 floor was false

The old component argument proved ranks of separately named tensors, but not the minimum **unique initializer count after cross-factor sharing**. The winning ladder exploits exactly that gap:

1. A joint rank-6 coordinate source replaces separate `D4/M4` tables.
2. An integer gauge makes the first five rows of `WD2` equal the old `B`, so one initializer serves both roles.
3. On the supported spatial factors, both old `P` uses equal the already-paid `T`; padding it to `U[3,6]` removes `P`.
4. `A = R3 @ U @ WD2`, trading `A[3,4]` for `R3[3,3]` inside the terminal Einsum.
5. With `R3=[[1,0,0],[-2,0,2],[1,0,-1]]`, `diag(R3) @ U @ WD2 = -e3`; contracting that with `S` supplies `-Y`. Deleting the explicit selector and negating `C` preserves the exact graph output.

Charged ledger: `Q2 180 + WD2 24 + WM2 24 + U 18 + R3 9 + C 40 + S 40 = 335`.

## Exact residual wall

Attempted route: further exact aliases/factor reductions within the same static one-Einsum linear topology.

Failed gate: algebraic/rank pre-gate, before building more variants. On the active 11 rows, `rank([D4 M4])=6`, forcing a 30x6 shared linear coordinate source in this formulation. Both channel factors remain rank 4 across all 10 channels. `WM2` is not obtainable from `WD2` by a cheaper common right map, and a left-map reconstruction costs more than it deletes. No remaining initializer has a value/shape-compatible alias.

Why dead: a second node would materialize a charged full output tensor, overwhelming any initializer saving; the remaining seven tensors are load-bearing in this linear shared-basis formulation.

One remaining hypothesis: a genuinely different nonlinear/sign-rank construction may reduce the spatial carrier from rank 6 to rank 5 using simplex block/local codes plus repeated-factor norms. It must still express separator union and the extra channel-bias term inside the single terminal output node. No concrete lower graph is known; do not re-run linear alias searches below 335.

The prompt's `floor_est 156` remains stale: it assumed a charged `[1,10,H,W]` prepad, while both live and candidate are already one-node free-output graphs with zero memory.
