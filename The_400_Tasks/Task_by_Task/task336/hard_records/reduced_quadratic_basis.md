# reduced_quadratic_basis

- Live/team-best bar: cost 400, static/profile 400 (`181` memory + `219` params), sha256 `418eee6e3107fe8c2c28ab6ecf0464fc47e7b07f007446ba5d25c1d83fcabc89`, source `Team_Submission/Backend/manifest.csv` and workbook row updated `2026-07-02T02:35:15Z`.
- Attempted route: remove the `z2coef` row from Vamshi's rank-2 quadratic final-Einsum render, and delete the matching feature row from `U`/`Vfac`.
- Probe artifact: `The_400_Tasks/Task_by_Task/task336/prototypes/basis_drop_z2.onnx`.
- Static cost: 375 (`169` memory + `206` params), strict-below 400 before gate.
- Failed gate: `python tools/ng_gate.py visible --task 336 --cand The_400_Tasks/Task_by_Task/task336/prototypes/basis_drop_z2.onnx` -> `[visible] 0/31 (0.00%)  FAIL`.
- Why dead: the static quadratic `z2` basis is load-bearing for the 30-wide free-output render; deleting it makes every visible pair wrong, so this reduced-basis family is a certain hidden zero.
- One remaining hypothesis: a different sub-400 win would need an exact algebraic fusion that preserves the full 5-row feature bank while deleting scalar/selector overhead, not a lower-degree basis truncation.
