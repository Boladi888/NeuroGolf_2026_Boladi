# task297 zdrop visible-zero route

- Bar used: workbook+manifest team-best cost 136, sha256 fefcaa50289d057f74b9cfcefa9188d236be69cb956ceed87c16d2a41f7d9d22, owner Vamshi, updated_utc 2026-07-02T02:35:15Z.
- Attempted route: delete one signed z term from the current single-Einsum free-output representation (`task297_zdrop0.onnx`, static cost 118).
- Failed gate: `python tools/ng_gate.py visible --task 297 --cand The_400_Tasks/Task_by_Task/task297/prototypes/task297_zdrop0.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: visible-zero means certain hidden zero; each z term carries required width/height/color cases, so do not run truth/AB or build further zdrop variants from this representation.
- Remaining hypothesis: preserve all three z terms but factor duplicated affine rows or shared bases inside the single Einsum, as in the cost-130 `factor_de` candidate.
