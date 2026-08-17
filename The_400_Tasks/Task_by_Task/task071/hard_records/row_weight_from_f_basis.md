# task071 dead route: row_weight from existing F basis

- Live bar used: authoritative live cost 784, profile cost 784, SHA `0dc63a9456a8f4bad4ccf8d343b45cb91296d1cb600a3e0734d3352cb9ae5674`, Boladi team-best from `python tools/ng_current_bar.py --task 71 --json` / manifest updated `2026-07-10T22:11:58Z`.
- Attempted route: remove `row_weight [30]` by fitting its active support with the existing `F [7,2,30]` basis, paired with the successful `axis_col_weight` replacement.
- Exact failed gate: `python tools/ng_gate.py visible --task 71 --cand The_400_Tasks/Task_by_Task/task071/prototypes/worker1_f_basis_weights_out/task071_row_axis_col_from_f.onnx` returned `[visible] 66/265 (24.91%)  FAIL`.
- Why dead: the current row selector uses extreme lexicographic weights (`row2=1`, rows 3..6 near `2^-36` powers) so lower rows do not perturb the axis/color ratios. Reconstructing that vector through the Fourier basis loses the tiny rows numerically; the only visible passes are the 66 examples whose top occupied row is 2.
- One remaining hypothesis: a different scalar top-row selector with stable, bounded weights might save bytes, but it must preserve rows 2..5 and prove visible before any truth/ab. Do not retry this direct row-weight-from-`F` fit.
