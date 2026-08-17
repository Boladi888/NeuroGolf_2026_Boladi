# task092 poly_carrier dead route

- live bar: 778, sha256 `531d8dee5c01da606810020fab34d1c0ecdeb3d3d579b85547f4166f45443866` (`python tools/ng_current_bar.py --task 92 --json`, manifest/team_best/workbook aligned).
- attempted route: replace live `Einsum -> HardSigmoid` orientation carrier with `raw_count` and `raw_count^2` polynomial rows, then feed `T_poly` into the existing final deep `Einsum`.
- candidate: `The_400_Tasks/Task_by_Task/task092/prototypes/work_task092_cdx2/task092_poly_carrier.onnx`, sha256 `a7421c6fb27a299d0b26040e665ae0845899cc08df2f735e49877bdfc254f4bc`, static/profile cost 777.
- exact failed gate: `python tools/ng_gate.py visible --task 92 --cand The_400_Tasks/Task_by_Task/task092/prototypes/work_task092_cdx2/task092_poly_carrier.onnx` -> `[visible] 0/265 (0.00%)  FAIL`.
- why dead: the live graph's saturated `HardSigmoid` makes the background channel's huge raw count a constant 1, then a tiny `T[0,0,0] ~= 5.8e-11` coefficient emits channel 0 on active-grid background cells. A polynomial in raw count either explodes on variable background counts or zeros that path; the probe lost almost all channel-0 positives.
- one remaining hypothesis: a real improvement must keep the saturated background path and target constant tensor factorization (`T`/coordinate bases), not replace the orientation nonlinearity with raw-count polynomials.
