# task272 hard record: biasless red-only scatter-conv

Live/team-best bar: cost 378, sha256 `f0302ce51976334b74f18b95e7cb9e455aa91d56d97a77bc47b1ee604000d526`, source `Team_Submission/Backend/manifest.csv` and `NG_OnnxlyFans_TaskStatus.xlsx` row `task272`.

Attempted route: remove the 2-element Conv bias from the team-best `Slice(ch2 5x5) -> Conv(two update planes) -> ScatterElements(output)` graph. Probe path: `The_400_Tasks/Task_by_Task/task272/prototypes/task272_worker3_no_bias.onnx`, sha256 `441eb036b4827a4222d614dcb02a7ff8475bf2b9bef9d5f4eaa672d6a52cf3bb`.

Exact failed gate: `python tools/ng_gate.py visible --task 272 --cand The_400_Tasks/Task_by_Task/task272/prototypes/task272_worker3_no_bias.onnx` returned `[visible] 0/267 (0.00%)  FAIL`. Static score was `(0, 300, 76)`, total 376.

Why dead: red-only no-bias channel-2 separation is impossible. With score `a*center_red + b*neighbor_red_count`, isolated red needs `a <= 0`, red with one orthogonal red neighbor needs `a + b > 0`, and a padding cell adjacent to one border red needs `b <= 0`. Contradiction. The visible-zero probe is the concrete failure.

One remaining hypothesis: a different producer that includes a center active/in-grid feature can avoid the bias algebraically, but the required additional ch0/IG slice raises memory above the 378 bar before params are considered.
