# task147 hard record — w2_moment_bias_shave

- live bar: 254, SHA-256 `b1b2a03ce731839a21ad8cfb63898b11c77cb7a577cdb6033af65be520e82427`, source `python tools/ng_current_bar.py --task 147 --json` (manifest, Sho, 2026-07-09T01:09:17Z).
- attempted route: follow the prompt's separable/moment direction, then test the only obvious sub-254 shave on Sho's 2-Conv graph: remove the final 10-element Conv bias, optionally moving one scalar bias to the first 1x1 Conv.
- exact failed gate: pre-build feasibility probe, not visible. `python The_400_Tasks/Task_by_Task/task147/prototypes/probe_task147_w2_moment_and_bias.py --fresh 1000 --random-trials 3000`
  - `moment_counterexample_key=(2, 2, (1, 1), (1, 1))`; grids `[[0, 3], [3, 0]]` and `[[3, 0], [0, 3]]` share row/column green marginals but require different spatial outputs, so row/column moments alone lose placement state.
  - fresh ch8 rank summary `{0: 17, 1: 100, 2: 261, 3: 486, 4: 280, 5: 108, 6: 15}`; output masks are commonly not low-rank separable.
  - final-bias removal search over 1,194 symbolic local patterns and 5,048 code trials returned `found=false`; last infeasible class was ch0 with 520 positive and 674 negative patterns.
- skeleton price: `python The_400_Tasks/Task_by_Task/task147/prototypes/build_task147_w2_marginal_skeleton.py` produced `task147_w2_marginal_skeleton.onnx`, score `(0, 216, 16)`, cost `232`. This leaves only 22 bytes below live before any reconstruction tail; the moment collision proves that tail needs nonseparable per-cell state.
- why dead: current best already pays the minimum one F32 `[1,1,6,6]` read (144 B) and a final output-writing 3x3 Conv (100 params). Removing the final bias would be the only byte-scale win found, but the no-bias homogeneous separators are infeasible on fresh generator local patterns.
- one remaining hypothesis: prove or disprove a legal way to make the first Conv output a narrower-than-F32 `[1,1,6,6]` tensor without materializing a full-input cast. If that is impossible under ORT Conv input dtype rules, Sho's 254 graph is likely the practical floor for this local-neighbor family.
