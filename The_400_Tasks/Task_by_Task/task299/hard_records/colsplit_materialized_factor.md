# task299 - column-factor split route DEAD

- **Live bar:** cost **50**, profile cost **50**, sha256
  `63937e8cccb8a0981bf1e4606b6807083bfab548f949e29a4243abf836467c24`
  (`ng_current_bar.py --task 299 --json`, authoritative source: team manifest,
  owner Aditya, updated 2026-06-19T12:21:50Z).
- **Attempted route:** `prototypes/build_colsplit_probe.py` builds
  `prototypes/task299_colsplit_probe.onnx`, sha256
  `22a6b28c821ff572f7b1486c0d7bbfca47aba7d153be06cdb45018d8d9bb24c2`.
  It splits the live `col_selector[10]` factor into:
  `active_col = Einsum(input, "bgxc->bc")`,
  `neg8_col = Einsum(input, neg8_selector, "bgxc,g->bc")`,
  `col_factor = active_col + neg8_col`,
  then feeds `col_factor` to a final output Einsum.
- **Gate result:** `python tools/ng_gate.py visible --task 299 --cand
  The_400_Tasks/Task_by_Task/task299/prototypes/task299_colsplit_probe.onnx`
  returned `[visible] 21/21 (100.00%)  PASS`.
- **Failed gate:** cost/profile no-gain. `score_network_v2` returned
  `(0, 360, 50)`, cost **410**. `profile_real_cost.py --arcgen 266`
  returned `static=410 profile=410 (memory 360 + params 50; 21 real inputs)`.
- **Named cost diff:** `ng_graph_cost_diff.py` vs live reported delta
  `memory +360, params +0, cost +360`; added charged tensors are
  `active_col`, `neg8_col`, and `col_factor`, each `float32 [1,30]` = 120 B.
- **Why dead:** this route does not reduce params and it materializes at least
  one `[1,30]` F32 column tensor before the free output. Even the minimal
  split starts above the live 50-cost single-Einsum floor; the tested exact
  version is 410.
- **Remaining hypothesis:** only an invalid-or-exotic scorer gap could help,
  such as hiding the column-factor construction inside the same output Einsum.
  The live graph already does exactly that with a 10-param channel selector, so
  any legal ONNX split that names the factor is expected to be strictly worse.
