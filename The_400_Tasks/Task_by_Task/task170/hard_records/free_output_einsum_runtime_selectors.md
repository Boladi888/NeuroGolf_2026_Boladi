# free_output_einsum_runtime_selectors — DEAD

Live bar: **1623**, SHA `a0463f0c05f62fe4c2cc1d0f34a7494a29a5a2517fea9edad643ef8e6ce0a4bd`, from `python tools/ng_current_bar.py --task 170 --json` (`manifest`, authoritative). The full 266-real-input parent profile is `1623 (memory 1524 + params 99)`.

Attempted route: one terminal free-output Einsum reading the cursor through runtime row/column `F32[4,30]` projection selectors and a shared static `F32[4,30]` output embed. The probe deliberately grants the locator for free and omits both motif selectors, size/background logic, and channel-0 completion, so its price is an optimistic lower bound.

Exact failed gate: **FAST PRICE**. `score_v2=(0, 1044, 132)` gives cost **1176**, which is **+376 above the 800 kill ceiling**. The two selector operands alone are `480 + 480 = 960` bytes. One real ARC-GEN-shaped `sess.run` took `0.389300 ms`, so latency is not the blocker. Per the prompt kill rule, visible/truth/AB/full profile/safety were not run.

Why dead: task170 cursor top/left are runtime values (observed top 15–24 and left 7–24), and the F32 graph input locks exact Einsum selectors to F32. They cannot be static initializers. The complete direct formulation also needs motif row/column selectors, so it is strictly more expensive. The crop sibling also loses: all-10-channel crop + motif sample/predicate/cast + shared embed is at least 904 before locator and background completion; the 9-channel crop plus channel embed is at least 930. The current 1623 graph already uses free-input row/column Einsums and a free-output `ConvInteger`; the supposed full projection planes are already absent.

Named diff: `ng_graph_cost_diff` identifies the removed `cropb` (−576 B) but adds `cursor_row_selector` (+480 B), `cursor_col_selector` (+480 B), and `embed4x30` (+120 params). Its structure estimator reports −467 versus its 1621 static live estimate; exact scorer/profile accounting is probe 1176 versus authoritative live 1623 (−447), but the probe is incomplete and misses the required <800 target.

One remaining hypothesis: reopen only if an exact analytic scalar encoding can recover all 16 arbitrary cursor colors and motif alignment without a crop or any runtime selector bank, or if a newly proven contract makes both cursor and motif anchors static.

Artifacts: `../prototypes/build_task170_einsum_selector_price_probe.py`, `../prototypes/task170_einsum_selector_price_probe.onnx`, and `../prototypes/task170_einsum_selector_price_probe.manifest.json`.
