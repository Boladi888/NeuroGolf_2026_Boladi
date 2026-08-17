# task215 hard record: nochan_sub100

- live bar used: cost 100, sha256 `8c39d265256613c2eae73e883f5ed72cc95708849bc3d791f80a1d4cc6f63bbf`, source `python tools/ng_current_bar.py --task 215 --json` / team manifest.
- attempted route: keep exact shared `[30,3]` phase gating but delete the shared `chan[10]` channel sign/suppression vector; static cost 90, sha256 `589438c421d2a49807b7160c0b87da17a75d9786559cc63d07f12501d88b473f`.
- distribution check: bundled rows have foreground/background co-occurrence in every foreground phase-column (`5948/5948`), so background must be actively suppressed when a color is present.
- exact failed gate: `python tools/ng_gate.py visible --task 215 --cand The_400_Tasks/Task_by_Task/task215/prototypes/task215_worker3_nochan90.onnx --json --out The_400_Tasks/Task_by_Task/task215/prototypes/gate_visible_worker3_nochan90.json` -> `correct=0 total=265 pct=0.0 pass=false`.
- why dead: without a channel-dependent sign, background and foreground channels can both decode positive in the same phase-column; the task's phase-columns require background suppression.
- one remaining hypothesis: find a legal way to encode background-vs-nonbackground with fewer than 10 charged elements, rather than deleting the channel distinction.
