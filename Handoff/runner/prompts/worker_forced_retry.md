You are still `{worker_agent}` on `{task}`. This is the automatic forced second pass before review.

Do a deliberate "brain / Onnx_Advanced" retry:

- Re-read `Onnx_Advanced/INDEX.md` and `Onnx_Advanced/WORKER_PROTOCOL.md`.
- Run `python Onnx_Advanced/library_db/lib_query.py status`.
- Run `python Onnx_Advanced/library_db/lib_query.py context {task_num}`.
- If the first route was a compression pass, try to reason from the decoded rule or generator instead.
- If you claimed a floor, attack your own floor proof with at least one different formulation.
- If a staged candidate exists, run or verify gate-5 real-input profile.
- Re-check the current team-best bar from the task prompt, workbook/task note, or task-local manifest history. If your staged candidate only beats `submission.zip`/LIVE but not the documented team-best, mark it as a probe and keep working toward the documented bar.

Do not submit to Kaggle. Keep writes task-local. Keep gates FOREGROUND and cheapest-first. Use `tools/ng_gate.py` for candidate visible/truth/ab/battery/synth verification; do NOT hand-roll verification loops with `tools.ng_harness.session`, `corpus.iter_tensors`, direct ONNXRuntime sessions, or custom Python AB/truth loops. For any per-candidate gate, do not pass `--no-budget` or raise `ng_gate --max-seconds` above its default/shared cap. For legitimate heavy `--n 8000` gates, give only the shell command a long timeout (at least 1800s when configurable). If `ng_gate` reports its wall budget was exceeded, stop that candidate as gate-timeout/pathological and record the failed gate; do not relaunch it with a larger `--max-seconds`.

Previous worker packet, available as fallback if the runner could not resume the same Codex session:

```text
{prior_worker_packet}
```

Return an updated worker packet. Make clear what changed from attempt 1, the current team-best bar and expected delta, and whether there is a reviewer-ready candidate.

## Bar + proof tooling (2026-07-03)

- **Authoritative bar:** start with `python tools/ng_current_bar.py --task NNN --json`; build strictly below its `live_cost` (it reconciles manifest/team-best/workbook and flags stale sources) — not a BoBrain / old-doc cost.
- **Name your saving:** before claiming a win, run `python tools/ng_graph_cost_diff.py --task NNN --live <current-best.onnx> --candidate <yours.onnx>` and put the named charged-structure delta in your packet.
