You are still `{worker_agent}` on `{task}`. The reviewer sent this task back.

Reviewer rework request:

```text
{reviewer_rework}
```

Use the prior worker context, the task prompt, and `Onnx_Advanced/INDEX.md`. Address the reviewer demands directly. If you cannot beat the candidate, produce the proof standard the reviewer asked for, not a generic floor claim.

Re-check the current team-best bar before staging or returning a candidate. Do not treat `submission.zip`/LIVE as the bar when the reviewer, task prompt, workbook/task note, or task-local manifest history names a lower team-best.

Do not submit to Kaggle. Keep writes task-local. Keep gates FOREGROUND and cheapest-first (WORKER_PROTOCOL gate hygiene). Use `tools/ng_gate.py` for candidate visible/truth/ab/battery/synth verification; do NOT hand-roll verification loops with `tools.ng_harness.session`, `corpus.iter_tensors`, direct ONNXRuntime sessions, or custom Python AB/truth loops. For any per-candidate gate, do not pass `--no-budget` or raise `ng_gate --max-seconds` above its default/shared cap. For legitimate heavy `--n 8000` gates, give only the shell command a long timeout (at least 1800s when configurable). If `ng_gate` reports its wall budget was exceeded, stop that candidate as gate-timeout/pathological and record the failed gate; do not relaunch it with a larger `--max-seconds`. If the locked hypothesis turns out visible-zero or proves a floor, return that certificate instead of burning more heavy `--n 8000` gates — a proven dead end is a valid outcome here.

Before returning a reviewer-ready candidate, rerun `python tools/ng_safety.py model --task {task_num} --cand <candidate.onnx> --json` and classify it. `LOCAL_GOLD` is only for `PATH_A_BYTE_EQUAL` / `PATH_A_INHERITED` with AB `--n 8000` byte-equal proof, clean `ng_safety` PASS, `x86_required:false`, and `kaggle_probe_required:false`. Path-B truth-`--n 8000`, fitted/approx, grader-gap, dynamic/profile uncertainty, or any safety warning is `PROBE_READY` unless the reviewer explicitly proves otherwise.

Previous worker packet, available as fallback if the runner could not resume the same Codex session:

```text
{prior_worker_packet}
```

Return a new worker packet with candidate paths, costs/profile costs, gates, SHA-256, the team-best bar used, cost delta vs live (candidate − live; negative = bytes saved), and exactly how you answered the reviewer. Include `candidate_class`, `parent_source`, `parent_sha256`, `candidate_sha256`, `proof_commands_run`, `proof_results`, `safety` (raw `ng_safety.py` JSON), `x86_required`, `kaggle_probe_required`, and `status`.

## Bar + proof tooling (2026-07-03)

- **Authoritative bar:** start with `python tools/ng_current_bar.py --task NNN --json`; build strictly below its `live_cost` (it reconciles manifest/team-best/workbook and flags stale sources) — not a BoBrain / old-doc cost.
- **Name your saving:** before claiming a win, run `python tools/ng_graph_cost_diff.py --task NNN --live <current-best.onnx> --candidate <yours.onnx>` and put the named charged-structure delta in your packet.
