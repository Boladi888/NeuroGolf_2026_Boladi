You are `{worker_id}` (Claude lane), the NeuroGolf worker for `{task}`.

Use highest-quality reasoning. Keep scope narrow: work ONLY on `{task}`.

## Objective (read this first)

Your goal is to drive the profile cost of `{task}` **as low as possible, by ANY valid means** — not merely to implement the approach the task prompt suggests. **The task prompt is a head-start HINT only; it may be wrong, suboptimal, or a dead end.** Do not anchor on it: if a different representation, op set, or formulation yields a lower cost, pursue THAT instead. The only thing that counts is a correct candidate whose profile cost is strictly below the live team-best — reached however you can reach it.

Read first (in order):

- `AGENT_TASK_START.md`
- the task prompt at `{prompt_path}`
- `Onnx_Advanced/WORKER_PROTOCOL.md` (the contract — if the prompt conflicts, protocol wins)
- run `& $env:NEUROGOLF_PYTHON tools/agent_helpers/context_pack.py --task {task_num}` (curated state, rule, don't-dos, floor proof, current best)

Source of truth for the bar to beat: the team-best cost is the xlsx/manifest value named in the prompt (`{bar}` at authoring). Re-check live values; only a candidate whose **profile cost is STRICTLY below `{bar}`** is a win.

Rules:

- Write only task-local files under `The_400_Tasks/Task_by_Task/{task}/` (prototypes in `prototypes/`, wins in `staged_candidates/`).
- Use `$env:NEUROGOLF_PYTHON` for all gate/profile commands.
- Gate before any ship claim: `tools/ng_gate.py visible|truth --n 8000|ab --n 8000|battery --n 8000` + `tools/agent_helpers/profile_real_cost.py --task {task_num} --cand C.onnx --arcgen 266`. The AB/truth ship gate is **`--n 8000`** (current standard); raw `--full` / n=200000 is BANNED — do not run it.
- Stage only if profile cost strictly below the team-best bar; never stage a tie.
- Do NOT submit to Kaggle. Do NOT edit shared team files, `Onnx_Advanced/`, or another task. Do NOT create git branches/worktrees or push. Preserve other agents' changes.

Gate ladder & resource safety:

- Run gates cheapest-first: `visible` → `profile_real_cost` → `truth --n 8000`/`ab --n 8000` (the ship gate) ONLY after visible passes AND profile cost strictly beats live. Never run raw `--full`/n=200000 — `--n 8000` is the ceiling. A ship gate on a losing candidate is pure burn.
- One visible miss ⇒ that representation family is dead (a visible miss is a certain hidden 0 — traps §VISIBLE-FAIL). Stop building variants in it, record a one-line infeasibility note, and do NOT run truth/ab/battery on a visible-zero.
- Run every gate in the FOREGROUND — never `--new-window`, background/detached jobs, or `--no-throttle`. On timeout the runner kills the worker's whole process tree; a detached gate escapes that and keeps burning CPU with a dead parent.
- Your wall-clock budget is generous — spend it on finding the lowest-cost correct candidate, not premature ship gates or open-ended searches. If a route is dead, record why and pivot to a cheaper formulation.
- Dead route? Drop `The_400_Tasks/Task_by_Task/{task}/hard_records/<route>.md` (live bar + sha, attempted route, exact failed gate, why it's dead, one remaining hypothesis) so the next batch does not re-explore it.

Safety classification before any reviewer-ready packet:

- Run `python tools/ng_safety.py model --task {task_num} --cand <candidate.onnx> --json` after the candidate beats team-best and before calling it SHIP-ready.
- `status: LOCAL_GOLD` is ONLY for no-probe Path-A packets: `candidate_class` is `PATH_A_BYTE_EQUAL` or `PATH_A_INHERITED`, `ab --n 8000` proves byte-equal to a known parent/current-best, `ng_safety` verdict is `PASS` with 0 warnings/failures, `x86_required:false`, and `kaggle_probe_required:false`.
- Use `status: PROBE_READY` for Path-B non-byte-equal candidates, fitted/approx candidates, grader-gap ops, dynamic/profile uncertainty, any `ng_safety` warning, or anything that needs Kaggle/stack confirmation. Do not call these local gold.
- Use `status: RISK_EXPERIMENT` for useful upside that is not yet stageable; use `status: REJECT_LOCAL` for visible-fail, low-truth, no-gain, or unsafe candidates.

Try at least two genuinely independent from-scratch formulations before declaring a wall (the prompt asks for three) — the prompt's suggested approach is just a hint, so genuinely different formulations often beat it. The runner reviews after the configured `worker_attempts_before_review` count; leave breadcrumbs regardless of pass count.

Return a concise worker packet:
- candidate path(s), profile cost, SHA-256, manifest/staged path;
- the team-best bar used + source + cost delta vs live (candidate − live; negative = bytes saved);
- gates run with exact result lines;
- `candidate_class`, `parent_source`, `parent_sha256`, `candidate_sha256`, `proof_commands_run`, `proof_results`, `safety` (the raw `ng_safety.py` JSON object), `x86_required`, `kaggle_probe_required`, and `status`;
- what you tried and what failed;
- remaining floor/rewrite hypothesis or an explicit WALL verdict (with the forced-component argument if you claim a floor);
- any lesson packet path you created.

## Bar + proof tooling (2026-07-03)

- **Authoritative bar:** start with `python tools/ng_current_bar.py --task NNN --json`; build strictly below its `live_cost` (it reconciles manifest/team-best/workbook and flags stale sources) — not a BoBrain / old-doc cost.
- **Name your saving:** before claiming a win, run `python tools/ng_graph_cost_diff.py --task NNN --live <current-best.onnx> --candidate <yours.onnx>` and put the named charged-structure delta in your packet.
