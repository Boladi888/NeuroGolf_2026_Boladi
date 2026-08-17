# task232 - SeparableMomentCore (205 -> floor_est ~99, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output/separable Einsum, row/column moments, cost_floor_reference plane floors. workbook+manifest `task232` = 205. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task232.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 205 sha12 f54c7425eeb6 updated 2026-07-02T02:35:15Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: _unknown - derive and document here. If a recipe exists in `Research_Archive/notes/`, summarize it._

Prize: EV +0.73 pts (team_best 205 -> floor_est ~99, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced row/column/color marginals or moments; dense per-cell intermediate planes are not forced when the statistic is separable. Aim near 99; if recon proves this estimate false, return DROP with priced evidence.

Direction: Collapse spatial evidence into row/column/color moments with Einsum/Reduce ops, choose the object/color from compact vectors, and write output only at the end. Do not drag a 10x10x10 stack through the graph. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task232 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task232 SeparableMomentCore" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task232`, `python -B -m neurogolf_provider.cli consistency task232`, and `python -B -m neurogolf_provider.cli search "task232 SeparableMomentCore"`.

Fast probe (BEFORE the full build): Run a numpy moment/marginal check on fresh ARC-GEN rows, then price the marginals-only ONNX skeleton before building the output tail. Kill rule: if the rule requires nonseparable per-cell state after the marginal step, abandon this route and report the counterexample.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; bc3a5378` - PR_BIG worker: 36/50 tasks staged, ~316k bytes saved; b7247d4f` - BIG_RSUM_a: 2 staged candidates, 8731 bytes saved. Prototype count signal: 15 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. No top-level staged team JSON was found in this pass; still check task-local staged_candidates before build. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 232`; `The_400_Tasks/Task_by_Task/task232.md`; task JSON/generator/previews; `inspect_graph.py task232`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 232 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 232 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 232 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 232 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 232 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task232/staged_candidates/task232_SeparableMomentCore_cost<C>/task232.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
