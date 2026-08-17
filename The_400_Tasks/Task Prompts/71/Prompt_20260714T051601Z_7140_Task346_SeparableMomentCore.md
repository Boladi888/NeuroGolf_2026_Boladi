# task346 - SeparableMomentCore (140 -> floor_est ~68, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output/separable Einsum, row/column moments, cost_floor_reference plane floors. workbook+manifest `task346` = 140. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task346.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 140 sha12 88660493df54 updated 2026-07-09T01:09:17Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **SOLVED 2026-06-07** (source-read: `external/ARC-GEN/tasks/task_d9fac9be.py` + corpus 200k): grid 5-12 x 5-12, **exactly two colors {c0,c1}**: ~20% noise randomly split between them, then a **solid 3x3 block of c1** stamped at an interior center (rows/cols `randint(2, dim-3)` = 2..9) and the block **center overwritten to c0**. Output...

Prize: EV +0.72 pts (team_best 140 -> floor_est ~68, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced row/column/color marginals or moments; dense per-cell intermediate planes are not forced when the statistic is separable. Aim near 68; if recon proves this estimate false, return DROP with priced evidence.

Direction: Collapse spatial evidence into row/column/color moments with Einsum/Reduce ops, choose the object/color from compact vectors, and write output only at the end. Do not drag a 10x10x10 stack through the graph. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task346 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task346 SeparableMomentCore" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task346`, `python -B -m neurogolf_provider.cli consistency task346`, and `python -B -m neurogolf_provider.cli search "task346 SeparableMomentCore"`.

Fast probe (BEFORE the full build): Run a numpy moment/marginal check on fresh ARC-GEN rows, then price the marginals-only ONNX skeleton before building the output tail. Kill rule: if the rule requires nonseparable per-cell state after the marginal step, abandon this route and report the counterexample.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; e56582d6` - OB_MID_SMALL worker session: 0 staged, 62 tasks documented; ## Algorithm sketch. Prototype count signal: 8 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 346`; `The_400_Tasks/Task_by_Task/task346.md`; task JSON/generator/previews; `inspect_graph.py task346`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 346 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 346 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 346 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 346 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 346 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task346/staged_candidates/task346_SeparableMomentCore_cost<C>/task346.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
