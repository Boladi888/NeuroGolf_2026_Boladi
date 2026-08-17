# task249 - SeparableMomentCore (219 -> floor_est ~106, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output/separable Einsum, row/column moments, cost_floor_reference plane floors. workbook+manifest `task249` = 219. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task249.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 219 sha12 a9ac53ab9cc2 updated 2026-07-09T01:09:17Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **Horizontal 2x tile.** For an HxW color grid, output is Hx2W with the active columns copied twice, left->right: `output[r,c] = input[r, c mod W]` for `c < 2W`, else background. Colors pass through (incl. color 0). W{3,4,5}, H{3,4,5} in visible+arc-gen (data-dependent W must be detected at runtime). Output MUST be the full...

Prize: EV +0.73 pts (team_best 219 -> floor_est ~106, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced row/column/color marginals or moments; dense per-cell intermediate planes are not forced when the statistic is separable. Aim near 106; if recon proves this estimate false, return DROP with priced evidence.

Direction: Collapse spatial evidence into row/column/color moments with Einsum/Reduce ops, choose the object/color from compact vectors, and write output only at the end. Do not drag a 10x10x10 stack through the graph. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task249 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task249 SeparableMomentCore" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task249`, `python -B -m neurogolf_provider.cli consistency task249`, and `python -B -m neurogolf_provider.cli search "task249 SeparableMomentCore"`.

Fast probe (BEFORE the full build): Run a numpy moment/marginal check on fresh ARC-GEN rows, then price the marginals-only ONNX skeleton before building the output tail. Kill rule: if the rule requires nonseparable per-cell state after the marginal step, abandon this route and report the counterexample.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 9d8f6c11` - SG_ALL worker session: 3 staged candidates, 57,328 bytes saved; c2141b9d` - crop-sweep Q: 3 candidates staged, 49877B saved, expected +3.49 LB. Prototype count signal: 24 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 249`; `The_400_Tasks/Task_by_Task/task249.md`; task JSON/generator/previews; `inspect_graph.py task249`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 249 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 249 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 249 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 249 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 249 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task249/staged_candidates/task249_SeparableMomentCore_cost<C>/task249.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
