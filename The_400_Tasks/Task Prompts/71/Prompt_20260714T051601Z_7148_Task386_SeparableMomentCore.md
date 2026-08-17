# task386 - SeparableMomentCore (78 -> floor_est ~38, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output/separable Einsum, row/column moments, cost_floor_reference plane floors. workbook+manifest `task386` = 78. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task386.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 78 sha12 9c54eca017da updated 2026-07-09T01:09:17Z; assigned_to is context only (Aditya), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **Horizontal NOR.** Input always 4x7: cols 0-2 = left panel, col 3 = separator (always color 1), cols 4-6 = right panel. Output always 4x3: `out[r,c] = 3 (green) if left[r,c]==0 AND right[r,c]==0 else 0`. Verified 268/268 (train+test+arc-gen). `pred = ((i[:,0:3]==0) & (i[:,4:7]==0)) * 3`.

Prize: EV +0.72 pts (team_best 78 -> floor_est ~38, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced row/column/color marginals or moments; dense per-cell intermediate planes are not forced when the statistic is separable. Aim near 38; if recon proves this estimate false, return DROP with priced evidence.

Direction: Collapse spatial evidence into row/column/color moments with Einsum/Reduce ops, choose the object/color from compact vectors, and write output only at the end. Do not drag a 10x10x10 stack through the graph. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task386 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task386 SeparableMomentCore" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task386`, `python -B -m neurogolf_provider.cli consistency task386`, and `python -B -m neurogolf_provider.cli search "task386 SeparableMomentCore"`.

Fast probe (BEFORE the full build): Run a numpy moment/marginal check on fresh ARC-GEN rows, then price the marginals-only ONNX skeleton before building the output tail. Kill rule: if the rule requires nonseparable per-cell state after the marginal step, abandon this route and report the counterexample.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; e56582d6` - OB_MID_SMALL worker session: 0 staged, 62 tasks documented; c53898ab` - crop-sweep V: 0 candidates staged, 0B saved, expected +0.00 LB. Prototype count signal: 8 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 386`; `The_400_Tasks/Task_by_Task/task386.md`; task JSON/generator/previews; `inspect_graph.py task386`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 386 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 386 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 386 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 386 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 386 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task386/staged_candidates/task386_SeparableMomentCore_cost<C>/task386.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
