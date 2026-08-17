# task024 - TinyEqualTail (56 -> floor_est ~27, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to #89 scorer-invisible final op, Equal-tail, Where-as-output, tiny output writers. workbook+manifest `task024` = 56. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task024.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 56 sha12 9090d7fc66fc updated 2026-07-12T15:15:54Z; assigned_to is context only (Vamshi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: (one row: `python tools/agent_helpers/show_task.py --task N` - NEVER Read task_state.csv whole)

Prize: EV +0.73 pts (team_best 56 -> floor_est ~27, gap 2.1x). Class SWEEP. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced output writer plus a few scalar rule tests; a full-color canvas or repeated mask stack is not forced. Aim near 27; if recon proves this estimate false, return DROP with priced evidence.

Direction: Build the smallest color-index or BOOL decision core, then let Equal-tail or Where-as-output produce the final grid. Replace any stored canvas/Expand chain with one terminal writer. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task024 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task024 TinyEqualTail" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task024`, `python -B -m neurogolf_provider.cli consistency task024`, and `python -B -m neurogolf_provider.cli search "task024 TinyEqualTail"`.

Fast probe (BEFORE the full build): Prototype just the terminal writer with constants from the live graph and price it. It must remove at least one charged canvas/Expand/one-hot family before full rule work. Kill rule: if the output still needs a materialized 10-channel canvas or prices within 5 percent of the current rung, stop and report the exact charged tensor family.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 3f966912` - PR_MID worker: 31/48 tasks staged, ~102k bytes saved; Verified on all 3 train + 1 test pairs. NOT a whole-grid geometric transform.. Prototype count signal: 12 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. No top-level staged team JSON was found in this pass; still check task-local staged_candidates before build. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 24`; `The_400_Tasks/Task_by_Task/task024.md`; task JSON/generator/previews; `inspect_graph.py task024`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 24 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 24 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 24 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 24 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 24 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task024/staged_candidates/task024_TinyEqualTail_cost<C>/task024.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
