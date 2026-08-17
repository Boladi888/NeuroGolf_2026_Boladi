# task298 - TinyEqualTail (60 -> floor_est ~29, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to #89 scorer-invisible final op, Equal-tail, Where-as-output, tiny output writers. workbook+manifest `task298` = 60. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task298.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 60 sha12 b57713b38953 updated 2026-07-12T15:15:54Z; assigned_to is context only (Aditya), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **perm = [2, 0, 1, 2]**: for each cell in the object, `output_ring_k = input_at_diag_pixel_perm[k]` where ring k = Chebyshev distance from border = min(r,c,H-1-r,W-1-c). - ring0 (border) -> perm[0]=2 -> pixel (2,2) - ring1 (second layer) -> perm[1]=0 -> pixel (0,0) - ring2 (third layer) -> perm[2]=1 -> pixel (1,1) - ring3 (center 4 cells,...

Prize: EV +0.73 pts (team_best 60 -> floor_est ~29, gap 2.1x). Class SWEEP. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced output writer plus a few scalar rule tests; a full-color canvas or repeated mask stack is not forced. Aim near 29; if recon proves this estimate false, return DROP with priced evidence.

Direction: Build the smallest color-index or BOOL decision core, then let Equal-tail or Where-as-output produce the final grid. Replace any stored canvas/Expand chain with one terminal writer. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task298 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task298 TinyEqualTail" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task298`, `python -B -m neurogolf_provider.cli consistency task298`, and `python -B -m neurogolf_provider.cli search "task298 TinyEqualTail"`.

Fast probe (BEFORE the full build): Prototype just the terminal writer with constants from the live graph and price it. It must remove at least one charged canvas/Expand/one-hot family before full rule work. Kill rule: if the output still needs a materialized 10-channel canvas or prices within 5 percent of the current rung, stop and report the exact charged tensor family.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 3f966912` - PR_MID worker: 31/48 tasks staged, ~102k bytes saved; 7a45a36e` - BIG_CONV_c: task298 noop_elim candidate + dead-end docs for task125/task251. Prototype count signal: 8 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 298`; `The_400_Tasks/Task_by_Task/task298.md`; task JSON/generator/previews; `inspect_graph.py task298`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 298 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 298 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 298 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 298 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 298 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task298/staged_candidates/task298_TinyEqualTail_cost<C>/task298.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
