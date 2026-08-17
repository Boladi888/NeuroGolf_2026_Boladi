# task108 - QLinearTinyCarrier (150 -> floor_est ~73, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to QLinearConv/U8 carrier, dtype narrowing, ORT dtype reality, #84/#85 tiny placer-conv routes. workbook+manifest `task108` = 150. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task108.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 150 sha12 8dfa6aeacd8d updated 2026-07-12T15:15:54Z; assigned_to is context only (Aditya), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: Input is always a 10x10 grid embedded top-left in the 30x30 one-hot tensor. The output takes the odd-indexed source cells `(1,3,5,7,9) x (1,3,5,7,9)` and expands each selected cell to a 4x4 block, producing a 20x20 active output with all tensor cells outside that active region left all-zero.

Prize: EV +0.72 pts (team_best 150 -> floor_est ~73, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced detector kernels plus one U8/color-index carrier; F32 activation planes and duplicate scales are not forced. Aim near 73; if recon proves this estimate false, return DROP with priced evidence.

Direction: Inspect the live graph for F32 masks, Conv carriers, or duplicate constants. Try a U8/QLinear or BOOL carrier first, reusing scales and zero-points only when ORT accepts it without cast bloat. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task108 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task108 QLinearTinyCarrier" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task108`, `python -B -m neurogolf_provider.cli consistency task108`, and `python -B -m neurogolf_provider.cli search "task108 QLinearTinyCarrier"`.

Fast probe (BEFORE the full build): Patch just the carrier dtype/weight layout and run visible plus static/profile price. The first probe must prove the dominant F32 plane disappears. Kill rule: on ORT dtype rejection, scorer-price regression, or any visible miss; do not add casts that erase the dtype win.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 9d8f6c11` - SG_ALL worker session: 3 staged candidates, 57,328 bytes saved; ## Algorithm sketch. Prototype count signal: 11 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 108`; `The_400_Tasks/Task_by_Task/task108.md`; task JSON/generator/previews; `inspect_graph.py task108`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 108 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 108 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 108 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 108 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 108 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task108/staged_candidates/task108_QLinearTinyCarrier_cost<C>/task108.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
