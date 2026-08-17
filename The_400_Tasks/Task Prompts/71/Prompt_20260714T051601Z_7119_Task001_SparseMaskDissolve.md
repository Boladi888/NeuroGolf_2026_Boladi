# task001 - SparseMaskDissolve (118 -> floor_est ~57, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to by_symptom sparse masks, Scatter/Gather thresholds, BOOL mask carriers, free-output writer routes. workbook+manifest `task001` = 118. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task001.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 118 sha12 5121f6312e9e updated 2026-07-09T01:09:17Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: Input is a 3x3 pattern P with one fg color k. Output 9x9 fractal self-product: `out[3h+bh][3w+bw] = k iff P[h][w]==k AND P[bh][bw]==k, else 0` (derived 2026-06-07 from officials; LIVE implements the same via tiled/upsampled bg masks).

Prize: EV +0.73 pts (team_best 118 -> floor_est ~57, gap 2.1x). Class SWEEP. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the sparse decision coordinates and selected colors; broad masks, priority stacks, and full shift maps are formulation artifacts. Aim near 57; if recon proves this estimate false, return DROP with priced evidence.

Direction: Find the largest charged mask/priority tensor and dissolve it to sparse coordinates plus one final Where/Scatter/Gather choice. Prefer BOOL/U8 masks and avoid per-color duplication. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task001 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task001 SparseMaskDissolve" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task001`, `python -B -m neurogolf_provider.cli consistency task001`, and `python -B -m neurogolf_provider.cli search "task001 SparseMaskDissolve"`.

Fast probe (BEFORE the full build): Run largest_intermediate on the live graph, then rebuild only the sparse mask decision and compare price to the original mask family. Kill rule: if the sparse path still touches most cells/colors or needs a bigger priority tensor, stop and document the density threshold.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 9d8f6c11` - SG_ALL worker session: 3 staged candidates, 57,328 bytes saved; ff3cf676` - Add needless090-v31 + konbu17-401-v117 to external blend candidate pool. Prototype count signal: 1 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 1`; `The_400_Tasks/Task_by_Task/task001.md`; task JSON/generator/previews; `inspect_graph.py task001`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 1 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 1 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 1 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 1 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 1 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task001/staged_candidates/task001_SparseMaskDissolve_cost<C>/task001.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
