# task047 - MicroOpFuseAudit (60 -> floor_est ~29, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to op_cost_patterns micro-op pricing, scorer-invisible final op, traps FREE-OUTPUT and class-check doctrine. workbook+manifest `task047` = 60. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task047.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 60 sha12 6b007e6d6665 updated 2026-07-02T02:35:15Z; assigned_to is context only (Aditya), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: Single 7-cell and single 8-cell on a 9x9 grid. Output: full row+col cross of 8s through the 8, cross of 7s through the 7; the two crossing cells (r8,c7) and (r7,c8) get color 2; rest bg (derived 2026-06-07).

Prize: EV +0.73 pts (team_best 60 -> floor_est ~29, gap 2.1x). Class PEEPHOLE. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the already-tiny rule core; the remaining opportunity is one distinct untried fusion, not a tie-shave. Aim near 29; if recon proves this estimate false, return DROP with priced evidence.

Direction: Use this as a capped floor-break audit: find one charged op chain that can be fused or moved into a free-output tail. Do not retry known equivalent micro-shaves; require a named eliminated op family. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task047 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task047 MicroOpFuseAudit" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task047`, `python -B -m neurogolf_provider.cli consistency task047`, and `python -B -m neurogolf_provider.cli search "task047 MicroOpFuseAudit"`.

Fast probe (BEFORE the full build): Make a single local patch that deletes or fuses the named op family, then run visible and score_v2 before broader cleanup. Kill rule: if the first patch does not beat live cost by a real margin, stop and report the exact no-gain proof.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 3f966912` - PR_MID worker: 31/48 tasks staged, ~102k bytes saved; in R (rank >=3 forced). Numeric: t=2 loss floor 3.99 / 180 restarts vs t=3 2.5e-3.. Prototype count signal: 15 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 47`; `The_400_Tasks/Task_by_Task/task047.md`; task JSON/generator/previews; `inspect_graph.py task047`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 47 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 47 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 47 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 47 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 47 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task047/staged_candidates/task047_MicroOpFuseAudit_cost<C>/task047.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
