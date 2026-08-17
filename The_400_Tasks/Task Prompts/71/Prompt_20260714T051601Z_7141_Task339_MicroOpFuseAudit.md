# task339 - MicroOpFuseAudit (70 -> floor_est ~34, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to op_cost_patterns micro-op pricing, scorer-invisible final op, traps FREE-OUTPUT and class-check doctrine. workbook+manifest `task339` = 70. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task339.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 70 sha12 bf0fcabec2cb updated 2026-07-09T01:09:17Z; assigned_to is context only (Vamshi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **count -> recolor -> render row** (verified 261/261 arc-gen, 2026-07-07): input 3x3 has a single non-zero color `c` appearing `k` times (k[1,9]); output is a **1xk** row filled with `c`. Data-dependent in BOTH width `k` and color `c` (single nonzero color always).

Prize: EV +0.72 pts (team_best 70 -> floor_est ~34, gap 2.1x). Class PEEPHOLE. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the already-tiny rule core; the remaining opportunity is one distinct untried fusion, not a tie-shave. Aim near 34; if recon proves this estimate false, return DROP with priced evidence.

Direction: Use this as a capped floor-break audit: find one charged op chain that can be fused or moved into a free-output tail. Do not retry known equivalent micro-shaves; require a named eliminated op family. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task339 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task339 MicroOpFuseAudit" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task339`, `python -B -m neurogolf_provider.cli consistency task339`, and `python -B -m neurogolf_provider.cli search "task339 MicroOpFuseAudit"`.

Fast probe (BEFORE the full build): Make a single local patch that deletes or fuses the named op family, then run visible and score_v2 before broader cleanup. Kill rule: if the first patch does not beat live cost by a real margin, stop and report the exact no-gain proof.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; c53898ab` - crop-sweep V: 0 candidates staged, 0B saved, expected +0.00 LB; count -> recolor -> render row** (verified 261/261 arc-gen, 2026-07-07):. No prototype count signal; start from generator/visible recon and live graph inspection. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. No top-level staged team JSON was found in this pass; still check task-local staged_candidates before build. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 339`; `The_400_Tasks/Task_by_Task/task339.md`; task JSON/generator/previews; `inspect_graph.py task339`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 339 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 339 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 339 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 339 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 339 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task339/staged_candidates/task339_MicroOpFuseAudit_cost<C>/task339.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
