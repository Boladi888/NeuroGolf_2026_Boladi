# task058 - ZeroParamIndexBasis (128 -> floor_est ~62, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to zero-param Range/ConstantOfShape/Trilu basis, Gather/GatherND compaction, op_cost_patterns indexing thresholds. workbook+manifest `task058` = 128. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task058.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 128 sha12 795ce7543fc7 updated 2026-07-02T02:35:15Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: _unknown - derive and document here. If a recipe exists in `Research_Archive/notes/`, summarize it._

Prize: EV +0.72 pts (team_best 128 -> floor_est ~62, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced coordinate/index relation and final lookup; stored position tables and dense per-cell masks are formulation artifacts. Aim near 62; if recon proves this estimate false, return DROP with priced evidence.

Direction: Regenerate row/column/index bases at runtime with Range or ConstantOfShape and use Gather/GatherND only at sparse decision points. Aim to delete stored coordinate tensors, identity grids, and broad one-hot selectors. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task058 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task058 ZeroParamIndexBasis" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task058`, `python -B -m neurogolf_provider.cli consistency task058`, and `python -B -m neurogolf_provider.cli search "task058 ZeroParamIndexBasis"`.

Fast probe (BEFORE the full build): Price only the generated basis plus one Gather/GatherND decision, using visible examples to confirm the index math before the full output path. Kill rule: if generated indices cost more than the stored table or ORT dtype support forces cast clutter, stop and record the cheaper table floor.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 37a20356` - feat(tasks): task069/task383/task358 +6.16 LB wins + prompts refresh + worker prototypes; 3f966912` - PR_MID worker: 31/48 tasks staged, ~102k bytes saved. Prototype count signal: 14 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. No top-level staged team JSON was found in this pass; still check task-local staged_candidates before build. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 58`; `The_400_Tasks/Task_by_Task/task058.md`; task JSON/generator/previews; `inspect_graph.py task058`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 58 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 58 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 58 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 58 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 58 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task058/staged_candidates/task058_ZeroParamIndexBasis_cost<C>/task058.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
