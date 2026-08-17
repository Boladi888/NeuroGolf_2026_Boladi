# task203 - ZeroParamIndexBasis (216 -> floor_est ~104, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to zero-param Range/ConstantOfShape/Trilu basis, Gather/GatherND compaction, op_cost_patterns indexing thresholds. workbook+manifest `task203` = 216. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task203.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 216 sha12 45f547e4ec9b updated 2026-07-12T15:15:54Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: _unknown - derive and document here. If a recipe exists in `Research_Archive/notes/`, summarize it._

Prize: EV +0.73 pts (team_best 216 -> floor_est ~104, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced coordinate/index relation and final lookup; stored position tables and dense per-cell masks are formulation artifacts. Aim near 104; if recon proves this estimate false, return DROP with priced evidence.

Direction: Regenerate row/column/index bases at runtime with Range or ConstantOfShape and use Gather/GatherND only at sparse decision points. Aim to delete stored coordinate tensors, identity grids, and broad one-hot selectors. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task203 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task203 ZeroParamIndexBasis" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task203`, `python -B -m neurogolf_provider.cli consistency task203`, and `python -B -m neurogolf_provider.cli search "task203 ZeroParamIndexBasis"`.

Fast probe (BEFORE the full build): Price only the generated basis plus one Gather/GatherND decision, using visible examples to confirm the index math before the full output path. Kill rule: if generated indices cost more than the stored table or ORT dtype support forces cast clutter, stop and record the cheaper table floor.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 648e1306` - docs+prompts: Onnx_Advanced discoverability restructure + drive-to-floor prompts; d5554bad` - CL_HUGE worker session: 1 staged f16 candidate, 131,592 bytes saved. Prototype count signal: 17 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 203`; `The_400_Tasks/Task_by_Task/task203.md`; task JSON/generator/previews; `inspect_graph.py task203`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 203 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 203 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 203 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 203 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 203 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task203/staged_candidates/task203_ZeroParamIndexBasis_cost<C>/task203.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
