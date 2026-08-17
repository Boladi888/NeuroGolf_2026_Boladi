# task098 - QLinearTinyCarrier (270 -> floor_est ~130, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to QLinearConv/U8 carrier, dtype narrowing, ORT dtype reality, #84/#85 tiny placer-conv routes. workbook+manifest `task098` = 270. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task098.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 270 sha12 42d0ac9b0b33 updated 2026-07-02T02:35:15Z; assigned_to is context only (Boladi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **outline_extract** family (per `Research_Archive/notes/pattern_catalog_2026-05-06.md`): keep only boundary cells of connected non-zero regions; interior cells become color 0. Ops: Conv, Slice, Greater, ReduceSum, Pad.

Prize: EV +0.73 pts (team_best 270 -> floor_est ~130, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced detector kernels plus one U8/color-index carrier; F32 activation planes and duplicate scales are not forced. Aim near 130; if recon proves this estimate false, return DROP with priced evidence.

Direction: Inspect the live graph for F32 masks, Conv carriers, or duplicate constants. Try a U8/QLinear or BOOL carrier first, reusing scales and zero-points only when ORT accepts it without cast bloat. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task098 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task098 QLinearTinyCarrier" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task098`, `python -B -m neurogolf_provider.cli consistency task098`, and `python -B -m neurogolf_provider.cli search "task098 QLinearTinyCarrier"`.

Fast probe (BEFORE the full build): Patch just the carrier dtype/weight layout and run visible plus static/profile price. The first probe must prove the dominant F32 plane disappears. Kill rule: on ORT dtype rejection, scorer-price regression, or any visible miss; do not add casts that erase the dtype win.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; e56582d6` - OB_MID_SMALL worker session: 0 staged, 62 tasks documented; 79ca80d0` - worker-8 session close: snapshot + pre-existing staged changes. Prototype count signal: 12 task-local prototype files found. Hard-record sidecars exist; read them as failed-path maps before choosing the new formulation. No top-level staged team JSON was found in this pass; still check task-local staged_candidates before build. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 98`; `The_400_Tasks/Task_by_Task/task098.md`; task JSON/generator/previews; `inspect_graph.py task098`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 98 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 98 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 98 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 98 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 98 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task098/staged_candidates/task098_QLinearTinyCarrier_cost<C>/task098.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
