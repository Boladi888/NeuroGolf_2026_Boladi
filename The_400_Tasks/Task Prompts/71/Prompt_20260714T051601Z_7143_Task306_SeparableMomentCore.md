# task306 - SeparableMomentCore (220 -> floor_est ~107, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output/separable Einsum, row/column moments, cost_floor_reference plane floors. workbook+manifest `task306` = 220. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task306.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 220 sha12 c935fce45d09 updated 2026-07-12T15:15:54Z; assigned_to is context only (Aditya), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **Quadrant replication.** The grid is divided by full color-4 (yellow) lines into a `width x 2` array of 9x9 quadrants (ARC-GEN `task_c444b776`: `height=2` fixed, `width{1,2,3}`, `size=9`  grids are 19 rows x {9,19,29} cols, embedded top-left in the 30x30 canvas). Exactly ONE quadrant contains content (colors 1-9 on a color-0...

Prize: EV +0.72 pts (team_best 220 -> floor_est ~107, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the forced row/column/color marginals or moments; dense per-cell intermediate planes are not forced when the statistic is separable. Aim near 107; if recon proves this estimate false, return DROP with priced evidence.

Direction: Collapse spatial evidence into row/column/color moments with Einsum/Reduce ops, choose the object/color from compact vectors, and write output only at the end. Do not drag a 10x10x10 stack through the graph. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task306 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task306 SeparableMomentCore" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task306`, `python -B -m neurogolf_provider.cli consistency task306`, and `python -B -m neurogolf_provider.cli search "task306 SeparableMomentCore"`.

Fast probe (BEFORE the full build): Run a numpy moment/marginal check on fresh ARC-GEN rows, then price the marginals-only ONNX skeleton before building the output tail. Kill rule: if the rule requires nonseparable per-cell state after the marginal step, abandon this route and report the counterexample.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 1911db7d` - OB_HUGE worker: final wrap-up notes for 8 unstaged tasks; 1c267a34` - untouched HUGE_RSUM_d: 2 staged candidates, 25688 bytes saved. Prototype count signal: 37 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 306`; `The_400_Tasks/Task_by_Task/task306.md`; task JSON/generator/previews; `inspect_graph.py task306`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 306 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 306 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 306 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 306 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 306 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task306/staged_candidates/task306_SeparableMomentCore_cost<C>/task306.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
