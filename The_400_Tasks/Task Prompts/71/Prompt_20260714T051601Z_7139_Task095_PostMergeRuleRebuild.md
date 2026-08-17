# task095 - PostMergeRuleRebuild (208 -> floor_est ~101, EV +0.72 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output, #89 scorer-invisible final op, generator rule rebuild, trap FAMILY-LABEL. workbook+manifest `task095` = 208. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task095.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Vamshi cost 208 sha12 05651c20d66a updated 2026-07-12T15:15:54Z; assigned_to is context only (Vamshi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: Every 9x9 input contains only background `0` and marker `5`. Each marker is at least one cell inboard. Output preserves each `5` and paints color `1` into the eight neighboring cells, leaving all other active cells as background and all canvas padding all-zero.

Prize: EV +0.72 pts (team_best 208 -> floor_est ~101, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the compact rule state plus one terminal output writer; teammate graph shape and lookup/template carriers are current rungs, not true floors. Aim near 101; if recon proves this estimate false, return DROP with priced evidence.

Direction: Decode the ARC-GEN rule first, then replace stored or overfit template logic with a small exact rule core and a terminal Equal-tail or Where-as-output writer. The charged materialized canvas should disappear into the output node. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task095 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task095 PostMergeRuleRebuild" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task095`, `python -B -m neurogolf_provider.cli consistency task095`, and `python -B -m neurogolf_provider.cli search "task095 PostMergeRuleRebuild"`.

Fast probe (BEFORE the full build): Implement only the numpy rule plus dummy terminal writer and test it on 3000 fresh ARC-GEN rows before graph work; then price the skeleton rule core. Kill rule: if fresh generator rows falsify the rule or the skeleton still needs a dense answer table, stop and report the failed formulation.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; ## Algorithm sketch; | 2026-06-10 | baseline live extract | visible 265/265, A/B 2265/2265, static/profile 900/900 | 900 | `prototypes/live_task095.onnx`, SHA256 `289D18394862E993BAEE9C3611D6D1D9567A2CB310CD10AFAE3C8AEE03F81294` |. Prototype count signal: 6 task-local prototype files found. No hard_records signal found in this pass. No top-level staged team JSON was found in this pass; still check task-local staged_candidates before build. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 95`; `The_400_Tasks/Task_by_Task/task095.md`; task JSON/generator/previews; `inspect_graph.py task095`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 95 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 95 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 95 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 95 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 95 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task095/staged_candidates/task095_PostMergeRuleRebuild_cost<C>/task095.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
