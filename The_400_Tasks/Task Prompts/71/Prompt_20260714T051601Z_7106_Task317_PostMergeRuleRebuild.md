# task317 - PostMergeRuleRebuild (110 -> floor_est ~53, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to #67 free-output, #89 scorer-invisible final op, generator rule rebuild, trap FAMILY-LABEL. workbook+manifest `task317` = 110. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task317.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 110 sha12 ebd09483ac75 updated 2026-07-09T01:09:17Z; assigned_to is context only (Aditya), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: **Generator-proven (ARC-GEN `ce22a75a`, `tasks/task_ce22a75a.py` read directly).** Grid is **ALWAYS 9x9** - `generate()` is called with no args, so `size=3` is hardcoded and `grid = common.grid(size*size, size*size) = 9x9`. Input contains only colors **{0 bg, 5 gray}**; output only **{0 bg, 1 blue}** (verified over 5000 fresh generator...

Prize: EV +0.73 pts (team_best 110 -> floor_est ~53, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the compact rule state plus one terminal output writer; teammate graph shape and lookup/template carriers are current rungs, not true floors. Aim near 53; if recon proves this estimate false, return DROP with priced evidence.

Direction: Decode the ARC-GEN rule first, then replace stored or overfit template logic with a small exact rule core and a terminal Equal-tail or Where-as-output writer. The charged materialized canvas should disappear into the output node. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task317 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task317 PostMergeRuleRebuild" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task317`, `python -B -m neurogolf_provider.cli consistency task317`, and `python -B -m neurogolf_provider.cli search "task317 PostMergeRuleRebuild"`.

Fast probe (BEFORE the full build): Implement only the numpy rule plus dummy terminal writer and test it on 3000 fresh ARC-GEN rows before graph work; then price the skeleton rule core. Kill rule: if fresh generator rows falsify the rule or the skeleton still needs a dense answer table, stop and report the failed formulation.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; Generator-proven (ARC-GEN `ce22a75a`, `tasks/task_ce22a75a.py` read directly).** Grid is **ALWAYS 9x9** - `generate()` is called with no args, so `size=3` is hardcoded and `grid = common.grid(size*size, size*size) = 9x9`. Input contains only colors **{0 bg, 5 gray}**; output only **{0 bg, 1 blue}** (verified over 5000 fresh generator samples: all shapes.... Prototype count signal: 8 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 317`; `The_400_Tasks/Task_by_Task/task317.md`; task JSON/generator/previews; `inspect_graph.py task317`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 317 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 317 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 317 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 317 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 317 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task317/staged_candidates/task317_PostMergeRuleRebuild_cost<C>/task317.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
