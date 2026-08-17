# task106 - PadWrapAffineTile (282 -> floor_est ~136, EV +0.73 pts)

START at `Onnx_Advanced/INDEX.md`; route to Pad(mode=wrap) opset19, affine/period basis, symmetry_and_affine, periodic-template traps. workbook+manifest `task106` = 282. DO NOT submit. Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

Guard: Work only in personal `K-NeuroGolf`; no team folders, no push, no Kaggle/team submit. Refresh `NG_OnnxlyFans_TaskStatus.xlsx`, `Team_Submission/Backend/manifest.csv`, `current_leaderboard_state.json`, and `Team_Submission/Staged_Boladi/task106.json` before building.

Post-update brief: workbook was refreshed after teammate work. Current best is Sho cost 282 sha12 7ad458457c93 updated 2026-07-09T01:09:17Z; assigned_to is context only (Vamshi), not a filter. `Onnx_Advanced/library_db` was fresh on 2026-07-14; still rerun `status` and fall back to routed Markdown if it drifts.

Rule: 4-fold rotational tile on 2x2 or 3x3 inputs. The output is the input quilted into a 2x2 square of rotations: TL=input, TR=90 CW, BL=90 CCW, BR=180. The current team-best formulation assumes active colors are 1..9 and selects between fixed 4x4/6x6 source maps for the 2x2/3x3 cases.

Prize: EV +0.73 pts (team_best 282 -> floor_est ~136, gap 2.1x). Class STRUCTURAL. Target is floor_est, not `rung - epsilon`; no 1-byte tie-shave.

Forced floor: the period/affine parameters and terminal crop; duplicated tiled templates or unrolled shifts are not forced. Aim near 136; if recon proves this estimate false, return DROP with priced evidence.

Direction: If output is shifted/repeated/symmetric, express the period or affine relation directly with Pad-wrap, Slice, Transpose, or generated position basis instead of storing every shifted copy. Before coding run `python Onnx_Advanced/library_db/lib_query.py status`, `python Onnx_Advanced/library_db/lib_query.py context task106 --limit 8`, `python Onnx_Advanced/library_db/lib_query.py search "task106 PadWrapAffineTile" --limit 8`; then from `Q:/AIz/BoBrain/Providers/NeuroGolf_Provider` run `python -B -m neurogolf_provider.cli context task106`, `python -B -m neurogolf_provider.cli consistency task106`, and `python -B -m neurogolf_provider.cli search "task106 PadWrapAffineTile"`.

Fast probe (BEFORE the full build): Build the period/affine coordinate skeleton and test it on visible plus fresh cases before adding color logic. Kill rule: if the generator has data-dependent periods that cannot be represented compactly, stop rather than storing all phases.

Dead ends / task-note signal: Linked research / prototypes** (auto-detected by filename):; 859624fe` - chore: session artifacts " task284 reports + task077 alt prototype + task382 v4; 37a20356` - feat(tasks): task069/task383/task358 +6.16 LB wins + prompts refresh + worker prototypes. Prototype count signal: 15 task-local prototype files found. No hard_records signal found in this pass. Task-local staged_candidates exists; read it before build and beat any local rung, do not rebuild it. Treat any floor/wall/rejected language as prior-path evidence only; name the different formulation you tested.

Recon: `extract_task.py 106`; `The_400_Tasks/Task_by_Task/task106.md`; task JSON/generator/previews; `inspect_graph.py task106`; `largest_intermediate.py --top 20`; task-local `staged_candidates/`; BoBrain provider `context/consistency/search`; visual-first rule derivation before graph work.

Gates: first classify `candidate_class`. Develop with `ng_gate visible --task 106 --cand C.onnx`; Path A/inherited must final-prove with `ng_gate ab --task 106 --cand C.onnx --n 8000` against a recorded parent hash; Path B exact rewrite must final-prove with `ng_gate truth --task 106 --cand C.onnx --n 8000` or an explicit fresh-window policy. Always run `score_v2`; full gate-5 profile `--arcgen 266` via `python tools/agent_helpers/profile_real_cost.py --task 106 --cand C.onnx --arcgen 266`; and `python tools/ng_safety.py model --task 106 --cand C.onnx --json`. One visible miss = stop unless the packet honestly says `RISK_EXPERIMENT`.

Stage only if <EFFECTIVE_BEST: `The_400_Tasks/Task_by_Task/task106/staged_candidates/task106_PadWrapAffineTile_cost<C>/task106.onnx` plus manifest with `candidate_class`, `intended_parent`, `parent_source`, `parent_sha256`, `candidate_sha256`, `candidate_cost_static`, `candidate_cost_profile`, `proof_commands_run`, `proof_results`, `safety_results`, `x86_required`, `kaggle_probe_required`, `status`, and ladder/floor notes.
