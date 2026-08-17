# task374 - qscale-share feature pack shave (1366 -> strict-below 1365)

Assigned: Boladi in the updated OnnxlyFans workbook. Current team-best owner: Vamshi. This is batch 47, Beta/drain lane, slot 02. Workbook row at authoring: cost 1366, points field 17.7804, sha12 `8b6fdcc9538d`, updated_utc 2026-07-02T02:35:15Z, attempt_entries 3, credited_entries 3, excluded_entries 0, attempted_by: Aditya, Vamshi. Workbook file was last written locally at 2026-07-03 01:20:51 local. Manifest agrees: creator Vamshi, cost 1366. Selection used `--all-active` exclusion against active prompt folders plus current leaderboard hard avoids. Baseline phrase: workbook+manifest `task374` = 1366. Re-check workbook and manifest before work; live values win.

Existing evidence: staged_candidates: task374_qscale_share_cost1365 | hard_records: stretch_pack_features_1150_1250.md | prototypes newest: task374_qscale_share.onnx; build_task374_qscale_share.py; label30_coloridx.onnx; constzeros.onnx; build_task374_risky_probes.py; __pycache__; task374_v8.onnx; build_task374_v8.py

## Read first
Start at `AGENT_TASK_START.md`, then `Onnx_Advanced/WORKER_PROTOCOL.md`, then `Onnx_Advanced/INDEX.md`. Route through `Onnx_Advanced/INDEX.md` to the relevant technique/trap concepts; do not cite stale moving paths as authority. DO NOT submit; the captain ships. Do not create branches/worktrees, do not run `git push`, and Do not edit/stage into `K-NeuroGolf-OnnxlyFans`.

## Source of truth refresh
- Workbook: `<TEAM_REPO>/NG_OnnxlyFans_TaskStatus.xlsx`
- Team manifest: `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`
- Leaderboard state: `<TEAM_REPO>/Team_Submission/Backend/current_leaderboard_state.json`
- Team-best ONNX: `<TEAM_REPO>/Team_Submission/Backend/Team_Best_Onnx/task374.onnx`
- Home task doc: `The_400_Tasks/Task_by_Task/task374.md`
- Staged-pending check: No direct Team_Submission/Staged_Boladi/task374.json at authoring; re-check before work.
- Current Submit Captain packet: Agenters/Submit_Captain update `20260703T104822Z-task374.json` reports candidate cost 1365; treat it as a rung to beat, not duplicate. Note: Byte-equal-to-LIVE compression of Vamshi's 1366 graph (removed redundant h_scale scalar; reused q_scale=0.2 with x5 count kernels). ab byte_equal=True over 4267 inputs + staged full 200267/200267; truth 2000/2000; profile 1365 strictly below live 1366. SHA a4b47de8 differs from live 8b6fdcc9538d, so no dup. Hidden-safe -1. SAFE_BATCH_QUEUE.
- Leaderboard note at authoring: Boladi safe18 staged batch was accepted and promoted. Follow-up risk probes for task243, task310, and task233 hidden-failed; stale task137 was rejected as a regression and was not submitted.

## Mandatory lookups
Run compact library lookup before opening large docs:
```powershell
python Onnx_Advanced/library_db/lib_query.py status
python Onnx_Advanced/library_db/lib_query.py context 374
python Onnx_Advanced/library_db/lib_query.py facts --task task374
python Onnx_Advanced/library_db/lib_query.py search "task374 QScaleSharePlus qscale-share feature pack shave" --limit 8
```
If `status` reports STALE, treat it as recall only and fall back to routed Markdown from `Onnx_Advanced/INDEX.md`, `techniques_catalog.md`, and `by_symptom.md`.

Pull BoBrain read-only context if mounted:
```powershell
cd Q:/AIz/BoBrain/Providers/NeuroGolf_Provider
python -B -m neurogolf_provider.cli --include-private-paths context task374
python -B -m neurogolf_provider.cli --include-private-paths consistency task374
python -B -m neurogolf_provider.cli --include-private-paths search "task374 QScaleSharePlus qscale-share feature pack shave"
```

## Assets - visual first, existing graph last
Read in this order: `GUI/previews/task374_preview.png`, `GUI/client/public/tasks/task374.json`, ARC-GEN generator via `context_pack.py`, `The_400_Tasks/Task_by_Task/task374.md`, task-local sidecars, then the team-best/current ONNX. Existing graphs are behavior oracles and cost baselines, not design templates.

```powershell
python tools/agent_helpers/show_task.py --task 374
python tools/agent_helpers/context_pack.py --task 374 --max-kb 14
python tools/agent_helpers/extract_task.py 374
python tools/agent_helpers/inspect_graph.py task374
python tools/agent_helpers/largest_intermediate.py --top 20
```

## Visual-first scratch plan
Before build code, write down: decoded rule, output-size relation, object/color roles, 2-3 falsification cases, and at least two graph families with rough memory+params estimates. Name the charged tensor/op you intend to delete. Anti-anchor: do not inherit the live graph representation before this scratch plan exists.

## Direction
Documented floor/wall claim as tried path: Stretch pack-to-one-feature is walled because two feature channels are required for four output classes; qscale_share gets to 1365.

New formulation to test: Try to remove the next scalar/scale in the qscale-share path by sharing q_scale/h_scale constants or folding one class threshold into the existing feature pack. Target strict-below 1365; if refreshed team/captain/staged facts find a lower effective rung, target strict-below that rung instead.

Concrete dead ends: Do not collapse to one feature channel, do not ignore the four output classes, and do not stage a one-byte tie. Also do not touch team folders, do not submit, and never stage a tie. Treat local stale-current/home-cost values as context only; workbook+manifest/captain refresh wins.

## Gates
```powershell
${NEUROGOLF_PYTHON} tools/ng_gate.py visible --task 374 --cand C.onnx
${NEUROGOLF_PYTHON} tools/ng_gate.py truth --task 374 --cand C.onnx --full
${NEUROGOLF_PYTHON} tools/ng_gate.py ab --task 374 --cand C.onnx --full
${NEUROGOLF_PYTHON} tools/ng_gate.py battery --task 374 --cand C.onnx
python tools/agent_helpers/profile_real_cost.py --task 374 --cand C.onnx --arcgen 266
python -c "from tools._score_v2 import score_network_v2; p=r'C.onnx'; r=score_network_v2(p); print(r, None if r is None else r[1]+r[2])"
```
Visible pass is smoke. If candidate is not byte-equal to a known-correct team graph, require strong truth/battery evidence; `<99% truth` is a learning artifact unless LIVE has the same known ambiguity. If profile_cost differs from static score_v2 by more than 5%, treat it as a credit-probe/grader-gap risk and build a boring-static sibling if pursuing it.

## Stage only if strictly better
Work only in `The_400_Tasks/Task_by_Task/task374/prototypes/`. Stage only below the current effective best (strict-below 1365; or below a lower refreshed staged/captain rung): `The_400_Tasks/Task_by_Task/task374/staged_candidates/task374_qscaleshareplus_cost<C>/task374.onnx` plus manifest with cost, profile_cost, sha256, raw ONNX bytes, gate results, baseline comparison, and next-lever or wall verdict. Never stage ties. Return the worker packet expected by the HandOff runner.
