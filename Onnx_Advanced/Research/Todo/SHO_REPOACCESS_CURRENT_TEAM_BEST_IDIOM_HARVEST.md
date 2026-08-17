# SHO Current Team-Best Idiom Harvest — REPO-ACCESS card

**Deliverable type:** repo-access research note plus reviewed `library-fact` candidate blocks.

**Goal:** harvest the reusable minimization idioms in the 2026-07-10 Sho Team_Best promotion and a small low-cost obscure-op scan, then make the results safe for the Librarian to promote into `Onnx_Advanced/library_db/neurogolf_lesson_facts.md`, routed docs, or BoBrain provider notes.

## Contract

You have full repo access to:

- `<REPO_ROOT>`
- `<TEAM_REPO>`
- `Q:\AIz\BoBrain\Providers\NeuroGolf_Provider`

Do not submit to Kaggle. Do not stage team submission changes. Do not edit `Onnx_Advanced/` directly. Produce a Markdown research note under `Logs/research/` and a section of copy-ready `library-fact` candidate blocks for the Librarian.

## Authority Order

Use these sources in this order:

1. `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`
2. `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/current_leaderboard_state.json`
3. `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/`
4. `K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Sho/_shipped/20260710T064919Z_Sho/`
5. `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Past_Submissions/20260710T064919Z_Sho/RECONCILE_NOTE.md`
6. `K-NeuroGolf/Onnx_Advanced/library_db/onnx_library.sqlite` only after `lib_query.py status` is fresh.

Treat `NG_OnnxlyFans_TaskStatus.xlsx`, `The_400_Tasks/task_state.csv`, `live_costs.json`, and existing `task_techniques_index.md` costs as contextual and possibly stale. Record mismatches explicitly.

## Required Targets

Inspect these current team-best artifacts and sidecars:

| Priority | Tasks | Evidence to inspect | Search terms to preserve |
|---|---:|---|---|
| P0 | `task008`, `task198` | `RECONCILE_NOTE.md`, `task008.json`, `task198.json`, current ONNX files | `bianfu697`, `summary gap`, `Shrink Mod16`, `C7 edge-repair`, `ring4`, `ring5`, `ring6`, `membership_diff` |
| P0 | `task351`, `task400` | shipped sidecars + current ONNX files | `digit-pack`, `base-11 re-encode`, `parabola QLC decode`, `Pad(K5)`, `Div/Mod digits`, `z,z^2 QLinearConv` |
| P1 | `task216` | current ONNX + any sidecar/snapshot; mark hidden-flag status | `borrow-table transplant`, `UINT32 lowbit`, `x & -x`, `Log2 Round`, `bit box extraction` |
| P1 | `task286` | current ONNX and existing cost-floor/library facts | `25x25 closure`, `matrix reachability`, `F16 adjacency Einsum`, `BOOL output` |
| P1 | `task358` | current ONNX | `one-node terminal Einsum`, `rank6`, `free-output contraction`, `no value_info memory` |
| P2 | `task087`, `task140`, `task223`, `task244` | current ONNX low-cost ROI-family examples | `RoiAlign`, `MaxRoiPool`, `static ROI`, `ROI table`, `half_pixel`, `pooled_shape` |

## Required Output

Write `Logs/research/SHO_CURRENT_TEAM_BEST_IDIOM_HARVEST_20260710.md` with this order:

1. **Bottom Line** — what should be promoted now, what needs a probe, and what should stay only as current-artifact evidence.
2. **Authority Reconciliation** — manifest/current leaderboard vs workbook/task_state/library route mismatches, with exact current costs and file paths.
3. **Idiom Table** — one row per idiom: task(s), current cost, graph pattern, why it lowers score, safety status, destination.
4. **Reviewed Fact Candidates** — copy-ready `~~~library-fact` blocks for durable facts only. Avoid volatile exact costs unless the fact is specifically about an authority mismatch; use `mutable: true` and `source_anchor` when appropriate.
5. **Task Index Corrections** — exact task route/cost corrections the Librarian should make or defer.
6. **Research Follow-Ups** — ROI-family probe card if needed, task216 hidden-status rehab requirements, task286 doctrine correction plan.

## Promotion Rules

- Promote as `library-fact` when the idiom is visible in a current authoritative ONNX/sidecar and has a reusable pattern independent of the exact current cost.
- Mark facts from hidden-flagged or sidecar-only artifacts as `confidence: current_artifact` or `confidence: needs_probe`, not as proof.
- Do not write volatile score snapshots into BoBrain as permanent truth. Instead, add a provider note if a command/source-policy change is needed.
- If `lib_query.py status` is stale during the harvest, record that and trust Markdown + manifest/provider live facts over DB recall.

## Verification Checklist

- `python Onnx_Advanced/library_db/lib_query.py status`
- `cd Q:\AIz\BoBrain\Providers\NeuroGolf_Provider; python -B -m neurogolf_provider.cli health`
- For at least two target tasks, compare `neurogolf_provider.cli team-score taskNNN` against `manifest.csv`.
- Confirm no submissions were run and no team files were staged or modified.
