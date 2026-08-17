# GPT-5.5 scope audit for Onnx_Advanced

Audit date: 2026-06-20

Request: check whether `Onnx_Advanced/` is still only for material GPT-5.5 would not already know. Findings only; no cleanup performed.

## Overall verdict

The active worker-facing library is mostly still in-bounds.

The routed surface in `INDEX.md` is dominated by NeuroGolf-specific facts, empirical ORT 1.24.4 CPU behavior, scorer quirks, project-local cost formulas, task exemplars, and measured rewrite/floor lessons. That is not generic ONNX knowledge. A model may know ONNX operator semantics, but it would not already know this repository's 2026 NeuroGolf measurements, local scorer behavior, exact task histories, or project-specific "what actually worked" evidence.

The literal folder tree is not strictly in-bounds if `_archive/`, captain logistics, and retired research prompts count. There are cold-storage and process/provenance files that contain generic or public-reference material a future model could plausibly already know or look up.

## Scope boundary found

`INDEX.md` already defines the right boundary:

- Active first-read: `WORKER_PROTOCOL.md`, `INDEX.md`, `core.md`, `traps.md`.
- Everything else is routed by citation, not linear-read.
- `_archive/` is provenance cold storage, excluded by `.rgignore`, and "browse deliberately, never by default".
- `00_COMPETITION_SPEC.md` is captain reference, not worker path.
- `Research/` is a research-card pipeline; durable findings are extracted into root docs.
- `.update/` is worker drop inbox plus maintenance tools.

That architecture is what keeps the useful surface from becoming a pile of generic ONNX notes.

## Active surface assessment

In-bounds examples:

- `core.md`: NeuroGolf scoring, encoder/decoder, validator bans, hidden-test workflow, local tool paths.
- `ort_reality.md`: project-pinned ORT 1.24.4 dtype/opset/kernel behavior and error fixes.
- `cost_floor_reference.md`, `scoring_internals.md`, `traps.md`: scorer-specific cost and failure laws.
- `techniques.md`, `techniques_catalog.md`, `by_symptom.md`, `task_techniques_index.md`, `recent_ships.md`: measured project techniques and task-specific routing.
- `exemplars/`: worked NeuroGolf case studies.
- `.update/DONE_2026-06-19*`: the six pending drops are all narrow measured task/ORT findings, not generic ONNX knowledge.

No active top-level file looked like a plain ONNX tutorial or basic operator primer. The files that mention ONNX specs generally do so to contrast spec-clean behavior with pinned ORT/grader behavior.

## Likely out-of-scope if the whole tree counts

These are not necessarily harmful because most are archived, but they are not "only things GPT-5.5 would not already know" in a literal sense:

- `_archive/Onnx_basics_worth_reviewing.md`
- `_archive/onnx_op_specs.md`
- `_archive/onnx_utilities.md`
- `_archive/QUICKSTART.md`
- `_archive/notes_arc-agi-1-spec.md`
- `_archive/notes_arc-agi-1-task-families.md`
- `_archive/notes_arc-task-families.md`
- `_archive/00_COMPETITION_SPEC_full_pre_phase1.md`
- `_archive/00_COMPETITION_SPEC_pre_refactor_2026-06-10.md`

`00_COMPETITION_SPEC.md` also contains public logistics such as deadlines, prizes, licenses, rules, and starter-notebook details. Some are future/current competition facts and some are locally verified, but as a category this is public/captain reference more than advanced project-only ONNX technique.

`Research/Done/cards/` contains retired prompt bodies and source-policy scaffolding. The answers/results may matter as provenance, but the card contracts themselves include generic "how to research ONNX" instructions.

## Pending .update queue

Lint reports 6 pending drops:

- `DONE_2026-06-19T160000Z_task158_plane_fold_and_stamp_separability.md`
- `DONE_2026-06-19T172600Z_task174_conv_coloridx_dissolves_f32_slices.md`
- `DONE_2026-06-19T180000Z_task324_opset12_u8_reducemax_and_diagonal_einsum_floor.md`
- `DONE_2026-06-19T200000Z_task079_narrowslice_crop_and_int32_index_floor.md`
- `DONE_2026-06-19T210000Z_task096_freeout_einsum_dissolves_canvas_not_bbox_marginal.md`
- `DONE_2026-06-19T220000Z_task250_98conv_1ch_crop_floor_and_idx_chain_collapse.md`

I sampled all six. They are in-scope: measured task lessons, ORT dtype/opset facts, cost-floor corrections, and explicit sorter notes.

## Structural check

Command run:

```powershell
python .\Onnx_Advanced\.update\tools\lib_lint.py
```

Result: 0 errors, 97 warnings, 1 info.

Important in-folder warnings:

- `Onnx_Advanced/techniques.md` routes to missing `Logs/research/R5_aditya_idiom_catalog.md`.
- `Onnx_Advanced/Research/Done/COMPLETED_CARDS_LEDGER.md` routes to missing `Logs/research/R2_selector_bank_rank.md`.
- Same ledger also routes to missing `Logs/research/R3_peephole_sweep.md`.
- Same ledger also routes to missing `Logs/research/R4_task312_deep_rank.md`.
- Same ledger also routes to missing `Logs/research/R6_onnx_op_candidate_matrix.md`.

Most other warnings are outside `Onnx_Advanced/`, mainly `Agenters/` historical run/proposal references. The lint output does not indicate a hard library integrity failure.

## Recommendation

If the intended rule is "the worker-facing active knowledge base should contain only project-specific, non-obvious material", the folder still mostly satisfies it. Keep enforcing `INDEX.md` routing, `.rgignore` on `_archive/`, and the sorter protocol.

If the intended rule is literal for every file under `Onnx_Advanced/`, then it is not currently true. The main candidates for later review are archived ONNX basics/spec/quickstart files, public competition logistics, and retired research-card prompt scaffolding. Do not delete anything until deciding whether archive/provenance is exempt from the rule.

## Sorter note: owner decision needed

No library edit recommended yet. First decide the scope rule:

- Option A: active/routed worker surface only. Then no major cleanup is needed; just sort the six pending drops and optionally fix the five `Onnx_Advanced` dead-reference warnings.
- Option B: literal whole directory. Then create a separate provenance location outside `Onnx_Advanced/` for generic ONNX basics/specs, public competition logistics, and retired prompt bodies, or rename the folder contract to allow cold storage.
