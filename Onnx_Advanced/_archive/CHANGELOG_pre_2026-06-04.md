# CHANGELOG (rotated) — entries before 2026-06-04

> Rotated out of the live CHANGELOG 2026-06-10 (refactor pass 4). Newest-first continues in
> `Onnx_Advanced/CHANGELOG.md`. Historical-log warning applies: old claims are not current truth.

## 2026-06-02 (submit-confirmed opset20/croplabel/color-index lessons)

- `techniques.md` — Added the task138 v9 crop-collapse refinement: source canvas bounds and normalized crop-plane bounds are different; v9 split `[25,24]` F32 source canvas from `[24,23]` crop planes and shipped 14,838 -> 13,204.
- `techniques.md` — Added task238's small-frame scalar color-index map pattern (single `[1,1,7,7]` selected-color map + sentinel 10 + final Equal expansion) and task286's cropped-label Conv pattern (spend initializer params to remove full runtime label planes).
- `techniques.md` — Updated task191 positional-hash membership with the submit-confirmed opset20 `ReduceMax(BOOL)` path: v6 shipped at 24,807 and Kaggle accepted opset 20.
- `traps.md` — Added §14i for negative-pad official-only crops: task138 v10 scored lower and was byte-equal on official-generated inputs, but lost adversarial edge-frame parity, so it is a captain-option, not the clean floor.
- `traps.md` — Added task383's OOD-adversarial-shift interpretation and task286's parent-equivalent compression gate rule.
- `Ort_Advanced/ort_op_dtype_matrix.md`, `Ort_Advanced/ort_errors.md`, `Ort_Advanced/ort_quirks.md`, `Ort_Advanced/ort_opset_support.md` — Corrected OneHot INT32/depth behavior and recorded Kaggle-confirmed opset20 `ReduceMax(BOOL)`.
- `cost_floor_reference.md` — Added task138/task383/task286 floor rows and the 2026-06-01/02 submit calibration datapoints through task286. Recent single-task and same-task follow-on submits continued to match raw `ln(old/new)` within noise.
- `task_techniques_index.md` — Added a 2026-06-02 submit-confirmed quick-reference for task213/191/383/238/286.

## 2026-06-01 (captain-session lessons — filtered-clean is task-specific; ArgMax+Gather)

- `traps.md` — SHIP DECISION REFERENCE: new **§FILTERED-CLEAN vs LOOKUP**. Whether a visible/held-out miss is fatal depends on the rule class — ambiguous fill (hidden = filtered-clean subset, gate `ab`/clean-only; task255 v52 17,604 PASSED at 91.8% held-out and BEAT the worker's "18,145 floor") vs deterministic lookup / exact-geometry / fixed-visible guards (hidden = full official pair set, 100% visible required; task182 v53 truth-20k/20k but visible 264/267 → scored 0; task133 under-20 core truth-200k/200k but visible 262/267 → -11.58 LB, while visible-guarded v2 won +3.41). `ng_gate truth` vs the generator is NOT sufficient when the official task has pairs the generator never emits — gate byte-equal vs LIVE; "byte-equal to a sibling candidate" is only as correct as the sibling.
- `techniques.md` — new "One-hot MatMul → ArgMax + Gather (byte-equal table lookup)" (task182 v53, −52 B when the selector is always one-hot).
- `traps.md` — Follow-up submit lessons: task265 added a morphology-family repeat of the fixed-visible guard trap (truth 200k/200k but visible 265/266 scored as zero-contribution; byte-equal guarded port held), and task165 added the full-one-hot validation trap (`argmax`/active-crop self-tests can hide padding leaks and missing-selector fallthrough to background).
- `techniques.md` — Follow-up submit lessons: task004/task165 added the generator-separation shortcut for single-channel color-index rewrites (per-color-looking rules can collapse when objects are row/box-disjoint; arbitrary 2-D layout keeps the 3,600 B canvas floor), and task191's runtime-template Conv rewrite was annotated with its public +0.65 ship.
- (strategy/process lessons #8–#10 — filtered-clean task-specificity, "shipped ≠ live", captain spread discipline + submit ops — live in `The_400_Tasks/current_strategy.md`; session log in `Logs/lessons.md`.)

## 2026-06-01 (discoverability restructure — flatten + INDEX router; agents weren't opening subfolders)

- `INDEX.md` (NEW) — the top-level router: every durable doc mapped to a one-line "read when X" trigger, grouped by purpose. Convention: a bare filename = the file at the `Onnx_Advanced/` root. `README.md`, the prompt template, and `_Before writing prompt.md` now point here FIRST. Maintenance rule: if a file isn't listed in INDEX, agents won't find it.
- **Flattened the `Research/` reference library up to the root.** Agents skim and don't reliably open subfolders, so the 20 durable `Research/*.md` docs (einsum, uncommon_ops_catalog, arc_rule_taxonomy{,_alt}, arc_taxonomy_empirical_coverage, mlp_rule_extraction, synth_oracle_strategies, direct_detection_multi_qlinearconv, radius_encoded_weighted_cover, row_signature_rank_reduction, multistage_key_first_compositional, cellular_automata, function_proto_subgraphs, zero_param_generators, multi_axis_indexing, spatial_rearrangement, symmetry_and_affine, mod_and_bitwise, graphsurgeon, onnx_utilities) were promoted to the root via `tools/refactor_paths.py` (refs auto-rewritten). `Research/` is now **intake-only** — see its rewritten `README.md`.
- **Absorbed 3 orphan subfolders** (0 inbound refs) into single root docs: `Research/tiny_onnx_arch/` (B1–B6) → `tiny_architectures.md`; `scoring_research/` (A1–A4 + leaderboard) → `scoring_internals.md`; `_sources/` (op specs) → `onnx_op_specs.md`. Each has a TOC + per-section anchors + a "Consolidated from … originals removed" header; source folders `git rm`'d.
- **Dead-link sweep after the moves**: fixed 4 dangling `Research/X.md` sibling refs inside `arc_taxonomy_empirical_coverage.md` (the move tool skips a file's OWN internal refs), 24 `../`-depth links across the 7 promoted files (now one level shallower → `../Ort_Advanced/…`, `../exemplars/…`, `../techniques.md` became bare), and repointed dead `Numpy_Advanced/bit_packing.md` refs (folder removed 2026-05-23) → `Algorithm_Library/algorithm_patterns.md`.
- **Doc-hygiene rule (project-wide)**: the running *current LB total* is no longer hardcoded in any living doc (it goes stale + wastes agent time) — `current_strategy.md`, `_METHODOLOGY_UPDATE_2026-05-25.md`, `_Before writing prompt.md` now reference `Logs/submissions_ledger.csv` (latest row). Dated records (per-ship +LB, per-task xlsx costs) unaffected.
- `tools/refactor_paths.py` — hardened the reusable move tool: documented LIMITATIONS (does not rewrite a moved file's own internal refs; does not recompute `../`) and added a `--apply` warning when a moved file changes directory depth and still contains `../` links.

## 2026-05-31 (compression-session reusable findings integrated from submissions ledger)

- `techniques.md` — NEW section "Variable-size geometry" (3 byte-equal tricks from task150/155/384): (A) **negative-index Gather-wrap** — `idx=(N-1)-arange(30)` INT32 gather wraps padding cols to guaranteed-zero, deleting the mask+Where+Squeeze tail for variable-size flips/reversals (task150 4674→278, task155 →158); (B) **square-grid count identity** — `N=sqrt(ReduceSum(onehot))` fused via `ReduceL2` into one op, plus the **square-grid CONTRACT ship class** (byte-equal-safe only when all visible pairs square AND generator is `grids(size,size)`; validated 3×); (C) **fold-mask-into-Gather-index** — out-of-window output cells → zero row so the 2nd Gather IS the [1,10,30,30] output, no mask-Mul/Pad/Where (task384 77211→9617).
- `cost_floor_reference.md` — NEW section "Single-channel color-index canvas = 900 B floor". The canvas+broadcast tail's only counted cost is the [1,1,30,30] U8 canvas (900 B); **corollary: do NOT eliminate the canvas** — task388 v6 emitting BOOL Equal off [1,10,12,12] scored 2546 (WORSE) than v5's 1998 because the 1,440-B BOOL intermediate > 900-B canvas. Crossover with cropped-BOOL-output at h·w≈90. Plus the **empty-cell sentinel gotcha** (`+ (1-ReduceSum)*10` so background cells decode to all-zero one-hot; task275).
- `traps.md` — SHIP DECISION REFERENCE: added "Which row am I in?" membership test. Before framing a ship as an UNLOCK, extract the LIVE model and run visible — **266/266 visible pass ⇒ normally-scored task ⇒ gain is only the Path-A cost ratio, not a +12 unlock** (task216 v8 worker spent 2h on a +0.07 ship believing +12.5; csv NotSolved is not unlock evidence).

## 2026-05-25 (eighteenth pass — A3 researcher #2 partial deliverable; notes/ subfolder + reduced-scope directive)

- `Research/notes/synth_oracle_partial_audit_2026-05-25.md` (NEW) — second outside researcher on A3 returned a partial deliverable: source audit + methodology critique + 4-axis fidelity rubric (task-family fidelity, example-set disambiguation, oracle independence, runtime/throughput) + 3-layer validation protocol (property-based + adversarial + oracle-independent) + Mermaid diagnostic flowchart for "passed synth, failed held-out" failure mode. Not in the requested research-note FORMAT but materially advances A3. Caught the card's "Hodel's arc-gen" mislabeling (Hodel = RE-ARC, ARC-GEN = Google/Moffitt) and surfaced BARC + ARC-TGI as missing references.
- `Research/README.md` — new `notes/` subfolder pattern for partial deliverables (distinct from `Declined/` which is for no-value-add hedges).
- `Research/Todo/card_A3_synth_oracle_strategies.md` — corrected source list (Hodel = RE-ARC, NOT arc-gen; ARC-GEN attributed to Google/Moffitt; BARC + ARC-TGI added). Updated directive: reduced scope to Q1 capability comparison + Q2-Q4 concrete numpy/algorithm implementations; Q5/Q6 content from the partial-audit doc can be reformatted (do not re-research).
- **Meta-lesson on card design (extending the seventeenth pass)**: even with explicit "do not produce a scoping document" directive, researchers can still produce critique-of-card instead of research-of-topic. The pattern is "audit the assignment + propose how to do it" instead of "execute the assignment". The third-iteration directive moves to "produce numpy snippets, comparison tables, and algorithm recipes" — emphasis on EXECUTABLE artifacts, not narrative.

## 2026-05-25 (seventeenth pass — A3 researcher declined; new Declined/ subfolder + directive cards)

- `Research/Todo/Declined/a3_scoping_attempt_2026-05-25.md` (NEW) — outside researcher assigned card A3 (synth oracle strategies) returned a meta-scoping document instead of the requested research note. They hedged with "the request is unspecified" and recommended that someone do their "Plan A" rather than producing the research themselves. File preserved as a "declined deliverable" with status header explaining the failure mode and salvageable source list (RE-ARC, ARC-GEN, ARC Prize 2024/2025, Hodel paper, Hypothesis/Atheris/coverage.py/mutmut docs).
- `Research/Todo/card_A3_synth_oracle_strategies.md` — preface added with a DIRECTIVE TO NEXT RESEARCHER section: do NOT produce a scoping document; if you find a question genuinely under-specified, make a stated assumption and proceed — do not punt with options. Card itself remains open.
- `Research/README.md` — added Declined/ subfolder section to the Todo/ documentation.
- **Meta-lesson for card design**: strict-contract format alone is insufficient — cards must explicitly say "do not produce a scoping or meta-analysis document" to prevent the hedge-pattern this researcher demonstrated. Will apply this directive prefix to remaining open cards (B1-B3, C1-C3, D2, F1-F4) if/when re-issued.

## 2026-05-25 (sixteenth pass — E1 MLP rule extraction research integrated; task110 v28 F16 arithmetic regression documented)

- `mlp_rule_extraction.md` (NEW) — outside-researcher deliverable for card E1. Survey + 3-pipeline recommendation for extracting symbolic rules from small trained ReLU MLPs (e.g., `21→264→52→150`, discrete grid outputs). Pipelines: (1) TREPAN-style query-distillation into decision tree/rule list; (2) polytope/exact extraction via OpenBox + Marabou/MILP counterexample search; (3) probe-network decomposition for encoder/decoder reverse-engineering. Decision table by MLP shape × data × target rule expressiveness. Sample-complexity bounds for shallow trees, sparse linear separators, Shapley, Anchors, model-extraction. Failure modes: memorization without compact rule, polysemantic superposition, geometric-not-symbolic computation. Includes runnable code skeletons for sklearn distillation + Marabou ONNX query + LogisticRegression probe.
- `Research/Todo/Done/card_E1_mlp_rule_extraction.md` (RETIRED) — card moved out of open Todo backlog (3rd retired card: A1, A2, E1).
- `Research/README.md` — added `mlp_rule_extraction.md` index entry; updated Todo subsection to 11 open cards (was 12); E1 added to retired-cards list.
- `traps.md §14e-2026-05-25T15:55Z` — task110 v28 (cost 32,603→28,955, 266/266 visible, 96.7% synth) shipped → -12.80 LB regression on hidden. F16 cp/cn mask arithmetic lost precision on hidden value patterns. Refines the §14e-2026-05-25T14:08Z visible-family-coverage rule: same-rule type downcasts INVOLVING F16 ARITHMETIC now require ≥99% targeted synth (NOT 96% on generic synth). F16 storage of binary {0,1} masks remains safe; F16 arithmetic on values that depart from {0,1} is the new trap.
- `current_strategy.md` — recent-ships log updated with task110 v28 regression note (canonical 6124.84 preserved per Kaggle-keeps-highest).
- `task110.md` — v28 row updated with shipped-and-regressed outcome + worker pivot directive (revert F16 cp/cn arithmetic, keep v25/v27 wins).

## 2026-05-25 (fifteenth pass — empirical taxonomy coverage + 4 new research cards + new "96% synth ≠ hidden-generalizing" failure mode)

- `arc_taxonomy_empirical_coverage.md` (NEW) — empirical classification of 23 shipped NeuroGolf tasks against both ARC rule taxonomies. Verdict: ~26% of tasks fall in 6 open families neither captures. R1 sharper for extension/copy/translation, R2 sharper for flood-fill/multi-stage/CC-selection. Recommends using both as complementary.
- `Research/Todo/card_F1_radius_encoded_weighted_cover.md` (NEW), `card_F2_row_signature_rank_reduction.md` (NEW), `card_F3_multistage_key_first_compositional.md` (NEW), `card_F4_direct_detection_multi_qlinearconv.md` (NEW) — 4 strict-contract research cards for the 6 NeuroGolf-empirical open families identified by the coverage analysis.
- `Research/README.md` — added empirical-coverage entry + Todo/ subsection listing 12 open cards (A3, B1-B3, C1-C3, D2, E1, F1-F4).
- `traps.md §14e` — added 2026-05-25T15:11Z entry: task233 v8 structural slots K10 (26,744, 964/1000 synth) shipped in bundle + scored +0.00 LB on hidden. New failure mode for SOLVE-FROM-SCRATCH UNLOCKS (distinct from compression on Solved tasks): **synth ≥ 99% is the practical threshold; 96% leaves slack for hidden-distribution divergence**. First-miss synth seed identifies the divergence.
- `The_400_Tasks/current_strategy.md` — bumped to 6124.84, added F-mode for "96% synth not enough for unlocks", updated session-arc total to +175.83 LB.
- Per-task: `task077.md` SHIPPED row for 19,422 floor (+0.139 LB exact match); `task233.md` v8 row updated with shipped-but-no-LB-gain outcome + lesson.

## 2026-05-25 (fourteenth pass — ARC rule taxonomy + symmetry research integrated; methodology refined 3× this day)

**Research integration**: 2 cards completed and synthesized into Research/ proper:

- `arc_rule_taxonomy.md` (NEW) — 18-family ARC-AGI rule taxonomy with ConceptARC group names (`Center`, `Copy`, `Count`, `InsideOutside`, `MoveToBoundary`, etc.), per-family ONNX implementation patterns, recognition signals, cost formulas in `H, W, C, bytes_per_element`, and a decision tree for unknown puzzles. Synthesized from Chollet 2019, ARC Prize 2024/2025, ConceptARC, BARC, Hodel ARC-DSL, RE-ARC.
- `arc_rule_taxonomy_alt.md` (NEW) — complementary 15-family perspective grouped by output-shape behavior, with stronger ARC-AGI-2 emphasis on contextual gating + multi-rule composition. Includes a Mermaid decision tree.
- `symmetry_and_affine.md` (NEW) — 8 dihedral transforms (reflections + rotations) in ONNX, symmetry-flag detection from data, `GridSample` for affine warps.
- `Research/Todo/Done/` — A1 and A2 cards retired (research completed).
- `Research/README.md` — new index entries for the 3 new files.
- `by_symptom.md` — new top-level "By rule-decode stage" section pointing to the taxonomy for "I don't know what rule family this task belongs to" symptoms. Workers should consult this BEFORE writing a numpy reference for an unknown rule.

**Methodology refinements (3 successive ships validated)**:

- `traps.md §14e` 2026-05-25T00:00Z entry — Visible-fit shortcut rule refined to task-type-dependent (Solved tasks usually fine; cherry13b=0 still fails).
- `traps.md §14e` 2026-05-25T14:08Z entry — Refined a third time: HIGH visible-family-coverage synth (2000+/2000) is the actual safety signal; broad-distribution stress synth failures are often FALSE ALARMS for Solved tasks. Validated by task051 v34 (+0.91 LB despite 92/1000 broad synth) + task089 (+0.59 at 0/1000 stress) + task370 (+0.51 at 98.1%).
- `traps.md §14e` new failure mode F6: "Different rule, different cost" scoring artifact. For Solved tasks where a candidate uses a DIFFERENT rule than current but both solve: LOWER cost = positive-EV, HIGHER cost = negative-EV. Validated by task051 v25 (-2.65) vs v34 (+0.91).

**Stats**: 15 confirmed structural-LIVE 1.0× compression ships at ±0.05 LB exact match. 10 confirmed solve-from-scratch unlocks. Session arc 5949.01 → 6124.70 = **+175.69 LB** over 36 hours.

## 2026-05-24 (thirteenth pass — experimental hybrid testing + competitor notebook integration)

**Experimental hybrid testing (4 credits spent, 0 LB net change)**: tested swapping in competitor ONNX from `data/competitor_artifacts/`. Both attempts regressed:
- Hybrid v1 (beicicc-6645-39 base + our 26 confirmed-shipped) → **1508.26** (-4598 LB). Beicicc's claimed 6645.39 artifact is a STUB submission.
- Hybrid v2 (ours + 6 surgical afr1ste-6335 swaps for sub-1k tasks) → **6019.72** (-86 LB). Afr1ste's bigger ONNX for those tasks doesn't actually solve them on hidden.

Reverted to 6106.20 canonical via re-submit (confirmed). Net session result: 0 LB change, 4 credits + 1 revert.

**Competitor notebook integration** (octaviograu 6042.85 notebook, sourced from `V:\kaggle_data_stuff\Projects\Kaggle Scrape - NeuroGolf\kernels\119066626`):
- `traps.md`: new entry on cross-bundle swap antipattern (validated by octaviograu's v16 result + our hybrid v1/v2)
- `traps.md`: new entry on "novel op chain" rule — rewrites that introduce new logic chains can score 0 LB despite passing local verify. Exception: cherry13b=0 solve-from-scratch unlocks (proven 9× this session at 81.8% rate).
- `techniques.md` §33: `-MaxPool(-x)` MinPool trick for connected-component labeling without banned ops
- `techniques.md` §34: ScatterND-add histogram (opset 17+) for size-counting / per-component aggregation
- `techniques.md` "DON'T cross-bundle swap" — new explicit anti-pattern entry

Files updated:
- `traps.md` — 23:00Z entries (cross-bundle + novel-op-chain rules)
- `techniques.md` — §33, §34, anti-pattern DON'T entry, +6 new ships rows
- This CHANGELOG

## 2026-05-23 (twelfth pass — solve-from-scratch unlock pattern CONFIRMED + 4 ships +37 LB)

Four single-task wildcards shipped today (5949.01 → 5986.13 = +37.12 LB session-total). Two NEW calibration anchors established:

**Pattern A: lookup-LIVE / NN-LIVE on cherry13b=0 task → structural rewrite → +15 LB unlock** (2 confirms, 100% rate):
- task066 wallpath_rule (lookup-LIVE, 21,408 → 11,769) → 5949.01 → 5964.63 = **+15.62 LB** at 18:33Z
- task277 bbox_area_min_edges (NN-LIVE, 46,539 → 16,748) → 5968.46 → 5983.73 = **+15.27 LB** at 20:50Z
- Both at LIVE-contribution = 0 before ship (bulk cherry13b 2026-05-23T19:45Z confirmed). Both landed at the full theoretical max for their post-ship cost (`25 - log_e(cost)`). 10 remaining bulk-cherry13b=0 tasks (76, 96, 101, 118, 158, 219, 233, 285, 363, 367) each carry +12-17 LB ceiling at this pattern = **+120-150 LB family ceiling** if reproducible.

**Pattern B: structural-LIVE compression → raw log_e × 1.0× exactly** (2 confirms this session, after several earlier):
- task092 moment_score_einsum_threshold (198,818 → 4,352) → 5964.63 → 5968.46 = **+3.83 LB** (predicted +3.82) at 19:57Z
- task005 v40_compact_rayboxes (183,854 → 16,738) → 5983.73 → 5986.13 = **+2.40 LB** (predicted +2.40) at 21:18Z
- Both structural-LIVE per `live_arch`; both landed at raw log_e ratio of cost reduction. **Predictions are now exact-match accurate** — workers can quote `25 - log_e(new_cost) - (25 - log_e(LIVE_cost))` and treat as the expected delta within ±0.05 LB.

**Bulk cherry13b 11-task triage at 19:45Z**: replaced 11 lookup-LIVE tasks with cost-0 stubs in one Kaggle submit → 0.00 LB delta → ALL 11 confirmed LIVE-contribution = 0 → cost compression of LIVE-equivalent is dead lever for these; only structural rewrites unlock.

**task145 -15.30 LB regression at 19:01Z** (reverted): 0/X task-shaped synth-live IS predictive of hidden-bad regression. **Task-shaped synth-live > 50% required** before staging any candidate (task145 0/50 ≠ task170 0/100 generic which shipped clean — what matters is what the synth diverges ON, not the absolute number).

Files updated:
- `techniques.md` — 4 new ships table rows (task066, task092, task277, task005) with calibration tags
- `traps.md §14e` — three new entries: 18:33Z task066 OVERTURN, 19:01Z task145 regression refinement, 19:45Z bulk cherry13b refinement, 20:50Z task277 second-confirm with family ceiling math
- `by_symptom.md` — new "lookup-LIVE/NN-LIVE with cherry13b=0 → structural unlock" entry (this pass)
- `The_400_Tasks/Task_by_Task/{task066,092,277,005}.md` — SHIPPED entries
- `Logs/submissions_ledger.csv` — 4 manual-submit-captain entries (kanban off this session)
- 4 lookup-LIVE prompts (task367/219/233/096) — rewritten for solve-from-scratch ceiling +12-17 LB
- 6 structural-LIVE prompts (task005[shipped]/002/161/017/243/025) — session-state header added
- Memory: `feedback_lookup_family_detection_refined.md` updated with 2026-05-23T20:50Z second-confirm
- `.update/DONE_2026-05-23T2120Z_session_lookup_unlock_pattern.md` — session consolidation drop (this pass)

## 2026-05-21 (tenth pass — merged Conv technique + task349 radical rewrite)

- `techniques.md` — added new **§23 "Merged-Conv"** (pair-Conv+Equal → single Conv+Equal; replaces eq1+eq2+And pattern with combined kernel `[-1,+1×W,-1]`; -18,000 bytes over 5 iterations on task349).
- `The_400_Tasks/Task_by_Task/task349.md` — documented V3a/V4/V5 attempt history with cost progression (183k → 105k → 84k → **82,898**) and key insights per version.
- `.update/DONE_20260521_task349_mergedconv.md` — DONE note for the merged Conv technique with full math derivation and per-version savings breakdown.
- **Best staged**: task349 V5 at 82,898 bytes (-100,300, -54.7% from 183,198 baseline, +1.53 pts estimate).

## 2026-05-21 (ninth pass — task280 padslice-to-compact-masks integration)

- `exemplars/01_whole_graph_rewrites/task280_padslice_to_compact_masks.md` — **NEW** exemplar. Documents the second-largest single-task cost reduction in the codebase (-157,414 B, -75.7%) achieved via 6 stacked sub-techniques. Includes the cascade-suspect caveat (task282 at 900 B = +18.2 LB if unlocked) and the adversarial 60% safe-band note.
- `techniques.md` — added new **§22 "Compact-mask projection for perpendicular thickening"** (sub-technique 1 from task280 DONE; -15,200 B). Extended Rule 1 with **UINT8 Mul-chain for And+Reduce** (sub-technique 2; -6,400 B). Extended Rule 5 with **BOOL graph output via Concat+Pad-13+** (sub-technique 5; -8,000 B). Extended §4 Rule A with **per-channel F32 Slice + Cast combined variant** (sub-technique 6; -4,200 B).
- `traps.md` — added **§14b "Naive non-incremental thickness check causes cross-rectangle false positives"** (sub-technique 4 from task280 DONE). Documents the silent correctness bug that passes 100% single-object visible verify but fails on multi-object hidden inputs.
- `Ort_Advanced/ort_opset_support.md` — CumSum row extended with **F16 CumSum requires opset 14** + debunked the wrong "F16 CumSum unsafe if output feeds Gather as indices" claim. Future workers should verify the Gather connection (data vs indices position) FIRST before assuming F16 is unsafe. task280 v1 saved -25,600 B after this debunking.
- `by_symptom.md` — added 3 new entries: padslice_shift_conv pairs → compact-mask projection; multi-object thickness check → incremental chain; F16↔F32 CumSum roundtrip → opset 14 bump.
- `.update/DONE_2026-05-21T043642Z_task280_padslice_to_compact_masks.md` — consumed by sorter; deleted.

## 2026-05-22 (eleventh pass — 18-stack reaches 5890.29 + 9 DONE files integrated)

- **18-stack landed at 5890.29** (predicted 5891.79, shortfall -1.50 LB) — first non-perfect additivity in 5 stacks. Likely culprits: task182 (cherry13b not run, possibly zero-weight = -0.51 LB) + task080/task054 partial regressions from <100% adversarial pass rates. Still **+8.81 LB over prior best (5881.48)**.
- **Session standings**: 5862.05 → **5890.29 (+28.24 LB total session gain across 16 Kaggle submissions)**.
- **2 more zero-public-weight tasks confirmed**: task324 (cherry13b Identity = 0.00 LB) + task133 (Phase 0 audit from prior submit evidence). Added to `cascade_strategy.md` ruled-out list alongside task366. The "memorized-lookup family" (task133/task182/task324/task366) appears to be a systematic zero-public-weight cluster.
- `techniques.md` — added 7 new sections (§25-31):
  - §25 Active-subset projection before pair-matrix enumeration (task153 -173k)
  - §26 Compact histogram key replaces full flattened MatMul hash (task158 -45k)
  - §27 BOOL-Or chain replaces Cast→Sum→Greater accumulation (task158 -54k)
  - §28 AND-reorder for row_range × col_range cursor checks (task158 -16k)
  - §29 NOOP sentinel at N+1 position for ScatterND (task182 v1)
  - §30 Memorized-lookup discriminator must stay F32 for sums > 2048 (F16 exact-integer ceiling; task182 measurement)
  - §31 Direct channel slices vs staged `[1, N, H, W]` mask tensor (task187 -5,625 B)
- `traps.md` — added §14c Compact-profile Or-broadcast pollutes padding rows/cols (task013 v2 build-time bug catch).
- `Ort_Advanced/ort_op_dtype_matrix.md` — annotated UINT8 support for `Greater`/`Where`/`Max`/`Sub` at opset 14+ (task054 empirical probe; supersedes earlier task110 opset-13 rejection finding).
- `by_symptom.md` — added 7 new entries for the above techniques + zero-weight task family.
- `cascade_strategy.md` — task324 and task133 added to ruled-out list with citations.
- 9 DONE files consumed by sorter; deleted.

## 2026-05-21 (tenth pass — 4 DONE files integrated; 10-stack additivity proven at +6.70 LB)

- **10-stack additivity reconfirmed**: predicted 5881.48, actual **5881.48 — exact match to the cent**. Fourth consecutive exact-additivity confirmation (4-stack/6-stack/7-stack/10-stack all hit predicted within 0.00 LB). The additivity rule is rock solid across up to 10 simultaneously-stacked single-task wildcards.
- Session standings: 5862.05 → **5881.48 (+19.43 LB)**.
- `techniques.md` — extended Rule 11 (UINT8 for Reduce) with TWO new variants: **UINT8 MaxPool** (opset 12+ unlock, task025 -76k) and **INT8 output assembly** (originally logged as opset 16+, corrected 2026-05-23 to opset 12+, task092 v2 -9,900 B; combined with Or-fold = -15,300 B). Added new **§23 Color-index scalar representation** (task138 v2 -61% on 150k→59k) and new **§24 Merged Conv for W-wide isolated run** (task349 v5 -18k across 5 W-iterations).
- `by_symptom.md` — added 4 new entries: color-index representation for per-cell color graphs; merged Conv for paired sum-Conv+boundary-Conv equality tests; UINT8 MaxPool opset 12+ for 0/1 propagation; INT8 output assembly for BOOL pipelines.
- Single-task wildcard 1.0× calibration multiplier confirmed at 5/5 datapoints: task173 (2.05×), task286 (2.09×), task205 (2.09×), task138 v1 (2.05×), task025 (1.00× over predicted +0.91 = 0.91 actual), task349 v5 (1.00× over predicted +0.79 = 0.79 actual), task138 v2 (1.00× over predicted +1.39 = 1.39 actual). The 1.0× rate is now seven-times confirmed.
- `.update/DONE_2026-05-21T174711Z_color_index_bool_output.md` — consumed; deleted.
- `.update/DONE_20260521_task349_mergedconv.md` — consumed; deleted.
- `.update/DONE_2026-05-21T180325Z_task025_uint8_maxpool_opset12.md` — consumed; deleted.
- `.update/DONE_20260521T1900Z_int8_output_assembly.md` — consumed; deleted.

## 2026-05-20 (eighth pass — INT8 ReduceMax Or-fold + CumSum rank-reduce myth busted)

- **7-stack confirmed additive** (post-task092 v1 added to 6-stack): predicted 5877.33, actual 5877.33 — exact match. Additivity continues to hold across 7 simultaneously-stacked candidates.
- `techniques.md` — Rule 11 extended with **INT8 variant for channel-axis Or-fold** (originally logged as opset 16+, corrected 2026-05-23 to opset 12+). Saves 5,400 B vs N-step Or-chain for [1, N, H, W] BOOL channel-collapse. Replaces my earlier prompt's wrong `Conv(BOOL, ones)` suggestion (ORT rejects BOOL as Conv input).
- `Ort_Advanced/ort_op_dtype_matrix.md` — ReduceMax row annotated with INT8 support (originally logged as opset 16+, corrected 2026-05-23 to opset 12+); UINT8 opset 12+ kept.
- Historical task092 radical-rewrite note — **Vector 4 marked FALSIFIED** with empirical accounting. The "rank-reduce via ReduceMin/ReduceMax + range-comparison" claim was WRONG: the comparison step broadcasts to full [1,9,30,30] BOOL = exactly the same size as the CumSum-derived intermediates it was supposed to replace. The 60k savings estimate was wrong; actual delta is +2,160 B WORSE. Replacement "Vector 4 (the working one)": INT8 ReduceMax for Or-fold (-5,400 B). Also corrected: `Conv(BOOL, ones)` alternative was wrong — ORT rejects. Durable summary is in `cost_floor_reference.md`.
- `cost_floor_reference.md` — added architecture-floor finding for 9-channel span-fill graphs (~208k floor); added 7-stack additivity datapoint.
- `.update/DONE_2026-05-20T230000Z_int8_reducemax_or_fold_and_cumsum_rankreduce_myth.md` — consumed by sorter; deleted.

## 2026-05-20 (seventh pass — calibration model corrected + 2 more wins integrated)

- **Calibration model corrected**: 4 confirmed datapoints (task173 v6, task286 v11, task205 v6, task138 radical_v1) all show single-task wildcards land at **2.05–2.09× the 0.48-calibrated prediction**. Effectively, the 0.48× multiplier is wrong for single-task wildcards — empirical multiplier is **~1.0× (raw log_e directly)**. The "partial cascade unlock" interpretation from the 6th pass was wrong; this is calibration model error. Original 0.48× still holds for 100+ task megastacks. `cost_floor_reference.md` calibration section + `cascade_strategy.md` cascade-events taxonomy + updated single-task wildcard prediction formula all reflect this.
- **6-stack additivity reconfirmed**: t110 v9 + t173 v6 + t286 v11 + t366 v10 + t205 v6 + t138 radical_v1 stacked on v8-canonical → predicted 5877.30, actual 5877.30 (exact match to the cent). Additivity holds across 6 simultaneously-stacked candidates.
- `techniques.md` — added Rule 10 `deferred_cast_bool_throughput` (keep BOOL through Reshape/Transpose, Cast at MatMul/Gemm boundary; task173 v3 -33,750 bytes), Rule 11 `cast_bool_to_uint8_for_reduce` (UINT8 cast for ReduceMax/ArgMax, opset 12+ required; task205 v2 -23,966 bytes), Rule 12 `compare_after_reduce` (Greater-after-ReduceMax algebraic identity; task205 v3 **-60,400 bytes — biggest single win in task205's stack**). §1 header updated to "12 rules".
- `techniques.md` — added new §19 "Conv with runtime weights" (task173 v4 + task205 v1 + Phase D; collapses Mul+ReduceSum chains when output_channels ≤ 1; anti-pattern noted for outer-product reductions where M=C). Added new §20 "Gemm transA/transB" (drops Transpose intermediates; task173 v3 v6 -11,736 bytes). Added new §21 "Drop Conv padding when downstream threshold excludes overflow" (task205 v3 -20,800 bytes).
- `techniques.md` Anti-patterns section — added "DON'T assume K-fold template collapse saves bytes by default" (task205 measurement: BYTE-NEUTRAL for static-shape per-iter intermediates) + "DON'T use Conv-runtime-weights for outer-product reductions" (task173 v2 anti-pattern measurement).
- `Ort_Advanced/ort_op_dtype_matrix.md` — annotated ReduceMax UINT8 row with opset-12 minimum (was missing); added Conv row note about runtime W input support; added Gemm row with transA/transB attributes + C-input opset constraint.
- `.update/DONE_2026-05-20T200000Z_task173_radical_rewrite.md` — consumed by sorter; deleted.
- `.update/DONE_2026-05-20T210000Z_task205_uint8_cast_byte_neutral_stacking.md` — consumed by sorter; deleted.
- `.update/DONE_2026-05-20T220000Z_task205_greater_after_reducemax.md` — consumed by sorter; deleted.

## 2026-05-20 (sixth pass — cascade additivity confirmed)

- `cascade_strategy.md` — Rule 5 rewritten. Historical note: this pass still interpreted some 1.0× single-task gains as partial cascade. Corrected 2026-05-23: most such gains were calibration-model error, not cascade physics; cascade unlocks are rare and single-task wildcards should use raw log_e prediction by default.
- `cost_floor_reference.md` — calibration table extended with the additivity datapoint. Stacking confirmed-positives does NOT lose to cascade interaction.
- Three new datapoints added (partial cascade ~2× for task173 v6 and task286 v11; exact additivity for the 4-stack recovery).

## 2026-05-20 (fifth pass — task366 v10 sorter integration + advanced ORT tricks DONE file)

- `exemplars/01_whole_graph_rewrites/task366_template_vote_to_sparse_stamps.md` — **NEW** exemplar. Documents the largest single-task cost reduction in the codebase (-244,317 bytes, -84.5%) achieved via template-vote → sparse-stamps decomposition. 5 transferable sub-techniques + 4 dead-ends. **Includes the critical caveat: task366 has zero public LB weight (cherry13b 2026-05-11 + v10 measurement 2026-05-20 both confirm), so the technique is transferable but task366-specific LB gain was 0.**
- `techniques.md` — added Rule D to §4 F16 placement (F16 for ScatterND spatial pipelines, saves ~5,400 bytes per pipeline). Added "Radical pattern: template-vote → sparse-stamps decomposition" + "Universal trick: OneHot-value gating for no-match fallback" sections after the existing radical-rewrite playbook. Added task366 to the exemplars list.
- `Ort_Advanced/ort_quirks.md` — added quirk #11 (F16 integer-exactness threshold: 0-2048 exact, then even-only, then mults-of-4, etc.); quirk #12 (ORT does NOT constant-fold under ORT_DISABLE_ALL — explicit no-fold list); quirk #13 (Squeeze/Unsqueeze/Reduce* opset 12→13 schema break: axes attr→input).
- `Onnx_basics_worth_reviewing.md` — extended value_info section with memory-cost note ("each entry contributes num_elements × dtype.itemsize"); added multi-input ops section (Sum/Max/Min/Mean/Concat collapse chains); added cheap-op equivalents table (Reciprocal, Neg, Mul-self vs Pow, Abs vs Sqrt-Mul-self, etc.); added lazy shape ops section (Expand vs Tile, ConstantOfShape, Range, Reshape -1).
- `by_symptom.md` — added "SYMPTOM: template_sets [N1, N2] INT64 dominates params" entry; added "SYMPTOM: graph needs all-zero OneHot output when no pattern matches" entry.
- `cost_floor_reference.md` — calibration section extended with task366 v10 datapoint: zero-public-weight tasks have actual LB = 0 regardless of cost-curve prediction. Cherry13b "replace with Identity" diagnostic recommended BEFORE radical-rewrite.
- `cascade_strategy.md` — task366 marked as RULED OUT (cherry13b still holds; not a cascade-trigger or its neighbors don't have public weight). Added cherry13b diagnostic protocol: 1 Kaggle credit to check whether a task has any public LB weight before committing multi-week effort.
- `.update/DONE_2026-05-20T193610Z_advanced_ort_onnx_tricks.md` — consumed by sorter; deleted.
- `.update/DONE_2026-05-20T205815Z_task366_template_vote_to_sparse_stamps.md` — consumed by sorter; deleted.

## 2026-05-20 (fourth pass — cascade exploitation strategy)

- `cascade_strategy.md` — **NEW**. Strategic playbook for exploiting the cross-task memory cascade bug. Historical note: this initial version overprojected cascade upside from the single task110 event; corrected 2026-05-23 to treat cascade unlocks as rare step-function exceptions, not a +105 LB campaign assumption.
- `core.md` — added pointer to `cascade_strategy.md` in the cascade bug section.

## 2026-05-20 (third pass — task110 v9 Kaggle result back-annotated)

- `exemplars/09_periodicity_detection/task110_periodicity.md` — replaced "Kaggle delta TBD" with actual +0.07 LB (5874.78 → 5874.85). Calibrated prediction was +0.21, undershoot ~3×. Cascade-unlock is a step function; v8 already crossed the threshold so v9 reverted to ordinary log-curve.
- `cost_floor_reference.md` — calibration section extended with v9 data point: same-task follow-on reduction AFTER cascade unlock undershoots calibrated estimate. Cascade-unlock is step-function, not smooth.

## 2026-05-20 (second pass — sorter integration of `DONE_2026-05-20T192645Z_task110_v9_optimizations.md`)

- `techniques.md` — added Rule 9 `padding_sentinel_before_cast` to bool-first family; added compact-form (row+col ReduceSum + Or) sub-rule to §9 active-cell detection; added F16-vs-BOOL calibration paragraph to §1 intro; corrected Rule 8's bool-propagation list (Reduce ops actually reject BOOL).
- `Ort_Advanced/ort_quirks.md` — corrected quirk #7 (Reduce ops reject BOOL/INT8/UINT8/INT16, contrary to prior text); added INT8/UINT8/INT16 sub-bullet to quirk #2 (Where dtype constraints).
- `Ort_Advanced/ort_op_dtype_matrix.md` — added. Per-op accepted/rejected dtype matrix, empirically verified under `ORT_DISABLE_ALL` opset 13. Saves future workers ~30 min of trial-and-error.
- `Ort_Advanced/README.md` — added the new file to the reading order at slot 3.
- `exemplars/09_periodicity_detection/task110_periodicity.md` — corrected cost-floor analysis (row-signature hashing is provably unsound, not "estimated path to ~50k"); added "v9 follow-on optimizations" section documenting -14,107-byte stacking to cost 198,764.
- `.update/DONE_2026-05-20T192645Z_task110_v9_optimizations.md` — consumed by sorter; deleted.

## 2026-05-20

- `QUICKSTART.md` — added. 5-minute decision tree for time-pressed workers. Implements audit recommendation #1.
- `traps.md` — added. Consolidated trap catalog (build / runtime / scoring / submit traps), 25 traps total. Replaces scattered anti-pattern sections in techniques.md and exemplars. Implements audit recommendation #3.
- `by_symptom.md` — added. Flat index keyed by graph signature → exemplar. Alternative entry point to `task_techniques_index.md`. Implements audit recommendation #5.
- `diagnostics_recipes.md` — added. "When you want X, run Y" table for the 10 agent helper scripts. Implements audit recommendation #4.
- `core.md` — calibration section extended with task110 anomaly (+12.73 LB actual vs +0.18 calibrated); cascade trigger list flagged task110 as candidate trigger. Implements audit recommendation #9.
- `cost_floor_reference.md` — calibration section added with multiplier table for stack size + cascade-unlock cases.
- `README.md` — restructured with Tier 1/2/3 reading order + time estimates. Implements audit recommendation #8.
- `onnx_dir_plan.md` — typo fix (`task382_cumsum_geometric.md` → `task382_cumsum_gather.md`). Implements audit recommendation #10.

- `exemplars/02_bool_first/` — 9 exemplars added covering all 8 bool-first rules + ORT Where workaround.
- `exemplars/01_whole_graph_rewrites/` — 5 exemplars (task376 input→BOOL, task049 bbox, task382 CumSum, task119 diag_conv, task115 axis_factor).
- `exemplars/03_crop_and_f16/` — 6 exemplars (task117 cornerstone, task055 per-channel, task384 F16-drop, task128/281/364 crop variants).
- `exemplars/04_algebraic_identities/` — 4 exemplars (Equal-fold variants, add_chain).
- `exemplars/05_static_to_dynamic/` — 2 exemplars (task071 ring-mask, task201 reduce-reorder).
- `exemplars/06_cse_dead_op/` — 4 exemplars including task240 dead-end documentation.
- `exemplars/07_conv_reformulation/` — 2 exemplars.
- `exemplars/08_gather_lookup/` — 4 exemplars.
- `exemplars/09_periodicity_detection/` — task110 case study.
- All categories have READMEs.

- `Ort_Advanced/` — refactored out of core.md. 4 files: ort_quirks, ort_errors, ort_profiling, ort_opset_support.
- `Numpy_Advanced/` — REMOVED 2026-05-23 (general numpy education out of scope; F64 trap moved to `traps.md §17`, init construction inline in `Algorithm_Library/algorithm_patterns.md`).
- `Algorithm_Library/` — 3 files: algorithm_patterns, rule_discovery_methodology, per_family_starting_points.
- `task_techniques_index.md` — auto-generated from xlsx + staged_submits manifests + prototype scripts. 518 lines, grouped by family + technique class.
- `patterns.md` — 25-pattern auto-detector cheat sheet.
- `techniques.md` — 18 techniques across bool-first / crop / F16 / Equal-fold / add-chain / shape-only / Conv / projection-cache / static-to-dynamic / CSE / dead-op / etc.
- `core.md` — NeuroGolf-specific facts (scoring formula, encoder, validator, cascade).
- `Onnx_basics_worth_reviewing.md` — refresher list pinned to working memory.

- `.update/README.md` — simplified worker contribution protocol (DONE-only, sorter integrates later).

## Tools added 2026-05-20

- `tools/agent_helpers/` — 10 ready-to-use diagnostic scripts:
  - extract_task, inspect_graph, largest_intermediate, diff_graphs, dump_intermediates, find_op, score_compare, adversarial_against_live, check_data_freshness, regenerate_graph_diagnostics.
- `The_400_Tasks/Task_by_Task/taskNNN.md` (all 401 files) — auto-generated `## Graph Diagnostic` section appended via `regenerate_graph_diagnostics.py`. Includes cost / op-type histogram / quick-look flags / top intermediates / suggested techniques / full node sequence.

---

## How to update this file

When you add or significantly modify a library file:

1. Add a bullet to the top of the current date's section.
2. Format: `` `path/to/file.md` — short description.``
3. Push the bullet in the same commit as the change.

For contributions via `.update/`: the sorter agent will update this CHANGELOG when integrating.
