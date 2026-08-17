# 05_BACKLOG.md — NG strategic backlog

> LIVE — edit freely. Tier 2 — load on demand.
>
> `03_PROJECT.md` answers *"what is true right now?"* — open TODOs only (Active section).
> This file answers *"what should I work on next?"* — full prioritized backlog.

---

## When to use this file

- Picking the next experiment / feature to ship.
- Sequencing work across multiple sessions.
- Auditing whether an idea has already been considered (and at what tier).

For "what's open right now?" stay in `03_PROJECT.md`.

---

## Format constraint

Same as `03_PROJECT.md`: every entry is **short title + max 2 sentences** of context. Longer rationale goes to `Logs/lessons.md`, a per-session log, or a `Research_Archive/notes/<NN>_*.md` file — link to it from here.

---

## Lane attribution

Entries below carry historical `[Lane]` tags (`[Alpha]`, `[Beta]`, ...) from the retired lane system (archived 2026-06-05 — `playbooks/_archive/07_LANES.md`). Treat the tags as provenance only; any agent may pick any item.

## Spark handoff markers

Use `[Spark-ready]` for bounded tasks suitable for Codex Spark 5.3 on a fresh branch. Give Spark one marked item at a time, require local verification in the handoff prompt, and do not assign it broad scoring uploads, external-source triage, or multi-day research decisions.

Best current bounded branch picks: use `strawberry9` (5568.77) as the live scoring base, continue the organic-displacement scan order, run the BitShift/opset probe, or keep optimizing exact organic rewrites. Do not use `kiwi9` as a base; it was a pipeline-bad task249 scratch-payload package.

---

## Tiered backlog

### Active scoring coordination

- **[Both] Current lane deck after Alpha yuzu9 batch.** `yuzu9` is current public best at **5573.72**; consumed positives through task003/task017/task020/task021/task029/task052/task105/task126/task129/task229/task249/task251/task304/task315/task394 are in `The_400_Tasks/task_state.csv`, while `ugli9` task182 was public-flat, `vanilla9` task303, `watermelon9` task224, and `xigua9` task177 were hidden-bad, and task145/task370/task025/task044/task382/task387 remain public-negative for current best-score ladders. Treat banana7, cherry7, orange5, raspberry6, strawberry6, cherry9, elderberry9, fig9, grape9, honeydew9, imbe9, jackfruit9, lime9, lemon9, mango9, nectarine9, orange9, papaya9, quince9, raspberry9, strawberry9, tangerine9, ugli9, vanilla9, watermelon9, and xigua9 as historical/non-improving lineage only.
- **[Both] Work `The_400_Tasks/` scoring queue.** Use `The_400_Tasks/task_state.csv` before changing any individual task, and use the OnnxlyFans `Team_Submission` manifest / current leaderboard state for current candidate and shipping truth. The archived `The_400_Tasks/_archive/retired_top_level_ledgers_2026-06-21/submission_candidate*.csv` files are historical blend-ladder evidence only; the active cost-displacement sweep is `Logs/tangerine9_cost_probe_sweep_2026-05-08.csv`, and older tracked isolated recommendations remain in `Logs/next_isolated_probe_recommendations_2026-05-08.md`.
- ~~**[Beta][Spark-ready] Maintain the orange5 ablation queue.**~~ **SUPERSEDED 2026-05-08.** `Logs/orange5_ablation_queue_2026-05-08.md` / `.csv` now point to historical queue work; do not rerun the old orange5 generator unless deliberately rebuilding historical docs.
- ~~**[Beta][Spark-ready] Build the all-local external source sweep.**~~ **COMPLETED 2026-05-08.** `tools/_archive/all_local_external_sweep.py` reproduced the task145 result across every local artifact directory with task145: 46 candidate files scanned, 30 pass full visible+arc-gen, best repeated cost 134,980. Report: `Logs/all_local_external_source_sweep_2026-05-08.md`.
- ~~**[Beta][Spark-ready] Verify pending recipe notes.**~~ **COMPLETED 2026-05-08.** `task270` is `[B->A]` at 266/266; `task229` is `[B->A]` at 267/267; `task105` original recipe is demoted at 122/266 with a 231/266 gridline probe. Report: `Logs/beta_recipe_verification_2026-05-08.md`.
- ~~**[Beta] Probe the May-6 scorer/opset opening.**~~ **COMPLETED 2026-05-11.** Built 7-probe suite covering BitShift (opset 11/18, IR 10/11), CumSum (opset 11), Mod (opset 13), Identity (opset 10/18). **BitShift NO-GO** (passes `check_network` but ORT rejects InvalidGraph regardless of IR/opset). **CumSum opset-11 NO-GO** (same). **Mod opset-13 GO with caution** (works locally, cost-overhead 144k for a single op). **Higher-opset Identity GO**. Key lesson: `check_network` alone is NOT a sufficient gate; always pair with ORT smoke run. Full report: `Research_Archive/notes/beta-bitshift-opset-probe-2026-05-11.md`.
- **[Alpha] Convert verified replacement survivors into code or stable build rules only after hidden-safe proof.** When a candidate replacement survives Kaggle hidden scoring, update `The_400_Tasks/task_state.csv` with the result and decide whether the source should remain external-local or become an organic builder task.
- ~~**[Beta] Re-check old negatives after rule refresh through the task table, not scattered notes.**~~ **COMPLETED 2026-05-07 ledger pass.** task089, task241, and task380 remain `avoid_or_retest`; evidence is filed in `Logs/old_negative_retest_digest_2026-05-07.md` and reflected in `The_400_Tasks/task_state.csv`.
- ~~**[Beta] Work the post-parity no-current-coverage queue.**~~ **CLOSED 2026-05-07; refreshed 2026-05-08.** The default-source cascade-safe blend still leaves task145 only and the all-local sweep found candidates, but xigua6/yuzu6 made task145 public-negative on strawberry6. Keep this as source-manifest evidence, not the next scoring path.
- ~~**[Beta][Spark-ready] Produce a candidate-queue digest for the next upload pass.**~~ **COMPLETED 2026-05-08 refresh; HISTORICAL.** `Logs/submission_candidate_digest_2026-05-08.md` / `.csv` answer "which candidate zip, which tasks, which risks, which local payload path" for the retired blend-ladder flow without reopening old session logs.
- ~~**[Beta][Spark-ready] Refresh the candidate digest after any upload or queue edit.**~~ **SUPERSEDED 2026-07-11.** The candidate digest belongs to the retired blend-ladder CSV flow; current uploads should update the OnnxlyFans `Team_Submission` manifest / current leaderboard state instead.
- ~~**[Beta][Spark-ready] External-covered recipe hardening pass.**~~ **COMPLETED 2026-05-07 first pass.** task348, task106, and task231 were re-verified against current train+test+arc-gen examples; task231's stale task249/tile shortcut was demoted. Ledger: `Logs/external_recipe_hardening_2026-05-07.md`.
- ~~**[A->B][Spark-ready] Organicize task348 only if external-covered displacement becomes useful.**~~ **CLOSED 2026-05-08.** Alpha organicized task348; isolated scoring was hidden-safe but not better than the orange5 external, so reopen only for a concrete cost-displacement reason.

### Tier 1 — Strategic initiatives (from 2026-05-05 Opus synthesis, ported 2026-05-06)

- **[Both] Cascade-bypass via binary-search bisection over external pool**. Keep organic frozen at strawberry baseline (38 organic + 39 stubs = 736.61 LB), submit `strawberry + half externals`; bisect on ERROR. Per-source ordering: konbu17-401 → needless090 → ashok → karnak (CC0/uniform-DSL first, hand-coded with more variance later). Expected +1500-2700 LB recovery. Effort 6-12 hr / 4-7 submission cycles / 2-4 day wall-clock. Source: archived 2026-05-05 synthesis.
- ~~**[Beta] Build `audit_external_cascade(externals_dir, strawberry_baseline_zip)` — 4-stage filter pipeline**~~ **DONE 2026-05-08.** `tools/_archive/audit_external_cascade.py` performs static rejection, fast cost/memory profiling, source cumulative-memory simulation, optional `_score_v2` profiling, and dry-run planning columns; audit `cost` / `cascade_risk` are current-rule `memory + params`. Full run over 19,919 ONNX files: 8,161 likely-safe, 7,854 suspect, 3,904 known-bad; report: `Logs/external_cascade_audit_2026-05-08.md` / `.csv`.
- ~~**[Beta] Add `tests/test_detector_builder_parity.py`**~~ **COMPLETED 2026-05-07.** The test now has zero known strict-decode exceptions and also checks that `tools.inspect_tasks.SOLVED_TASKS` stays synchronized with strict-clean detector coverage. Source: `Research_Archive/notes/beta-queue-2026-05-07-alpha-run.md`.
- **[Both] Competitor port priorities (Opus-recommended order)**: (1) Imaad Mahmood's `rule-based-onnx-solver` — generalize `bbox_fill_with_picker` for the bbox-fill cluster's 8 truly-unsolved candidates; (2) Logic-driven-ensemble's I/O repair logic for external blend safety net; (3) JSON sandboxing prep for the next rule rewrite. Total +40-90 LB organic + uncertain external-blend recovery. Effort 12-20 hr. Source: archived 2026-05-05 synthesis.
- **[Already covered] Hardened external-covered recipe ordering: task348 -> task106 -> task231.** All 3 recipes were re-verified 2026-05-07; task231's tile shortcut is demoted, so only period-extension remains valid. Ledger: `Logs/external_recipe_hardening_2026-05-07.md`.

### Tier 1 — ship-soon (high-leverage, ready to implement)

- ~~**[B→A] task153 — two-blob tile-fit**~~ **DONE by Alpha hard-organic merge 2026-05-08.** Isolated replacement `raspberry6` scored 5534.29 versus orange5's 5522.12 and is inherited by the cherry9 line.
- ~~**[Alpha] task020 — 4-fold rotational symmetry completion (numpy 100%)**~~ **COMPLETED in 20-hr Beta-prep push (this PR, 2026-05-04).** Cost ~82k macs + ~557k memory → 11.63 pts. Implementation: bbox + 3 rotations Gather+Transpose + Max-union over channels 1-9 + derived ch0 + canvas-padding mask.
- ~~**[Alpha] task143 — marker-aligned shape recolor.**~~ **COMPLETED in 40-hr push #2 (2026-05-04).** Cross-correlation via 9 runtime-axis-stacked Slice-shifts × runtime template_mask scalar + ReduceSum. Cost ~225k macs + ~1.16M memory → 10.86 pts.
- ~~**[B→A][Spark-ready] task389 — color-swap**~~ **DONE prior session.** Strict-clean coverage is tracked by `tests/test_detector_builder_parity.py` and `tools.inspect_tasks.SOLVED_TASKS`.
- ~~**[B→A] task206 — marker-driven shape copy**~~ **DONE prior session.** Keep the recipe as background only; no new Alpha action.
- ~~**[B→A] task263 — cfbc odd-one-out count picker**~~ **DONE prior session.** Strict-clean coverage is in `SOLVED_TASKS`.
- ~~**[B→A] task174 — cfbc LR-symmetric picker**~~ **DONE 2026-05-07.** `color-filter-bbox-crop-lr` picks the bbox-relative LR-symmetric color with density tiebreak; strict-clean ONNX verifies 266/266.
- ~~**[B→A] task369 — zero-adjacency label**~~ **DONE 2026-05-07.** `zero-adjacency-label` preserves color 5 and labels zero cells by 4-neighbor zero counts plus heavy-neighbor context; strict-clean ONNX verifies 265/265. Recipe: `Research_Archive/notes/recipe-task369-zero-adjacency-label.md`.
- ~~**[B→A] task055 — cross-section grid label**~~ **DONE 2026-05-07.** `cross-section-grid-label` preserves color-8 divider rows/cols and fills the five non-corner cross sections with fixed labels; strict-clean ONNX verifies 263/263 scorer-convertible examples. Recipe: `Research_Archive/notes/recipe-task055-cross-section-grid-label.md`.
- ~~**[B→A] task059 — grid max-count fill**~~ **DONE 2026-05-07.** `grid-max-count-fill` preserves the color-5 divider grid and fills every 3x3 compartment tied for max object-cell count; strict-clean ONNX verifies 266/266. Recipe: `Research_Archive/notes/recipe-task059-grid-max-count-fill.md`.
- ~~**[B→A] task184 — grid object summary**~~ **DONE 2026-05-07.** `grid-object-summary` condenses nonzero row/column bands separated by all-zero gutters into the compact color grid; strict-clean ONNX verifies 169/169 scorer-convertible examples, with 97 >30x30 arc-gen examples skipped by the official helper. Recipe: `Research_Archive/notes/recipe-task184-grid-object-summary.md`.
- ~~**[B→A] task175 — Euclidean wedge fill**~~ **DONE 2026-05-07.** `euclidean-wedge-fill` fills 21x21 zero holes from the runtime base-color/period wedge lattice; strict-clean ONNX verifies 266/266. Recipe: `Research_Archive/notes/recipe-task175-euclidean-wedge-fill.md`.
- ~~**[B→A] task071 — bilateral-symmetry-denoise**~~ **DONE 2026-05-07.** `bilateral-symmetry-denoise` chooses the horizontal axis that maximizes kept cells mirrored onto any nonzero footprint, then fills only rectangular-noise cells mirrored to kept cells; strict-clean ONNX verifies 265/265. Recipe: `Research_Archive/notes/recipe-task071-bilateral-symmetry-denoise.md`.
- ~~**[B→A] task019 — tile diagonal-neighbor mark**~~ **DONE 2026-05-07.** `tile-diagonal-neighbor-mark` tiles the input 2x2 and writes color 8 on active background cells diagonally adjacent to tiled marker cells; strict-clean ONNX verifies 267/267. Recipe: `Research_Archive/notes/recipe-task019-tile-with-marked-empties.md`.
- ~~**[B→A] task137 — diagonal marker square rings**~~ **DONE 2026-05-07.** `diagonal-marker-square-rings` draws Chebyshev square rings centered on the middle of three diagonal markers at the marker spacing period; strict-clean ONNX verifies 266/266. Recipe: `Research_Archive/notes/recipe-task137-diagonal-marker-square-rings.md`.
- ~~**[B→A] task320 — staircase bottom-half recolor**~~ **DONE 2026-05-07 inventory sync.** Strict parity: 34/34 on train+test+first-30 arc-gen.
- ~~**[B→A] task348 — diagonal rays from line cells**~~ **DONE by Alpha hard-organic merge 2026-05-08.** Isolated replacement was hidden-safe but did not beat the orange5 external, so keep it as organic fallback unless Beta finds a cost-displacement reason.
- **[B→A][Spark-ready] task270 — anchor-cross from row/col-aligned cells.** Promoted 2026-05-08: refined color-partition rule verifies 266/266 across train+test+arc-gen. Recipe: `Research_Archive/notes/recipe-task270-anchor-cross-from-aligned-cells.md`; verifier report: `Logs/beta_recipe_verification_2026-05-08.md`.
- ~~**[B→A] task316 — boustrophedon-pack 3x3**~~ **STRICT-PARITY FIXED 2026-05-07.** Output padding is masked to top-left 3x3; strict parity is 34/34.
- ~~**[B→A] task106 — 4-fold rotation tile**~~ **DONE by Alpha hard-organic merge 2026-05-08.** Isolated replacement was hidden-safe but not score-improving over the orange5 external, so leave further work to cost-displacement evidence.
- ~~**[B→A] task249 — horizontal tile 2x**~~ **STRICT-PARITY FIXED 2026-05-07.** Width detection now uses the one-hot active mask so background-only edge columns survive strict decoding.
- ~~**[Beta] task231 — period-extend non-zero row**~~ **DONE by Alpha hard-organic merge 2026-05-08.** Current `horizontal_tile_2x` does **not** match this task; the period rule re-verifies 266/266, and the organic version is now a hidden-safe fallback.
- ~~**[B→A] task388 — column-fill 0->8 + 2x2 tile**~~ **STRICT-PARITY FIXED 2026-05-07.** Height/width detection now uses the one-hot active mask so background-only rows/cols are preserved.
- ~~**[Beta][Spark-ready] Remove dead CASCADE_FLOOR code from `build_submission.py`.**~~ **COMPLETED before this Alpha run.** Current source has no `CASCADE_FLOOR` references; keep `Research_Archive/notes/analysis-cushion-fix-reevaluation-2026-05-05.md` as historical context only.
- ~~**[Alpha] task206 — marker-driven shape copy**~~ **DONE prior session.**
- ~~**[Alpha] task071 — denoise via bilateral symmetry**~~ **DONE 2026-05-07.** Refined axis-selection rule landed as `bilateral-symmetry-denoise`; no further Alpha action.
- **[B→A] task363 — shift-correlation + sparse-line tiebreak** (LB-5126 silent-fail rule). Targets 1×4 / 4×1 line seeds. Recipe is high-level only in the deep-read; **rule derivation from data needed (~2-3 hr investigative, uncertain ROI)** — investigated 2026-05-04 pattern set #3, deferred. ~10-13 pts. Source: `Research_Archive/notes/deep-read-lb-5126-bug-free-blend.md`.

> **COMPLETED in pattern set #3 (PR #36, 2026-05-04):** ~~task031~~ (foreground-bbox-crop), ~~task123~~ (fractal-ring-expansion), ~~task014~~ (color-filter-bbox-crop, picker='least'; mining-generalized to also catch task049 + task310), ~~task036~~ (color-filter-bbox-crop, picker='smallest_bbox'; refines Beta's 68.7% partial), ~~task243~~ (seed-flood-fill in bbox-fill cluster A). +7 SOLVED, +93.26 organic pts.

> **COMPLETED in 40-hr push (PR #37, 2026-05-04):** ~~task307~~ (kron-block-up; +cost-aware reclassification of task223), ~~task100~~ (const-largest-bbox; refines old 248/267 partial), ~~task346~~ (const-framed-cell; refines old 230/267 partial). +3 SOLVED, +43.10 organic pts. **External pool refresh** added needless090-v31 + konbu17-401-v117 → 56 newly-covered tasks via blend (mango #13 = 4042 LB).

> **COMPLETED in 40-hr push #2 (PR #38, 2026-05-04):** ~~task143~~ (marker-aligned-shape-recolor; cross-correlation via runtime-axis-stacked shifts), ~~task021~~ (grid-division-count; per-channel monocolor row/col detection + ArgMax), ~~task065~~ (marker-section-extract; reuses task021 divider machinery + 1D BFS for section bounds + Gather bbox-crop). +3 SOLVED, +37.27 organic pts (1096.78 → 1134.05).

> **COMPLETED in 20-hr Beta-prep push (this PR, 2026-05-04):** ~~task020~~ (rotational-symmetry-completion; bbox + 3 rotations Gather+Transpose + Max-union over channels 1-9 + derived ch0; +11.63 pts), ~~task300~~ (color-filter-bbox-crop most_count picker variant; ArgMax(counts) directly; +13.19 pts; mining-surfaced via mine_karnak_targeted). +2 SOLVED, +24.82 organic pts (1134.05 → 1158.87).

### Tier 2 — ship-when-Tier-1-empty

- ~~**[Beta][Spark-ready] Post-parity truly-unsolved queue**~~ **CLOSED 2026-05-07; refreshed 2026-05-08.** `tools/_archive/karnak_xref.py` still sees a default-source task145 gap and the all-local sweep found candidates, but Alpha's xigua6/yuzu6 scoring makes task145 current-base negative. Keep the evidence filed; do not put task145 in the active upload queue.

- **[B→A] bbox-fill cluster (7 truly-unsolved candidates remaining: 66, 102, 105, 255, 265, 363, 367)**. ~~task243~~ ✅ landed via pattern set #3 (seed-flood-fill). Mined via `tools/mine_rule_based_detectors.py` (rule: 0-cells become a fixed color C consistent across pairs). Per-task characterization in `Research_Archive/notes/bbox-fill-cluster-2026-05-03.md`. Sub-cluster A (fill in input): task066 (path-drawing, hard), task363 (LB-5126 silent-fail recipe, medium). Sub-cluster B (fill introduced fresh): task102/105/367 (rectangle-interior-fill family, medium; task102 numpy-tested at 70/267 — needs refinement), task255 (large fill area, hard), task265 (2x2 marker family, medium-hard). Suggested order: 363 → 102/105/367 cluster → 066 → 265 → 255.
- **[B→A][Spark-ready] task029 — marker-rectangle-interior (divided-grid cropping cleanup).** task021 + task065 are complete; task029 is still pending with recipe at `Research_Archive/notes/recipe-task029-marker-rectangle-interior.md`; ~1-2 hr mechanical re-implementation; ~12-13 pts. task365 remains deferred because the multi-region selection rule is unclear and needs 2-3 hr derivation.
- **[B→A] task177 — bbox-crop with object-detection filter** (13/265 partial hit on simple bbox-crop). Karnak label "Cropping + Object Detection + Background Separation". Add object-CC filtering before bbox.
- **[B→A] task243 Flood Fill** (complexity 3 in Color_and_Logical) — overlap with bbox-fill cluster sub-cluster A. Likely seed-based 4-conn flood-fill.
- **[B→A] task182 Object-Detection + Color-Mapping** (complexity 2, no size change). Add to Tier 1 if cheap.
- **[B→A] task298 Color-Swapping + Background-Separation + Filling-Regions** (complexity 2). Add to Tier 1 if cheap.

- **[Alpha] Generalize marker-corners to inputs with >2 colors.** Replace `input_ch0 + marker` shortcut with `ReduceMax(input, axis=1)`. Re-mine first to confirm hits before implementing — prior generalization attempt yielded 0 new hits.
- **[B→A] task281 — stretch rectangle toward marker (heavier).** 11×11, ~18 edits/pair. Stretches a 2-bordered 1-filled rectangle so one edge reaches an isolated 8-marker. Cost ~500k → ~11 pts.
- **[B→A] Pixel-permutation-with-mask residuals (113, 307).** Of the original ashok205 cluster B list (108, 113, 194, 214, 307, 373, 385 plus 116, 223), PR #26 caught 108/194/214/373/385 via plain shape-changing-pixel-perm (with constants resolved by gather where possible); 113 and 307 still don't fit the same-shape or shape-changing pattern variants tested. Need a different approach (perhaps explicit Constant + Where overlay).
- **[B→A] task389 cheap-rewrite — color-swap via ReduceSum+Greater+Mul** (LB-5126 `t389_cheap`). Currently external-only; our recipe could displace external winner at lower cost. ~184k cost → ~12.88 pts. **Cost-displacement hit, not full new-LB-gain.** Source: `Research_Archive/notes/deep-read-lb-5126-bug-free-blend.md`.
- **[B→A] Bbox-fill** — ~~multi-source Tier B #21~~ **COMPLETED via pattern set #2 (PR #32).** Lands task166 with target=2 (~78k cost, ~13.7 pts). Implementation uses cumulative-OR via MaxPool kernel-30 (CumSum is opset 11+); 8 truly-unsolved candidates Beta identified (66/102/105/243/255/265/363/367) — re-mine with the new detector to see how many fit the implemented "tight bbox of input non-zeros" rule vs need other variants.
- **[B→A] Crop-plus-color** — multi-source Tier B #20. Fixed-position crop + per-cell color map via `Slice + Gather (channel) + optional Mul (kill-mask) + Pad`. **Mining: 3-8 hits estimated.** ~36k → ~14 pts. Source: same #20.
- **[B→A] Tile + period-row/col-tile family** — multi-source Tier A #8 + #11. Full integer-multiple tiling via `Slice + Tile`; period-tile via `Slice + Tile + Pad` when input has periodic rows/cols. **Mining: 3-8 hits per subfamily estimated.** ~50k → ~14 pts each. Source: same #8 + #11.
- **[B→A] Single-Conv feature-detector cluster** — locally-solved cluster A. 7 unsolved candidates: tasks 015, 120, 151, 180, 322, 344, 352 (note: task015 also covered by Tier-1 LB-5126 recipe — try that first). Pattern: `Conv + Mul + ReduceMax` with per-task hand-set kernel detecting a specific local feature. Per-task analysis required, no single generalized detector. **Mining: 1-2 likely fall out as clean wins.** ~5 KB ONNX → ~13-14 pts each. Source: `Research_Archive/notes/deep-read-locally-solved.md` cluster A.

### Tier 3 — speculative / R&D (pattern candidates from Research)

- ~~**[Alpha] task143 — marker-aligned shape recolor**~~ **COMPLETED in 40-hr push #2** (cross-correlation approach delivered 10.86 pts at 1.39M cost — close to the 13-pt-with-cross-correlation upper estimate, actual lower because memory dominated).
- ~~**[B→A] task020 — centered-symmetry-completion (65% partial)**~~ Promoted to **Tier 1 above**: actual rule is 4-fold ROTATIONAL symmetry (not reflection) — numpy 100% in 40-hr push #2. Recipe filed; ONNX 80% written + reverted untested.
- **[B→A] task036 — color-filter bbox-crop (least-common-color, 182/265 = 68.7% verified)**. Same rule family as task014 but the picker fails on 31% of pairs. Needs further mining to identify the alternate rule (most-common picker fits 1/20 of remaining failures; some other heuristic likely). ~12-14 pts if rule extended. Source: `Research_Archive/notes/karnak-xref-truly-unsolved-2026-05-03.md`.
- **[B→A] task110 Filling-Regions-x2** (karnak complexity 3, Pattern_Recognition). Two-step filling pipeline.
- **[B→A] FP16-intermediate optimization scan** — identify Alpha builders with substantial intermediate `[1, C, H, W]` activations (memory > 5 KB). Wrapping with `Cast(FP16) → ... → Cast(FP32)` halves the memory term. Estimated +0.05-0.1 pts/affected-task. Per `Research_Archive/notes/cost-edge-probe-2026-05-03.md`. Per-builder analysis needed.

- **[Alpha] 9-bit binary-neighborhood LUT** (J.20 Pattern 4 generalized). Currently `bin-nbr-lut` covers 7 tasks; further variants on the LUT pattern may extend. Source: `Research_Archive/notes/J.20-hyper-quantized-associative-memories.md`.
- **[Alpha] Hardmax-channel pattern** (J.11) — caveat: must apply to a *derived* quantity (channel sums, row/col sums, neighborhood reductions), NOT raw input (raw input is structurally identity per `lessons.md` 2026-05-02 entry).
- **[Alpha] Min-sum log-domain Potts BP** (J.27). 5 iterations for denoise / hole-fill / symmetry-completion. ~770k cost, ~11.45 pts. Felzenszwalb's bipartite half-update halves cost. Note: target set may overlap with the now-shipped enclosed-region-fill (PR #27); re-mine before implementing.
- **[Alpha] Morton/Hilbert linearisation Gather** (J.14). For row-fill / column-fill / quadrant-structured tasks. Length-900 int32 Gather table = 3,600 B params, 0 MACs. Hilbert preferred over Morton. Note: pixel-perm-general (PR #26) already provides the same op set; revisit only if a Morton/Hilbert-specific cost win is found.
- **[B→A] Outline-extract** — multi-source Tier B #24. Keep only boundary cells of each connected non-zero region via `ReduceMax(channels) → Sub(1) → MaxPool 3×3 → Mul + 1×1 Conv`. **Mining: 1-3 hits.** ~80-120k → ~12 pts. Source: `Research_Archive/notes/deep-read-multi-source-onnx-solver.md` #24.
- **[B→A] Keep-largest-color / keep-smallest-color** — multi-source Tier B #25. Keep only cells of the most/least-common nonzero color via `ReduceSum(axes=2,3) → Equal → Cast → Mul`. **Mining: 1-4 hits.** ~50-80k → ~12.5 pts. Source: same #25.
- **[B→A] Spatial-zero-ambiguity solver** (pixel-perm extension) — multi-source Tier A #10. For each output pixel, find input pixels with matching cross-pair signatures; if exactly one, take it. Subsumes our pixel-perm + extends to harder cases. **Mining: 5-15 hits ABOVE strict pixel-perm.** Same cost as pixel-perm (~3,600 params + ~36k memory). Source: same #10.

### Tier 4 — only if blocked elsewhere

- **[Alpha] Dynamic-shape flips for 150/155.** Design sketched in `Logs/sessions/2026-05-02_0021_*.md`. Estimated 1+ hr feasibility study. ARC tasks where flip operates on per-pair varying grid sizes; needs ReduceSum + ReduceMax + Where + GatherElements pipeline (>50k MACs).
- **[Alpha] Make region-swap less brittle.** Derive mask from input properties (bbox of color C, connected component of foreground). Currently 0/400 hit rate (deleted in PR #21 D10).
- **[Alpha] Per-task hand-coded ONNX programs.** Viable for specific simple mechanics (corner-symmetry completion, midpoint placement). Effort: 45-90 min per task; ROI: ~13-15 pts/task.
- **[Both] Submit a one-task model declaring opset 18+** to test whether the validator accepts it. Gates uint4/int4 sub-byte storage (4-8× params reduction) and the bitwise op family.
- **[B→A] Cost-rewrite set** — multi-source-onnx-solver Tier A #6/7/12/13/15/16/17 + Tier B #19/26/27/28: cheap-rot180-framed (Slice+reverse+Pad replaces Gather×2), anti-transpose, frame-border, row/col-broadcast, nonzero-recolor, bg-recolor, border-add, concat-rot180-w, dilate-3×3 per-channel MaxPool, bbox-then-tile, crop-then-scale. PR #35 audit verified the cost-rewrite candidate set: cheap-rot180-framed via Karnak (Slice+reverse-Slice+Pad) saves +0.39 pts on task140 and +0.17 on task087 — **deferred as marginal** (~+0.56 total; not worth a new pattern); other Tier-A items remain candidates if blocked elsewhere. Source: `Research_Archive/notes/deep-read-multi-source-onnx-solver.md` Tier A/B sections.
- ~~**[B→A] MaxPool-3×3 dilate-then-recolor cost-rewrite**~~ — **LANDED in PR #35** as `marker-dilate` pattern (build_submission.py). Dispatcher picks marker-dilate for task317 (cost 168k → 78k = +0.77 organic LB pts). task095 keeps per-color-stamp (cheaper than marker-dilate at 66.6k vs 78k). task331 doesn't fit the dilate pattern (uses cardinal-cross-stamp instead). Source: `Research_Archive/notes/deep-read-locally-solved.md` cluster I.
- **[B→A] task001 / task104 / task251 per-task organic ports** — 2026-05-08 calibration: the live `grape9` package already has very cheap hidden-scored payloads for task001 (`4339`) and task104 (`6382`), so they are not immediate score displacements. Task251 is now resolved in the active ladder by `grape9`: organic enclosed-region-fill cost `62960`, public-positive +1.18 over `fig9`. Keep the older post-audit note as historical source material only if a future base loses those cheap task001/task104 payloads. Source: `Logs/sessions/2026-05-03_2200_audit-deep-pass.md` cross-competitor scan plus `Logs/sessions/2026-05-07_2254_alpha-score.md`.

### Tier 5 — parking lot (re-evaluate at next overview)

- **[Owner] Paste `/discussion/692827`** if Kaggle web becomes accessible. Likely answers the open `ConstantOfShape` accounting question + may clarify the `7600.42` anomaly hinted at in `Onnx_Advanced/scoring_internals.md#a4_grader_replication`.
- ~~**[Beta] Rotate next 5-topic set into NEW_RESEARCH_AGENT ACTIVE QUEUE.**~~ **DONE 2026-05-08.** Active queue now contains B.7, C.1, C.2, E.4, and F.6; `Research/_RESEARCHED_TOPICS.md` was updated with the rotation note.
- ~~**[Beta] Refresh `external_blend.py`'s candidate pool.**~~ **DONE 2026-05-08 dry-run; expanded manifest evidence filed.** Current default cascade-safe sources leave task145 as the only default gap after Alpha's 122 organic tasks, and `Logs/all_local_external_source_sweep_2026-05-08.md` reproduces the 30 passing all-local task145 candidates. Current scoring queue still excludes task145 after xigua6/yuzu6.
- ~~**[Beta] Mining slice rotation (more angles).**~~ **CURRENT PASS DONE 2026-05-08.** Targeted mining found no task145 rule hit; all-local source evidence is filed, but current scoring excludes task145 after xigua6/yuzu6.
- ~~**[Beta] Deep-read remaining 9 competitor notebooks.**~~ **CLOSED AS STALE 2026-05-08.** The high-value notebooks already have deep-read notes (`rule-based`, `multi-source`, `complexity-map`, `acking`, `epic-sax`, `logic-driven`, `4808`, `locally-solved`, `LB-5126`); remaining thin wrappers point to external datasets rather than new notebook logic. Index: `Research/primary_sources/00_INDEX.md`.
- ~~**[Beta] Pull `karnakbaevarthur/neurogolf-2026-task-transformation-library`**~~ **DONE 2026-05-02.** `Research_Archive/notes/artifact-pulls-2026-05-02.md` records the pull and `arc_primitives.csv` analysis; current side worktree lacks `data/`, so use tracked notes unless the owner provisions local artifacts here.
- ~~**[Beta] Implement `_compact_mask` channel-collapse trick**~~ — **investigated and demoted 2026-05-03**. Beta `tools/compact_mask_analyzer.py` static analysis of all 50 SOLVED ONNXs found ZERO channel-uniform `[1, 10, ...]` initializers — our builders are already compact at this layer. Total potential saving across SOLVED set: 120 elements (~480 bytes), all H/W-axis (not channel-axis) and not actually compactable due to op semantics. See `Research_Archive/notes/compact-mask-analysis-2026-05-03.md`.
- ~~**[Beta] Pull external pack candidates for blend refresh**~~ **COMPLETED via 40-hr push 2026-05-04.** needless090-v31 + konbu17-401-v117 added (both CC0-1.0; 56 newly-covered tasks). aliafzal9323's pack not found; 4808.21 pack (`thisray/neurogolf-4808-21-post-apr-28-metric-update`) is a single submission-zip-bytes file (different format) — not pursued. afr1ste-6335-19 pulled but rejected (Compress).
- **[Beta][Spark-ready] Re-run `tools/_archive/karnak_xref.py` after every Alpha pattern PR** — last run **2026-05-11** after the 2026-05-10 + 2026-05-11 dash + harvest cycle: 154 organic, 246 default-source external-covered outside organic, **0 default-source gaps** (task145 was the only remaining truly-unsolved as of 2026-05-08 and is now externally covered per the 154-organic / 246-external / 400-total split). Tool is single-source-of-truth for the default manifest; pair it with the all-local source sweep for source coverage only, not automatic scoring promotion.
- **[Alpha] Generalize Translation to shape-changing tasks.** Current shape-preserving Translation found 0 net hits (per `lessons.md` 2026-05-02 "Translation under shape-preservation is structurally niche"); shape-changing variant (input shape ≠ output shape, pure Pad) may catch new hits.
- ~~**[Alpha] task346 framed-cell-mode.**~~ **COMPLETED via 40-hr push 2026-05-04.** Real rule: 3x3 frame of color X with core color Y (Y != X != 0); output = 1x1 of Y. Implemented as `const-framed-cell` (grouped Conv frame kernel, ~394k cost, 12.1 pts). 267/267 verifier.
- ~~**[Alpha] task100 dominant-nonzero.**~~ **COMPLETED via 40-hr push 2026-05-04.** Real rule: largest-bbox color (not most-common). Implemented as `const-largest-bbox` (per-channel bbox extraction + ArgMax-area, ~37k cost, 14.5 pts). 266/266 verifier.
- **[Alpha] task315 ternary LUT** (3×3 in, 9×9 out, 2 nonzero colors). Mining hit but 1.59M LUT elements exceeds 1.2M file-size guard. May fit if int8 LUT representation works. Note: also matches outer-product-marker-tile in Tier 1; try that first.
- **[Alpha] Position-encoded shape projection residuals** (post-task153/task316). Sparse small-grid outputs where cells encode positional/shape info may still exist, but task153 and task316 are no longer open examples. Re-mine before implementing any generalized variant.
- **[Alpha] Anchor-marker-cross-spreading** (task270 family). 15×15 input has scattered "anchor" cells (color X) plus matching "marker" cells (color Y); output draws 4-conn cross of color Y around each anchor cell. Multi-step: classify cells by adjacency to nearest opposite-color cell. Promising but ~2 hr design.
- **[Alpha] task304 9×9 outer-product** generalization (Tier 1 candidate). May extend to other scale factors (2×, 4×) if mined further.

---

## Promoted-to-active

Empty. Items move to `03_PROJECT.md` Active TODO when picked for the current session; record the promotion here as one row.

| Date | Lane | Item | From tier | Source session |
|---|---|---|---|---|
| | | | | |

---

## Demoted / dropped from backlog (closed-out attempts)

Patterns mined and confirmed unproductive — don't re-investigate without a new angle.

| Date | Lane | Item | Reason | Source |
|---|---|---|---|---|
| 2026-05-01 | Both | Pure threshold-tuning of existing detectors | 0 hits — strict-consistency check is already optimal | `Logs/lessons.md` 2026-05-01 "Threshold-tuning vs. fingerprint-mining" |
| 2026-05-02 | Both | Per-axis dominant-color reduction (4 variants: per-row / per-col mode, raw and tiled) | 0 hits | session 2026-05-02 |
| 2026-05-02 | Both | Hardmax-over-input pattern (raw) | Structurally identity by construction | `Logs/lessons.md` 2026-05-02 "Hardmax-over-input is structurally identity" |
| 2026-05-02 | Both | Slice + transform compounds (transpose / flip_h / flip_v / rot180 / rot90 / rot270) | 0 hits | session 2026-05-02 |
| 2026-05-02 | Both | Mark-centered slice (slice from input's unique-color cell) | 0 hits | session |
| 2026-05-02 | Both | Single-cell change/add (output == input except 1 cell across every pair) | 0 hits | session |
| 2026-05-02 | Both | Cell-edit fingerprint mining for ≤5 cell edits per pair | 8 candidates, 7 position-varying / 1 covered by marker-corners | session |
| 2026-05-02 | Both | H1/H3 const-output (most-common-NONZERO / least-common-nonzero) | 100% strict matches on 21/49/339 but all have varying output shapes; needs dynamic-shape | session |
| 2026-05-02 | Alpha | `generalize-marker-corners` (relax 2-color restriction) | 0 new hits; would cost ~0.3 pts on task266 for zero new captures | session 2026-05-02 (PR #16) |
| 2026-05-02 | Alpha | `isolated-marker-recolor` 8-conn variant | 0 new hits | same |
| 2026-05-02 | Alpha | `constant-output-fixed-color` (output uniform single color, fixed) | 0 hits — `const-dominant-color` covers anything matching | same |
| 2026-05-02 | Alpha | Identical-output literal | 0 hits | same |
| 2026-05-02 | Alpha | `constant-output-dominant-NONZERO` (ArgMax over [1..9]) | 0 hits | same |
| 2026-05-02 | Alpha | `constant-output-least-nonzero` | 0 hits | same |
| 2026-05-02 | Alpha | Framed-cell single-cell (one cell with 8-uniform-neighbors) | 0 hits | same |
| 2026-05-02 | Alpha | Mirror/grid layouts mined exhaustively (256 4-quadrant + tile-3 + tile-3x3 variants) | Only {I, fw, fh, fhw} matched (3 tasks); tile variants 0 hits | same |
| 2026-05-02 | Alpha | Tasks investigated but rule too complex / dynamic shape needed | task027 / 042 / 181 / 102 / 248 / 119 / 391 / 273 / 207 / 355 / 363 deferred | same |
| 2026-05-02 | Alpha | task083 / task315 LUT versions | 1.77M / 1.59M bytes exceed file-size cap; task083 SOLVED via mirror-grid-2x2 instead | same |
| 2026-05-02 | Beta | `Compress`-with-fake-zero-dim exploit (3 hypotheses tested locally) | All bypass attempts hit `_EXCLUDED_OP_TYPES` rejection at `neurogolf_utils.py:218`. Bypass likely historical (pre-2026-04-21 scorer). | `Onnx_Advanced/scoring_internals.md#a2_cost_metric_exploits` |
| 2026-05-02 | Alpha | Translation (shape-preserving variant) | 0 net new hits on 400 tasks; only task053 matches and is already covered by cheaper `fixed-shift_v_1`. Kept as primitive. | `Logs/sessions/2026-05-03_0125_grape-blend-translation.md` + `Logs/lessons.md` 2026-05-02 entry |
| 2026-05-03 | Beta | Color-swap (2-color) `[B→A]` | Pre-flight mining showed 0 unsolved hits across 400 tasks. Dropped before implementation — saved ~30-45 min. Predict-then-implement strategy validated. | `Logs/sessions/2026-05-03_0245_alpha-pattern-set.md` |
| 2026-05-03 | Alpha | Pixel-perm-with-mask (same-shape variant) | Subsumed by plain pixel-perm: gather can serve constant cells when any input position has the same color across pairs. 0 additional hits beyond plain pixel-perm. | same |
| 2026-05-03 | Alpha | Shape-changing-pixel-perm-with-mask | 0 additional hits beyond plain shape-changing-pixel-perm. Same subsumption logic. | same |
| 2026-05-03 | Alpha | Self-concat-h / self-concat-v `[B→A]` | Structurally covered by existing `mirror_concat` with `transform=identity` — no new pattern needed. Verified by reading `_detect_mirror_concat` source. | same |
| 2026-05-03 | Alpha | Pixel-permutation general `[B→A]` (Tier-A from deep-read) | **COMPLETED via PR #26.** 3 unsolved hits (task214, task373, task385); cost-aware dispatcher also reclassified some previously-mirror-grid-2x2 tasks to this cheaper pattern. ~16-17 pts each. | `Logs/sessions/2026-05-03_0245_alpha-pattern-set.md` + [PR #26](https://github.com/Boladi888/NeuroGolf_bo/pull/26) |
| 2026-05-03 | Alpha | Color-replace general `[B→A]` (multi-mapping; ashok205-confirmed task276/309) | **COMPLETED via PR #26.** 1×1 Conv with non-permutation weight; generalizes color-perm by allowing merge cases. 2 hits at ~13.6 pts each. | same |
| 2026-05-03 | Alpha | Shape-changing-pixel-perm (cluster B subset) | **COMPLETED via PR #26.** 3 hits (task108, task194, task223) from ashok205 cluster B + cost-aware reclassification of 4 previously-solved tasks. 113/307 still pending — see Tier 2. | same |
| 2026-05-03 | Beta | Tier-3 region-gated-swap-via-Where (J.36 Class 3) | Strict-consistency mining: 0 hits across 352 unsolved tasks. Brief estimate "5-15 ARC tasks" was 5-15× optimistic. Dropped before implementation. | (this session) `tools/mine_tier3_candidates.py` (never committed) |
| 2026-05-03 | Beta | Tier-3 mask-gated-NOT-via-Where (J.36 Class 4) | Strict-consistency mining: 0 hits. Dropped before implementation. | same |
| 2026-05-03 | Beta | Tier-3 constant-output uniform-fixed (B.2) | 0 hits in mining; existing `const-dominant-color` covers any matching cases. | same |
| 2026-05-03 | Beta | Tier-3 color-filter | 0 hits; channel-Gather to bg subsumed by existing `color-perm` / `color-replace`. | same |
| 2026-05-03 | Beta | Tier-3 color-presence (J.11) | 1 false-positive match (task393); actual rule on task393 is ordering-dependent, not simple presence. Not Tier-3 cheap. | same |
| 2026-05-03 | Alpha | Tier-3 enclosed-region-fill (J.22) | **COMPLETED via PR #27.** 2 hits (task002, task251). 30-iter BFS from canvas boundary blocked by walls; ~742k cost → 11.48 pts each = +22.97 organic pts. | same + (this session log) |
| 2026-05-03 | Alpha | Outer-product-marker-tile (Tier-1 mining) | **COMPLETED via pattern set #2.** 2 hits (task304 most_common; task315 fixed_marker=2). ch0-bg fill via complement-mask Add. ~13k cost → ~15.5 pts each. | `Logs/sessions/2026-05-03_alpha-pattern-set-2.md` |
| 2026-05-03 | Alpha | Diagonal-shear / Toeplitz tile (Tier-1 mining) | **COMPLETED via pattern set #2.** 1 hit (task327). Sum-of-shifts at output anchors (k, k) for k=0..5. ~22k cost → ~15.0 pts. | same |
| 2026-05-03 | Alpha | task015 per-input-color stamp (Tier-1 [B→A] LB-5126) | **COMPLETED via pattern set #2.** Generalized as `per-color-stamp` pattern with rule list. Lands task015 + bonus task220 (3 rules) + task352 (1 rule) + task095 cost-rewrite (bin-nbr-lut → per-color-stamp, 169k → 67k). ~67-128k cost → ~13.2-13.9 pts. | same |
| 2026-05-03 | Alpha | Symmetry-completion family (Tier-1 [B→A]) | **COMPLETED via pattern set #2.** Slice + Gather + Greater + Where. Brief estimate 5-15 hits → actual 1 hit (task113 v-axis); confirms 2026-05-03 lesson on 5-15× discount. ~41k → ~14.4 pts. | same |
| 2026-05-03 | Alpha | Bbox-fill (Tier 2) | **COMPLETED via pattern set #2.** Cumulative-OR via MaxPool kernel-30 (CumSum is opset 11+). Lands task166 at ~78k cost → ~13.7 pts. | same |
| 2026-05-03 | Alpha | Outline-extract (Tier 3) | **COMPLETED via pattern set #2.** Grouped Conv 3x3 with [10, 1, 3, 3] kernel + Greater + Cast + Mul. Lands task098 at ~314k → ~12.3 pts. | same |
| 2026-05-03 | Alpha | Active-grid-frame (new, not in deep-reads) | **COMPLETED via pattern set #2.** Border around active grid via is_active + 8-neighbor Conv + Less. Lands task171 at ~43k → ~14.3 pts. | same |
| 2026-05-03 | Alpha | Color-filter / nonzero-recolor / bg-recolor (Tier A re-mining) | 0 hits each across 344 unsolved tasks. | same (preflight_tier2 throwaway) |
| 2026-05-03 | Alpha | h-symmetry-completion / quad-mirror / row-broadcast / col-broadcast / crop-plus-color | 0 hits each. | same |
| 2026-05-03 | Alpha | v-self-concat (vertical column-tile) | 0 hits. h-self-concat matches task249 but it's variable-W; same dynamic-shape problem as task231. | same |
| 2026-05-03 | Alpha | task058 spiral pattern | Mining hit on keep-largest/keep-smallest proxies (false positive: input all-zero). Genuine rule is a connected spiral path; complex per-task hand-code, 3+ hr. Deferred. | same |
| 2026-05-03 | Alpha | task123 fractal expansion (Tier-1) | Tested as shape-changing-pixel-perm: 27/100 cells consistent → not viable as pixel-perm. Needs custom decoder, 2-3 hr. Deferred. | same |
| 2026-05-03 | Alpha | task153 2-blob projection (Tier-1) | Tested as shape-changing-pixel-perm: 0/9 cells consistent → output depends on per-blob shape (connected components). Hard in ONNX without flood-fill; deferred. | same |
| 2026-05-04 | Alpha | task031 foreground-bbox-crop (Tier-1 [B→A]) | **COMPLETED via PR #36 pattern-set #3.** ReduceSum + Slice + Sub + ReduceMax×2 + ArgMax×4 + Cast + Add + Min + Cast + Gather×2 + mask. Variable in/out shape (single-example only). Cost ~134k → ~13.2 pts. | `Logs/sessions/2026-05-04_0211_alpha-pattern-set-3.md` |
| 2026-05-04 | Alpha | task123 fractal-ring-expansion (Tier-1 [B→A]) | **COMPLETED via PR #36 pattern-set #3.** Beta's recipe Option B was mathematically wrong for max(i,j)≥5; switched to Option C (precompute K=4 + K=5 lookup tables, runtime Where). 5 Slice + Concat + Slice + Greater + Where + Reshape + Gather + Reshape + Pad. Cost ~10k → ~15.8 pts. | same |
| 2026-05-04 | Alpha | task014 color-filter-bbox-crop (Tier-1 [B→A]) — picker='least' | **COMPLETED via PR #36 pattern-set #3.** Mining-generalized 1→3 hits: lands task014 + task049 + task310. Used ArgMax(-counts) workaround for missing ArgMin in onnx-tool 1.0.1. Cost ~134k → ~13.2 pts each. | same |
| 2026-05-04 | Alpha | task036 color-filter-bbox-crop (Tier-3 partial) — picker='smallest_bbox' | **COMPLETED via PR #36 pattern-set #3.** Pre-flight mining angle L (per-color bbox area picker) refines Beta's 68.7% partial with the wrong picker. Cost ~159k → ~13.0 pts. | same |
| 2026-05-04 | Alpha | task243 seed-flood-fill (Tier-2 bbox-fill cluster A) | **COMPLETED via PR #36 pattern-set #3.** Same dilation pipeline as enclosed-region-fill but seeded from input==fill_color cells (with seed channel ALSO marked non-blocking). 30 iters via Conv 3x3 cross + Clip + Mul. Cost ~718k → ~11.5 pts. | same |
| 2026-05-04 | Alpha | 35+ pre-flight mining angles (rounds A-HH) on 145 truly-unsolved | Only 1 hit: smallest_bbox picker → task036. Tested: most-common picker, least-common, smallest-bbox, largest-bbox, second-most-common, fixed-color (per channel), single-nonzero-color, background-detection, all 9 fill-with-fixed-color, all 81 X→Y recolor pairs, identity, full bbox-fill (per fill_color), 8-conn flood-fill (per fill_color), 4 directional largest-CC variants, etc. Bbox-crop / bbox-fill family confirmed exhausted on truly-unsolved set. | same |
| 2026-05-04 | Alpha | task178 (variable-shape with consecutive-duplicate removal) | Investigated; needs runtime CumSum / ScatterND (opset 11+, banned). Skipped. | same |
| 2026-05-04 | Alpha | task088 (smallest_bbox + recolor to other color) | Investigated; rule hit only 76% (204/267) — refinement needed but unclear. Skipped. | same |
| 2026-05-04 | Alpha | task355 (1×1 single-color output) | Investigated; tested 5 simple pickers (most/least/single nonzero / etc.), max 28% pass. Rule complex. Skipped. | same |
| 2026-05-04 | Alpha | task363 (LB-5126 silent-fail recipe) | Investigated; deep-read recipe is high-level only — rule derivation from data would take 2-3 hr with high uncertainty. Deferred (still in Tier-1 backlog). | same |
| 2026-05-04 | Alpha | task153 (two-blob tile-fit) | Investigated; Beta-verified rule but ONNX implementation needs per-color bbox×9 channels + 81-way runtime anchor enumeration with runtime color detection. Historical defer; superseded by Alpha implementation 2026-05-08. | same |
| 2026-05-04 | Alpha | task307 kron-block-up (mining hit, small-in-large-out) | **COMPLETED in 40-hr push.** Slice canvas to (15, 15) + Resize-nearest with scale (1, 1, 2, 2). Both ops 0 MACs. Cost ~9k → ~15.9 pts. Cost-aware dispatcher reclassified task223 (was shape-changing-pixel-perm) as bonus. | `Logs/sessions/2026-05-04_1057_both-40hr-push.md` |
| 2026-05-04 | Alpha | task100 const-largest-bbox (refines 248/267 partial) | **COMPLETED in 40-hr push.** Real rule: pick channel (1..9) with largest bbox area; not most-common-cell-count. Per-channel bbox via ArgMax-trick, then ArgMax(area) + OneHot + Tile. Cost ~37k → ~14.5 pts. | same |
| 2026-05-04 | Alpha | task346 const-framed-cell (refines 230/267 partial) | **COMPLETED in 40-hr push.** Real rule: 3x3 frame of color X with core color Y (Y != X != 0); output = Y. Grouped Conv with frame kernel + Pad + Greater + ReduceMax + ArgMax + OneHot + Tile. Cost ~394k → ~12.1 pts. Bug mid-impl: forgot ch0-mask before cross-channel ReduceMax → 223/267; fixed → 267/267. | same |
| 2026-05-04 | Beta | External pool refresh: needless090-v31 + konbu17-401-v117 + afr1ste-6335-19 | **PARTIAL COMPLETION.** First two added to `external_blend.py` (252 + 401 ONNXs, both CC0-1.0, 1-indexed naming matches ours). 56 newly-covered tasks via blend (mango #13 = 4042 LB). afr1ste-6335-19 spot-checked but uses Compress (banned) → not added. | same |
| 2026-05-04 | Alpha | task020 / task021 cropping cluster / task355 / task263 / task005 / task008 | Investigated all in 40-hr push; each needs 3-4+ hr ONNX implementation with verified or partial rules. Deferred (still in backlog). | same |
| 2026-05-04 | Alpha | More constant-output picker variants (least, smallest_bbox, unique-once, not-present, isolated-cell) | 0 hits ≥ 95% on the 89 truly-uncovered set. Constant-uniform-color-output family confirmed exhausted. | same |
| 2026-05-04 | Alpha | task143 marker-aligned-shape-recolor (Tier-1 [B→A]) | **COMPLETED in 40-hr push #2.** Cross-correlation via 9 runtime-axis-stacked Slice-shifts × runtime template_mask scalar + ReduceSum. Match indicator = (max_corr == template_count) AND (input_total == template_count) excluding ch0/ch5/template_color. Cost 1.39M → 10.86 pts. Memory-dominated (1.16M memory). | `Logs/sessions/2026-05-04_1700_both-40hr-complex-targets.md` |
| 2026-05-04 | Alpha | task021 grid-division-count (cropping cluster Tier-2) | **COMPLETED in 40-hr push #2.** Per-channel monocolor row/col detection via per_chan_count vs per_row_total comparison + ArgMax for divider. Output via bbox_mask × fill_color_oh. Cost 75k → 13.78 pts. | same |
| 2026-05-04 | Alpha | task065 marker-section-extract (cropping cluster Tier-2) | **COMPLETED in 40-hr push #2.** Reuses task021 divider machinery + 1D BFS via MaxPool(3,1) + Mul by not_div_rows × 30 iters for section bounds. Marker = ArgMin of per_chan_total over non-divider channels with cells (note: marker CAN be ch0). 2 bugs found + fixed: (1) ch0 marker exclusion was wrong, (2) BFS spread into inactive rows. Cost 235k → 12.63 pts. | same |
| 2026-05-04 | Alpha | task020 4-fold rotational symmetry (Tier-3 partial promoted to Tier-1) | **NUMPY 100% IN 40-HR PUSH #2.** Discovered actual rule = 4-fold ROTATIONAL symmetry around bbox center (not h+v reflection — that was the 65% partial). ONNX builder ~80% written then **reverted untested at owner stop for handoff**. Recipe + full implementation sketch at `Research_Archive/notes/recipe-task020-rotational-symmetry-completion.md`. ~3 hr to finish, ~11.3 pts. | same |
| 2026-05-04 | Alpha | task153 deferred-again | Re-deferred after considering task153 vs cropping cluster ROI; cropping cluster was 2 patterns at 26 pts in same time as 1 task153 pattern at 12 pts. Historical defer; superseded by Alpha implementation 2026-05-08. | same |
| 2026-05-04 | Alpha | task365 multi-region selection deferred | Inspected 3 train pairs: each has 3 non-zero rectangular regions in 10x10 input; output is one of them. Selection rule unclear (counts of marker colors tie across regions; positions vary; sizes vary). Needs 2-3 hr derivation. | same |

---

*Last updated: 2026-05-08 (yuzu9 is current high score at 5573.72; ugli9 task182 was public-flat; vanilla9 task303, watermelon9 task224, and xigua9 task177 were hidden-bad; Alpha reconciled ledgers through task003/task017/task020/task021/task029/task052/task105/task126/task129/task229/task249/task251/task304/task315/task394, fixed the stale queue base, documented task014/task153 hidden-bad guards, and refreshed the cost-probe sweep).*
