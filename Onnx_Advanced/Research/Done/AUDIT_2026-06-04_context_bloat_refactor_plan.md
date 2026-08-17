# AUDIT 2026-06-04 — Onnx_Advanced context-bloat + refactor plan

**What this is**: full audit of `Onnx_Advanced/` (~1.7 MB, ~150 files) against the folder's charter — *"only info a Claude-Sonnet-class model doesn't already know."* Commissioned by the owner; executed by 11 parallel auditor agents (every file read in full) + spot-check verification of all load-bearing claims. **This is an internal audit deliverable, not a ChatGPT research card.**

**Rubric used** (every file scored): **A** keep (paid-for empirical fact: measured costs, ORT-version behavior, validator bans, scoring internals, task outcomes) / **B** compress (real signal in verbose wrapping) / **C** generic (Sonnet answers it cold — cut to a pointer) / **D** debris (superseded, already-integrated, process-log).

---

## 1. Headline

| Metric | Today | After refactor |
|---|---|---|
| Active tree (worker-visible) | ~1.7 MB | ~600 KB |
| Every-task required set | 127–290 KB (depending which spec a worker follows) | **~65 KB** (one spec) |
| Overall content split | ~35% A / 20% B / 20% C / 25% D | ~85% A |
| Entry points | 3 (INDEX, README, QUICKSTART — overlapping, drifting) | 1 (INDEX) |
| Known cross-file contradictions | 4 confirmed (see §2) | 0 |

The library is **better than feared in places** (by_symptom.md, Ort_Advanced/, exemplars/, algorithm_patterns.md are 70–90% A — these are the crown-jewel assets) and **worse than feared in others**: the single biggest problem is not generic-knowledge bloat but **process-log accretion and append-only supersession** — ship ledgers, dated incident diaries, and "ADDENDUM/UPGRADE" blocks stacked on top of text they obsolete. Second biggest: ~300 KB of generic ONNX/ARC tutorial (the op-reference shelf, the two ARC taxonomies, the ChatGPT deliverable preambles). Both are fixable without losing a single empirical fact.

**Read-frequency matters more than raw size.** Consumption-model mapping (playbooks, live prompts, task files) shows workers are INDEX-routed and §-cite the big files rather than reading cover-to-cover. So the costliest KB are in the *every-task set* (core, SPEC, cost_floor, traps, task_techniques_index) — that's where eviction pays per-task, every task. `Research/`, `.update/`, `CHANGELOG.md` are already outside worker context; cleaning them is hygiene, not context relief.

---

## 2. Correctness bugs found (fix regardless of any refactor)

These are drift, not bloat — they actively mis-teach workers. All verified by direct read, file:line cited.

1. **MACs ghost**: `scoring_internals.md:29,135,562` still defines `cost = macs + memory + params`. MACs were removed from scoring 2026-05-04 (`core.md:24`, confirmed by `00_COMPETITION_SPEC.md:5` banner). The entire A.1 section (op-by-op MAC table, "sub-9k-MAC floor" reasoning) prices a dead term. ~10–12 KB of obsolete framing in the cost-golf ground-truth doc.
2. **OneHot contradiction**: `QUICKSTART.md:26` says "OneHot needs INT64, not INT32"; `traps.md:128-129` corrected this ("the old INT64-only rule was a task124 v3 misdiagnosis — INT32 indices work when depth dtype matches"). The duplicated hard-rules copy went stale — the canonical fix is to stop duplicating rules in entry files.
3. **Taxonomy family-count swap**: `INDEX.md:42` and `by_symptom.md:13` call `arc_rule_taxonomy.md` "the 15-family taxonomy" — it declares **18** families (its line 3, 707); the *alt* is the 15-family one. Routing labels are swapped.
4. **techniques.md § routing is broken**: duplicate top-level numbers confirmed — `# 23.` at lines 1279 *and* 1556, `# 40.` at 2395 *and* 2472, `# 41.` at 2440, 2501, *and* 2586; plus four literal unnumbered `# N.` headers (2193, 2213, 2246) and out-of-order runs (§25→§22→§23 after §31). An HTML comment at :2434 admits "three workers independently used # 41 today." Workers grep by §number per INDEX instructions — `grep "^# 41\."` returns three unrelated techniques.
5. **Actively-wrong numbers preserved in-tree**: `00_COMPETITION_SPEC.md:297` carries the wrong scoring guess `max(1, 2500-length)` and the obsolete `0.001` floor — inside the 65% of the file that its own banner (line 153) says "Do not act on... retained for historical research only." It is in the every-task read set anyway.
6. **Two near-duplicate trap entries**: `traps.md` §21 (~:1107) and §26 (~:1156) document the same crop_to_size_advisory false-positive with the same six task IDs — merge.

---

## 3. What actually costs context (consumption model)

- Workers are told three different "required" sets: INDEX "Start here" (~127 KB), README Tier-1 (~227 KB, dominated by traps.md 141 KB billed as "5 min"), prompt template Tier-0+1 (~290 KB). Live prompts are leaner — they §-cite into techniques/traps. **Pick ONE spec (INDEX) and make it small.**
- The scoring formula is restated ~25× repo-wide; the banned-op list 7×; the hard-rules block 3× near-verbatim (README:58-68, QUICKSTART:71-81, core.md:234-242 — which is how bug #2 happened).
- `traps.md` (1,447 lines, ~55 distinct traps ≈ 26 lines/trap vs the ~4 a trap needs) cannot be "scanned in 5 min" as QUICKSTART instructs; lines 521–1029 are a dated incident diary whose lessons were already consolidated into §27.
- `Research/` (~280 KB), `.update/` (~61 KB), `CHANGELOG.md` (~57 KB) are already excluded from worker reading — their cleanup frees repo hygiene + grep noise, not per-task tokens.
- Blast radius for any move/rename: **456 files, ~1,100 refs** (368 are `Task_by_Task/*.md`, largely auto-generated). `tools/refactor_paths.py` is purpose-built for this (anchored-path rewrite, dry-run default). Two generators re-emit library paths and must be re-run after moves: `tools/agent_helpers/regenerate_graph_diagnostics.py`, `tools/_build_task_techniques_index.py`.
- **§-number hazard**: hundreds of task files + live prompts cite `techniques.md §37/§39/#44-47` and `traps.md §14e/§27`. Therefore: fix *colliding* numbers only (assign fresh numbers, never reuse retired ones); do NOT mass-renumber.

---

## 4. Per-file disposition table

Verdicts: **KEEP** (as-is / light trim) · **TRIM** (rewrite in place to target size) · **MERGE→X** · **SPLIT** · **ARCHIVE** (move to `_archive/`, out of worker tree) · **DELETE**.

### 4a. Every-task set (highest leverage)

| File | KB | A% | Verdict → target |
|---|---|---|---|
| `core.md` | 25 | 70% | **TRIM → ~18 KB.** Stays the worker bible + becomes the ONLY home of: scoring formula, banned ops, hard-rules block, calibration pointers. Cut: 41-line adversarial-gate essay → 6-line verdict; spread-tooling ops (→ diagnostics_recipes); key-paths process roster (~40%); sparse-proof one-liner + pointer. |
| `00_COMPETITION_SPEC.md` | 48 | 22% | **SPLIT + ARCHIVE.** Lines 153–443 (self-disowned draft) → `_archive/`. Worker facts (~7 KB: caps, encode/decode, bans, pinned versions, cost-lever probes) → fold INTO core.md. Captain rules (deadlines/prizes/licensing/submit mechanics, ~7 KB) → `The_400_Tasks/` captain doc. File disappears from worker set. |
| `cost_floor_reference.md` | 44 | 40% | **TRIM → ~22 KB.** Evict "Shipped floors" dated batch logs (lines ~291–444, ~13 KB, pure ship-ledger) → CHANGELOG/per-task files. Compress 50-datapoint calibration dump → 4-line regime table + pointer to core.md. Byte table → formula + 3 anchors (KEEP the scalar-itemsize row — that one is empirical). KEEP verbatim: plane-floor taxonomy, cost-per-op table, floor verdicts, decision tree. |
| `traps.md` | 141 | 33% | **SPLIT.** New `traps.md` = ≤200-line trap TABLE (one row per trap: pattern/symptom/fix/ref) + the ~6 irreducible blocks (§8b-f score→None cluster, reduce-dtype matrix, cascade trigger set). Incident archaeology worth keeping (task110 saga, task233, Einsum-tiebreak root cause) → `traps_appendix.md` (grep-only, ~30 KB). Ship-decision block (lines 12–81) → `current_strategy.md` (it's captain strategy; it even cites its real home). Dated diary lines 521–1029 + cherry13b campaign ledgers → appendix or delete (lessons already consolidated in §27). |
| `task_techniques_index.md` | 62 | 55% | **TRIM → ~45 KB.** Strip the 77-line dated "submit-confirmed lessons" prose header (19% of bytes — 38 tasks' battle reports every worker pays for) → one-line-per-task ledger or `recent_ships.md` outside the set. De-dup the 47 prose entries to router one-liners. Run link-liveness sweep on ~177 `build_*.py` pointers (file self-flags staleness). |
| `INDEX.md` | 6 | 80% | **KEEP — promote to SOLE entry point.** Absorb QUICKSTART's 10-min path + decision tree. Fix the 15/18 family-count swap (bug #3). Add per-file size budgets. |
| `QUICKSTART.md` | 4 | 35% | **MERGE→INDEX, delete.** Fix bug #2 in passing (its hard-rules copy dies with it). |
| `README.md` | 4 | 20% | **Reduce to 5-line stub** → "start at INDEX.md". Tier tables duplicate INDEX; hard-rules block duplicates core.md. |

**New every-task set: INDEX (~7) + core (~18) + cost_floor (~22) + traps table (~10) + your taskNNN.md ≈ 65 KB ≈ 16k tokens.** Down from 127–290 KB. The README/template/playbook reading instructions must be updated to say exactly this.

### 4b. The toolkit (grepped per-section — keep, repair, compress)

| File | KB | A% | Verdict → target |
|---|---|---|---|
| `techniques.md` | 208 | 50% | **TRIM + REPAIR → ~100 KB** (Phase 3, see §6). Fix § collisions (fresh numbers for the dup #23/#40/#41s + the four `# N.` placeholders; never reuse old numbers). Hoist a 10–15 KB verdict-index front-matter (ORT dtype×opset matrix pointer, grader cost mechanism, F16-2048 ceiling, zero-weight task list, cascade slots). Evict §0 ship-log table (54 rows → current_strategy.md / task files). Collapse the ~14 §39–42 ADDENDUM/UPGRADE blocks into one consolidated verdict + one gap-ledger table (final MECHANISM CONFIRMED supersedes the rest). Delete exact dup §23(=§24); merge §17d/§17e into §5/§10. Compress case-study narrative to pattern→rewrite→safety→measured→exemplar. |
| `by_symptom.md` | 29 | 85% | **KEEP.** The gold-standard format (symptom → fix → exemplar → confirmed delta). Trim only lines 50–79 ("By LIVE architecture signature" restates traps §14e/§27 — make it a pointer). |
| `scoring_internals.md` | 68 | 30% | **TRIM → ~30 KB, stays on-demand.** Purge MACs-scored framing throughout (bug #1). Archive competitor-exploit decode + onnx-tool bug history (~10 KB of "decoded then rejected; do not exploit"). Grader-replication recipe → 3-line pointer to `tools/_score_v2.py`. KEEP: Where/OneHot=0 verdicts, kq5y `shape_infer(None)` gap, 2026-04-28 metric-update date, #19594 anomaly. |
| `cascade_strategy.md` | 35 | 18% | **TRIM → ~7 KB worker stub** (threshold model T≈250–300k, cherry13b protocol, Rule 1 submit-singly, raw-log calibration, "1 confirmed cascade event ever"). The 140-line submission diary + stale 2026-05-20 watchlists + 4× repeated +105-LB retraction → `Logs/` captain campaign log. |
| `diagnostics_recipes.md` | 9 | 80% | **KEEP.** Trim 2 duplicated blocks (synth_multicolor python dup of traps §14d; 0.48× note dup). |
| `patterns.md` | 7 | 85% | **KEEP** (verified: 25 rows = exactly 25 `pass_*.py` files; author gotchas are battle-scars). |
| `empirical-cost-probes.md` | 6 | 35% | **Fold ~1.5 KB** (memory-additivity correction + one Conv validation row) into cost_floor_reference; **archive the J-cluster** (60% of file; zero inbound references; all-negative verdicts on a dead research thread). Delete file. |

### 4c. Op-reference shelf (the confirmed generic-bloat zone, ~120 KB → ~35 KB in 3 files)

Per-file: `onnx_op_specs.md` (65% C — ONNX schema reprint) **DELETE**; `einsum.md` (60% C tutorial) **DELETE** after promoting the task191 F16-Einsum-not-byte-equal finding; `graphsurgeon.md` (75% C API tour) **DELETE**; `onnx_utilities.md` (65% C API tour) **DELETE** after promoting 3 facts (functions-banned dup, value_info-counts-memory dup, version_converter caveat); `spatial_rearrangement.md` (50% C) **DELETE** after salvaging 4 one-liners; `mod_and_bitwise.md`, `multi_axis_indexing.md`, `uncommon_ops_catalog.md` → **MERGE** A/B residue into new **`op_cost_patterns.md`** (~10 KB: cost-priced patterns + opset-boundary insurance lines, ZERO spec reprints — dtype questions point at the matrix).

**Legacy ORT shard docs → consolidate into one `ort_reality.md` (~30 KB, ~90% A)** — the library's most valuable content: the dtype matrix, quirks, error→fix table, and the non-duplicate opset-support material. `scoring_internals.md` + `Onnx_basics_worth_reviewing.md` (9 lines, 90% A — the best file in the library pound-for-pound) → merge into a `scoring_reality.md` section or into the trimmed scoring_internals. All spec-reprint dtype tables elsewhere become pointers to the matrix.

**Algorithm_Library/** — KEEP all 3 (algorithm_patterns 60% A core IP — trim "for completeness" entries, cross-link its misfiled ORT facts to ort_reality instead of duplicating; rule_discovery_methodology → keep gates/bands/trap-cases, cut ~80 lines of generic python; per_family_starting_points → keep, point BOOL accept/reject list at the matrix).

### 4d. Rule-decode + build-pattern docs (ChatGPT deliverables in worker tree)

| File | KB | Extracted to techniques.md? | Verdict |
|---|---|---|---|
| `arc_rule_taxonomy.md` | 46 | n/a | **REWRITE → ~10 KB** project-calibrated taxonomy: spine = the 23-task empirical table + 6 NeuroGolf-only families (from arc_taxonomy_empirical_coverage.md), one-line-per-family reference, ONE decision tree, salvaged task-ID facts. ~90% of current content is textbook ConceptARC with self-admitted "inferred, not measured" ONNX sketches + `citeturn` artifacts. |
| `arc_rule_taxonomy_alt.md` | 42 | n/a | **DELETE.** Head-to-head (per empirical_coverage) it wins on only 3 of 23 tasks — absorb those 3 family rows. 42 KB for 3 rows is indefensible. |
| `arc_taxonomy_empirical_coverage.md` | 10 | n/a | **MERGE** into the rewritten taxonomy (it becomes the spine). |
| `tiny_architectures.md` | 63 | NO | **TRIM → ~25 KB.** Cut textbook blocks (boolean-via-min/max, 40-row activation menu, Sobel kernels printed twice, weight-sharing literature). KEEP: depthwise 10× cost table, power-iteration dilation-doubling (O(log) flood), unroll cost law, pad-color-10 sentinel rules, Gather color-perm baseline. |
| `mlp_rule_extraction.md` | 42 | NO — addendum unique | **COMPRESS → ~5 KB.** Keep the 2026-05-25 addendum (RBF-hash variant voids Marabou; hash-MLP fingerprint; probe-first advisory) ~verbatim; stub the 263-line generic interpretability survey; strip ~30 bare URLs + `citeturn` litter. **Promote the hash-MLP fingerprint to techniques.md FIRST** (it exists nowhere else). |
| `synth_oracle_strategies.md` | 47 | practice embedded | **COMPRESS → ~8 KB.** DELETE lines 266–418 (verbatim reproduction of its own assignment card). Keep: 4-axis fidelity rubric, self-consistency-trap mitigation, evolutionary-search recipe, diagnostic table. |
| `cellular_automata.md` | 19 | **YES (fully, §1935-1960)** | **TRIM → ~9 KB** backing-store (formulas + priced tables; strip citeturn walls, CA-theory recaps, 3 of 4 code blocks). |
| `symmetry_and_affine.md` | 17 | NO | **TRIM → ~11 KB.** Mostly priced constructions (suspect flag only partly right). **Promote first**: GridSample-vs-Transpose+Slice priced tradeoff + BOOL-incompat verdict, boolean-safe symmetry scaffold, square-gate trap. Cut D4 theory preamble + duplicated snippet appendix. |
| `multistage_key_first_compositional.md` | 24 | YES (5 Wins) | **ARCHIVE** (worked example self-flagged speculative; zero task refs). |
| `radius_encoded_weighted_cover.md` | 24 | YES (6 Wins) | **ARCHIVE or 15-line stub** (family shipped task118 +15.51 via the extraction, not the file). |
| `direct_detection_multi_qlinearconv.md` | 24 | YES (6 Wins + examples) | **ARCHIVE** (examples double-tagged SPECULATIVE in file + Todo README). |
| `row_signature_rank_reduction.md` | 27 | YES (7 Wins incl. the uint32-limbs correction) | **ARCHIVE.** |
| `function_proto_subgraphs.md` | 23 | correctly NOT | **DELETE.** Validator ban is categorical (core.md:59-69 three independent gates; file's own DEAD-END banner; Todo README declined list; task255 "not usable"). The why-dead note survives at `Research/notes/b3_..._critique`. |
| `zero_param_generators.md` | 9 | cross-ref'd | **TRIM → ~5 KB** (the one native keeper: priced Trilu/Range/ConstantOfShape wins; stub EyeLike per J2's 34×-better diagonal-read finding; resolve-or-delete the [NEEDS EMPIRICAL] TODO block). |

### 4e. exemplars/ (~150 KB, healthy — the format to imitate)

**KEEP the tree** (avg 69 lines/file, before→insight→after→measured-delta format, high by_symptom reuse). Actions: dedup task117's twice-pasted surgery script (~30 lines); task384 → 15-line pointer at task117 (~50 lines); task366/task280 → cut the 3-4× restated LB-postmortem/adversarial caveats (~40 lines); task266/task167 (08_gather) → 10-line stubs (near-generic); strip category READMEs' "Decision flow" sections (~30% each — duplicates by_symptom routing; by_symptom owns ALL symptom routing).

### 4f. Non-worker tiers (hygiene, not context)

| Target | Verdict |
|---|---|
| `.update/` 16 DONE files | 15/16 verified integrated → **ARCHIVE**. **EXCEPTION — promote first**: `DONE_2026-06-01...task013...md` — its 3 named techniques (`aspect_ratio_branch`, `modgather_periodic_fill`, `broadcast_and_dual_orientation`) never landed in techniques.md/by_symptom.md (verified absent; only the cost datapoint + traps §14c made it). Keep `.update/README.md` (live protocol). |
| `CHANGELOG.md` 57 KB | **KEEP** (sorter ledger, outside worker context; optionally split pre-June to `CHANGELOG_archive.md`). |
| `Research/` root deliverables (7 files, 149 KB) | **ARCHIVE** — all 7 extractions verified present in techniques.md (§37-ADD/§39-UPG/§40/#44/#45/#46/#47) at ~85-90% decision-content fidelity; originals are the provenance layer. |
| `Research/notes/` | **ARCHIVE**: b1 attempt1+2 (67 KB failed deliverables), synth_oracle_partial_audit, research_card_cull. **DELETE**: arc-agi-1-spec, arc-agi-1-task-families, arc-task-families (generic; salvage the ~5 task-ID lines into the rewritten taxonomy). **KEEP**: k_arc2_cross_reference (Todo README points at it), b3_function_proto critique (guards a dead line), Research/README.md. |
| `Research/Todo/Done/` 8 card bodies | **ARCHIVE** (README's Done/Declined tables are the operational guard; bodies are template-corpus only). Also: de-dup the stray repeated "Card bodies: all delivered…" lines and fragmented tables at the bottom of `Todo/README.md`. |

---

## 5. Target structure (end state)

```
Onnx_Advanced/
├── INDEX.md                  ~7 KB   sole entry: 10-min path + routing table + size budgets
├── core.md                   ~18 KB  worker bible (scoring/encoder/validator/rules — THE canonical home)
├── cost_floor_reference.md   ~22 KB  floors + per-op costs (ledger evicted)
├── traps.md                  ~10 KB  the ≤200-line trap table (5-min scannable, for real this time)
├── traps_appendix.md         ~30 KB  incident archaeology (grep-only)
├── by_symptom.md             ~27 KB  router (unchanged role)
├── task_techniques_index.md  ~45 KB  router one-liners (prose header evicted)
├── techniques.md             ~100 KB renumber-repaired, verdict-first, ship-log evicted
├── ort_reality.md            ~30 KB  dtype matrix + quirks + errors + opset (Ort_Advanced merged)
├── scoring_internals.md      ~30 KB  MACs purged; + scorer-mechanics merge (ort_profiling, Onnx_basics)
├── op_cost_patterns.md       ~10 KB  priced op patterns, zero spec reprint (shelf salvage)
├── arc_rule_taxonomy.md      ~10 KB  project-calibrated (empirical spine; alt deleted)
├── cascade_strategy.md       ~7 KB   worker stub (campaign log → Logs/)
├── patterns.md, diagnostics_recipes.md, zero_param_generators.md   (kept, trimmed)
├── tiny_architectures.md ~25 | mlp_rule_extraction.md ~5 | synth_oracle_strategies.md ~8
├── symmetry_and_affine.md ~11 | cellular_automata.md ~9
├── Algorithm_Library/        3 files, trimmed
├── exemplars/                tree kept; 5 files deduped; READMEs = file lists only
├── README.md                 5-line stub → INDEX
├── CHANGELOG.md, .update/ (README + inbox), Research/ (intake role unchanged)
└── _archive/                 everything ARCHIVE-verdicted above (greppable, never auto-loaded)
```

Worker-visible ≈ 600 KB (was ~1.7 MB); every-task ≈ 65 KB (was 127–290 KB); zero empirical facts lost — every crown jewel named in §7 has a designated survivor location.

---

## 6. Migration plan (phased; each phase ships independently)

**Phase 0 — correctness hotfixes (½ session, no moves, do immediately)**
1. scoring_internals: banner + strike MACs from the three formula sites (bug #1).
2. QUICKSTART:26 OneHot line → match traps §2 (bug #2). (Dies anyway in Phase 2 — fix now regardless.)
3. INDEX:42-44 + by_symptom:13 family-count swap (bug #3).
4. techniques.md: assign fresh numbers to the five colliding/placeholder headers (#23-dup → delete as exact dup of §24; #40-dup, #41-dups, `# N.`s → next free numbers ≥48, with "(formerly second #41)" breadcrumbs). Update the handful of by_symptom/INDEX refs. Do NOT touch non-colliding numbers.
5. Promote the 4 orphaned jewels: task013 trio (.update) → techniques.md + by_symptom.md; hash-MLP fingerprint (mlp_rule_extraction addendum) → techniques.md; GridSample tradeoff + bool-safe scaffold (symmetry_and_affine) → techniques.md; merge traps §21/§26 dup.
6. Mechanical `citeturn…`/`fileciteturn…` strip across all ex-ChatGPT files (hundreds of dead tokens).

**Phase 1 — evict process-logs from the every-task set (1 session, biggest per-task win)**
SPEC split/archive · cost_floor ledger eviction · cascade diary → Logs/ · traps split (table/appendix/ship-block→current_strategy) · task_techniques_index header strip. Then update INDEX/README/template reading instructions to the new ~65 KB set.

**Phase 2 — kill/merge the C-heavy shelves (1–2 sessions)**
op shelf → op_cost_patterns + 5 deletions · Ort_Advanced → ort_reality · taxonomy collapse (delete alt) · entry-point collapse (QUICKSTART/README → INDEX) · deliverable compressions (mlp/synth/tiny/CA/symmetry) · archive sweep (4 extracted deliverables, function_proto delete, Research/notes, Todo/Done, .update DONE files, J-cluster) · exemplar dedups.

**Phase 3 — techniques.md deep rewrite (1–2 sessions, last; nothing depends on it)**
Verdict-index front-matter · §39-42 addendum consolidation · §0 ship-table eviction · narrative compression. Title-greppability is the compatibility contract: never delete a section title workers may grep; leave one-line tombstones for moved content.

**Every phase**: `tools/refactor_paths.py --apply` for moves → hand-update the 2 generators → re-run them → update INDEX + CHANGELOG entry → commit. Blast-radius checklist for moves/renames: INDEX.md, README/QUICKSTART(template), `.update/README.md`, `agent_feedback_rules.md` (18 refs), `_TEMPLATE_Prompt_TaskNNN_Radical Rewrite.md` (22 refs), `regenerate_graph_diagnostics.py` (6 hardcoded paths), `_build_task_techniques_index.py`, `current_strategy.md`, `Logs/lessons.md`. Run phases between campaign waves; commit the working tree first.

---

## 7. Crown jewels — must survive any edit (refactor executor's checklist)

Grader/scoring: cost mechanism `max(static, runtime-on-real-inputs)` per tensor (30+ exact confirmations; dynamic repriced up to 78×, task094) · input AND output excluded from memory (basis of F16/BOOL-output + free-input Einsum) · params = element-count, dtype irrelevant · scalar value_info = itemsize · value_info re-inferred strict_mode and counted · calibration regimes (single-task raw log_e ×1.0, megastack ×0.48, zero-weight 0.00) · kq5y shape_infer(None) gap · 2026-04-28 constant-folding metric date.
ORT/validator: full dtype×opset matrix (ort_op_dtype_matrix + Rules 6/8/11) — Where(bool,bool,bool) rejected, Equal floats@11, ReduceMax(BOOL)@20, CumSum no-F16-no-U8, GatherND INT64-only, Conv same-dtype binding, OneHot no-BOOL/U8 out + depth-dtype match, ScatterElements reduction verdicts (add-F16 rejected, min/max accepted) · no constant-folding under ORT_DISABLE_ALL · axes attr→input per-op transition · banned-op list + Sequence-substring + kernel_time + sub-INT32 gather floor + Div/Mod-by-input SIGFPE + sparse 3-gate rejection · ClearField("value_info") mandate · F16 integer-exactness ceiling 2048 (+octaves) · F16 multi-operand Einsum not byte-equal (0.042, flips ArgMax).
Competition: byte cap 1,509,949 exclusive · encoder/decoder + 1-ch0 trap + pad-color-10 sentinel rules · cross-task cascade trigger set + threshold T≈250–300k · zero-public-weight task family {366,324,133,182,018,157,255,066,076,277} + cherry13b protocol · verify gates (100% M/M; adversarial advisory; task008 3/86 exception class) · negative-pad Conv crop grader-confirmed +1.685 · free-input Einsum (1-2 operands) scores 0/0 · power-iteration dilation doubling · depthwise 10× table · plane-floor taxonomy table · hash-MLP fingerprint + RBF-hash caveat · 25-pattern auto-detector index + author gotchas · Trilu/Range/ConstantOfShape priced generators.

---

## 8. Anti-re-bloat charter (add to `.update/README.md` + `agent_feedback_rules.md` §8)

1. **Litmus at the door**: would Sonnet produce this cold? Then it's a pointer, not a paragraph. (Empirical *verdicts* that merely look spec-like — "ORT rejects X", opset boundaries — always pass; *explanations* of what ops do never do.)
2. **Verdict-first formats**: trap = pattern/symptom/fix/ref in ≤4 lines (incident detail → appendix). Technique = pattern → rewrite → safety → measured saving → exemplar path. No truth-table traces, no session narration.
3. **Supersede in place**: the sorter EDITS the parent section; "ADDENDUM/UPGRADE/ESCALATION" appendages are banned. Old text dies when corrected (CHANGELOG records the change).
4. **No ship-ledgers in reference docs**: per-task outcomes live in `taskNNN.md` / `current_strategy.md`; reference docs cite one anchor example, not 54-row tables. Dated "batch N completion" logs go to CHANGELOG.
5. **One canonical home per fact** + pointers: scoring formula/banned ops/hard rules → core.md; dtype facts → the matrix; symptom routing → by_symptom. A fact stated twice WILL drift (bugs #1–#3 are all duplication-drift).
6. **Section keys are stable**: never reuse a retired §number; new sections take the next free number; title-grep is the contract.
7. **Extraction closes the loop**: when a Research deliverable is extracted to techniques.md, the original moves to `_archive/` in the same commit.
8. **Strip `citeturn` artifacts on paste**; delete deliverable self-appendices (cards reproducing their own assignment).
9. **Every-task set has a size budget** (~65 KB), printed in INDEX; adding a file to the set requires evicting equivalent KB.

---

## 9. Suggested immediate actions (if only 3 things happen)

1. **Phase 0 hotfixes** — 6 items, ½ session, fixes active mis-teaching (MACs, OneHot, taxonomy swap, §41 collisions, orphaned jewels).
2. **Phase 1 every-task eviction** — ~1 session, cuts what every single worker pays from ~130–290 KB to ~65 KB.
3. **Adopt charter rules 2/3/4** for the sorter immediately — they stop the two growth mechanisms (append-only supersession, ship-ledger accretion) that produced most of today's D-mass.

*Full per-file evidence (line numbers, quotes, A/B/C/D splits) is preserved in the 11 auditor reports from this session; this file is the distilled plan of record.*
