# AUDIT 2026-06-10 - Onnx_Advanced navigation bloat and refactor recommendation

Scope: read-only audit of `Onnx_Advanced/` as it exists on 2026-06-10. The only file created by this audit is this report.

Worktree note: `Onnx_Advanced` already had local changes before this audit:

- Modified: `CHANGELOG.md`, `Research/Todo/README.md`, `WORKER_PROTOCOL.md`, `ort_reality.md`, `techniques.md`, `traps.md`
- Deleted from Todo / added under Done: `M3_min_cost_integer_linear_map_onnx.md`, `R4_REPOACCESS_task312_deep_rank_harvest.md`
- Added under Done: `R5_REPOACCESS_aditya_idiom_harvest.md`, `m3_min_cost_integer_linear_map.md`

## Bottom line

`Onnx_Advanced` is not bloated in the disk-space sense. It is bloated in the worker-navigation sense.

The directory has become a high-value but append-heavy knowledge base. The current shape is good for grep by an expert who already knows the section names, but it is not good for a single-task worker trying to quickly find "what could lower cost on this graph." The key failure is that several files have become blended surfaces: quick rules, historical evidence, ship logs, research provenance, stale redirects, and deep technique notes live in the same worker-visible area.

I would do a full refactor, but not a destructive one. Preserve the material. Change the worker-facing surface so the first pass is a small decision system, and move full research/provenance behind exact pointers.

## Size snapshot

Measured markdown/text files under `Onnx_Advanced`:

| Area | Files | Markdown | Size |
|---|---:|---:|---:|
| `_archive/` | 69 | 69 | 1,436.6 KB |
| root docs | 33 | 33 | 1,157.3 KB |
| `Research/` | 36 | 36 | 497.6 KB |
| `exemplars/` | 49 | 49 | 148.9 KB |
| `.update/` | 22 | 22 | 86.6 KB |
| `Algorithm_Library/` | 4 | 4 | 36.1 KB |
| `ort_reality.md` | 6 | 6 | 1.9 KB |

Top worker-visible root files:

| File | Size | Lines | Words | Audit read |
|---|---:|---:|---:|---|
| `techniques.md` | 339.9 KB | 4,341 | 49,436 | Core bloat source; should become an index plus split family pages. |
| `traps.md` | 79.9 KB | 787 | 11,362 | Valuable, but too long for every-task read. |
| `CHANGELOG.md` | 79.8 KB | 578 | 10,137 | Historical, not task guidance; should not be root-worker-visible at this size. |
| `traps_appendix.md` | 70.4 KB | 777 | 10,043 | Properly marked grep-only, but still root-visible. |
| `scoring_internals.md` | 68.3 KB | 734 | 9,721 | Still contains old/scoring-history bulk behind banners. |
| `task_techniques_index.md` | 66.9 KB | 659 | 5,848 | Useful router mixed with dated battle prose. |
| `tiny_architectures.md` | 63.0 KB | 987 | 10,113 | Deep note, not first-pass routing. |
| `arc_rule_taxonomy.md` | 52.9 KB | 828 | 7,644 | NeuroGolf mapping is valuable; generic taxonomy bulk is less so. |
| `ort_reality.md` | 49.9 KB | 632 | 7,454 | Valuable canonical ORT layer. Needs better entry routing. |

## Worker read-path problem

The documented every-task set is:

| File | Size | Words |
|---|---:|---:|
| `INDEX.md` | 8.8 KB | 1,233 |
| `core.md` | 25.4 KB | 3,312 |
| `cost_floor_reference.md` | 41.4 KB | 6,123 |
| `traps.md` | 79.9 KB | 11,362 |

That is 155.5 KB / 22,030 words before the task file, prompt, or cited technique sections. Including `WORKER_PROTOCOL.md` makes it 162.9 KB / 23,092 words.

That contradicts the stated 5-15 minute path. A strong worker can skim it, but weaker or cheaper workers will either over-read, skip the important parts, or grep randomly. The existing `tools/agent_helpers/context_pack.py` is the right direction, but the docs still tell humans and agents to read a large doctrine set.

Recommended target: every task should start from:

1. `WORKER_PROTOCOL.md` or a 1-page `START_HERE.md`
2. `context_pack.py --task N`
3. one graph-symptom router hit
4. 1-3 exact technique pages

Budget target: 35-50 KB total before code inspection, not 160 KB.

## Highest-value content to preserve

Do not flatten this knowledge into generic docs. The most valuable content is exactly the stuff a base model is unlikely to know:

- NeuroGolf scorer mechanics: `max(static, runtime-on-real-inputs)`, params counted by elements, final output free, dynamic repricing classes.
- ORT 1.24 quirks: dtype support, `Where(bool,bool,bool)` rejection, OneHot depth/index dtype matching, TopK details, F16/Gemm/Conv execution reality.
- Cost-floor measurements and proof style.
- Verified graph idioms: free-output families, Einsum-final, negative-pad Conv crop, QLinearConv gadgets, bit/nibble packing, selector-bank rank tests, no-Mul color decode, TopK compaction, ScatterElements routing.
- Bad-hidden patterns: lookup/MLP hash families, visible-pass traps, stale value_info, class-check and dual-truth methodology.
- Concrete exemplars where a worker can copy an approach.

The problem is not "too much knowledge"; it is weak separation between quick routing, canonical recipes, evidence, and history.

## Ranked findings

### P0. `techniques.md` has outgrown the "toolkit" shape

Evidence:

- 339.9 KB, 4,341 lines, 49,436 words.
- 139 numbered techniques by the end of the file.
- It mixes: quick rules, long worked histories, research-card extractions, ship logs, anti-patterns, addenda, superseded evidence, and worker-draft renumber breadcrumbs.
- Numbering is navigable by grep but not semantically clean. Examples:
  - `Rule 9` appears after `Rule 12` inside the bool-first section.
  - Top-level numbering jumps and backtracks (`#25` after `#31`, `#22` after `#25`, `#95` before `#90`).
  - Several sections carry "renumbered from worker-drafted" notes because parallel additions collided.

Impact:

Workers with a single graph cannot read this file. They need the right 10-30 lines. Today they need a router, grep luck, or prior knowledge of the section number.

Recommendation:

- Convert `techniques.md` into `TECHNIQUE_INDEX.md` <= 30 KB.
- Split actual content into family pages, for example:
  - `techniques/bool_first.md`
  - `techniques/free_output.md`
  - `techniques/einsum_final.md`
  - `techniques/spatial_crop_and_pad.md`
  - `techniques/packing_and_bitwise.md`
  - `techniques/qlinearconv.md`
  - `techniques/rank_and_selector_banks.md`
  - `techniques/dynamic_cost_and_pricing.md`
  - `techniques/graph_algebra_peepholes.md`
- Give every technique a stable ID independent of section order: `T067_free_output_where`, `T089_einsum_final`, etc.
- Each technique entry should follow the same schema:
  - Trigger signature
  - Rewrite recipe
  - Cost equation / crossover
  - ORT/opset/dtype caveats
  - Verified tasks
  - Contraindications
  - Search terms / detector command

### P0. The every-task read set is too expensive

Evidence: current documented read set is 155.5 KB / 22k words before the task file.

Impact:

The competition now needs many cheap iterations, not a few heroic reads. A 160 KB doctrine preload is too high when workers are assigned narrow tasks and looking for the one cost lever.

Recommendation:

- Create `START_HERE.md` <= 4 KB:
  - run `context_pack.py --task N`
  - run graph inspection / show_task
  - use `by_symptom.md`
  - read only cited technique pages
  - mandatory gates
- Replace the every-task doctrine set with a "doctrine quick table" <= 15 KB:
  - scoring formula
  - input/output encoding
  - banned ops
  - dtype trap table
  - visible/truth/AB/profile gates
  - pointers to deep docs
- Keep `core.md`, `traps.md`, and `cost_floor_reference.md` as deep references, not first-read files.

### P1. `ort_reality.md` is actively misleading

Evidence:

- It says `# Ort_Advanced - required reading`.
- It tells workers to read `ort_reality.md`, `ort_reality.md`, `ort_reality.md`, `scoring_internals.md`, `ort_reality.md`.
- Those files are now stubs pointing to `ort_reality.md` or `scoring_internals.md`.

Impact:

A worker browsing the folder follows a dead old route. The bytes are tiny, but the navigation cost is real.

Recommendation:

- Replace `ort_reality.md` with a redirect only:
  - "Do not read this folder; canonical ORT layer is `../ort_reality.md`; profiling/cost is `../scoring_internals.md`."
- Optionally move the stubs to `_archive` after reference rewrites are clean.

### P1. Research/Done mixes deliverables, retired prompts, audits, and process notes

Evidence:

`Research/Done` contains:

- Long deliverables: `k1_superoptimization_survey.md`, `k2_ort_exact_semantics.md`, `k3_unexploited_op_audit.md`, `l1_exact_cp_decomposition.md`, `m1_opset19_22_op_audit.md`, etc.
- Retired card bodies: `K1_equality_saturation_superoptimization.md`, `M1_opset19_22_unexploited_op_audit.md`, `R*_REPOACCESS_*.md`, etc.
- Audits and verification reports.

Impact:

The current `Research/README.md` explains this, but a worker hunting "brand-new" ideas sees a mixed folder with no physical separation.

Recommendation:

- Split `Research/Done` into:
  - `Research/Done/deliverables/`
  - `Research/Done/card_bodies/`
  - `Research/Done/audits/`
  - `Research/Done/verifications/`
- Keep `Research/Todo/README.md` short. It is currently 12 KB and the board says all K/L/M/R cards are done. Move the old ledger to Done.

### P1. `task_techniques_index.md` is part router, part ship diary

Evidence:

- 66.9 KB.
- Begins with long dated shipped sections before the family/task index.
- It has an integrity note admitting it mixes current exemplars, shipped entries, historical prototype paths, and staged-manifest pointers.

Impact:

The file is supposed to answer "what was done to task N or similar task?" It does that, but it also makes the worker cross a battle-log layer first.

Recommendation:

- Make the first 80% of the file a generated task lookup table:
  - task
  - current family
  - current cost
  - best known technique IDs
  - exact exemplar paths
  - stale/historical flag
- Move dated ship prose to `recent_ships.md` or `Logs/submissions_ledger.csv` summaries.
- Keep "recent ship lessons" as technique IDs, not prose blocks.

### P1. `traps.md` is valuable but too long for first-read

Evidence:

- 79.9 KB, 11,362 words.
- Already has a pointer to `traps_appendix.md`, but the main file still includes long doctrine and historical blocks.
- Numbering has at least one duplicated visible section number (`15` appears in runtime traps and scoring traps), which is manageable for humans but brittle for prompts.

Impact:

Workers need a trap checklist, not a full incident archaeology read.

Recommendation:

- Create `TRAP_TABLE.md` <= 10-15 KB:
  - symptom/error
  - exact cause
  - fix
  - deep pointer
- Move long narratives to `traps_deep.md` or keep them in `traps.md` but no longer mark it every-task.
- Preserve stable named anchors: `VISIBLE-FAIL`, `CLASS-CHECK`, `DUAL-TRUTH`, `CROP-CONTRACT`, `INPUT-DTYPE-LOCK`.

### P2. `CHANGELOG.md` is root-visible history, not active knowledge

Evidence:

- 79.8 KB.
- Many broken or intentionally historical references, old path strings, and dated operational notes.

Impact:

It is useful for provenance, but it should not compete with active worker docs in root.

Recommendation:

- Keep a short root `CHANGELOG.md` with last 10-20 changes and links to archived monthly logs.
- Move older entries to `_archive/changelog_YYYY_MM.md` or `Research/Done/audits/history/`.

### P2. `scoring_internals.md` still mixes current scorer facts with old scoring-era material

Evidence:

- 68.3 KB.
- It contains correction banners and archived-MAC references.
- It contains public evidence/history sections, grader replication, ban workarounds, profiling details, and old exploit discussions.

Impact:

The current scoring facts are critical, but they are buried with historical evidence.

Recommendation:

- Split:
  - `scoring_quick.md` <= 12 KB: exact current cost model and profile workflow.
  - `scoring_internals.md`: deep source/read evidence.
  - archive old MAC-era and public-discussion material.

### P2. Root contains several tiny redirect/stub files

Examples:

- `einsum.md`
- `multi_axis_indexing.md`
- `function_proto_subgraphs.md`
- `direct_detection_multi_qlinearconv.md`
- `radius_encoded_weighted_cover.md`
- `row_signature_rank_reduction.md`

Impact:

The stubs are not a byte problem. They are useful if old prompts cite them. But root-level stubs increase the apparent surface area and make directory browsing noisier.

Recommendation:

- Keep stubs only where active prompts still cite them.
- Add an expiry rule: once `rg` finds no live citations outside `_archive`, move the stub to `_archive/redirects/`.

### P2. Generic research notes should be routed, not root-visible by default

Examples:

- `tiny_architectures.md` at 63 KB
- `synth_oracle_strategies.md` at 45 KB
- `mlp_rule_extraction.md` at 37 KB
- `cellular_automata.md` at 17 KB
- `symmetry_and_affine.md` at 16.5 KB

Impact:

These are high-value when relevant, but they are not every-task context. Some contain content a base model partially knows. The NeuroGolf-specific extraction should be a short routed page; the full research note can stay behind it.

Recommendation:

- For each, keep a 1-2 KB root summary with:
  - when to use
  - NeuroGolf-specific delta from base knowledge
  - top 3 recipes
  - deep note path
- Move long form under `Research/Done/deliverables/` or `deep_notes/`.

## What is working well

- `INDEX.md` is the right concept: one router, no folder browsing.
- `by_symptom.md` is a useful worker entry point. Its graph signatures are concrete and cost-oriented.
- `WORKER_PROTOCOL.md` is a good central contract, especially gate 5 real-input profile.
- `context_pack.py` exists and is the right ergonomic lever.
- `exemplars/` is relatively compact and high-signal. The issue is broad "required reading" language, not the exemplars themselves.
- `_archive/` keeps provenance, which matters because many claims are empirical and competition-specific.

## Suggested refactor plan

### Phase 1: Stop the first-read bleed

1. Add `START_HERE.md` <= 4 KB.
2. Change `INDEX.md` and `WORKER_PROTOCOL.md` to make `context_pack.py --task N` the default task entry path.
3. Replace "every-task set" language with a budgeted "first task pack" target <= 50 KB.
4. Fix misleading subfolder READMEs:
   - `ort_reality.md`
   - `Algorithm_Library/README.md`
   - `exemplars/**/README.md`

### Phase 2: Split the big catalogs

1. Convert `techniques.md` into an index.
2. Move technique bodies into family pages using stable IDs.
3. Convert `traps.md` into a quick trap table plus deep trap notes.
4. Split `scoring_quick.md` from `scoring_internals.md`.
5. Trim `cost_floor_reference.md` into quick price tables plus deep calibration notes.

### Phase 3: Separate provenance from active guidance

1. Split `Research/Done` by role.
2. Move retired card bodies out of the same folder as deliverables.
3. Shorten `Research/Todo/README.md`; move completed ledger to Done.
4. Keep root `CHANGELOG.md` short; archive old history.
5. Move root stubs once no active references remain.

### Phase 4: Enforce it mechanically

1. Extend `tools/repo_lint.py` checks:
   - root doc size ceilings
   - every-task pack budget
   - "required reading" phrase only allowed in `START_HERE.md` / `WORKER_PROTOCOL.md`
   - stale redirect stubs with no live citations
   - duplicate visible technique IDs
   - `ort_reality.md` citation ban except stubs
2. Make prompt generation cite `WORKER_PROTOCOL.md` and exact technique IDs only.
3. Add a generated `TECHNIQUE_INDEX.md` from per-technique metadata headers, so routing stays current.

## Proposed target architecture

```
Onnx_Advanced/
  START_HERE.md
  WORKER_PROTOCOL.md
  INDEX.md
  core_quick.md
  scoring_quick.md
  cost_floor_quick.md
  by_symptom.md
  TECHNIQUE_INDEX.md
  TRAP_TABLE.md
  ort_reality.md
  techniques/
    bool_first.md
    free_output.md
    einsum_final.md
    spatial_crop_and_pad.md
    packing_and_bitwise.md
    qlinearconv.md
    rank_and_selector_banks.md
    graph_algebra_peepholes.md
    dynamic_cost_and_pricing.md
  exemplars/
    ...
  deep_notes/
    tiny_architectures.md
    synth_oracle_strategies.md
    mlp_rule_extraction.md
    cellular_automata.md
    symmetry_and_affine.md
  Research/
    Todo/
    Done/
      deliverables/
      card_bodies/
      audits/
      verifications/
  _archive/
```

## What I would not prioritize

- Do not delete `_archive/` just to save disk. It is useful provenance. Hide it from default routing/search instead.
- Do not compress `exemplars/` aggressively. It is only 148.9 KB and often more actionable than the big catalogs.
- Do not spend much effort on the tiny root stubs until citations are rewritten. They are annoying but cheap.
- Do not turn ONNX/ORT empirical facts into generic prose. The value is in exact opset, dtype, scorer, and task-verified constraints.

## My recommendation

Yes, do the full refactor.

The current folder is still packed with valuable lessons, but its shape is now optimized for cumulative memory, not fast task execution. The competition pressure argues for a smaller first-read path and more exact routing, because remaining gains are likely spread across many medium tasks. The winning structure is not "less knowledge"; it is "pay only for the knowledge this task can use."

If I were scheduling it, I would do Phase 1 immediately, then split `techniques.md` and `traps.md` next. Those two changes alone should cut most worker context waste while preserving the advanced material that actually gives this project an edge.
