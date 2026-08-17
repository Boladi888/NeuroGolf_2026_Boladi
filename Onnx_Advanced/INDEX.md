# INDEX.md — the Onnx_Advanced router (READ THIS FIRST)

The map of the whole knowledge base. **You do not need to browse folders** — find your situation below and open the files it names. One row per file, present tense, no archaeology; if a file isn't listed here, agents won't find it (two-way contract, lint-enforced). A bare filename means the file at the root of `Onnx_Advanced/`. `_archive/` is provenance cold-storage — excluded from search via `.rgignore`, never required.

**Scope contract (2026-06-20 audit):** the "advanced / non-obvious" boundary applies to the active routed worker surface, not every byte of cold storage. Public specs, generic ONNX notes, retired prompt bodies, and old snapshots may exist only as captain reference or provenance in `_archive/`/`Research/Done`; durable project-specific findings must be extracted into the routed root docs before workers rely on them.

## The read model (two layers)

1. **FIRST-READ (every task, ~60 KB total)**: `WORKER_PROTOCOL.md` → this INDEX → `core.md` → `traps.md` (tables + the doctrine blocks your prompt cites; §VISIBLE-FAIL and §14e always). Then your task file / `context_pack.py --task N`.
2. **EVERYTHING ELSE IS CITED, NOT READ**: prompts cite `techniques.md §N` and trap anchors; you grep those §s directly. For discovery beyond your prompt, run `library_db/lib_query.py status` + `context N` first, then use the routers below — never linear-read `techniques.md` (420 KB).

## First-read set

| File | Read it for |
|---|---|
| `WORKER_PROTOCOL.md` | the worker contract: read order, the 5 gates, staging schema, no-submit rule, 10 most-recent library verdicts |
| `core.md` | scoring formula, encoder/decoder + palette sentinels, the 1-ch0 trap, validator bans, pinned versions, cascade bug — the canonical home of the hard rules |
| `traps.md` | what breaks your model: 3 trap tables (build/runtime/scoring-submit) + the doctrine blocks (§VISIBLE-FAIL, §14e, §CLASS-CHECK, §DUAL-TRUTH, §GRADER-GAP, §FREE-OUTPUT, §FAMILY-LABEL, §25 calibration) |

## Discovery — "what could lower cost on THIS graph?" (the three routers)

| File | Entry point |
|---|---|
| `library_db/lib_query.py` | generated SQLite lookup CLI (`status`, `task`, `op`, `search`, `facts`) over the routed library; query this before opening large docs |
| `techniques_catalog.md` | **the complete one-line-per-§ index of techniques.md (GENERATED — all 140+ techniques)**. Grep by keyword/tag here FIRST, then open the § |
| `by_symptom.md` | "I see pattern X in my graph" → symptom → technique/exemplar route (incl. the #63–#139 new-wave section) |
| `task_techniques_index.md` | "what was done to task N (or a similar task)" → task ID → family → exemplar |
| `recent_ships.md` | dated ship lessons + battle prose, newest first — read when your task resembles a recent ship |

## The toolkit (grepped per-section via the routers)

| File | Read it for |
|---|---|
| `techniques.md` | the full §-numbered catalog (#0–#139). Navigate by §-number from the catalog/by_symptom/your prompt — `grep "^# 89\."` |
| `traps_appendix.md` | grep-only: full trap bodies + incident archaeology behind every traps.md row |
| `cost_floor_reference.md` | price your op chain BEFORE building: formula, plane-floor taxonomy, byte tables, per-op costs, floor laws (recolor=C, canvas=900B, shift=axis-length), floor-verdict ledger, calibration regimes |
| `scoring_internals.md` | how the scorer actually works: profiler mechanics, max(static,runtime) law, accounting levers (corrected), decoded exploits (ethical-play), ban workarounds, cost ladder |
| `ort_reality.md` | THE ORT layer: canonical dtype×opset matrix, behavioral quirks, error→fix table, opset support, K2 source-read. Build against this, not the spec |
| `patterns.md` | the 25 auto-detector passes (`pass_*.py`) + author gotchas |
| `diagnostics_recipes.md` | the `tools/agent_helpers/` catalog (extract_task, inspect_graph, largest_intermediate, context_pack, ng_fill_hang_safe, …) |
| `op_cost_patterns.md` | priced patterns for uncommon ops (GatherND/ScatterND thresholds, mod/bitwise, multi-axis indexing salvage) |
| `zero_param_generators.md` | generate masks/indices at runtime with no param cost (Trilu/Range/ConstantOfShape, priced) |
| `cascade_strategy.md` | only if cost > 150k: cascade-unlock status + submit rules (1 confirmed event ever) |

## Decode the rule (Phase 1 — "I don't know what this task does")

| File | Read it for |
|---|---|
| *(always)* | **decode the ARC-GEN generator FIRST — the family label is a hint, not the rule (traps §FAMILY-LABEL)** |
| `arc_rule_taxonomy.md` | the NeuroGolf-validated coverage spine (23-task mapping + 6 NeuroGolf-only families) + condensed 18-family table + decision tree |
| `Algorithm_Library/algorithm_patterns.md` | bbox, mode-color, permutation, hash, CC — the numpy-level pattern library |
| `Algorithm_Library/rule_discovery_methodology.md` | the decode procedure + its trap cases |
| `Algorithm_Library/per_family_starting_points.md` | family → best-first technique |

## Build patterns (after the rule is decoded)

| File | Read it for |
|---|---|
| `tiny_architectures.md` | hand-crafting sub-1k graphs: hand-set kernels, depthwise 10× law, power-iteration dilation-doubling, unroll cost law, static-shape strategies, iteration without `Loop` |
| `mlp_rule_extraction.md` | LIVE is a small MLP → hash-MLP fingerprint FIRST (#56), then the extraction pipelines |
| `symmetry_and_affine.md` | reflections, rotations, D4, GridSample-vs-Transpose pricing (live-maintained) |
| `cellular_automata.md` | CA/flood quick-ref: neighbor kernels, decision table, priced example |
| `synth_oracle_strategies.md` | building an INDEPENDENT synth oracle — the self-consistency trap + 4-axis rubric |

## Captain-reference (not in any worker path)

| File | Read it for |
|---|---|
| `00_COMPETITION_SPEC.md` | competition logistics: deadlines, prizes, licensing, rules citations, verified-facts provenance |
| `CHANGELOG.md` | library change record, newest first (pre-June-4 rotated to `_archive/`) |

## Subfolders

| Folder | Role |
|---|---|
| `exemplars/` | worked case studies by category (01_whole_graph_rewrites … 09_periodicity_detection) — by_symptom links to the exact one |
| `Algorithm_Library/` | routed above (3 files) |
| `Research/` | research-card pipeline: `Todo/` = the board + open cards; `Done/` = audits + `cards/` + the completed-card ledger. Durable findings are EXTRACTED into the root docs; full deliverable bodies are archived after extraction (rule 7) |
| `.update/` | session-end contribution drops (workers) + **`tools/`** (the library's own maintenance scripts: `build_techniques_catalog.py`, `lib_lint.py`) |
| `library_db/` | generated SQLite lookup layer: schema, builder, query CLI, reviewed fact blocks, manifest, and `onnx_library.sqlite`; Markdown remains canonical |
| `_archive/` | cold storage: pre-refactor snapshots, evicted bulks, extracted deliverable bodies, retired stubs. `.rgignore`d; browse deliberately, never by default |

## Size budgets (lint-enforced — `python Onnx_Advanced/.update\tools\lib_lint.py`)

Adding bytes to a budgeted file beyond its budget requires evicting equivalent KB (charter rule 9). The first-read sum (PROTOCOL + INDEX + core + traps) must stay ≤ 75 KB.

| file | budget KB |
|---|---|
| INDEX.md | 11 |
| WORKER_PROTOCOL.md | 8 |
| core.md | 30 |
| traps.md | 25 |
| cost_floor_reference.md | 30 |
| scoring_internals.md | 20 |
| by_symptom.md | 48 |
| techniques_catalog.md | 30 |
| techniques.md | 420 |
| task_techniques_index.md | 55 |
| recent_ships.md | 35 |
| ort_reality.md | 60 |
| CHANGELOG.md | 60 |

*techniques.md crossing ~420 KB = the standing trigger to revisit the family-page split (audit 2026-06-10 §1b Phase-E); until then, single file + generated catalog is the architecture.*

---

*Maintenance contract: when you add or promote a durable doc, add ONE row here (one row per file — no parenthetical archaeology; when content moves, rewrite the row, don't annotate it). After ANY techniques.md edit: regenerate the catalog. After library edits, rebuild `library_db/onnx_library.sqlite`; `lib_lint.py` enforces freshness. New techniques take the next free §-number via the sorter (`.update/README.md`) — never reuse a retired number; renumbers leave a `formerly #N` breadcrumb. Run `lib_lint.py` before committing library changes. Full maintenance doctrine: `.update/README.md` §sorter-duties.*
