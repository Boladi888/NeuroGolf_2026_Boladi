# AUDIT 2026-06-10 - Onnx_Advanced refactor information preservation

Scope: strict post-refactor audit of `Onnx_Advanced`, focused on whether the refactor lost information and whether the new navigation surface is safe for task workers.

Short verdict: **no content loss found**, but **route hygiene is not clean yet**. The deleted tracked files I checked are preserved byte-for-byte somewhere else, and the project lint passes. The remaining problem is that several current worker-facing pages still cite files that were moved into `_archive`. That is not data loss, but it can behave like loss for agents because the named path no longer opens.

## What I Checked

- Current `git status --short -- Onnx_Advanced`.
- Current size/layout inventory for root, `Research`, `exemplars`, `.update`, `Ort_Advanced`, and `_archive`.
- Exact-content preservation of tracked files deleted by the refactor.
- Pre-refactor snapshot presence for modified root and maintenance docs.
- Generated catalog and project lint status via `Onnx_Advanced/.update/tools/lib_lint.py`.
- Independent `rg` scans for stale references to moved root stubs, moved research deliverables, and moved update notes.

I made no code/doc changes other than writing this audit file.

## Preservation Result

### Deleted tracked files

Result: **51 deleted tracked files, 51 exact-preserved, 0 missing exact copies**.

| Deleted source bucket | Preservation target | Result |
| --- | --- | --- |
| `.update/_integrated/*` | `_archive/update_integrated_2026-06/*` | Exact copies found |
| Root stub docs such as `einsum.md`, `multi_axis_indexing.md`, `radius_encoded_weighted_cover.md` | `_archive/stubs_2026-06-10/*` | Exact copies found |
| Full root deliverables behind the stubs | `_archive/*.md` | Present for checked F/einsum/multi-axis/function-proto docs |
| `Research/Done/K*`, `L*`, `M*`, `R*` card bodies | `Research/Done/cards/*` | Exact copies found |
| Full research deliverables | `_archive/Research_*.md` | Exact copies found |
| `Research/Todo/M3_*`, `Research/Todo/R4_*` | `Research/Done/cards/*` | Exact copies found |

Conclusion: I did not find a deleted tracked file whose content disappeared.

### Modified tracked files

The refactor archives pre-refactor snapshots for the large edited docs. Most snapshot checks were exact against `HEAD`. Three files were not exact against `HEAD`:

- `Research/Todo/README.md`
- `WORKER_PROTOCOL.md`
- `traps.md`

This looks consistent with the refactor preserving the live pre-refactor working copy, not clean `HEAD`, because these files already had uncommitted edits before the refactor. I would not call this information loss, but it is worth knowing if someone later tries to audit only against `HEAD`.

### Lint

`python Onnx_Advanced/.update/tools/lib_lint.py` passed:

```text
lib_lint: 0 error(s), 0 warning(s), 0 info - checks: dup-sections, catalog-stale, catalog-coverage, index-two-way, size-budgets, routing-coverage, stub-expiry, done-archival, changelog-cap, integrated-clean, required-reading, update-inbox
```

This is good, but it does not catch every inline Markdown path that a worker might follow.

## Navigation / Size Result

The refactor did reduce the mandatory worker surface.

| Surface | Current state |
| --- | --- |
| First-read path | `WORKER_PROTOCOL.md` + `INDEX.md` + `core.md` + `traps.md`, about 60.5 KB total |
| First-read budget in `INDEX.md` | 75 KB cap |
| Root docs | About 950.6 KB |
| `_archive` | About 2.6 MB, hidden by `.rgignore` |
| `techniques.md` | Still huge: about 348 KB / 4,341 lines |
| `traps.md` | Now compact: about 21 KB |
| `traps_appendix.md` | Now large: about 132 KB, acting as long-form archaeology |

This is a real improvement for workers: the default read path is much smaller, and the archive is out of normal `rg` results. The cost is that any stale path into an archived file must be corrected explicitly, because `_archive` is now intentionally hidden.

## Findings

### P1 - Current pages still point at archived root stub names

These files no longer exist at the root:

- `direct_detection_multi_qlinearconv.md`
- `einsum.md`
- `function_proto_subgraphs.md`
- `multi_axis_indexing.md`
- `multistage_key_first_compositional.md`
- `radius_encoded_weighted_cover.md`
- `row_signature_rank_reduction.md`

They do exist in `_archive/stubs_2026-06-10/`, and the fuller deliverables exist in `_archive/`, so this is not loss. But current docs still send workers to the old root paths:

| Current file | Lines | Stale refs |
| --- | ---: | --- |
| `arc_rule_taxonomy.md` | 79-82 | `radius_encoded_weighted_cover.md`, `row_signature_rank_reduction.md`, `multistage_key_first_compositional.md`, `direct_detection_multi_qlinearconv.md` |
| `techniques.md` | 1964, 2027, 2089, 2133 | Same four deliverable names as "Full deliverable in ..." |
| `op_cost_patterns.md` | 6, 44, 59 | `einsum.md`, `multi_axis_indexing.md` |
| `recent_ships.md` | 52 | `einsum.md` provenance note |
| `Research/Done/COMPLETED_CARDS_LEDGER.md` | 44-45 | root deliverables described as promoted root docs/stubs |

Risk: a worker following a current page gets a missing file and may conclude the source is gone.

Recommended fix: rewrite active references to either:

- the extracted `techniques.md` sections plus `techniques_catalog.md`, or
- the real archive targets, e.g. `_archive/radius_encoded_weighted_cover.md`.

I would prefer current technique-section references for worker-facing docs, with archive links only as provenance.

### P1 - Current technique sections cite old `Research/...` paths

The full research deliverables are preserved under `_archive/Research_*.md`, but current pages still cite the old `Research/*.md` paths:

| Current file | Lines | Old refs |
| --- | ---: | --- |
| `ort_reality.md` | 91 | `Research/nary_fanin_catalog.md` |
| `techniques.md` | 2371, 2379, 2470, 2693, 2714, 2817, 2836 | `Research/ort_dynamic_shape_profiling.md`, `Research/log_round_flood_reconstruction.md`, `Research/nary_fanin_catalog.md`, `Research/spatial_data_movement_catalog.md`, `Research/bit_packing_calculus.md`, `Research/einsum_equation_catalog.md`, `Research/row_packed_uint32_morphology.md` |
| `techniques_catalog.md` | 62, 66-69 | generated rows copied from `techniques.md` titles |

The matching archive files exist:

- `_archive/Research_ort_dynamic_shape_profiling.md`
- `_archive/Research_log_round_flood_reconstruction.md`
- `_archive/Research_nary_fanin_catalog.md`
- `_archive/Research_spatial_data_movement_catalog.md`
- `_archive/Research_bit_packing_calculus.md`
- `_archive/Research_einsum_equation_catalog.md`
- `_archive/Research_row_packed_uint32_morphology.md`

Risk: same as above. The content is preserved, but the path a worker sees is stale.

Recommended fix: update the source headings/notes in `techniques.md` and rerun the catalog builder, or change the wording to "extracted from archived Research_* deliverable" without presenting a dead path as a current route.

### P2 - One active `by_symptom.md` citation appears stale from before this refactor

`by_symptom.md:224` cites:

```text
.update/DONE_2026-05-22T204400Z_task383_directional_padslice_to_maxpool.md
```

That file does not exist in `.update` or `_archive/update_integrated_2026-06`. `git show HEAD:Onnx_Advanced/by_symptom.md` shows the same citation before this refactor, so I do not think this was lost during the refactor. It is still a current worker-facing dead link.

Recommended fix: either find the actual task383 walkthrough in logs/history and point to it, or remove the "See ..." clause and keep only the distilled pattern.

### P2 - Archive visibility is now intentional, so stale links matter more

`.rgignore` contains only:

```text
_archive/
```

That is good for normal worker search. It also means an agent will not naturally discover archived content unless a current doc gives the correct archive path. The hidden archive makes the P1 stale route refs more important to fix.

### P2 - `techniques.md` is still the big remaining bloat object

`techniques.md` remains about 348 KB. The new `techniques_catalog.md` makes it navigable, and the project lint says the catalog is current. I would not split it immediately unless workers are still reading it whole. The safer next step is to fix stale refs first and then watch whether workers actually obey the catalog route.

If workers keep loading all of `techniques.md`, split by family after the refactor settles. The family pages should be generated or mechanically checked from the catalog to avoid creating a new drift problem.

## Suggested Next Patch

I would make one small hygiene patch before calling the refactor done:

1. Update stale root-stub refs in `arc_rule_taxonomy.md`, `techniques.md`, `op_cost_patterns.md`, `recent_ships.md`, and `Research/Done/COMPLETED_CARDS_LEDGER.md`.
2. Update old `Research/*.md` refs in `techniques.md` and `ort_reality.md` to `_archive/Research_*.md` or to extracted section references.
3. Regenerate `techniques_catalog.md` after editing `techniques.md`.
4. Extend `lib_lint.py` with a "dead local markdown refs" check for active docs. It should ignore `CHANGELOG.md`, `_archive/`, and audit reports, but it should catch the exact stale refs listed above.
5. Rerun `lib_lint.py`.

This is a routing cleanup, not a recovery operation. I do not see evidence that the refactor lost information.

## Bottom Line

The refactor appears preservation-safe: deleted content was archived, research cards were retained, and the project lint is clean. The new worker flow is also materially smaller.

The main remaining defect is stale citation paths. For humans this is annoying; for autonomous workers it is worse, because a dead path can make them stop looking or hallucinate a replacement. Fix those stale routes and add a lint check for them, then this refactor is in much better shape.
