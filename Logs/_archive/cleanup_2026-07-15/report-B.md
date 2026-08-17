# Agent B cleanup report - Onnx_Advanced

## Scope and inventory

Owned path: `Onnx_Advanced/` plus this required `_cleanup/report-B.md` and `_cleanup/proposals-B.md`.

Inventory source of record:
- `git ls-files -- Onnx_Advanced`: 466 tracked files before cleanup.
- `Get-ChildItem -LiteralPath 'Onnx_Advanced' -Recurse -Force | Measure-Object`: 496 visible entries before cleanup.
- `git ls-files -- Onnx_Advanced`: 455 tracked files after staged cleanup.

Reconciliation:

| bucket | count | notes |
|---|---:|---|
| before | 466 | tracked files at start |
| deleted | 11 | 1,937,590 bytes removed from tracked tree |
| moved | 5 | malformed `.update/_archive/update_integrated_2026-07/` timestamp names normalized |
| integrated | 32 | retained files with content edits, route fixes, provenance fixes, or regenerated catalog |
| kept unchanged | 418 | untouched tracked files still present |

Check: 11 + 5 + 32 + 418 = 466.

## Lint

Live baseline in this worktree differed from the prompt snapshot:

| linter | before | after |
|---|---:|---:|
| `python tools/repo_lint.py` | 32 errors, 859 warnings | 27 errors, 859 warnings |
| `python Onnx_Advanced/.update/tools/lib_lint.py` | 2 errors, 586 warnings | 1 error, 586 warnings |

Hard gate satisfied: neither error nor warning count increased. `repo_lint` dropped the 5 requested bad tool-path errors. `lib_lint` still reports `library-db-fresh`; I changed fact Markdown, so `Onnx_Advanced/library_db/onnx_library.sqlite` and `build_manifest.json` need regeneration by the normal library-db path.

## Deleted

- `Ort_Advanced/` (6 tracked redirect stubs, 1,340 bytes). Repointed owned live references to `ort_reality.md` or `scoring_internals.md` first.
- `team_best_snapshot_20260714.md` (7,046 bytes). It was mutable late-competition state; fact anchors now mark it as git-history-only.
- `_archive/arc_rule_taxonomy_alt.md` (38,813 bytes). The live taxonomy already keeps the head-to-head result; the alt won only 3 of 23 tasks.
- `_archive/notes_b1_quantized_ops_verification_attempt1_2026-05-25.md` and `attempt2` (61,189 bytes combined). Abandoned verification attempts with no durable edge over the retained cull note.
- `Research/Boladi_NeuroGolf_Knowledge_Pack_20260703.zip` (1,829,220 bytes). Generated repo-material bundle; the repo itself is the source.

## Moved

Renamed five tracked malformed archive drops from truncated `T####Z` to six-digit `THHMMSSZ` names:

- `T3708Z` -> `T033708Z`
- `T3745Z` -> `T033745Z`
- `T3820Z` -> `T033820Z`
- `T4145Z` -> `T034145Z`
- `T4200Z` -> `T034200Z`

All inbound fact anchors in `library_db/neurogolf_lesson_facts.md` were updated.

## Integrated

- Fixed the five bad `tools/...` path lint errors. Note: `repo_lint` matches the literal substring `tools/foo.py`, so first-read docs use Windows separators for `Onnx_Advanced/.update\tools\...`.
- Repointed owned ORT references away from `Ort_Advanced/` stubs.
- Changed false "purged" wording in `scoring_internals.md` and `CHANGELOG.md` to "moved to `_archive/`".
- Made the hardcoded local paths in `core.md` repo-relative.
- De-linked private K-Arc2 absolute links while preserving the prose.
- Removed line-numbered credential-map detail from the infrastructure audit.
- Fixed fact provenance:
  - `late_competition_lesson_facts_20260714.md`: 48 resolving file anchors, 2 external sibling-repo anchors, 2 git-history-only anchors for the removed mutable snapshot.
  - `neurogolf_lesson_facts.md`: 49 resolving file anchors, 28 corrected ZIP-internal anchors, 15 explicitly marked `source not committed`.
- Regenerated `techniques_catalog.md` after `techniques.md` edits.

## Kept and why

- `ort_reality.md`: keep. It is the most portable file: dtype/opset matrix, ORT quirks, and error-to-fix knowledge transfer outside ARC. Coupling count: `The_400_Tasks=0`, `tools=1`, `Logs=0`, `taskNNN=73`, `NeuroGolf=38`.
- `op_cost_patterns.md`: keep. Mostly portable for ONNX cost-golfing and uncommon-op pricing. Coupling count: `The_400_Tasks=0`, `tools=0`, `Logs=0`, `taskNNN=25`, `NeuroGolf=11`.
- `zero_param_generators.md`: keep. Portable static-ONNX generator tricks with light project coupling. Coupling count: `The_400_Tasks=0`, `tools=0`, `Logs=0`, `taskNNN=4`, `NeuroGolf=10`.
- `core.md`: keep as NeuroGolf-specific doctrine, not a portable lift-out. Coupling count: `The_400_Tasks=17`, `tools=29`, `Logs=5`, `taskNNN=35`, `NeuroGolf=57`.
- `techniques.md`: keep as the flagship local corpus, but it is the least portable as a whole. Coupling count: `The_400_Tasks=29`, `tools=4`, `Logs=2`, `taskNNN=780`, `NeuroGolf=199`.
- `cascade_strategy.md`: kept. It is only 1,349 bytes and still has meaningful inbound references; recommend archiving only after another pass updates those refs.
- `Research/Done/_archive/Creator_Aditya_NeuroGolf_Knowledge_Pack_20260703.zip`: kept. Unlike the deleted Boladi ZIP, this is a unique source packet for 28 corrected fact anchors.
- `_archive/` generally: kept. Only targeted debris was removed; live docs and provenance still cite the archive.

## Could not decide / recommendations

- `library_db/onnx_library.sqlite` is now stale and should be regenerated by the normal library DB builder. I did not update generated DB artifacts by hand.
- The prompt's "42 of 46 dead" source-anchor count did not match this worktree. Using `git ls-files` and filesystem truth, the late-competition anchors mostly resolve to tracked root `.update/_archive/...` files outside Agent B scope. If another agent deletes root `.update/`, those fact anchors should be converted to source-not-committed or moved into `Onnx_Advanced/.update/`.
- Lift-out recommendation for a future non-ARC project: start with `ort_reality.md`, `op_cost_patterns.md`, and `zero_param_generators.md`. Treat `core.md` and `techniques.md` as NeuroGolf case-study corpora, not direct transferable docs.

## Out-of-scope noticed

See `_cleanup/proposals-B.md` for concrete text/path follow-ups in `Logs/`, `Agenters/`, and `The_400_Tasks/` that still mention deleted `Ort_Advanced/` or `arc_rule_taxonomy_alt.md` paths.
