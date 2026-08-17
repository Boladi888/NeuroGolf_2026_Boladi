# .update/ — contribution protocol

You should only be reading this file if you have a transferable insight to contribute back. Do NOT load this during the build phase of your task. Read it at session END, after staging your candidate.

## Protocol (simple — workers drop, sorter integrates)

You do not edit library files directly. You drop a file here. A separate sorter agent (smarter, multi-file context) reads accumulated drops and integrates them into the right library files.

### To contribute

Create one file in this folder:

```
DONE_<UTC-timestamp>_<short-topic>.md
```

Timestamp format: `YYYY-MM-DDTHHMMSSZ` (UTC, lexically sortable, no colons — Windows-safe).
Topic: 2-4 words, snake-case. Examples: `task110_periodicity`, `onnx_op_quirk_split`, `algorithm_diagonal_conv`.

Filename example: `DONE_2026-05-20T214530Z_task110_periodicity.md`

### File content

Whatever insight you want to contribute, in whatever form is clearest to you. The sorter will translate. Minimum useful content:

- **What you discovered** — the technique / quirk / pattern / dead-end in your own words.
- **Where you encountered it** — task ID, file path to your build script, staged candidate dir.
- **Why it matters** — what gain it produced, or what trap it avoided.
- **What's missing from the library** — if you searched for this and didn't find it, say so.

Long prose is fine. Code snippets are fine. The sorter handles formatting.

### Before you submit — quick checks (60 seconds)

These two checks save the sorter from chasing stale claims:

1. **Refresh your library snapshot.** The library evolves between when you started and when you submit. Run a quick `ls Onnx_Advanced/` and `ls Onnx_Advanced/Ort_Advanced/` and `ls Onnx_Advanced/exemplars/` before claiming any file is missing. (Examples of past stale claims: workers reporting `INDEX.md` / `traps.md` / `scoring_internals.md` as missing when they already existed at submit time.)
2. **Spot-check existing files when you have a "should be added to X" claim.** If you write "should be added to `techniques.md` §9", briefly grep §9 first — it might already cover your insight under a different name, in which case the sorter note becomes "extend §9 with this measurement" rather than "add new rule".

If you skip these and the claim is stale, the sorter will just drop that part of your contribution. The rest still integrates.

### Make sorting mechanical: use `## Sorter note:` markers

For each insight in your file, end the section with a clearly-marked sorter directive. The sorter agent grep's for these. Format:

```
## Sorter note: [target file path] — [add as new section / extend existing / replace existing]
```

Example: `## Sorter note: techniques.md §1 — add as Rule 9 "padding_sentinel_before_cast"`.

This is not enforced — long prose still works. But files with explicit sorter notes integrate ~3× faster.

### One DONE file can carry multiple insights

If a single session uncovered three related optimizations and two dead-ends, put them all in one DONE file with one `##` header per insight. The sorter splits them at integration time. Don't fragment into five separate files — concurrent drops are fine but they don't help the sorter when they're all from the same session.

### Do NOT

- Edit any existing file in `Onnx_Advanced/`. The sorter agent does that.
- Delete files in `.update/`. The sorter handles cleanup after integration.
- Create files with names other than `DONE_<timestamp>_<topic>.md`.
- Wait for a lock or coordinate with other workers. Just drop your DONE file. Concurrent drops are fine — files have unique timestamps.

### Sorter responsibility (not yours)

The sorter agent reads accumulated `DONE_*.md` files, decides which library file each insight belongs in, edits the appropriate file, then archives the consumed DONE file. You don't worry about that.

## SORTER DUTIES (the library's maintenance contract — added 2026-06-10, refactor pass 4)

The sorter is the library's ONLY direct writer. Per integration pass, in order:

1. **Integrate each drop** into its target file. Supersede in place — no ADDENDUM/UPGRADE stacking (charter §8b). Verdict-first format: pattern → rewrite → safety → measured saving → exemplar pointer.
2. **Assign §-numbers**: a new technique takes the **next free number** in `techniques.md` — the sorter assigns it, never the worker (parallel workers self-assigning numbers caused every collision to date). Never reuse a retired number; a renumber leaves a `<!-- formerly #N -->` breadcrumb.
3. **Regenerate the catalog**: `python Onnx_Advanced/.update/tools/build_techniques_catalog.py` — MANDATORY after any techniques.md edit (lint fails on a stale catalog).
4. **Route it**: add/extend the `by_symptom.md` route for any new technique; update the WORKER_PROTOCOL "Ten most-recent verdicts" by **REPLACING the oldest line** (hard cap 10 — the digest must never grow).
5. **Archive the consumed drop** to `_archive/update_integrated_<YYYY-MM>/` (don't leave it in `.update/`).
6. **Run the lint**: `python Onnx_Advanced/.update/tools/lib_lint.py` — must exit 0 errors. Budgets are in INDEX §Size budgets; going over means evicting equivalent KB (charter rule 9), not editing the budget.
7. One `CHANGELOG.md` entry per pass. Rotate CHANGELOG to `_archive/` past ~60 KB.

Research-deliverable filing (captain) additionally follows `Research/Todo/README.md` §filing-protocol (extract → catalog → route → archive body per rule 7 → ledger row → lint).

### `tools/` subfolder

`.update/tools/` holds the library's own maintenance scripts (`build_techniques_catalog.py`, `lib_lint.py`) — owner-sanctioned exception to the DONE-files-only rule. Workers: don't add scripts here; propose them via a DONE drop.

### When to drop a DONE file

Drop one when you discover something the library should have had when you started. Examples worth contributing:

- A new technique that landed bytes (and isn't in `techniques.md`).
- A new ORT error message you debugged (and isn't in `ort_reality.md`).
- A new algorithm pattern (and isn't in `Algorithm_Library/algorithm_patterns.md`).
- A documented dead-end so future workers don't redo it.
- A cost-floor measurement that calibrates `cost_floor_reference.md`.
- A fix for outdated info you spotted while reading the library.

If your contribution is just "task110 done at 31% reduction" with no transferable insight, that's a stage-result not a library contribution — skip the DONE file, just leave the staged candidate.

**Anti-re-bloat charter in force (2026-06-05)**: see `agent_feedback_rules.md` §8b. Sorter EDITS parent sections (no ADDENDUMs); verdict-first; one home per fact; stable §numbers; strip PUA citation tokens on paste.

**Phase-2/3 note (2026-06-05)**: filing targets updated to the refactored tree (op shelf -> `op_cost_patterns.md`; Ort_Advanced -> `ort_reality.md`; entry point = `INDEX.md`).
