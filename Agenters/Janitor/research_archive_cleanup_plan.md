# Research Archive Cleanup Plan

Date: 2026-06-12

Intent: decide whether `Research_Archive` and `Competitor_Notebooks` can be removed, and define the safe path if they can.

No files have been moved or deleted by this plan.

## Short Answer

Do not delete `Research_Archive` as-is. It is small, currently referenced, and still contains recipe notes and research ledgers that task docs and tooling comments point at.

`Competitor_Notebooks` is a better cleanup candidate. The raw notebooks look low-value now, and the directory is small. If the decision is to remove it, preserve only the useful catalog/deep-read conclusions first, then delete the raw notebooks and stray submission zips.

If `Research_Archive` is going away too, then moving `Competitor_Notebooks` into it is only a temporary staging step. The cleaner final state is to migrate the useful summaries into current docs, then delete both top-level folders.

## Inventory

`Competitor_Notebooks`:

- About 21 files
- About 3.5 MB total
- Main content: `Competitor_Notebooks\competitor_notebooks_2026-05`
- Notable untracked root files:
  - `Competitor_Notebooks\submission_1.zip`
  - `Competitor_Notebooks\submission_2.zip`
  - `Competitor_Notebooks\submission_3.zip`
- Internal index: `Competitor_Notebooks\competitor_notebooks_2026-05\00_INDEX.md`
- Internal notes say `_stale_archives\files.zip` and `_stale_archives\files1.zip` were already extracted and are safe to delete after spot-check.

`Research_Archive`:

- About 146 files
- About 0.83 MB total
- Main content:
  - `Research_Archive\00_INDEX.md`
  - `Research_Archive\_RESEARCHED_TOPICS.md`
  - `Research_Archive\notes`
  - `Research_Archive\organic_prototypes`
  - `Research_Archive\overview`
  - `Research_Archive\sources`
- `00_INDEX.md` says this folder is mostly a redirection map after a previous `Research/` rename and keeps historical dispatch briefs plus niche topic surveys.

## Current Reference Risk

Deleting `Research_Archive` now would break references in active and historical files.

High-signal active references include:

- `build_submission.py` comments cite recipe notes in `Research_Archive\notes`
- Multiple task docs under `The_400_Tasks` cite `Research_Archive\notes\recipe-*`
- Several prototype/task notes cite `Research_Archive\organic_prototypes`
- `.ai_instructions` still references both `Research_Archive` and `Competitor_Notebooks` until that folder is retired
- The Janitor `.ai_instructions` retirement plan references both as migration targets

The reference count includes a lot of historical notes, so this does not mean every file must be edited. It does mean a blind delete will leave broken breadcrumbs.

## Recommended Final State

Preferred cleanup path:

1. Delete `Competitor_Notebooks` after preserving useful notes.
2. Migrate the few still-useful `Research_Archive` notes into current homes.
3. Delete `Research_Archive` only after references are updated or intentionally left as historical dead links.

Suggested canonical homes:

- Task-specific recipe notes -> corresponding `The_400_Tasks\taskXXX\...` docs or prototype notes.
- General ONNX/runtime findings -> `Onnx_Advanced\...` docs.
- Submission/build policy observations -> `submissions\README.md` or `build_submission.py` comments if they are still operational.
- Research ledgers that are only useful for provenance -> `Agenters\Librarian` or a short retained index in `Agenters\Janitor`, depending on who owns ongoing repo memory.

Avoid moving raw competitor notebooks into another permanent source-controlled folder unless they are still actively used. They are third-party reference artifacts and are better treated as disposable once the conclusions have been captured.

## Proposed Cleanup Steps

1. Preserve the competitor notebook catalog:
   - Read `Competitor_Notebooks\competitor_notebooks_2026-05\00_INDEX.md`.
   - Copy any useful conclusions into the current research/task docs.
   - Record that the raw notebooks were retired.
2. Remove obvious disposable files first, if authorized:
   - `Competitor_Notebooks\submission_1.zip`
   - `Competitor_Notebooks\submission_2.zip`
   - `Competitor_Notebooks\submission_3.zip`
   - `_stale_archives\files.zip`
   - `_stale_archives\files1.zip`
3. Audit live `Research_Archive` references:
   - Prioritize `build_submission.py`.
   - Prioritize `The_400_Tasks` task docs that cite `recipe-*` notes.
   - Ignore purely historical logs unless the user wants a clean reference sweep.
4. Migrate active notes:
   - Move or summarize task recipes into their task folders.
   - Move or summarize general findings into `Onnx_Advanced` or `submissions`.
   - Replace live references with the new locations.
5. Delete `Competitor_Notebooks` once the notebook catalog has been captured or explicitly deemed unnecessary.
6. Delete `Research_Archive` only after the active references above are migrated.

## Validation

Before deleting:

```powershell
rg -n "Competitor_Notebooks|Research_Archive" .
git status --short -- Competitor_Notebooks Research_Archive
```

After migration and deletion:

```powershell
rg -n "Competitor_Notebooks|Research_Archive" README.md WHERE_DATA_LIVES.md build_submission.py tools The_400_Tasks Onnx_Advanced submissions Agenters
git status --short
```

Acceptable remaining references should be either:

- historical notes that intentionally mention removed paths, or
- Janitor cleanup records explaining what was removed.

## Decision Point

The practical choice is not disk space; both folders are small. The choice is whether the repo should keep old research provenance in source control.

Recommended decision:

- Delete raw competitor notebooks and stray zip files.
- Preserve distilled findings where they are still useful.
- Keep `Research_Archive` only until active references are migrated, then delete it.

