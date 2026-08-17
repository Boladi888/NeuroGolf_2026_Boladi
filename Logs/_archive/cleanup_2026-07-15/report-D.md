# Agent D Cleanup Report

Scope: `Logs/`, `Research_Archive/`, `Prompts/`.

## Inventory Reconciliation

Tracked files before cleanup:

| Path | Tracked before | Final tracked |
|---|---:|---:|
| `Logs/` | 952 | 26 |
| `Research_Archive/` | 146 | 39 |
| `Prompts/` | 34 | 32 |
| **Total** | **1132** | **97** |

Reconciliation:

- Before: 1132 tracked files.
- Deleted outright: 1035 tracked files.
- Moved: 0.
- Integrated/replaced: 1 file (`Research_Archive/_RESEARCHED_TOPICS.md` became a 264-byte redirect stub instead of the old 53 KB stale ledger).
- Kept as surviving original content: 96 tracked files.
- Final: 97 tracked files in Agent D scope.

Force-count sanity:

- Before filesystem count: `Logs/` 983 items, `Research_Archive/` 151 items, `Prompts/` 48 items.
- After filesystem count: `Logs/` 28 items, `Research_Archive/` 40 items, `Prompts/` 46 items.
- No junctions were found under the checked delete parents before directory removals.

## Lint Counts

Live before this cleanup:

- `python tools/repo_lint.py`: 32 errors, 859 warnings.
- `python Onnx_Advanced/.update/tools/lib_lint.py`: 2 errors, 586 warnings.

After cleanup:

- `python tools/repo_lint.py`: 32 errors, 859 warnings.
- `python Onnx_Advanced/.update/tools/lib_lint.py`: 2 errors, 586 warnings.

The prompt's older master baseline said `repo_lint` 32/869 and `lib_lint` 0/573, but the live worktree baseline was already different when this pass started. This cleanup did not increase either live count.

## What I Deleted

- `Logs/` root scratch: per-task probe dumps, silent-fail CSVs, megastack JSONs, active lane ledgers, empty `runs.jsonl`, linter output, and root cleanup blow-by-blow logs.
- `Logs/scans/`, `Logs/sessions/`, `Logs/spread_scan_results/`, `Logs/audits/`, `Logs/cleanup/`, `Logs/delta_packets/`, `Logs/multi_task_sweeps/`, `Logs/worker5_session_2026-05-18/`, and most of `Logs/_archive/2026-05/`.
- `Prompts/WebPrompts/*.zip` only. The unpacked Task374 pack and workflow remain.
- `Research_Archive/` run-history: old research-agent dispatch briefs, thin `digest-*` stubs, most spent `recipe-*` task handoffs, source fragments duplicated by surviving J-notes, empty markers, and obsolete overview/prototype folders.

## What I Kept

- `Logs/lessons.md` and `Logs/_archive/lessons_pre0601.md`: the canonical durable lesson stream plus the pre-June archive.
- `Logs/submissions_ledger.csv`: the only file that traces the Kaggle score arc. I added an honest note that rows are submit-by-submit through 2026-06-08 and the 2026-06-21 tail is a resync.
- `Logs/README.md`: kept as a reusable convention pattern, but rewritten to say the convention decayed and the repo is archived.
- `Logs/cascade_campaign_log.md`: borderline, but kept because `Onnx_Advanced/cascade_strategy.md` still points here for the campaign diary.
- `Logs/_archive/CASCADE_WORK_LOG.md`: retained because old task notes and lesson entries still reference it.
- `Logs/research/`: all 20 structured research cards/data files (`R1` through `R6`) stayed. This means Agent B's proposed `Ort_Advanced` repoints still have their targets.
- `Research_Archive/notes/J.*`: 20 portable representation surveys with verdicts, costed arithmetic, and "does not pay" guidance.
- `Research_Archive/notes/deep-*`: 9 competitor deep reads.
- Three recipe exemplars: task270, task206, task071. They are the clearest examples of the old recipe format because they include verified rules, failed hypotheses, and implementation/verification context.
- `Research_Archive/notes/spread_knowledge_*`: kept because `Onnx_Advanced/core.md` still points to the spread-knowledge report/tools.
- `Prompts/`: all non-zip prompt contracts stayed, including the 4 root supervisor contracts, 4 HandOff supervisor contracts, HandOff README, lean prompter request, the web-prompt workflow, and the unpacked Task374 worked example.

## Could Not Decide / Recommendation

- `Logs/cascade_campaign_log.md`: keep for now. If someone later folds its one useful diary pointer into `Onnx_Advanced/cascade_strategy.md`, it can be deleted in a separate cross-scope pass.
- `Research_Archive/_RESEARCHED_TOPICS.md`: I recommend keeping the tiny redirect stub until the two Agenters docs that reference it are updated. It prevents new `lib_lint` warnings without restoring the stale ledger.

## Out-of-Scope Notes

- `Logs/lessons.md` should point at `Onnx_Advanced/library_db/late_competition_lesson_facts_20260714.md` once Agent B fixes that file's provenance. It is fresher than the latest `lessons.md` entry and sounds like the right late-competition lesson companion.
- Agent B proposal note: I kept `Logs/lessons.md`, `Logs/cascade_campaign_log.md`, and `Logs/research/R6_onnx_op_candidate_matrix.*`; send the `Ort_Advanced` repoint proposal against those files if still needed.
- `Onnx_Advanced/core.md` still mentions `Logs/spread_scan_results/...` scan outputs. I deleted `Logs/spread_scan_results/` per Agent D instructions; a core-doc owner should either make those references historical or point them to a surviving summary.
- Old Handoff run logs outside my scope still mention `Prompts/WebPrompts/Task374_Creative_WebPrompt_Pack_20260703.zip`. I fixed the live Prompt docs and the unpacked pack handoff post, but did not rewrite historical board/run logs.

## Verification Limits

I did not run project tools, tests, `ng_gate`, `build_submission.py`, or anything that needs `.venv/` or `data/`. Only the required stdlib linters were run.
