# Cleanup Report - Agent F

Date: 2026-07-15
Worktree: `<CODEX_WORKTREES>\dcfe\K-NeuroGolf`
HEAD: `0f15a1ac86c4cc628375411e9c3fd4a71d92789b`

## Scope

Owned paths: `tools/`, `tests/`, `ml/`, `.ai_instructions/`.

Allowed deliverable paths used: `_cleanup/report-F.md`, `_cleanup/proposals-F.md`,
`_cleanup/UPSTREAM_LIFT/`.

Also added root `pytest.ini` because the updated test ground truth explicitly requested a
pytest runner config. No other root files were edited.

## File Counts

Tracked files before:

| Path | Before tracked | After tracked | Action |
|---|---:|---:|---|
| `.ai_instructions/` | 21 | 21 | kept; 5 files copied into upstream-lift bundle |
| `tools/` | 235 | 231 | deleted 4 tracked `tools/api/` files |
| `tests/` | 39 | 39 | kept; edited 3 files |
| `ml/` | 40 | 40 | kept; replaced top-level `ml/README.md` content |
| **Total owned tracked** | **335** | **331** | **335 = 4 deleted + 0 moved + 0 consumed/integrated-away + 331 kept** |

New files added:

| Path | Count | Why |
|---|---:|---|
| `_cleanup/UPSTREAM_LIFT/` | 6 | 5 copied AI-instruction files plus copy-map README |
| `_cleanup/proposals-F.md` | 1 | Agent A NOTICE text and Orbit Wars/A770 lore proposal |
| `_cleanup/report-F.md` | 1 | this report |
| `pytest.ini` | 1 | documents and pins pytest as the real runner |

Filesystem counts after cleanup (`Get-ChildItem -Recurse -Force`):

| Path | FS entries | FS files | Python files |
|---|---:|---:|---:|
| `.ai_instructions/` | 24 | 21 | 0 |
| `tools/` | 240 | 231 | 219 |
| `tests/` | 39 | 39 | 38 |
| `ml/` | 47 | 40 | 22 |
| `ml/ngs/` | 43 | 37 | 20 |

Generated `tests/__pycache__` from syntax checking was removed after a junction check on `tests/`.

## Lint Counts

Both linters were run before and after. Exit codes were ignored; counts are the gate.

| Linter | Owner baseline in brief | Live before | Live after | Delta vs live before |
|---|---:|---:|---:|---:|
| `python tools/repo_lint.py` errors | 32 | 32 | 32 | 0 |
| `python tools/repo_lint.py` warnings | 869 | 859 | 859 | 0 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` errors | 0 | 2 | 2 | 0 |
| `python Onnx_Advanced/.update/tools/lib_lint.py` warnings | 573 | 586 | 586 | 0 |

Note: this worktree is exactly `0f15a1ac`, but live lint counts did not match the prompt's
baseline. I used live before/after as the hard gate and did not increase either count.
`repo_lint.py` rewrites `Logs/lint_last.md`; I restored that generated side effect and did not
commit it.

## Changes Made

### Preserved AI Instructions Upstream-Lift Bundle

Created `_cleanup/UPSTREAM_LIFT/` with exact SHA-256 matching copies of:

| K-NeuroGolf source | Proposed upstream destination |
|---|---|
| `.ai_instructions/04_DATA.md` | `Archive\AI_Instructions\04_DATA.md` |
| `.ai_instructions/05_BACKLOG.md` | `Archive\AI_Instructions\05_BACKLOG.md` |
| `.ai_instructions/playbooks/agent_feedback_rules.md` | `Archive\AI_Instructions\playbooks\agent_feedback_rules.md` |
| `.ai_instructions/playbooks/SpreadSweepWorker.md` | `Archive\AI_Instructions\playbooks\SpreadSweepWorker.md` |
| `.ai_instructions/playbooks/WorkerAgent_Setup.md` | `Archive\AI_Instructions\playbooks\WorkerAgent_Setup.md` |

I followed `Agenters/Janitor/ai_instructions_retirement_plan.md` for inventory and reference-risk
awareness, but did not follow its destination map. That plan was for live-project context reduction;
this cleanup is archival, so upstream preservation is the right first move. I did not delete
`.ai_instructions/`.

### Deleted `tools/api/`

Deleted 4 tracked files:

- `tools/api/debug_task388.py`
- `tools/api/verify_pr46_strict.py`
- `tools/api/verify_pr46_tasks.py`
- `tools/api/verify_t174.py`

Evidence:

- `tools/` parent junction check: no `LinkType` entries.
- Not listed in `tools/README.md`.
- Last touched by old PR/task work: `bbaefdac 2026-05-07 Add task174 LR symmetry picker`.
- Search after deletion finds one stale out-of-scope historical pointer:
  `The_400_Tasks/task_state.csv:176` mentions `tools/api/verify_t174.py`.

Recommendation: leave the CSV as historical provenance or let the owning agent decide whether to
remove that one tool mention. I did not edit outside scope.

### Tests

Added `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
```

Updated `tests/README.md` to document pytest as the real runner. It now warns that
`python -m unittest discover` silently under-collects pytest-style tests, and documents the data/
dependency portability problem.

Patched stale assertions in:

- `tests/test_neurogolf_orphan_candidates.py`: `package_in_staged_submits` ->
  `package_in_staged_candidates`, and `require_staged_submits` ->
  `require_staged_candidates`.
- `tests/test_agenters_hardening.py`: expected task-local
  `The_400_Tasks/Task_by_Task/<task>/staged_candidates/` hint instead of the old root
  `staged_submits/` layout.

Verification I could run here:

- `python -m py_compile tests/test_neurogolf_orphan_candidates.py tests/test_agenters_hardening.py`

Not run:

- `pytest`
- `unittest`

Reason: no `.venv/`, no `data/`, and the brief explicitly said not to run the suite in this worktree.
Mark these test fixes as **UNVERIFIED-BY-RUN**.

### `ml/`

Kept all of `ml/ngs/` (37 tracked files). It is fresh, honest, and useful: neural-guided ONNX
program search, ML as search policy, not shipped artifact.

Replaced the foreign top-level `ml/README.md` with a short pointer to `ml/ngs/README.md`.
The old content described Orbit Wars / Arc A770 local ML venvs and shadowed the current NGS doc.
I preserved the useful A770/XPU/OpenVINO lore in `_cleanup/proposals-F.md` as a proposed move back
to Orbit Wars or a machine-environment archive.

Kept `ml/verify_ml.py`. It is not NGS-specific and belongs with the old hardware lore if the owner
wants `ml/` to be NGS-only, but it is a small useful backend validator and not a first-read hazard
once `ml/README.md` points to `ml/ngs/`.

Important bug, not fixed here:

- `ml/ngs/score.py::_load_repo_scorer()` resolves `REPO_ROOT / "tools" / "_score_v2.py"` by file
  path and swallows all exceptions.
- If `_score_v2.py` moves, renames, or fails to import, `exact_cost()` silently falls back to
  `static_cost()` and only changes the tag to `static`.
- I did not move or rename `tools/_score_v2.py`.
- Recommended patch: make scorer load failures fail loudly or return an explicit error object, then
  allow a deliberate opt-in static fallback. Do this before any future scorer relocation.

`ml/ngs/backends.py:23` still hardcodes `Q:/AIz` scan roots. I report-only: do not blind-fix because
the current README says this is an intentional local-env discovery bridge.

## What I Kept And Why

| Item | Kept because |
|---|---|
| `.ai_instructions/` | upstream lift must land first; live code/docs still assume it exists |
| `tools/_score_v2.py` | canonical scorer; load-bearing for docs and `ml/ngs/score.py` |
| `tools/ng_harness/` | not duplicate; `tools/ng_gate.py` imports `killjob`, `throttle`, `corpus`, `modes` |
| `tools/spread_patterns/` | live plugin package covered by tests and used by spread/polish tools |
| `tools/agent_helpers/` | documented new-agent entry point in `tools/README.md` and root README |
| `tools/_archive/` | quarantined; includes `task014_synthetic_guard.py` and `task153_synthetic_guard.py` still cited by `tools/recommend_isolated_probes.py` as required gates |
| `tools/onnx_tool_1.0.1_reference/` | reference/provenance evidence for scorer history; licensing notice needs Agent A action |
| `tools/multi_task_sweeps/` | report-only recommend deletion later; campaign-specific but not as cleanly isolated as `tools/api/` |
| `ml/ngs/` | fresh research bet; completed 400-task sweep evidence belongs in archive |
| `tests/` | owner said tests are good; fixed stale assertions and runner docs instead of deleting |

Root README existence claims checked:

| README claim | Exists |
|---|---:|
| `tools/agent_helpers/show_task.py` | yes |
| `tools/ng_gate.py` | yes |
| `tools/_score_v2.py` | yes |
| `tools/swap_and_submit.py` | yes |
| `tools/build_dispatcher_task.py` | yes |

## Hermes / Board-Coupled Tool Triage

Literal `rg -l -i "hermes" tools --glob "*.py"` after deletion finds 47 files. A wider board/control
plane search (`hermes|.hermes|kanban|candidate_registry|submit_queue|submit_credit_state|task_outcome_registry|task_backlog_cursor|submit_picker_cache|next_submit_recommendation`) finds 54 files.

Because `.hermes/board/submit_credit_state.json` was reportedly written today and Agent E owns `.hermes/`,
I deleted **none** of this cluster. Triage:

| Category | Files | Recommendation |
|---|---|---|
| A: Hermes-only, safe to delete once retirement is confirmed | `tools/dashboard.py`, `tools/neurogolf_kanban_python.py`, `tools/neurogolf_pool_b_auto_unblock.py`, `tools/neurogolf_stale_run_reclaim.py`, `tools/reset_stale_crash_counters.py`, `tools/route_repair_cards.py`, `tools/scripts/submit_gate_wrapper_template.py`, `tools/_archive/_audit_tier_health.py`, `tools/_archive/daily_telegram_digest.py`, `tools/_archive/patch_hermes_cprint.py`, `tools/_archive/patch_hermes_logging.py`, `tools/_archive/tune_hermes_profiles.py`, archived staged-submit campaign scripts in `tools/_archive/` that only read/write `.hermes/board/staged_submits` | Keep for now; bulk-delete only after Agent E confirms Hermes retirement and no board replay is needed |
| B: Hermes/board-coupled but useful concepts remain | `tools/auto_triage_blocked.py`, `tools/neurogolf_auto_sync_current_best.py`, `tools/neurogolf_backlog_autoforeman.py`, `tools/neurogolf_backlog_candidates.py`, `tools/neurogolf_blocked_mechanical_cleanup.py`, `tools/neurogolf_board_policy.py`, `tools/neurogolf_cron_snapshot.py`, `tools/neurogolf_dependency_serializer.py`, `tools/neurogolf_orphan_candidates.py`, `tools/neurogolf_paths.py`, `tools/neurogolf_pool_b_autoforeman.py`, `tools/neurogolf_preflight.py`, `tools/neurogolf_promote_orphans.py`, `tools/neurogolf_queue_audit.py`, `tools/neurogolf_review_packet_builder.py`, `tools/neurogolf_smart_cleanup.py`, `tools/neurogolf_stack_picker.py`, `tools/neurogolf_submit_picker_cache.py`, `tools/neurogolf_submit_policy.py`, `tools/neurogolf_submit_trigger.py`, `tools/neurogolf_sync_current_best.py`, `tools/refill_submit_credits.py`, `tools/submit_gate.py`, `tools/submit_gate_kaggle.py`, `tools/submit_gate_mechanical.py`, `tools/submit_gate_runner.py`, `tools/task_outcome_registry_io.py`, `tools/update_current_state.py` | Keep; future patch should extract durable state-machine ideas away from `.hermes/board` paths |
| C: Still-live support/lint/reference code | `tools/repo_lint.py`, `tools/refactor_paths.py`, `tools/state.py`, `tools/algorithm_family_directive.py`, `tools/neurogolf_task_worktree.py` | Keep; some mentions are lint/search/path semantics rather than Hermes runtime dependence |

Known example confirmed: `tools/update_current_state.py` remains broken/coupled to Hermes (`hermes kanban`
and `.hermes/board`). I recommend replacing it later with a pure archive-summary generator or deleting it
after extracting any still-accurate status text.

## Hardcoded Path Analysis

Do not blindly make `tools/neurogolf_paths.py` repo-relative. Its docstring explains why the hardcoded
canonical root existed: Hermes workers ran from `.worktrees/<task-id>` and needed to avoid stale board
copies. That rationale is obsolete only if Hermes is truly retired.

Recommended future patch, after Hermes decision:

1. Keep `NEUROGOLF_CANONICAL_ROOT` override.
2. Change default canonical root to `Path(__file__).resolve().parents[1]` only after board-state readers no
   longer need main-checkout `.hermes/board`.
3. Update or delete the other hardcoded-root tools in one reviewed patch:
   `tools/neurogolf_sync_task_state.py`, `tools/neurogolf_auto_sync_current_best.py`,
   `tools/neurogolf_sync_current_best.py`, `tools/neurogolf_cron_snapshot.py`,
   `tools/scripts/submit_gate_wrapper_template.py`, and the hardcoded test fixture in
   `tests/test_neurogolf_orphan_candidates.py`.
4. Remove the personal plan-file reference in `tools/submit_gate.py:16`:
   `<CLAUDE_HOME>/plans/lets-go-overview-for-async-fiddle.md`.
5. Decide `tools/submit_gate_runner.py:78` only with the Hermes retirement patch, because its lock path is
   under `<HERMES_HOME>/`.

## Other Recommendations

| Item | Recommendation |
|---|---|
| `tools/multi_task_sweeps/` | delete or move to a historical archive after owner review; completed 2026-05-23 campaign and `_common.py` hard-pins a fragile backup zip |
| `tools/onnx_tool_1.0.1_reference/` | Agent A should update NOTICE; see `_cleanup/proposals-F.md` |
| `ml/verify_ml.py` | optional future move with old A770 environment notes if `ml/` should become NGS-only |
| `The_400_Tasks/task_state.csv:176` | stale historical reference to deleted `tools/api/verify_t174.py`; out of Agent F scope |
| `repo_lint.py` | consider a `--no-write` flag so required lint checks do not dirty `Logs/lint_last.md` in scoped cleanup worktrees |

## Hygiene Checks

Scoped scans over `tools/`, `tests/`, `ml/`, `.ai_instructions/`:

- High-confidence key/private-key signatures: no real hits. Initial broad `sk-` scan false-positive matched
  the substring in `task-transformation-library`; PCRE2 boundary scan was clean.
- Kaggle credential filenames/env vars: no credential material found. One linter pattern in `tools/repo_lint.py`
  references `credentials.json` / `secrets.json` as filenames to guard against.
- Profanity scan: zero hits.
- TODO/FIXME/XXX/HACK excluding third-party reference dumps and the NGS cheap-op backlog: 3 benign hits
  (`t_XXX` examples in tests/tools and one explicit TODO marker in a completed multi-task sweep).

## Out Of Scope Noticed

- Live lint mismatch with prompt baseline on exact HEAD `0f15a1ac`: prompt said `repo_lint 32/869`,
  `lib_lint 0/573`; this worktree reports `repo_lint 32/859`, `lib_lint 2/586`.
- `Onnx_Advanced` has pre-existing `lib_lint` errors: stale catalog SHA and stale/missing library DB.
- `The_400_Tasks/task_state.csv` still names deleted `tools/api/verify_t174.py`.
- `tools/submit_gate.py` has a personal `.claude/plans/...` doc pointer.
- `tools/onnx_tool_1.0.1_reference/` needs third-party attribution in NOTICE.
- Full pytest verification should be run in the main checkout with `.venv` and `data/`.
