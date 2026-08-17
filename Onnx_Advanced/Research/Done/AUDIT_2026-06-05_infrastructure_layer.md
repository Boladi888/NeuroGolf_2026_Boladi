# AUDIT 2026-06-05 — infrastructure layer (tools, build_submission.py, GUI, .hermes, repo hygiene)

**What this is**: audit #3, companion to the library audit (#1) and task-memory audit (#2). Covers everything not yet audited except `.ai_instructions/` (owner handles separately): `tools/` (108 root scripts + 4 subdirs), `build_submission.py` (965 KB), `GUI/` (288 MB), `.hermes/` (230 MB), and misc dirs + repo root. 6 parallel auditors, read-only, file:line evidence throughout. **Internal audit deliverable.**

Character of this layer: less context-bloat, more **broken-machinery + disk-bloat + drift**. The headline risks are silent-wrong-output tools that agents trust, two latent token bombs without guards, one dangerous legacy CLI path, and ~400 MB of reclaimable/duplicated disk.

---

## 1. Headline

| Area | Size / count | Core finding |
|---|---|---|
| tools/ root | 108 scripts | ~12 dead/broken (1 still-broken, **4 hardcoded-dead-worktree landmines**, 7 orphans, 1 superseded); 8 stale catalog pointers; 7 live-but-invisible tools; hygiene otherwise CLEAN (0 tracked .pyc repo-wide — audit#2's 352 got cleaned) |
| build_submission.py | 965 KB / 21,978 lines / 311 defs | **LIVE library** (dispatcher + 3 test suites import it), NOT legacy, NOT a data blob (380 hand-coded ONNX builders/detectors). But: unguarded ~240k-token Read hazard + a **dangerous dead `--hybrid` CLI** that would clobber the live submission.zip with a 103/400-correct baseline |
| GUI/ | 288 MB disk / 430 tracked | Well-built agent-facing satellite app ("NeuroGolf Lab" + "Doors" game) that is a **complete island** (zero inbound refs); its design doc falsely claims the shipped feature is unbuilt; **94 MB of duplicated task JSON is git-tracked**; 216 MB regenerable (node_modules+dist) |
| .hermes/ | 230 MB / 1,519 files | 96% = `staged_submits/` (548 dirs): every dir stores a FULL 401-task zip even for 1-task changes; 223 dirs double-store; **zero retention mechanism**; ~105 MB reclaimable today; one latent git footgun (blanket ignore lives only in local `.git/info/exclude`) |
| Misc + root | — | `.env.handoff` holds real secrets but is **properly untracked** (no leak); README/CURRENT_STATE stale (5002/5874-era vs LB 6523); models/ tracking contradicts its own PLAN.md; 2 stray .onnx at root |

---

## 2. SECURITY

1. **A local `.env.handoff` contains live Handoff MCP credentials**. **It is NOT git-tracked** (covered by `.gitignore:32` `.env.*`; never committed — verified via `git log --all`). No leak. Residual note: plaintext local credentials still require backup/sync hygiene so they are not shipped off-machine.
2. **Fresh-clone footgun**: `.hermes/` (230 MB), `repo/`, and `.notongit/` are ignored only via **local** `.git/info/exclude:14`, not the committed `.gitignore`. On any other clone, `git add -A` can stage 193 MB of board zips (the committed ignore only blocks state.json/tick_log/locks/baks/onnx). Fix in §6 R3.

---

## 3. Broken machinery (agents trust these — fix first)

1. **`tools/split_alpha_lanes.py` is STILL broken** (the 2026-06-04 fix in commit 17c8a9c7 was incomplete — it fixed the task-ID parse but not the schema drift): `:51` reads `cost` (column is `current_cost`), `:62` reads `best_known_source` (column is `current_src`). Live proof: every lane row prints `cost=0 headroom=0.00`, so `alpha_sub_lanes.md` — called "source of truth" by 7 playbook refs — currently publishes all-zero garbage rankings in task-ID order. **Two-token fix + re-run.** (Correction to the verify report: `cascade_report.py` is already fixed and verified working.)
2. **Four more tools carry the exact hardcoded-dead-worktree landmine that broke cascade_report** (`...\.claude\worktrees\beautiful-shockley-865962`, a deregistered worktree that still half-exists on disk — tools run without crashing but read/write a STALE tree): `cascade_targets.py:25`, `promote_task_work.py:41,47,48`, `uncashed_prototypes.py:49`, `verify_submission.py:8`. Same one-liner fix each (`Path(__file__).resolve().parents[1]`). `promote_task_work` is especially nasty — it's the tool that preserves worker artifacts.
3. **Live code instructs running archived scripts**: `recommend_isolated_probes.py:161,173` tells workers to run `tools/task014_synthetic_guard.py` / `task153_synthetic_guard.py` — both moved to `tools/_archive/` → file-not-found. Also cataloged as live in `04_DATA.md:67-68,91-92` and `The_400_Tasks/README.md:111`.
4. **Stale catalogs advertise 8 archived tools as live** (`tools/README.md:26,34-40` + `04_DATA.md`: external_blend, blend_diff, karnak_xref, mine_karnak_targeted, audit_external_cascade, all_local_external_sweep, + the 2 guards above). Zero-cost doc edits.
5. **`show_task.py` is missing from `diagnostics_recipes.md`** — the helper catalog whose whole purpose is "find the existing tool before writing your own". The most-common worker question ("status of task N?") has no route in that catalog; `agent_helpers/README.md` is also stale (3 of 12 scripts missing). Soft ambiguity: two adversarial entry points (`adversarial_against_live.py` vs `silent_fail_check.py`) across catalogs with no "canonical" marker.
6. **7 orphan tools** (zero refs anywhere: `_audit_tier_health`, `cumulative_memory_sim`, `daily_telegram_digest`, `document_live_algorithm`, `patch_hermes_logging`, `tune_hermes_profiles`, `worktree_prune_readiness_audit` — 39 KB) + **1 superseded** (`_score_zip_to_json.py`) → archive. **7 live-but-invisible analysis tools** (cascade_targets, catalog_organic_prototypes, batch_stage_polish, spread_polish_sweep, _scan_new_patterns, _spread_audit, scan_unsolved) → one catalog line each.
7. Minor: `cost_predictor.py:146` comments the pre-migration path `Research/sources/onnx_tool_1.0.1_node.py` (now `tools/onnx_tool_1.0.1_reference/`).

## 4. build_submission.py — verdict: LIVE library, two ~20-line guards needed, split deferred

- **Identity**: 21,978 lines = 266 `*_network` builders + 113 `_detect_*` heuristics + dispatch/build/zip orchestration. Imports the central scorer (`tools/_score_v2.py:62` — no duplication; spread_patterns is a different layer, no overlap). Importers: `tools/build_dispatcher_task.py:19` (live per-task submit path) + 3 active test suites. Alpha lane owns editing it grep-first (`07_LANES.md:54`, `alpha_a_marathon_plan.md:20`).
- **The `--hybrid`/`--all` whole-zip CLI is dead and dangerous**: last edit 2026-05-18; zero references from the June campaign; produces 103/400-correct (~1,627 pts vs live ~6,523); its embedded recipes are obsolete by orders of magnitude (task137: 536k bytes embedded vs 2,516 shipped); and `make_zip()` (:21913) would **overwrite the live submission.zip**. Doc drift: `03_PROJECT.md` still teaches `--hybrid` as "build all" and claims "73 patterns" (actual: 113/266).
- **Actions**: (1) ~15-line guard banner at L1 — "965 KB / ~240k tokens, DO NOT Read whole; grep `_task<NNN>_` and Read ±60 lines; live flow is per-task dispatcher, NOT --hybrid". (2) ~5-line clobber guard in `make_zip()`/`main()` (refuse to overwrite non-empty live zip, or write `submission_hybrid_baseline.zip`). (3) Fix `03_PROJECT.md`/`04_DATA.md` to stop teaching `--hybrid`. (4) The detector-shard split (option b, core driver →~5 KB) only if Alpha-lane monolith editing resumes — multi-day, merge-conflict-heavy, defer.

## 5. GUI/ — functional satellite app, drifting docs, 94 MB tracked duplication

- **Identity**: "NeuroGolf Lab" — FastAPI+React local app (task viewer, visual ONNX editor with competition-exact validation via the project scorer) + "Doors" gamified rule-discovery mode that compiles winning door-paths to ONNX or emits recipe JSONs for agents to golf. `project_bridge.py` reads main state read-only and writes only additive timestamped files into `Task_by_Task/<task>/prototypes/` (good design). It has already produced real output (task177 doorpath recipe, 2026-06-05).
- **It is an island**: zero inbound references from playbooks/prompts/INDEX; `SKILL.md` defines a `neurogolf-lab` skill registered nowhere. Decide: wire it in (register the skill + strip the duplicated worker-protocol lines in AGENTS.md/SKILL.md to pointers) or mark it in the main INDEX as "satellite app — not part of the core loop". Either way one line in INDEX.
- **Doc drift to fix**: `DOORS_DESIGN.md:3,314` says "nothing in this document is built yet" — **false**, the feature shipped (door_engine.py + 4 live server routes + campaign JSON + production output). `WORKFLOW_EVAL.md` (2026-05-13) describes the dead HuggingFace-upload era (server.py has zero HF refs) → archive. `scripts/hf_auth.py` → delete. Uncommitted WIP: `client/src/doors/DoorsView.jsx` (untracked — the doors frontend!) + modified main.jsx/style.css — commit or fence.
- **Repo weight**: `client/public/tasks/` = **400 task JSONs, 94 MB, git-tracked**, duplicating `data/competitions/` — generate at build time instead (biggest single repo-size win). Disk: node_modules (121 MB) + dist (95 MB) regenerable, properly gitignored, deletable locally.

## 6. .hermes/ — 230 MB with no retention; fix the staging pattern

- **What it is**: `staged_submits/` = 548 dirs / 193 MB (96%). Each dir stages a FULL 401-task submission.zip even when changing ONE task (verified: non-target onnx byte-identical across zips) + 223 dirs ALSO store the loose .onnx → triple duplication vs `prototypes/` and `submissions/`. Board root: live `candidate_registry.json` + `submit_queue.json` (1.6 MB each) + **19 `.bak*` snapshots (~17 MB)**. Empty scaffolding dirs (handoffs/processed/review_*/snapshots) + an `archive/` convention used once in 35 days.
- **No token-bomb risk**: all access is CLI/tool-mediated (`hermes kanban` is "the only transport"); no instruction routes an agent to Read board files. Tracked-file lifecycle (6 inbox/outbox directives, committed-then-deleted-when-consumed) is intentional and clean (2 commits ever).
- **Recommendations**: **R1** retention cron (delete staged dirs for closed/superseded tasks OR >7d-and-not-in-submit_queue; archive-then-delete; size cap ~50 MB) — one-shot reclaim today: 275 stale dirs (~88 MB) + 19 baks (~17 MB). **R2** stage only the changed taskNNN.onnx + manifest (SpreadSweepWorker.md:84 already says the zip is optional); build the full zip at submit time — kills the 399× intra-zip duplication. **R3** move the blanket `.hermes/` ignore from `.git/info/exclude` into committed `.gitignore` with `!inbox/ !outbox/ !*.md` carve-outs (closes the fresh-clone footgun). **R4** one guard line in KanbanWorkerLoop.md: board state files are machine-only. **R5** delete the 6 empty scaffolding dirs or wire `archive/` into R1.

## 7. Misc + root

| Item | Finding → action |
|---|---|
| `README.md` (root) | STALE: references deleted `Research/` (renamed Research_Archive 2026-05-28), quotes 5002.95-era LB and "484 tests" → fix paths + drop hardcoded numbers (charter rule) |
| `CURRENT_STATE.md` | Auto-generated but generator not re-run since 2026-05-25 (shows 5874.78 vs actual 6523.20) → re-run `tools/update_current_state.py`; consider adding to the post-submit refresh sequence |
| `task198.onnx`, `task201.onnx` at root | Untracked worker scratch → delete (belongs in scratch dirs) |
| `models/` 45 force-tracked .onnx | **Contradicts `tools/ng_harness/PLAN.md:16`** ("parents are not tracked") — they were `git add -f`'d. Pick one: un-track (honor the doc) or fix PLAN.md. Doc/reality must agree |
| `tmp/` | Not gitignored (only its *.onnx are) → add `tmp/`; then delete contents |
| `repo/`, `.notongit/` | Ignored only in local exclude → promote to committed `.gitignore` |
| `work/` (16.8 MB untracked scratch) | Already ignored; no rotation — periodic purge note |
| `Research_Archive/` | KEEP as-is — its 00_INDEX redirect map is accurate and is the only thing that resolves stale `Research/...` links |
| `Competitor_Notebooks/`, `Team/`, `ml/`, `docs/`, `tests/` | KEEP (tests are clean — no refs to moved docs; the `etc/passwd` string is an intentional security-test fixture) |
| `.claude/` | 1 file, properly ignored — fine |
| Tracked .pyc / __pycache__ / .pytest_cache repo-wide | **0 / 0 / 0** — clean ✓ |

---

## 8. Execution plan

**Phase A — broken-machinery hotfix (½ session, highest trust-impact)**: split_alpha_lanes 2-token fix + re-run · 4 worktree-landmine one-liners · repoint recommend_isolated_probes:161,173 · catalog fixes (8 stale pointers; add show_task.py + the 7 invisible tools to diagnostics_recipes/04_DATA; refresh agent_helpers/README; mark canonical adversarial tool) · archive 7 orphans + _score_zip_to_json · cost_predictor.py:146 comment.

**Phase B — guards (½ session)**: build_submission.py banner + make_zip clobber guard + fix 03_PROJECT/04_DATA `--hybrid` teaching · KanbanWorkerLoop board-state guard line · re-run update_current_state.py + fix root README.

**Phase C — disk + git hygiene (1 session)**: .hermes retention tool + one-shot reclaim (~105 MB) + stage-only-changed-task pattern (R2) · .gitignore patch (tmp/, repo/, .notongit/, .hermes carve-out) · GUI: commit-or-fence the doors frontend WIP, fix DOORS_DESIGN false banner, archive WORKFLOW_EVAL + delete hf_auth.py, decide skill wiring + INDEX line, un-track client/public/tasks (94 MB) in favor of build-time copy · models/ tracking decision · delete root .onnx strays.

**Charter additions (§8d proposal)**: (1) no absolute paths in tools — repo root = `Path(__file__).resolve().parents[1]`, enforced by a 5-line test that greps `tools/*.py` for `Path(r"C:`; (2) every new agent-facing tool ships with its catalog line in the same commit; (3) generated state docs (CURRENT_STATE, alpha_sub_lanes, CASCADE_ANALYSIS) carry a generated-at header and join the post-submit refresh sequence; (4) staging dirs hold deltas, not full bundles; (5) satellite apps (GUI) keep protocol content as pointers to the canonical docs, never copies.

## 9. If only 3 things happen

1. **Phase A items 1-2** — split_alpha_lanes + the 4 worktree landmines: six one-liners that stop tools from silently publishing wrong data into "source of truth" docs.
2. **build_submission.py's two guards** — the banner (kills a latent 240k-token context bomb) and the make_zip clobber guard (kills a one-command path to destroying the live submission).
3. **.hermes retention + gitignore hardening** — reclaims ~105 MB now, stops unbounded growth, and closes the only way this repo accidentally commits 200 MB.

*Evidence trail: 6 auditor reports this session (tools-root liveness, agent_helpers/catalogs, build_submission deep-dive, GUI, .hermes, misc+root+security). No files modified.*
