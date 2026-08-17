# AUDIT 2026-06-06 — further optimizations (beyond the three structural audits)

**What this is**: audit #4 — forward-looking. The three structural audits + verify punch-list are largely implemented (confirmed against today's tree). This report covers what ELSE can be optimized: the one live regression, mechanical enforcement so the fixed things stay fixed, worker-loop ergonomics, storage reclaim, a quantified leaderboard-headroom analysis, and the consolidated open-items register. 5 parallel auditors, read-only, evidence cited. **No changes were made.**

---

## 1. Ranked opportunities

| # | Opportunity | Payoff | Effort |
|---|---|---|---|
| 1 | **Bound the whale-growth engine mechanically** (§2) | Stops the one active regression — task018/233/366 re-grew to 245/191/142 KB the same day one was compacted | S-M |
| 2 | **`repo_lint.py` + `refresh_all.py`** (§3) | The 17-check lint would have caught ~37 of the ~45 findings across all past audits; refresh_all ends the stale-generated-docs class permanently (CURRENT_STATE is 12 days stale again TODAY) | ~11-17 h total |
| 3 | **`context_pack.py` + prompt pointer-ification** (§4) | One command replaces a worker's 4-6 task-specific file reads (~13 KB curated pack); kills the per-wave prompt-boilerplate drift at the root (live prompts still embed gate blocks + a dated library digest, zero cite WORKER_PROTOCOL) | M |
| 4 | **Storage reclaim** (§5) | ~550 MB in-repo/junction now + ~1.5 GB owner-gated cold storage | S-M |
| 5 | **Aim workers with the headroom numbers** (§6) | Headroom is FLAT: +50 LB requires ~17 mid-band tasks at the 1500-floor; 3 of the top-30 "targets" are already floor-proven per their files (csv `solved_status` has ~226 false NotSolved) | data provided |
| 6 | **Finish the every-task-set trims** (§7) | Worker context bill is ~146 KB vs the 65 KB target — gap = traps table (68.5), cost_floor calibration dump (41.4), SPEC exit (16.7) | M-L each |

---

## 2. The whale-growth engine — regression root-caused, mechanical fix

**Diagnosis (corrects the obvious read):** the 31e49fe7 compaction did NOT defeat itself — it correctly externalized task233's history to `_archive/task233_pre_compaction.md` (83.9 KB) with a pointer. The regression is **entirely uncommitted working-tree growth appended AFTER the compaction, the same day**: task018 +131 KB, task233 +108 KB (~70 version entries v9→v80, each re-pasting an identical ~50-line gate-dump block — the same boilerplate ×30-35 within one file), task366 +46 KB. Committed sizes still match the audit-era numbers.

**The engine has three locked-in parts**: (i) `WORKER_PROTOCOL.md:16` — "**Ladder every improvement rung**" with per-rung exact-margin gate dumps = unbounded append by contract; (ii) `agent_feedback_rules.md §4b` — grind-the-same-task-for-weeks = maximal re-appending to exactly these files; (iii) charter §8b routes all per-attempt ledgers INTO taskNNN.md and its size budget covers only the shared reference set — **there is no per-task file budget anywhere, and no lint exists**. Manual compaction provably loses this race.

**Fix (minimal interlocking set — emission-time bounds, not cleanup):**
1. **Tripwire**: `lint_task_md.py` check — manual (non-marker) content of any taskNNN.md warn ≥15 KB, fail ≥20 KB; wire into `ng_gate` staging so a worker cannot stage a rung while over budget without archiving first.
2. **Relief valve**: standardize the convention task233 already proved — `_archive/taskNNN_history.md` + 1-line `## COMPRESSED HISTORY` pointer; `_archive/` excluded from worker first-read.
3. **Protocol amendment** (`WORKER_PROTOCOL.md:16`): rungs are **one-line table rows** (version | cost | Δ | verdict); full prose/gate-dump only for the CURRENT best + current floor proof; superseded-rung prose is not kept inline; **at session close, if file >15 KB → move history to the archive file**. Re-state: retracted floor proofs are deleted, not stacked (task018 currently stacks 4, ~3 retracted).
4. Optional backstop: `compact_task_md.py` (captain batch tool moving dated sections older than N days). (1)+(3) are what actually hold.

Note: mid-size active files (task367, task270) show the same pattern at an earlier stage — this is corpus-wide physics, not a whale quirk. The STATUS line (shipped) helps readers but does nothing to bound size.

## 3. Enforcement: `tools/repo_lint.py` + `tools/refresh_all.py` (full spec researched; build-ready)

**Today there is NO CI, no pre-commit lint (only 3 footgun-guard hooks), no session-start check, and the hermes cron surface is dormant** — which is why CURRENT_STATE.md is 12 days stale again despite two audits flagging it. The fix is two small tools:

**`repo_lint.py`** (~550-750 LOC, 8-12 h, registry-of-checks + one ALLOWLISTS dict, output: compact table + `Logs/lint_last.md`): 17 checks; the five highest catch-rate (counted against actual past findings): **dead-doc/script-refs (~14 past hits)**, **duplicate-§/duplicate-H2 (~7)**, **generated-doc staleness (~5)**, **hardcoded-worktree-path (~6 — the §8d-mandated test; must allowlist ~17 intentional canonical anchors like `neurogolf_paths.py:14`, NOT a blanket `Path(r"C:` flag)**, **stale-supersession/false-"moved-to" + INDEX two-way contract (~5, the P2 class)**. Plus regression guards: marker-pair integrity, token-bomb banners, catalog sync, secret/.env tracked-file scan, citeturn, LB-number-hardcoding (info-only), retired-system refs (alpha_sub_lanes — confirmed deliberately retired with the lane system, README tombstone), taskNNN oversize (feeds §2), doc/reality contracts (models/ vs PLAN.md). Runs: session-start fast set (~8 s) via a SessionStart hook in `.claude/settings.json` (currently has no hooks — clean addition) + pre-commit (error-severity only) + optional daily.

**`refresh_all.py`** (~180-260 LOC, 3-5 h): wraps the EXISTING canonical 5-step sequence (`The_400_Tasks/README.md:96-102`: enrich → classify → merge → cascade_report → generate_task_mds) + `update_current_state.py`, with a submission.zip-hash skip guard, `--check` dry-run, `--full` for the heavy HIST/SPREAD/GDIAG re-runs, and a uniform **`generated-at + source-hash` header** stamped on every generated doc (makes staleness exactly checkable instead of date-guessing). Omit `split_alpha_lanes` (retired). Wiring: **last step of the captain ship flow** (it IS the reconcile step currently done by hand and forgotten) + daily `--check` advisory. Hazard handled: `update_current_state` degrades gracefully if hermes is down.

## 4. Worker ergonomics: `context_pack.py` + finish the pointer-ification

- **Live prompts have NOT adopted WORKER_PROTOCOL.md**: the 06-06 wave is ~2 KB (template shrink shipped ✓) but every prompt still embeds a byte-identical `## Gate` block + a dated "2026-06-05 LIBRARY" digest (~1.3 KB × 20/wave) and zero prompts cite the protocol file — the template points at it, the generated wave doesn't. The digest is exactly the drift mechanism audit#2 predicted. Fix: regen prompts with one `## Protocol → WORKER_PROTOCOL.md` line.
- **`tools/agent_helpers/context_pack.py --task N`** (spec verified buildable — every input is machine-locatable today): assembles ~12-15 KB from (a) taskNNN.md STATUS line + RULE/DON'T-DO/OPEN-IDEAS H2 sections (byte-stable parser contract), (b) `show_task.py` row, (c) by_symptom.md entries matching the task's GDIAG quick-look flags, (d) the techniques.md §s the prompt cites (anchors now collision-free; resolve renumbers via breadcrumbs), (e) the trap table when it exists. Failure modes: placeholder on missing sections; whale fallback to show_task. **It complements (not moots) the §7 trims** — the pack cites into core/traps/cost_floor, so the doctrine set still wants its restructures, but the per-task read collapses from 4-6 files to one command, and the prompt digest becomes unnecessary.

## 5. Storage reclaim (prioritized; verified live dependencies first)

| Action | Freed | Risk / gate |
|---|---|---|
| Relocate the ONE live dependency (`karnak-task-transformation/arc_primitives.csv`, 37 KB → `data/reference/`; repoint `classify_task_family.py:47`), then zip `data/competitor_artifacts/` (570 MB, 19,919 onnx, measured 5.1× compression) to `.notongit` cold zip (~111 MB) and delete the live tree | **~459 MB** | Low — only consumer tooling is archived; origin is a NAS pull (no re-fetch script), so keep the cold zip |
| Owner-gated: purge `.notongit\…\worktrees\` (5 stale May worker trees, 972 MB) + the 2026-05-16 `submissions` snapshot (563 MB) **EXCEPT `_alpha_a/b_scratch`** (promote_task_work dependency) | **~1.5 GB** | Med — outside git, owner cold storage; flag-only |
| Archive ~55 old `_canonical_*_backup.zip` (keep live, blend, staged, `_best`, last-3 by score, **and `_canonical_5911_36_backup.zip` — hard dependency of `multi_task_sweeps/_common.py:26`**) + delete the stray 7.4 MB ort-profile json | **~57 MB** | Low — grep sweeps for pinned filenames first |
| `work/` closed-task scratch (task366/379), gzip 2 May blend logs + Logs/scratch npz, delete redundant `neurogolf-2026.zip`, root `task*.onnx` strays | **~35 MB** | Very low |
| Defender exclusions | none needed | Real-time protection measured OFF — not taxing the 36k-file churn |

Retention policies: competitor_artifacts = one cold zip, expand on demand · canonical backups = keep set above, archive rest · Logs scratch = promote summaries, purge raw on closeout (existing convention — enforce via lint).

## 6. Leaderboard headroom — the numbers (computed from task_state.csv, cross-checked vs live files)

- **Sanity**: csv `current_points` Σ=6494 (runs ~29 low); formula-from-cost Σ=6557 (~34 high); real LB ≈6523 between. Cost drift ≤1.5% on sampled tasks — rankings robust.
- **Shape**: no task >150k anymore. 176 tasks (44%) sit in the 10-50k band — **that band IS the remaining opportunity**. 66 tasks ≤1.5k are done (pushing them is worthless).
- **Ceilings**: everything at the 1500 shape-floor → **~7163 total (+606 from today)**; aggressive 200-floor → ~7909 (+1352). Max single-task prize in the entire corpus: **3.57 pts** (task286 — and its file disputes that floor).
- **Concentration: FLAT.** Top-10 = 5.2% of the +618 actionable pts; top-30 = 14.6%; top-100 = 44%. **+50 LB ≈ ~17 tasks driven to 1500.** There is no high-leverage head; it's a breadth grind — which argues for cheap per-task worker iterations (→ §4 ergonomics) over deep sieges.
- **Data quality warning**: `solved_status` is unreliable (≈226 false "NotSolved"; task158 shipped +14.56 and reads NotSolved); `live_arch` is uniformly "structural" (no signal); only the per-task .md STATUS/floor sections are trustworthy. → Either have `refresh_all` recompute `solved_status` from ships, or mark the column advisory in SCHEMA.md.
- **Skip (csv says open, files say floor-proven)**: task173 ("0 bytes recoverable"), task158 ("within 0.01 of theoretical max"), task286 (D\*=20 floor, disputed). **Best fresh targets** (27-35k cost, ~3 pts each, no floor/ship notes): **264, 351, 93, 281, 11, 132, 154, 91, 178, 349**.

## 7. Consolidated open-items register (verified against today's tree)

**Closed since the verify pass** (sample-verified): P1 core.md OneHot ✓, P4 #33/#34 renumber ✓, WORKER_PROTOCOL created ✓, template→1 KB ✓, worktree landmines fixed ✓, build_submission banner ✓, orphans archived ✓, DOORS false banner fixed ✓, lessons.md 289→45 KB ✓, Logs sweep ✓, hermes staged sweep ✓, GDIAG node dumps gone (non-whale) ✓, STATUS lines emitting ✓, lanes retired (OBE) ✓.

**Still open** (item — current size — effort):
- **P0**: prompts adopt WORKER_PROTOCOL (S, regen) · whale bound §2 (S-M) · context_pack (M) · whale compaction itself (L — blocked on fenced workers closing; task018/233/366 also still carry node dumps + lack STATUS lines)
- **Every-task set → 65 KB** (now ~146 KB): traps table restructure (68.5→~10, L) · cost_floor calibration dump (41.4→~22, M) · SPEC exit from Start-here (16.7, M) · techniques §39-stack consolidation + deep-compress (205 KB, L) · scoring_internals MAC body (67.8→~30, M) · taxonomy generic bulk (52.8→~10, M) · task_techniques_index header eviction (63.7, M — recent_ships.md is an honest tombstone awaiting it) · current_strategy ≤15 KB (50.8, M) · deliverable compressions (tiny 64.5/mlp 38.2/synth 46.3/CA 17.8/symm 16.9, L) · exemplar dedups (S)
- **One-liners**: CURRENT_STATE regen (or just build refresh_all) · show_task into diagnostics_recipes · GUI hf_auth delete + WORKFLOW_EVAL archive · `.gitignore` += tmp/, repo/, .notongit/ (fresh-clone footgun persists) · GUI client/public/tasks un-track (94 MB, M) · models/ vs PLAN.md:16 reconcile · make_zip explicit clobber guard (current guard is completeness-only) · Already_Ran residue (59 files) · README.md stale Research/ refs + LB numbers

## 8. If only 3 things happen

1. **§2 whale bound** (lint tripwire + archive convention + protocol line) — the only item actively regressing right now; everything else holds.
2. **Build `repo_lint.py` + `refresh_all.py`** (§3, ~2 days total) — converts all four audits from "findings someone fixed once" into standing invariants; ends the staleness class that has now recurred three times.
3. **`context_pack.py` + prompt regen with the protocol pointer** (§4) — biggest per-worker ergonomic win, and with §6's flat-headroom finding (success = many cheap task iterations), per-iteration context efficiency is the throughput lever for the remaining +600.

*Evidence: 5 auditor reports this session (whale forensics incl. git-diff attribution; enforcement-tool spec incl. automation inventory; data/storage incl. measured compression + dependency greps; LB headroom computed from task_state.csv with live cross-checks; ergonomics + register verified file-by-file). Read-only throughout.*
