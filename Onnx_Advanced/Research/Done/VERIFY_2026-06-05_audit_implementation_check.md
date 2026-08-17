# VERIFY 2026-06-05 — implementation check of both context-bloat audits

**What this is**: independent verification of the implementation of `AUDIT_2026-06-04_context_bloat_refactor_plan.md` (library) and `AUDIT_2026-06-04_task_memory_layer_bloat_refactor_plan.md` (task-memory layer). 6 parallel verifiers, read-only, against commits 65a28f8d…ab8862ce. Every claim below carries file:line evidence from this session's verifier reports.

**Bottom line**: what shipped, shipped well — no data loss, no broken machinery, honest commit messages. But the implementation is **partial** (audit#1 Phases 0–2 + a thin Phase 3; audit#2 Phases 0 + 1a only), and verification found **15 defects**, of which 8 are one-line fixes and 1 is a false in-file claim that actively mis-routes workers.

---

## 1. Verified CORRECT (the load-bearing stuff)

| Check | Result |
|---|---|
| Crown-jewel survival (44 facts from audit#1 §7) | **44/44 alive in live tree**, sensible canonical homes (dtype→ort_reality.md, grader→core/scoring_internals, floors→cost_floor). Zero found-only-in-archive. |
| 396-file GDIAG regen | **Zero manual-content damage** (whale jewels byte-preserved: task286 D\*=20, task110 −12.80 scar + v50/v51, task205 floor proofs, task233 trap writeup). Marker contract 0/402 imbalances; 401/402 carry all 5 H2 headers. |
| GDIAG staleness fix | Proven: task205 GDIAG went from describing a ~17-versions-dead graph (177,538) to current v17 (27,525); task338 matches shipped v8 @ 8,469; task005 183,854→16,738. Node tables: 0 remaining except the 5 fenced in-flight files (task018/185/198/270/366 — correctly skipped, coherent state). |
| § renumber | Clean: dup #23→#51, dup #40→#50, dup #41s→#48/#49, placeholders→#52-54; breadcrumbs; **no old-number reuse**; all prompt-cited anchors (#44-47, §39 MECHANISM, §41, §42, §37, §8b, traps §CLASS-CHECK/§VISIBLE-FAIL/§14z) resolve; 13 sampled by_symptom routes all land on correct content. |
| Jewel promotions | #55 task013 trio ✓, #56 hash-MLP fingerprint ✓, #57 bool-safe scaffold ✓, GridSample tradeoff in #44 ✓. traps §21→§26 merge ✓. citeturn strip: 0 live hits ✓. |
| Token-bomb guards (audit#2 §2) | `show_task.py --task 110` runs and prints one row ✓; all 3 instruction rewrites in place (incl. `_Before writing prompt.md:2`) ✓; both banners (SCHEMA.md:1, artifact_manifests/README.md:24) ✓. |
| current_strategy contradiction | Rejected-theory block (108 lines) deleted, 0 live hits of "Complete zero-weight family"/"unlock ceiling" as guidance ✓; SHIP DECISION REFERENCE landed at L65, **single-homed** (not duplicated in traps) ✓. |
| Data hygiene | live_costs.json regenerated (401 keys, stamped) ✓; -06-01 package CSV deleted ✓; runs.jsonl rotated (archive `Logs/runs.jsonl.2026-06-05T012735Z.archive`) ✓; METHODOLOGY_UPDATE folded+deleted ✓. |
| Archive sweep | 66 files in `_archive/`, all expected items present, stubs point correctly, 3 KEEP files survive, wrong SPEC formulas (`max(1,2500-length)`, 0.001 floor) gone from live tree ✓. |

---

## 2. DEFECTS — punch list (ordered by severity)

**P1 — `core.md:131` still teaches "OneHot needs INT64".** The audit-#1 bug-#2 contradiction wasn't killed — it survives in the worker bible (every-task file), now contradicting traps.md §2 and ort_reality.md (INT32 works when depth dtype matches). Fix the line.
**P2 — `task_techniques_index.md` header eviction did not happen, and line 14 FALSELY says it did.** The ~12 KB dated "submit-confirmed lessons" block (lines 14–92) is still there; `recent_ships.md` is an empty 0.2 KB stub; the in-file note "battle reports moved to recent_ships.md" is untrue. Either do the move or fix the note — a false claim in a router is worse than no move.
**P3 — `scoring_internals.md` got no redirect stub** (the other 4 Ort files did). 4 live dangling refs: `ort_reality.md:10`, `task018.md:620`, `task185.md:198`, `task366.md:411`. Fix: add the 5th stub pointing at `scoring_internals.md` (profiling section) — the form core.md:133 already uses — or rewrite the 4 refs.
**P4 — two MORE techniques.md collisions the original audit missed: `# 33.` (lines ~1759/1827) and `# 34.` (~1779/1871) each appear twice.** Pre-existing, actively cited by 3 Already_Ran prompts (§33/§34). Renumber one of each pair with breadcrumbs, same recipe as #48-54.
**P5 — prompt template still points at deleted `_METHODOLOGY_UPDATE_2026-05-25.md`** (`_TEMPLATE...Radical Rewrite.md:45`). Every future prompt cloned from the template inherits the dead link. Point it at current_strategy.md.
**P6 — stale internal pointers (3)**: `current_strategy.md:21` → "see traps.md SHIP DECISION REFERENCE" (the block now lives in current_strategy itself); `core.md:148` → cascade_strategy.md "full playbook" (now `Logs/cascade_campaign_log.md`); `core.md:129-135` → "read the whole `ort_reality.md` folder" + per-file paths (now stubs; should say ort_reality.md).
**P7 — `by_symptom.md:15`** still references `_archive/arc_rule_taxonomy_alt.md` and mislabels it "18-family" (the alt is the 15-family one) — a fresh mislabel introduced next to the bug-#3 fix.
**P8 — 5 small salvage facts dropped in the op-shelf merge** (live-tree losses; all recoverable from `_archive/`): (a) And-vs-BitwiseAnd mnemonic (bool→bareword, int→Bitwise prefix — mod_and_bitwise.md:146); (b) ReverseSequence batch_axis/time_axis defaults-are-(1,0) trap (spatial_rearrangement.md:208); (c) graphsurgeon doesn't-enforce-banned-ops caveat; (d) DepthToSpace blocksize-must-divide-30 ({1,2,3,5,6,10,15,30}); (e) GatherND/ScatterND explicit win-thresholds (~50 cells; N<~280). One paragraph in op_cost_patterns.md restores all five.
**P9 — live prompt `Task Prompts/1/...Task018...md:11`** cites archived `multi_axis_indexing.md` bare (folder-2 prompts were fixed; this one wasn't). Point at `op_cost_patterns.md (multi_axis salvage)`.
**P10 — MACs purge is banner-over-stale-text**, the audit's named anti-pattern: formula corrected at all 3 sites ✓, but the MAC-era body remains (68.3 KB vs ~30 target; ":72 the only path to sub-9k-MAC tasks" still present; only 83 lines actually removed). Acceptable as deferral; flagging because the commit message reads as if archived.
**P11 — traps.md anchor gap**: INDEX advertises `§CROP-CONTRACT` but it's not in traps.md (lives at current_strategy.md:81). A worker grepping traps misses it. Add a one-line anchor/pointer in traps.md.
**P12 — INDEX contract violations (2)**: `recent_ships.md` and `Algorithm_Library/per_family_starting_points.md` exist but aren't routed in INDEX ("if a file isn't listed here, agents won't find it").
**P13 — missed in-phase deletions**: synth_oracle_strategies.md still carries its 152-line verbatim self-appendix (planned deletion); `research_card_cull_2026-05-26.md` still live in Research/notes (plan: archive); exemplar dedups (task117 twice-pasted script, task384→pointer) not done.
**P14 — charter-inconsistent leftover**: `live_costs_2026-05-23_superseded.json.bak` kept at The_400_Tasks root (charter §8: dated snapshots die when canonical refreshes). Delete or move to _archive.
**P15 — stray file**: `Task_by_Task/task228_delta_t_e06c43ad.md` (1.8 KB, hash-named scratch from 2026-05-18, breaks the 5-H2 contract count). Archive it; canonical task228.md is healthy.

---

## 3. Deferred-work register (consistent with commits — nothing silently dropped, but still open)

**Audit #1**: techniques.md deep compression (204 KB vs ~100 target; verdict-index + ship-table eviction done, §39 stack only partially consolidated — 6 SUPERSEDED-tagged blocks remain); scoring_internals MAC-content removal (68.3 vs ~30); traps ≤200-line table restructure (82.3 KB long-form vs ~10-12; diary→appendix done); cost_floor calibration-dump compression (40.6 vs ~22; ship-ledger→33-row table done); taxonomy rewrite (52.8 KB — merge done, generic 18-family bulk remains); SPEC worker/captain split (16.7 KB single file); §4d deliverable compressions (tiny 64.5/mlp 38.2/synth 46.3/CA 17.8/symmetry 16.9). **Every-task set currently ≈154 KB (INDEX budgets honestly record current sizes) vs the ~65 KB target — the headline goal lands only when the traps-table + cost_floor + techniques restructures ship.**

**Audit #2**: Phase-1b generator trims NOT started (SPREAD no-op rows, HIST infra-commit filter, AUTO **STATUS-line emission** — `tools/generate_task_mds.py`, note: at tools/ root, not agent_helpers/ as the plan implied); Phase-2 whale compaction (top whales now task018 **109.2 KB**, task366 **90.1** — both fenced in-flight AND absorbing heavy 06-04 worker appends; compact + regen them when their workers close); Phase-3 (WORKER_PROTOCOL.md absent; template still 21.8 KB; Already_Ran 154 files; CASCADE_WORK_LOG 43 KB + Candidate_Briefs still at root; live prompts still carry the duplicated boilerplate block); Phase-4 (lessons.md 289.4 KB / 240 headers — grew; 93 auto_triage + 18 ledger .baks at Logs root; loose task264 zips). **Two regen tools confirmed broken and bannered, blocking two Phase-0 closures**: `cascade_report.py` (worktree path bug) and `split_alpha_lanes.py` ("no rows" schema drift) — repair these to refresh CASCADE_ANALYSIS.md and alpha_sub_lanes.md.

---

## 4. Suggested order

1. **P1–P9 in one hotfix pass** (~30 min: eight one-liners + one paragraph + one renumber pair) — they're all active mis-routing or contradiction, and P2's false note is the worst kind of doc-debt.
2. Repair the 2 broken regen tools (unblocks alpha_sub_lanes + CASCADE_ANALYSIS freshness).
3. Audit#2 Phase-1b generator trims (SPREAD/HIST/AUTO-STATUS — biggest remaining mechanical win, −~600 KB + the STATUS line all 402 files need).
4. Then the deferred restructures in audit-plan order (traps table → cost_floor → techniques/scoring deep trims → whales when fenced workers close → prompts/Logs).

*Evidence trail: 6 verifier reports in this session (Phase-0 content checks, Phase-1/2 moves, reference-integrity sweep, audit#2 Phase-0/1, 44-jewel survival sweep, whale-integrity + remaining-work). No edits were made during verification.*
