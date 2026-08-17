# AUDIT 2026-06-04 — The_400_Tasks + Logs (task-memory layer) context-bloat + refactor plan

**What this is**: companion to `AUDIT_2026-06-04_context_bloat_refactor_plan.md` (the Onnx_Advanced library audit, now being implemented). This one covers the next-largest offender: **the task-memory layer** — `The_400_Tasks/` (8.22 MB md, 721 files) + `Logs/` (1.94 MB md, 489 files / 56.7 MB total). Executed by 7 parallel auditors (all 6 whale files + generators + protocols read in full) with owner-side spot-verification of every load-bearing claim. **Internal audit deliverable, not a ChatGPT research card.**

Same rubric as audit #1 (A keep / B compress / C generic / D debris), adapted for *memory* files: A = current state, decoded rule, don't-do verdicts with evidence, floor proofs, hidden-correctness measurements; D = version-by-version sagas, stacked retracted proofs, git-log dumps, campaign lore retold per-task.

---

## 1. Headline

| Metric | Today | After refactor |
|---|---|---|
| Task_by_Task corpus (401 taskNNN.md) | 5,676 KB, avg 14.5 KB | ~2,400 KB, avg ~6 KB |
| — of which machine-generated | **55.5% (3,152 KB)** | ~21% (1,208 KB) |
| — of which GDIAG node dumps | **1,104 KB = 19% of corpus** | 0 (one generator edit) |
| Whale files (top 8, the most re-picked tasks) | 570 KB | ~93 KB (−84%) |
| Strategy layer (current_strategy + cascade docs + briefs) | 162 KB | ~42 KB live |
| Task Prompts tree | 657 KB | ~80 KB live |
| lessons.md | 289 KB, never deduplicated | ~100 KB consolidated |
| Token bombs agents are pointed at | 2 unguarded (412k + 272k tokens) | 0 |

**The single highest-leverage edit in this entire layer**: stop emitting the GDIAG "Nodes (in order)" `<details>` table in `tools/agent_helpers/regenerate_graph_diagnostics.py` (~lines 315–327) and re-run the pipeline. **−1,104 KB across 401 files for one edit, zero information loss** (the .onnx is canonical; verified: task205.md is 1,060 lines of which 687 are node-table rows for a graph ~17 versions stale).

**Key structural difference from audit #1**: this layer's bloat is mostly *machine-emitted* (4 generators own 55.5% of the bytes) and *append-protocol-driven* (workers are mandated to append every attempt; nobody is mandated to compact). That means most of the fix is generator edits + protocol rules, not manual rewriting — only the ~38 whale files need skilled hand-compaction.

---

## 2. Token bombs (most urgent — these end sessions instantly)

1. **`task_state.csv` (1,064 KB ≈ 272k tokens) — unguarded AND actively pushed.** `Task Prompts/_Before writing prompt.md:2` literally says "read task_state.csv" (verified); `.ai_instructions/04_DATA.md:13` and `The_400_Tasks/README.md:107` also route agents at it. There is **no per-task row-query helper** in `tools/agent_helpers/` (`tools/state.py` exists but prints a global snapshot, not `--task N`). An agent that follows the written instructions loads ~272k tokens → instant context kill. **Fix: add `tools/agent_helpers/show_task.py --task N` (or `state.py --task N`), rewrite the 3 instruction lines to call it, add a "⚠ never Read whole — query one row" banner to SCHEMA.md/04_DATA.md.**
2. **`artifact_manifests/2026-05-07-preserved-files-manifest.txt` (1.6 MB ≈ 412k tokens)** — the largest latent bomb; only its README references it today, but nothing warns against Reading it. Add the same do-not-Read guard line.
3. Honorable mention: `NeuroGolf_Task_Status.xlsx` (220 KB binary, low practical risk), `submission_candidate_tasks.csv` (207 KB, tool-only — fine).

## 3. Correctness / contradiction bugs (fix regardless of refactor)

1. **current_strategy.md contradicts itself on the live strategy.** Header (L9/L12, 2026-06-01 reconciliation): `solved_status` flags STALE, "no +full-points unlocks to chase", zero-weight-family theory **REJECTED**. Yet L150–164 (2026-05-27 era) still presents "Outstanding unlock candidates… Combined remaining unlock ceiling: up to +44 LB… Complete zero-weight family (13 confirmed members)" as live guidance (verified L164). A captain who scrolls past the header gets a rejected strategy. **Delete L150–164** (the 13-member family list survives in traps.md §27 per audit #1's crown-jewel list).
2. **Copy-paste duplication in the captain doc**: the same "BATCH-7 COMPLETE 2026-06-04 night: 10/10 delivered, +21.4 actual…" sentence appears twice verbatim (~L77 and ~L79).
3. **`live_costs.json` inversion**: the canonical-named file (mtime 05-23) is OLDER than the orphaned dated copy `live_costs_2026-05-30.json` (which matches current package costs). Tools point at the stale one. Regenerate `live_costs.json` (or promote the 05-30 data) and archive the dated copy; delete superseded `latest_package_costs_2026-06-01.csv` (the glob picks `-06-02`).
4. **`alpha_sub_lanes.md` stale numbers in an every-worker-session doc** (7 playbook refs call it "source of truth"): the lane *partition* is current, but the cost/"worst-scoring first action" tables are May snapshots (its own L3 says re-run after each batch; it wasn't). **Re-run `tools/split_alpha_lanes.py`.**
5. **`_METHODOLOGY_UPDATE_2026-05-25.md` mis-states task outcomes**: claims task216 "EXHAUSTED" — task216.md:73 shows v7 SHIPPED +1.97 LB. The file's own banner defers to current_strategy; it's a stale D-layer in the prompt-authoring chain (template:45 → METHODOLOGY_UPDATE:5 → current_strategy — three stacked "ignore the above" hops).
6. **`runs.jsonl` is 6.41 MB, over its own documented 5 MB rotation threshold** (04_DATA.md:32); `tools/rotate_runs_log.py` exists — run it.
7. **task110.md has TWO different "v51" sections** (L1060 crop-29 @15,202 and L1104 drop-P2 @22,059) — same label, parallel branches, both with full floor proofs.
8. **lessons.md superseded-in-place corpses**: L1226 titled "[REVISED — now we know it's RAW]" restates the lesson L1186 already corrected; "Kaggle CLI requires literal submission.zip" duplicated verbatim at L487 and L1414; ordering is non-monotonic (06-04 entries at the BOTTOM of a "newest-first" file).
9. **Prompt boilerplate drift is live**: `Task Prompts/1/...Task198...md` carries the identical library-upgrade block **twice** (original + "CAPTAIN REFRESH" footer) because the library changed mid-wave — the exact failure mode the per-prompt copy-paste design guarantees.

---

## 4. Consumption + ownership model (what makes this layer safe to cut)

- **taskNNN.md**: read partially (diagnostic + Attempts + Don't-do), once per claim, by exactly one worker; **never cross-task** (KanbanWorkerLoop.md:181 "your only universe is task209"). Bloat multiplies through **re-picks of the same whale** — and policy maximizes whale re-picks (agent_feedback_rules §4b "grind for weeks… release only on proof-of-floor"; Batch-9/10 re-targets 233/286/018/366/157/198/270).
- **Four generators own marker-bounded blocks, all replace-in-place (verified: zero duplicate markers in 401 files)**: `generate_task_mds.py` → `AUTO:`, `mine_task_history.py` → `HIST:`, `spread_annotate_notes.py` → `SPREAD:`, `regenerate_graph_diagnostics.py` → `AUTO-GRAPH-DIAG`. Growth in the manual sections comes from the mandated append-protocol ("update with every attempt", "≥3 attempts documented" gate) with **no compaction step anywhere**.
- **Restructure contract (the blast radius)**: keep the 4 marker pairs AND the five H2 names (`## Rule`, `## Algorithm sketch`, `## Attempts`, `## Cost analysis`, `## Don't-do log`) byte-stable — `generate_task_mds.py` both emits them and **parses legacy files by H2 name** (key_map), and `regenerate_graph_diagnostics.py` seeds `## Attempts` on new files. Everything else (prompt wording, README) is doc-drift, not code breakage. `uncashed_prototypes.py` reads HIST content — keep HIST markers, trim contents.
- **Strategy docs**: current_strategy.md = captain per-session read (95 refs repo-wide); CASCADE_WORK_LOG.md = **zero inbound references, orphaned**; Candidate_Briefs = orphaned 05-16 dispatch snapshot; alpha_sub_lanes.md = every worker session (regenerated, keep fresh).
- **Task Prompts**: live prompts are 3–5 KB; **2,692 B of each is a byte-identical boilerplate block (md5-verified across all 20 active prompts) ≈ 54 KB/wave of hand-synced duplication**. `Already_Ran/` (541 KB, 82% of the tree) has **zero programmatic or instructed readers** and spot-checks show all verdicts already migrated to taskNNN.md.
- **Logs**: load-bearing = lessons.md, submissions_ledger.csv (live, row appended today), runs.jsonl, README, PROBES ledgers (aging). Everything else at root — 93 `auto_triage_*.md`, 12 worker packets, 18 `submissions_ledger.csv.bak_t_*` — has zero inbound refs. The 56.7 MB non-md mass is regenerable scans/spread dumps (~35–40 MB compressible) + ignored scratch.

---

## 5. Disposition by area

### 5a. The four generator blocks (one-time edits, then re-run pipeline — saves ~1,944 KB / 34% of corpus)

| Block | Now | Action | After |
|---|---|---|---|
| GDIAG (1,811 KB) | header + flags + histogram + initializers + **node-by-node dump** + suggested-techniques w/ hardcoded exemplar paths + ORT-note sentence ×401 | **Drop the node table** (−1,104 KB); keep header/flags/histogram/top-3 initializers; replace per-file technique-prose + ORT-note with one pointer line (routing lives in by_symptom.md); regenerate **against the staged/shipped graph, not stale canonical** — or skip regeneration for tasks where staged ≠ canonical (whales' diagnostics currently describe graphs ~17–70 versions dead) | ~578 KB |
| SPREAD (748 KB) | 8-pattern verdict table ×400, **66% of rows are "no X detected" no-ops**, + legend + re-run footer ×400 | Emit only non-null verdicts + "Do NOT apply" + "Free LB" rows; legend/footer → one line | ~357 KB |
| HIST (464 KB) | git-log mining incl. **~8 repo-wide infra commits repeated in all 401 files** (38% of block) | Filter infra commits; cap at task-touching commits; keep markers (uncashed_prototypes.py reads it) | ~245 KB |
| AUTO (129 KB) | xlsx pointer + mostly-empty "linked research" | Emit a **STATUS line** instead: `LIVE cost / best staged / verify / hidden-class / last LB` from task_state.csv — this becomes line 2 of every file (fixes the "no current state at top" problem for free) | ~28 KB |

### 5b. Manual sections — whales first (hand-compaction, ~38 files, saves ~1,340 KB)

Verified anatomy: **no whale has current-state at top** — state is buried at L114–567+ behind boilerplate and chronology. Growth engines: append-only version sagas (task366 narrates v1→v198 TWICE; task018 has 60+ attempt rows), stacked retracted floor proofs (task018 carries 4, ~3 retracted; task286 keeps a full floor proof ABOVE its own "⚠ THE ABOVE FLOOR CLAIM WAS WRONG" stamp), duplicate template sections from workers re-pasting headers (task233 has THREE `## Rule` blocks, TWO Attempts, THREE Cost-analysis), and campaign lore retold per-file (cherry13b story in **22 task files**; task110's −12.80 scar re-narrated in ≥4 other files; "ORT OneHot INT32/INT64" restated 5× within task110 alone).

Standard template (hard budget ≤15 KB): `# taskNNN — STATUS:` line (machine-emitted) → `## RULE` (≤8 lines) → `## CURRENT BEST` → `## DON'T-DO` (one row per ruled-out approach: approach / version / delta / why) → `## OPEN IDEAS` → `## FLOOR PROOF` (**the current one only — delete retracted ones**) → `## COMPRESSED HISTORY` (one line per version-GROUP with pivotal anchors) → trimmed auto blocks. Keep the five H2 names for the parser contract; the new sections nest under them where needed.

Per-whale targets: task018 94→14, task233 81→13, task255 79→13, task366 78→12, task110 68→11, task205 61→8, task286 59→12, task158 50→10 KB. **Crown jewels that MUST survive compaction** (per-file examples): task286 "D\*=exactly 20 reach-2sym macros, pinned on the hidden test"; task018 dual min+max anchor correctness requirement + F32-scatter floor; task233 self-consistency-trap writeup + v6/v7 hidden-distribution verdicts; task110 v28 −12.80 scar + v50/51 orbit-max rewrite; task366 zero-public-weight confirmation + cardinality-first rule; task205 QLinearConv count-detector lever + no-crop proof. Rule of thumb: every don't-do with an LB delta, every decoded rule, every hidden-measured fact survives; every blow-by-blow dies into one history line.

### 5c. Strategy/captain layer (162 KB → ~42 KB live)

| File | Verdict |
|---|---|
| current_strategy.md (82 KB) | **TRIM+RESTRUCTURE → ≤15 KB live** (Current state / SHIP DECISION REFERENCE / Active strategy / Don't-do rules / Open batches / Failure-mode pointers) + appendix-or-Logs for the 05-25→27 attempt diary (~18 KB, duplicates task files) and completed-batch ledgers (~12 KB). **Delete L150–164** (bug #1). **Create the `## SHIP DECISION REFERENCE` landing zone BEFORE audit #1 moves the traps.md ship-block here** — otherwise that move adds restatement #90 of the same vocabulary instead of consolidating. Fix the stale doc-hierarchy pointer to cascade_strategy.md (being stubbed). |
| CASCADE_WORK_LOG.md (43 KB) | **ARCHIVE → Logs/** — orphaned May 9–12 diary; its one durable lesson (visible≠hidden) is in lessons.md + meta-lesson #8. Must land in the SAME Logs home as cascade_strategy.md's diary (audit #1) — ONE cascade campaign log, not two. |
| CASCADE_ANALYSIS.md (9 KB) | KEEP (auto-generated by cascade_report.py) + add a one-line freshness banner; re-run after batches. |
| README.md (9 KB) | TRIM → ~7.5 KB (cut Migration-history block → git/CHANGELOG). Tool-runbook section is accurate and load-bearing — keep. |
| SCHEMA.md (5 KB) | KEEP + one-line advisory that unlock/cherry13b columns are now advisory-only (post-reconciliation). |
| alpha_sub_lanes.md (5.5 KB) | KEEP + **REGENERATE** (bug #4). |
| Candidate_Briefs/ (7.5 KB) | ARCHIVE (stale 05-16 scout dispatch; roster/locks/LB all expired). Future scout briefs → Logs/sessions/. |

### 5d. Task Prompts (657 KB → ~80 KB live)

1. **Create `Onnx_Advanced/WORKER_PROTOCOL.md`** (one canonical home: ng_gate catalog, verify/adversarial gates, staging+manifest schema, DO-NOT-submit rules, current library-upgrade digest). New prompt template ≤2 KB: task ID/objective + decoded-rule direction + task-specific §-pointers + ONE protocol pointer + deliverables. Kills the 2,692 B × 20 per-wave boilerplate (~54 KB/wave + the mid-wave drift failure mode, bug #9) and collapses audit #1's blast radius (template's 30 path-strings → 1). |
2. **Template**: rewrite to ≤2 KB shell (its Tier-0/1/2/3 reading tables restate INDEX.md's job; its cascade/unlock framing is banner-superseded; ~45% C + ~25% D). The A-content (hard constraints, forbidden moves, staging schema, reporting format) moves to WORKER_PROTOCOL.md. |
3. **`_METHODOLOGY_UPDATE_2026-05-25.md`**: fold surviving A-content (canonical-extract A/B discipline, F16-arithmetic trap, hash-MLP fingerprint — audit #1 already promotes the fingerprint) into core.md/traps.md; delete the file (bug #5). `_Before writing prompt.md`: keep (75% A) — it's the authoring-policy gold; fix line 2 (bug: token bomb instruction) to use the row-query helper. |
4. **Already_Ran/ (541 KB)**: archive out of the working tree (keep last ~7 days). Zero readers, no un-harvested lessons (spot-verified 3: verdicts all in taskNNN.md). |

### 5e. Logs (1.94 MB md → ~0.6 MB live; 56.7 MB → ~20 MB)

1. **lessons.md (289 KB, 220 entries)**: consolidate to 5 thematic sections (§GRADER & SCORING DOCTRINE / §SHIP-DECISION CALIBRATION / §TRAP TAXONOMY pointers / §FLOOR-PROOF STANDARD / §PROCESS), one dated entry per distinct lesson, strict newest-first. The 6× "Kaggle scores raw tensor" retellings → 1; the retired ≥95%-era adversarial trilogy → 1 line; per-task technique narratives → pointers to techniques.md/taskNNN.md. Target ~100 KB / ~85 distinct entries. **Preserve dated anchors that live prompts cite** ("2026-05-27", "2026-05-28 stub-probe campaign") — keep those dates as entry headers or update the 2 citing prompts. The 2026-06-01+ "Pointer entry" style (L2018) is the model — it's already self-correcting; codify it. |
2. **Root sweep → `Logs/_archive/2026-05/`**: 93 auto_triage_*.md + 12 task-packet reports + 16 of 18 ledger .baks + the 2 giant generated worker-group files (353 KB, generators already in tools/_archive). All zero-inbound-ref (verified). Extend the existing `sessions/_archive/` precedent to pre-June sessions. |
3. **Mass**: run `tools/rotate_runs_log.py` (bug #6); gzip the >1 MB scans/spread JSON/CSVs (~35 MB → ~4 MB); move the 3 loose `task264_v*/submission.zip` (2.9 MB) to staging dirs; `orphan_worktree_*_recovered_tools/` → archive; scratch/ self-clears (already ignored). |

### 5f. Data files + subdirs

- `task_state.csv` / manifest: guard per §2 (the #1 and #2 actions of this whole audit).
- `Task_by_Task/taskNNN/` subdirs: md layer is healthy (LIVE_ALGORITHM.md + prototypes/README.md are governed and load-bearing — KEEP; 5 orphaned `opus_rule_hypothesis_2026-05-05.md` → archive). The 289 MB of committed .onnx/.py/.pyc/.zip variants (task255 alone: 81 MB / ~60 build variants, +352 committed .pyc) is repo-weight, not context-weight — separate housekeeping pass: dedup old variants, gitignore __pycache__.
- Decoder scratch report bundle: superseded by durable task notes (`The_400_Tasks/Task_by_Task/task025.md`, `task198.md`) and removed in the 2026-06-12 cleanup.

---

## 6. Migration plan (phases ship independently; commit between each)

**Phase 0 — hotfixes (½ session, no moves)**: §2 token-bomb guards + `show_task.py` + rewrite 3 instruction lines · delete current_strategy L150–164 + dup Batch-7 sentence · rotate runs.jsonl · regen alpha_sub_lanes + CASCADE_ANALYSIS · fix live_costs inversion + delete superseded package-costs CSV.

**Phase 1 — generator trim (1 session, −1.94 MB, zero manual rewriting)**: edit the 4 generators per §5a (incl. STATUS-line emission) → re-run the full refresh sequence (README.md:94–102) → verify markers/H2 names unchanged → commit. Highest ROI in the layer.

**Phase 2 — whale compaction (1–2 sessions, manual, −1.3 MB)**: top ~38 files by size, standard template per §5b, crown-jewel preservation rules. Do AFTER Phase 1 so the auto-block noise is already gone.

**Phase 3 — strategy + prompts (1 session)**: current_strategy restructure (landing zone FIRST, coordinate with audit #1's traps move) · WORKER_PROTOCOL.md + minimal template · archive Already_Ran / CASCADE_WORK_LOG / Candidate_Briefs · fold METHODOLOGY_UPDATE.

**Phase 4 — Logs (1 session)**: lessons.md consolidation · root archive sweep · gzip/rotate/move mass.

Safety rails: the 4 marker pairs + 5 H2 names are byte-stable contracts (generate_task_mds key_map parses them); `tools/refactor_paths.py` for any path moves; live prompts cite lessons.md entries BY DATE — preserve dated headers; whale compaction happens between campaign waves (workers mid-grind hold these files in context).

---

## 7. Anti-re-bloat charter additions (extend agent_feedback_rules.md §8b with §8c — task-memory rules)

1. **STATUS line is line 2 of every taskNNN.md** (machine-emitted from task_state.csv). A reader learns LIVE/best/verify/hidden-class without scrolling.
2. **History = one line per version-group.** Full chronology lives in git; the file keeps verdicts + pivotal anchors (ship deltas, scars, unlocks). The "≥3 attempts documented" gate is satisfied by table ROWS, not narrative sections.
3. **Floor proofs supersede in place.** A retracted proof is deleted (CHANGELOG/git remembers), never stacked under a ⚠-banner.
4. **Campaign lore lives once.** cherry13b / zero-weight / calibration stories live in traps.md+current_strategy; task files cite, never retell (currently retold in 22 files).
5. **One `## Rule`/`## Attempts`/etc. per file** — workers EDIT existing sections, never paste a fresh template header (task233 has three Rule blocks).
6. **lessons.md: dedup-before-append** (grep task-id + keyword first; edit-and-date-bump on hit), ≤3 sentences + pointer per entry, strict newest-first, consolidation every ~40 entries.
7. **Token-bomb rule**: any file >50 KB that an instruction tells agents to read MUST ship with a row/section query helper and a do-not-Read-whole banner.
8. **Generated files are regenerated, never hand-edited**; dated snapshot copies (live_costs_YYYY-MM-DD, latest_package_costs_*) are deleted when the canonical refreshes.
9. **Prompts carry only task-specific content + one WORKER_PROTOCOL.md pointer**; dispatched prompts age out of the working tree after ~7 days.

---

## 8. If only 3 things happen

1. **Guard the token bombs + ship `show_task.py`** (§2) — the only items here that can kill a session TODAY, and the instructions actively cause it.
2. **Phase 1 generator trim** — one session, −1.94 MB (34% of the corpus), removes the stale-graph node dumps every worker wades through, and plants the STATUS line that fixes the buried-state problem across all 401 files at once.
3. **Resolve the current_strategy.md contradiction** (delete L150–164) — the captain's primary doc currently teaches a rejected strategy to anyone who scrolls.

*Full per-file evidence (line numbers, quotes, md5s, per-block KB accounting) is in this session's 7 auditor reports; this file is the distilled plan of record.*
