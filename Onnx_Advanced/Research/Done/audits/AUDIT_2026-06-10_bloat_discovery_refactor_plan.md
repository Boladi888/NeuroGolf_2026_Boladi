# AUDIT 2026-06-10 — Onnx_Advanced bloat + discovery-layer refactor plan (audit #5)

**What this is**: owner-commissioned full audit of `Onnx_Advanced/` (~3.3 MB, 219 files) against the folder charter — *"only info a Claude-Sonnet-class model doesn't already know"* — with the specific question: **is the library still navigable for a single-task worker hunting anything that gets cost down?** Executed by 5 parallel auditors (every live file read in full) + spot-verification of all load-bearing claims by the coordinating session. **No changes were made; this file is the only write.** Figures marked [est.] are auditor-sampled estimates; everything else was measured directly.

Prior art this builds on (all in `../Done/`): `AUDIT_2026-06-04_context_bloat_refactor_plan.md` (the master plan, "audit #1"), `VERIFY_2026-06-05_audit_implementation_check.md`, `AUDIT_2026-06-06_agent_context_bloat.md`, `AUDIT_2026-06-06_further_optimizations.md`. This audit does not re-litigate them — it measures what happened since, and what's newly broken.

**Sibling audit, same day**: `AUDIT_2026-06-10_onnx_advanced_navigation_bloat.md` (this folder) reached the same core diagnosis independently. §1b reconciles the two — its adopted findings are folded into the plan below; the one structural disagreement (split techniques.md vs single-file + generated catalog) is argued explicitly with a decision rule, not glossed. §3b adds the prompt-layer analysis (owner request, 2026-06-10): how `The_400_Tasks/Task Prompts/` actually constructs the worker read-path, and what that implies for both the library and the prompt template.

*Correction note (2026-06-10, same day)*: an earlier revision claimed `einsum.md` had zero live citers — wrong, an artifact of a verification command that hung (rg invoked with `--glob` filters but no path argument reads stdin forever in a non-tty shell; two such orphans were created and killed this session). Re-run with an explicit path: `einsum.md` is cited by INDEX.md, op_cost_patterns.md, techniques.md, and task_techniques_index.md. §4c and the stub recommendation are corrected accordingly.

---

## 1. Headline

**The content is winning; the plumbing is losing.** The June 6–10 campaign added ~81 techniques (#59→#139) that are overwhelmingly empirical, charter-compliant, decision-grade material [est. 85–90% A/B-grade] — exactly the moat the endgame needs. But every surface that requires a *human/sorter to update after content lands* is now 60–100 sections behind, and that — not raw KB — is why workers can't find what helps them.

| Metric | 2026-06-04 (audit #1) | post-refactor 06-05 | **today 06-10** | target |
|---|---|---|---|---|
| `techniques.md` | 208 KB | 206 KB | **348 KB, 140 §s (#0–#139)** | see §5 |
| …its verdict-index covers | — | §0–§42 (complete) | **§0–§42 (97 §s unindexed)** | all 140, generated |
| `by_symptom.md` routes of the ~81 new §s | — | — | **~10** [est.] | all high-value §s |
| exemplars covering #78+ | — | — | **0** | top new families |
| every-task required set | 127–290 KB | ~154 KB | **163 KB** (INDEX 8.8 + core 25.4 + cost_floor 41.4 + traps 79.9 + PROTOCOL 7.6) | ~75 KB |
| `WORKER_PROTOCOL.md` "library verdicts" | n/a | 1 line, ~1.5 KB | **one 4,035-byte run-on line, 35 `·`-clauses** | ≤10 lines + pointer |
| § number collisions | 5 fixed Phase-0 | 0 | **4+ recurred 06-06→06-08** (`techniques.md:2884,2908,2960,3687`) | 0, by construction |
| extracted deliverables awaiting rule-7 archive | 0 | 0 | **~275 KB** (k1–k5/l1–l2/m1–m3 in `Research/Done/`) | 0 |
| grep pollution (hits landing in `_archive/` etc.) | unmeasured | unmeasured | **31–34%** on `§39`/`Einsum` probes; **no `.rgignore` exists** | ~0 |

**One-line diagnosis**: growth is fine — *unindexed* growth is the bloat. The refactor that matters is (a) make the discovery layer **generated, not maintained**, (b) finish the June-4 every-task-set diet that never shipped, (c) turn the lifecycle rules into lint checks so they hold under campaign pressure.

### 1b. Reconciliation with the sibling audit (`AUDIT_2026-06-10_onnx_advanced_navigation_bloat.md`)

Two independent audits, same day, same bottom line: *"not bloated in the disk-space sense; bloated in the worker-navigation sense"* / *"the plumbing is losing."* Both say: full refactor yes, destructive no; preserve every empirical fact; the win is routing, not deletion. Where the plans differ:

**Adopted from the sibling audit into this plan** (it found these; this audit missed or under-weighted them):
1. **`ort_reality.md` is actively misleading** — it still says "required reading" and routes to the 5 dead stubs. Fixed in Phase A.
2. **First-read budget ≤50 KB** as the headline worker-entry target (sharper than this audit's 75 KB doctrine-set framing — see Phase B for how both compose: 75 KB is the *doctrine surface that exists*, ~53 KB is *what a worker reads before code*).
3. **Research/Done physical role-split** (`deliverables/` `card_bodies/` `audits/`) — composes with this audit's rule-7 archival: extracted bodies leave entirely, the remainder gets role-foldered. Phase C.
4. **Stub expiry rule** ("keep while live-cited; archive when rg-clean") — strictly better than this audit's original "delete now," which rested on a wrong measurement (see correction note above). Phase C + lint.
5. **`Research/Todo/README.md` ledger hygiene** — completed K/L/M/R ledger tables move to Done; Todo README stays a thin board. Phase C.
6. **"required reading" phrase lint** — the phrase is only allowed in WORKER_PROTOCOL.md/INDEX.md; everything else must route, not conscript. Phase D.
7. Additional evidence folded into §4a/§4b: physical §-ordering disorder in techniques.md (#25 after #31, #95 before #90, Rule 9 after Rule 12) and a duplicated visible trap number (§15 appears in two trap categories).

**Held against the sibling audit's recommendation, with reasons** — *split techniques.md into ~9 family pages + new `T067_*` stable IDs, techniques.md becomes a ≤30 KB index*:
- **Live prompts cite §-numbers TODAY** (`Task Prompts/2/...Task042...md:10` cites "#81"; `...Task130...md:12-13` cites "#67/#81", "#4 Rule A"). Renaming the key-space mid-campaign breaks the running citation layer for zero discovery gain — discovery is delivered by the *index*, which both plans require anyway (the sibling's own Phase 4 concedes a generated TECHNIQUE_INDEX).
- June-4 measured the blast radius: ~456 files / ~1,100 refs cite library paths; hundreds cite `techniques.md §N`. A split pays that cost now, during the endgame, against tombstone-maintenance forever.
- Nine family pages re-create the maintenance surface ×9 (per-page coherence + cross-file §-uniqueness = harder lint), and family boundaries are genuinely fuzzy for the best techniques (#81 is pad-floor AND output-assembly AND crossover-law; #95 is dtype AND TopK AND scatter).
- **Decision rule adopted instead**: catalog-first (Phase A). The generated catalog assigns family tags to every § — which makes a later physical split *mechanical and cheap* (refactor_paths.py + per-§ tombstones + the tag map) **if** single-file pain persists. Concrete trigger for revisiting: techniques.md exceeds ~450 KB, or post-catalog worker feedback still reports §-grep noise. Parked as optional Phase E, not killed.

Net: the two plans are ~85% the same refactor. This file remains the plan of record; the sibling audit's architecture sketch is the reference design for Phase E if the trigger fires.

---

## 2. What happened to the June-4 plan (honest scorecard)

| June-4 item | Status today |
|---|---|
| Phase-0 hotfixes (drift bugs, renumbers, jewel promotions, citeturn strip) | ✅ shipped, held |
| Entry-point collapse (QUICKSTART/README → INDEX), op-shelf merge, Ort_Advanced merge, archive sweep | ✅ shipped — but left **tombstone debris in INDEX rows + 11 stub files** (see §4c) |
| traps.md → ≤200-line table (~10 KB) | ❌ never started; still 80 KB long-form |
| cost_floor_reference → ~22 KB (calibration dump + proofs evicted) | ❌ still 41.4 KB |
| 00_COMPETITION_SPEC exits every-task set (worker facts → core) | ❌ still in "Start here" |
| scoring_internals MAC-era body purge (→ ~30 KB) | ❌ still 69.9 KB; lines ~99–170 even self-mark RETIRED 2026-06-07 [est.] |
| arc_rule_taxonomy rewrite (→ ~10 KB empirical spine) | ❌ still 52.9 KB, ~90% textbook ConceptARC |
| deliverable compressions (tiny 63, mlp 37, synth 45, CA 17, symmetry 16.5) | ❌ all untouched |
| techniques.md deep-compress (→ ~100 KB) | ❌ overtaken by events: +142 KB of *new empirical content* since; the 100 KB target is obsolete (see §5) |
| anti-re-bloat charter §8b | ⚠️ **partially holding**: core.md is disciplined (+0.3 KB in 6 days), supersede-in-place mostly followed in new §s; but ship-ledger leakage recurred (techniques.md:2768–2797 [est.]), ADDENDUM stacks survive in the §39/§41 era (7 blocks, techniques.md:2569–2798), and **rule 7 + rule 9 are dead letters** (275 KB unarchived; INDEX "budget" table was edited to match actual sizes instead of evicting) |
| repo_lint.py / context_pack.py (audit #4) | ⚠️ built but thin: lint has 11 checks — **none enforce library contracts** (no index-coverage, no routing-coverage, no set-size budget, no rule-7, no INDEX two-way check; `Logs/lint_last.md` shows only `dup-anchors` runs). context_pack.py shipped **without** the by_symptom/§-slice features its spec called for (`tools/agent_helpers/context_pack.py` packs state+STATUS+prompt+staged+prototypes only) |

Pattern worth naming: **everything that shipped was a one-time fix; everything that required *ongoing* discipline regressed.** Audit #4 predicted exactly this ("findings someone fixed once" vs "standing invariants"). The plan below therefore pairs every cleanup with the lint check that keeps it fixed.

---

## 3. What a single-task worker pays today

Required reading per WORKER_PROTOCOL: INDEX (8.8) + core (25.4) + cost_floor (41.4) + traps (79.9) + PROTOCOL itself (7.6) ≈ **163 KB ≈ 41k tokens before touching the task** — 2.5× the June-4 target. Then discovery:

- **"What was done to similar tasks?"** → task_techniques_index.md (68 KB) — works, but ~40 KB of it is dated ship-battle prose a current worker doesn't need (`task_techniques_index.md:16–117`).
- **"I see pattern X in my graph"** → by_symptom.md (33 KB) — gold-standard format, but blind to ~71 of the 81 newest techniques [est.], including the current campaign's core levers (#82 free-output multi-term Einsum, #89 einsum-final contract, #104 linear-map decision procedure, #109 whole-one-hot output, all of Aditya #111–#127 / Boladi #128–#139).
- **"Anything else that could help?"** → no surface answers this. The verdict-index stops at §42; the only "complete" view is the WORKER_PROTOCOL 4 KB run-on line (35 clauses, unreadable) or reading 348 KB linearly. Keyword grep into techniques.md returns ~1,200 hits for `Conv` [est.]; 31–34% of repo-wide § greps land in `_archive/`/`Research/Done` copies.

Realistic session cost observed: **60–100k tokens of context before the first build attempt**, much of it spent *failing to find* things that exist. With ~44% of remaining LB points sitting in the 10–50k mid-band (audit #4 §6 — breadth grind, many cheap iterations), per-iteration context efficiency is the throughput lever. This is the cost the refactor buys back.

### 3b. How prompts actually construct the read-path (owner request — and it changes the design)

Read for this section: `The_400_Tasks/Task Prompts/_Before writing prompt.md` (9.8 KB authoring policy), `_TEMPLATE_Prompt_TaskNNN_Radical Rewrite.md` (1 KB Phase-3 shell), live batch-17/18 prompts in `1/` and `2/`.

**The prompt shell is the healthiest layer in the whole pipeline.** Current prompts are ~1.5–2.4 KB: protocol pointer first (`WORKER_PROTOCOL.md` + `INDEX.md`), why-this-pays, target, a captain-decoded **Direction with hand-picked §-cites** (`Task042:10` cites "#81 Where-as-output" + a `task341.md:94–115` exemplar pointer; `Task130:12–13` cites "#67/#81", "#4 Rule A", "traps §FAMILY-LABEL"), recon commands, gates, staging. The audit-#4 complaints (embedded gate boilerplate, dated library digests in every prompt) were fixed by the Phase-3 template. Keep this shape.

**The implication the library plan must absorb: discovery happens at prompt-WRITE time, by the captain.** The worker is routed to 2–5 §s chosen by the captain from memory of a 140-§ library whose index froze at §42 and whose digest is a 4 KB run-on line. Two failure modes follow:
1. **Captain-side misses** — the prompt under-cites because no complete, scannable § inventory exists. A missed cite costs a whole worker-session of grep-luck or a missed lever. *The generated catalog's first customer is the prompt author, not the worker.*
2. **Worker-side dead-ends** — when the prompt's angle fails mid-session ("anything else that could help?"), today's fallback surfaces are the frozen index, the run-on digest, or a 348 KB linear read. This is the owner's reported pain, and it is exactly the gap between "prompt-cited §s" and "library contents."

Also confirmed by the prompt layer: **§-numbers are the live citation API.** Active prompts cite `#81`/`#67` bare. Any renumber/rename scheme (including the sibling audit's `T067_*` IDs) breaks prompts already in flight — §-key stability is non-negotiable (see §1b).

**Prompt-structure amendments recommended** (owner confirmed the template is amendable; all land in Phase A):
- `_Before writing prompt.md` gains a mandatory **DISCOVERY step**: before writing the Direction, grep `techniques_catalog.md` (once it exists) by the task's family/GDIAG flags + scan `by_symptom.md` routes; the prompt must carry 2–5 §-cites + ≥1 exemplar/task-file pointer (codifying what the best current prompts already do).
- Template gains one fallback line: *"Angle dead-ends? → `by_symptom.md` + `techniques_catalog.md` (complete one-liners, ~10 KB). Do NOT linear-read techniques.md."* That single line converts the worker's worst-case from 60–100k tokens of spelunking to a 10 KB scan.
- Prompts stop restating gate doctrine (the `Task130:20` gates paragraph re-explains §14e/truth-gate semantics that WORKER_PROTOCOL owns — drift class): one line, `Gates: WORKER_PROTOCOL §Gates — Path A: ab · Path B: truth ≥99% · gate-5 profile`, plus only task-specific gate notes (e.g. task042's "gate `ab`, not truth").
- `context_pack.py --task N` embeds the §-texts the prompt cites + their catalog family-neighbors (finishing its original spec) — the worker's first command then materializes the whole curated slice, and the prompt stays a 2 KB shell.

---

## 4. Layer-by-layer findings (evidence)

### 4a. techniques.md — 348 KB, 140 §s, the index froze at §42

- **Inventory**: 140 numbered headers (#0–#139, measured). Newest blocks: einsum-final family (#89–#95), M-card extractions (#104…), **Aditya idiom harvest #111–#127 (~18 KB)**, **Boladi harvest #128–#139 (~12 KB)** — dated through 2026-06-09.
- **Verdict-index** (lines 1–14) is Phase-3 front-matter dated 2026-06-05, covering §0–§42 only. No TOC. **97 sections have no index entry anywhere in the file.**
- **Collisions recurred post-refactor**: `#58→#62`/`#58b→#62b`/`#59→#63` renumbered 2026-06-06 (lines 2884/2908/2960), a collided `#92` renumbered 2026-06-08 (line 3687). Root cause: parallel workers append directly to the file and self-assign "next free number," bypassing the `.update/` single-writer sorter design (`.update/README.md:5–7,60` explicitly forbids direct edits). Breadcrumb convention is being followed (good), but the collision class is structural until number-assignment is single-writer.
- **Physical ordering disorder** (sibling-audit find, verified): top-level numbers backtrack (#25 after #31, #22 after #25, #95 before #90; "Rule 9" after "Rule 12" inside the bool-first block) — harmless to §-grep, hostile to linear reading, and another reason the catalog (sorted, complete) must be the browse surface, not the file.
- **ADDENDUM-stack residue** (charter rule 3): the §39 cluster spans lines ~2569–2798 with 7 ESCALATION/VERDICT/GAP-LEDGER/MECHANISM blocks (one self-marked SUPERSEDED, still in place); §41 carries 3 addenda (lines ~1911, 2206, 2646). New-era sections (#93+) are largely clean — the discipline improved, the old stack was never consolidated.
- **Ship-ledger leakage** (charter rule 4): per-task repricing tables at lines ~2768–2797 [est.] (task094 trilogy, task014/065/398 rows) belong in task files.
- **Format by era** [sampled]: early §1–§20 are verbose (~100–130 lines/§, ~20–30% compressible); mid-era ~10%; #93–#139 are already tight (10–27 lines/§, ~0–5%). **The new content is not the bloat problem.**
- **Generic-knowledge share**: ~12–15% [est.] (#0 strategy doctrine, #17 opt_safe micro-patterns, parts of #5's algebra, K1 survey prose). Low — the charter is mostly working at the door; the file's problem is structure, not junk.
- **Redundancy**: BOOL family restated across #1/#27/#57; F16 across #4/#77/F16-CONV-VERDICT; one-hot across #8b/#23; §39 gap-ledger appears 4×. ~5–10 KB recoverable, more value in coherence than KB.

### 4b. traps.md (80 KB) + traps_appendix.md (72 KB) + by_symptom.md (33 KB)

- traps.md: 48 sections (40 numbered traps + 8 doctrine blocks), **avg 11.2 lines/trap vs the ~4 a table row needs**; ~15–18% is dated incident diary again [est.]. ~60% paid-for empirical, and the "generic" remainder is mostly competition *methodology* doctrine that must stay (it's project-specific, just mislabeled generic by sampling). Sibling-audit find: visible trap number **§15 appears in two categories** (runtime + scoring) — brittle for prompt cites; renumber one side during table-ization.
- Table-ization (the June-4 order) is feasible: 40-row table (~6.5 KB) + 8 irreducible doctrine blocks (~4 KB) + 1–2 worked examples per block → **realistic 14–16 KB**, not the plan's 10. Load-bearing grep anchors that must survive as section keys: §VISIBLE-FAIL, §CLASS-CHECK, §DUAL-TRUTH, §GRADER-GAP, §FAMILY-LABEL, §14z, §CROP-CONTRACT (note: CROP-CONTRACT still lives in current_strategy.md — the VERIFY P11 pointer gap is still open).
- traps_appendix.md: ~150 lines of **live doctrine leaked into the grep-only file** — §K2 ACT-NOW PAIR, §FAMILY-LABEL detail, §GENERATOR-INVARIANT-vs-OFFICIALS-QUIRK should promote to traps.md; trap §5 is fully duplicated in both files; ~200–250 lines are pure-debris dated calibration entries [est.].
- by_symptom.md: 66 routing entries, format still gold-standard; references reach #88 and stop. **The discovery gap of §1 lands here** — backfilling ~25 high-value routes costs ~1.5–2 KB of file and one session.

### 4c. Entry/routing layer — INDEX decayed into a tombstone map

- `INDEX.md:35–36` routes `scoring_internals.md` twice with contradictory roles ("merged jewel" vs cost-golf ground truth); `INDEX.md:42–44` lists `arc_rule_taxonomy.md` **three times**; `INDEX.md:64–72` lists `op_cost_patterns.md` **five times** with "(salvage; archived)" archaeology; `INDEX.md:84` names `ort_reality.md` **four times in one row**. A new worker cannot tell what these annotations mean — they're merge history, not routing.
- **Two-way contract violations** (INDEX's own rule, line 92): `WORKER_PROTOCOL.md` — the canonical worker contract — **is not routed in INDEX at all** (measured, zero hits); `ort_reality.md` (51 KB crown jewel) has no top-level row; `recent_ships.md`, `traps_appendix.md`, `per_family_starting_points.md` unrouted.
- The EVERY-TASK SET table (`INDEX.md:95–107`) was **edited to record actual sizes (traps 82 KB) instead of enforcing budgets** — charter rule 9 inverted ("budget" now documents the bloat).
- `WORKER_PROTOCOL.md:21` — the "Current library verdicts" digest — is one 4,035-byte line with 35 `·`-separated clauses, appended every wave. It has become the *only* complete-ish index of recent techniques, which is exactly backwards: the protocol file grows without bound while the real index sits frozen.
- **11 stub files** (7 root: einsum.md, function_proto_subgraphs.md, multi_axis_indexing.md, row_signature_rank_reduction.md, radius_encoded_weighted_cover.md, direct_detection_multi_qlinearconv.md, multistage_key_first_compositional.md; 4–5 in Ort_Advanced/) totaling ~4.5 KB. **[Corrected 2026-06-10]** einsum.md IS still cited live (INDEX.md, op_cost_patterns.md, techniques.md, task_techniques_index.md); an earlier "zero citers" claim came from a hung verification command. Adopt the sibling audit's **expiry rule** instead of immediate deletion: a stub lives while `rg` finds live citers outside `_archive/`; repoint citers as routers get rewritten (Phase A), then archive the stub — lint `dead-doc-refs` guards the transition.
- **`ort_reality.md` actively mis-routes** (sibling-audit find, verified): it still opens "required reading" and instructs workers to read the five files that are now 0.2 KB stubs. Replace with a 2-line redirect to `../ort_reality.md` (+ `../scoring_internals.md` for profiling) in Phase A.
- task_techniques_index.md (68 KB): header note at line 14 is now honest (P2 fixed) but the ~40 KB battle-prose eviction it promises is still pending; `recent_ships.md` remains a 0.4 KB tombstone awaiting it.
- Healthy small files (keep as-is): patterns.md, diagnostics_recipes.md, zero_param_generators.md, cascade_strategy.md (1.3 KB worker stub — the June-4 trim actually landed here), op_cost_patterns.md, core.md (**the discipline showcase: +0.3 KB in 6 days of heavy campaigning**).

### 4d. Domain shelf — the June-4 verdicts still apply, unexecuted (~378 KB → ~250 achievable)

| File | KB | Verdict (unchanged from June-4 unless noted) |
|---|---|---|
| scoring_internals.md | 69.9 | TRIM → ~25–30. MAC/sub-byte era body persists; part self-marked RETIRED 2026-06-07 [est. lines ~99–170]; grader-replication + exploit verdicts are the irreducible core |
| arc_rule_taxonomy.md | 52.9 | REWRITE → ~8–10 empirical spine (the 23-task coverage table + 6 NeuroGolf-only families); archive the 18-family textbook bulk |
| tiny_architectures.md | 63.0 | TRIM → ~35 (LUT-as-network block, activation menus, twice-printed Sobel kernels out; depthwise 10×, unroll law, power-iteration stay) |
| synth_oracle_strategies.md | 45.2 | minimum: delete the ~150-line verbatim self-assignment appendix (planned since June-4, VERIFY P13); full: compress to rubric+recipes ~8–12. Auditors split on depth — decide via the K/L/M precedent: extract-then-archive the deliverable body |
| mlp_rule_extraction.md | 37.3 | COMPRESS → 5–25 (range = same split); **elevate the hash-MLP fingerprint addendum to the top** — it's the load-bearing part and it's buried at the end |
| cellular_automata.md | 17.4 | TRIM → ~9 (extractions confirmed in techniques.md) |
| symmetry_and_affine.md | 16.5 | KEEP — actively maintained (2026-06-10 GridSample grader validation landed here) |
| 00_COMPETITION_SPEC.md | 16.7 | SPLIT + exit every-task set (worker facts → core.md, captain logistics → The_400_Tasks/) |
| ort_reality.md | 51.1 | **KEEP — best file in the library.** K2/M1/wave merges verified coherent, matrix still single+canonical, no internal contradictions found |
| Algorithm_Library/ (3 files) | 36 | KEEP all |
| Worst cross-file duplication | — | scoring formula in 3 homes; cost ladder in 2; boolean-via-min/max in 2 — consolidate to core.md + pointers (drift class that produced 3 of the 4 June-4 bugs) |

### 4e. Lifecycle layers — the contracts exist; nothing enforces them

- **Research/Done (568 KB, 34 files)**: all 10 K/L/M full deliverables (~275 KB) have **verified-faithful extractions** in techniques.md/ort_reality.md (4/4 spot-checks: K2→ort_reality:563+, L2→#89 @3449, M3→#104 @3824, M1→#94+ort_reality — extraction quality is genuinely good), yet **all 10 bodies still sit in Done**, violating charter rule 7 ("archive in the same commit"). The 10 summary cards + 5 R-cards + 6 audit/process notes legitimately stay.
- **.update/ (87 KB)**: `_integrated/` holds 20 consumed drops. The protocol's own line (`.update/README.md:67`) says the sorter **deletes** consumed files. Either delete/archive them or amend the protocol — today it's an unbounded keep-everything bin that also pollutes grep.
- **CHANGELOG.md (81.7 KB)**: +20 KB in 6 days. Nothing reads it programmatically; cap at ~40 recent entries, rotate the rest to `_archive/CHANGELOG_pre_2026-06.md`.
- **_archive/ (1.4 MB, 69 files)**: correct as cold storage, but **31–34% of worker-style greps (`§39`, `Einsum`) land in it** (plus `.update/_integrated` and superseded copies in `Research/Done`). **No `.rgignore` exists anywhere in the folder.** One 2-line `.rgignore` (`_archive/`, `.update/_integrated/`) fixes the entire pollution class for every rg-based agent.
- **exemplars/ (149 KB, 49 files)**: healthy format, June-4 dedups mostly moot on inspection; zero coverage of #78+ families (consistent with the router lag, same fix wave).

### 4f. What's healthy — protect it during any refactor

`ort_reality.md` (canonical matrix + coherent merges) · `core.md` (size-stable, single-home discipline works where applied) · by_symptom's format · exemplar format · `.update/` protocol *design* (drop→sort→delete; it failed by bypass, not by design) · §-stability + breadcrumb convention (every renumber left a `formerly #N` comment — this is why old citations still resolve) · the new-era technique writing style (#93+ are near-charter-perfect) · cascade_strategy.md as the model of a successful trim · the charter §8b text itself.

---

## 5. The refactor plan

**Design principles** (the load-bearing decisions):

1. **§-anchors are the API. Never renumber, never shard techniques.md.** Hundreds of task files/prompts cite `techniques.md §N`; title-grep is the compatibility contract (June-4 rule, still right). Sharding into per-family files would break every citation for zero discovery gain once the catalog exists.
2. **The discovery layer becomes GENERATED, not maintained.** Anything a sorter must remember to update after landing content will lag again within a week — June 5→10 proved it at 97 sections of debt. A ~120-LOC generator + lint check cannot lag.
3. **Every cleanup ships WITH its lint check.** Otherwise this is audit #5 of 6.

### Phase A — discovery layer (1 session; the user-facing win)

1. **`tools/build_techniques_catalog.py`** → generates `techniques_catalog.md`: one line per § (`#N | title | one-line verdict | family tags | task anchors`), parsed from headers + first bold verdict sentence. ~140 lines ≈ **10 KB that replaces the 348 KB linear read for discovery**. Same run regenerates the in-file verdict-index header (or replaces it with a pointer to the catalog). Lint check `catalog-coverage`: every `^# N.` header has a catalog row, build fails otherwise.
2. **by_symptom backfill**: route the ~25 highest-value uncovered §s (#82, #89, #91–#95, #104, #109, #111–#127 cluster entries, #128–#139 cluster entries, #81/#83/#86 already partial). +~2 KB. Lint check `routing-coverage` (warn): any § younger than 7 days with no by_symptom/catalog-tag route.
3. **INDEX.md rewrite (~1 hour)**: kill tombstone parentheticals (one row per file, present-tense role), collapse the op_cost_patterns×5 and taxonomy×3 rows, route WORKER_PROTOCOL.md + ort_reality.md + traps_appendix.md + techniques_catalog.md, restore the budget table as *budgets* (with rule-9 teeth via lint check `set-size-budget`).
4. **WORKER_PROTOCOL digest cap**: replace the 4 KB run-on line with "10 most-recent verdicts + `techniques_catalog.md` pointer." The catalog is now the complete view; the protocol stops being an index.
5. **context_pack.py finishes its spec**: append (a) by_symptom rows matching the task's GDIAG flags, (b) catalog one-liners for the §s the prompt cites + same-family §s. The pack becomes the worker's discovery surface — "anything that could help" answered in ~14 KB.
6. **§-number assignment goes single-writer**: workers return to the `.update/` drop protocol for *new* techniques (it exists, it's good, it's being bypassed); the sorter assigns numbers at integration. During hot waves where direct edits are tolerated, the rule is "claim the number by committing the catalog row in the same edit" — the dup-anchors lint already catches the failure.
7. **Prompt-layer co-design (§3b)**: `_Before writing prompt.md` gains the mandatory DISCOVERY step (grep catalog + by_symptom before writing the Direction; 2–5 §-cites + ≥1 exemplar pointer required); the template gains the dead-end fallback line ("by_symptom + catalog, never linear-read techniques.md"); prompt gate sections collapse to a one-line WORKER_PROTOCOL pointer + task-specific gate notes only.
8. **`ort_reality.md`** → 2-line redirect to `../ort_reality.md` (+ `../scoring_internals.md` for profiling); kills the last "required reading" mis-route into stub files.

### Phase B — every-task-set diet (1 session; 163 → ~75 KB)

1. traps.md → 40-row table + 8 doctrine blocks (**~15 KB**); promote the 3 leaked appendix blocks (§K2 ACT-NOW, §FAMILY-LABEL, §GENERATOR-INVARIANT); add the §CROP-CONTRACT pointer line (VERIFY P11); de-dup trap §5; appendix sheds ~20 KB of debris.
2. cost_floor_reference → **~22 KB**: evict task328/054/080/367 floor-proof bodies to their task files (the 338/054/080 standard requires proofs live there anyway), price-shopping examples out, calibration dump → 4-line regime table (core.md anchors it).
3. 00_COMPETITION_SPEC exits the set (worker facts → core.md ~3 KB; captain logistics → The_400_Tasks/).
4. task_techniques_index: execute the promised battle-prose move (lines 16–117 → recent_ships.md/Logs ledger), file drops to ~35 KB (it stays out of the required set; it's a router).
5. Result, two layers (adopting the sibling audit's framing): the **doctrine surface** that exists as required-citable reference ≈ INDEX 7 + core ~26 + cost_floor 22 + traps 15 + PROTOCOL ~5 ≈ **75 KB**; the **first-read before code** ≈ PROTOCOL 5 + INDEX 7 + core 26 + traps-table 15 ≈ **53 KB ≈ 13k tokens** — cost_floor demotes to priced-on-demand (prompts/catalog cite its floor-taxonomy §s), SPEC exits entirely. That lands the sibling audit's ≤50 KB target through the *existing* two entry points. core.md deliberately stays first-read: encoder, validator bans, and the scoring formula must be known cold, and at 26 KB with proven size-discipline it's cheap insurance. The lint budget check keeps both layers at target.

### Phase C — shelf + lifecycle sweep (1 session, almost all mechanical)

1. `git mv` the 10 extracted K/L/M deliverable bodies → `_archive/` (−275 KB from Done; extraction fidelity already verified). Add the move to the Todo/README filing protocol as an explicit step; lint check `done-archival` (warn): full-deliverable-sized file (>15 KB) in Research/Done whose card is marked extracted. Role-folder what remains (sibling-audit adoption): `Done/cards/`, `Done/audits/` — so a browser sees provenance classes, not a mixed pile. Move the completed K/L/M/R ledger tables out of `Todo/README.md` into Done; the Todo board stays a thin open-cards file.
2. Delete `.update/_integrated/` contents (protocol says sorter deletes; −87 KB) or `git mv` to `_archive/` if provenance wanted. Add `SORTER_LOG.md` one-liner per integration run.
3. **Write `Onnx_Advanced/.rgignore`** (`_archive/`, `.update/_integrated/`) — 2 lines, kills the 31–34% grep-pollution class.
4. scoring_internals purge (→ ~25–30 KB), taxonomy rewrite (→ ~8–10), tiny_architectures trim (→ ~35), mlp reorder+compress (hash-MLP addendum first), synth self-appendix delete, CA trim (→ ~9). Single-home the scoring formula / cost ladder (core.md canonical; others become pointers).
5. Apply the stub **expiry rule** to the 11 stubs: repoint the live citers (einsum.md has 4 — INDEX/op_cost_patterns/techniques/task_techniques_index; most others are already rg-clean), then archive each stub once rg-clean; lint dead-refs guards it. Rotate CHANGELOG (keep ~40 entries).
6. techniques.md *light* repair only: consolidate the §39 stack into one MECHANISM block + dated one-line amendments, move the 2 ship-ledger tables to task files, merge §-pairs (#22/#25, #8b/#23 cross-ref) — **~20–30 KB and major coherence**, leaving the file ~310–320 KB of dense, indexed, catalog-fronted reference. (A deeper old-era narrative compression to ~250 KB is available later; with the catalog in place it stops being urgent — grep lands on the right § first try.)

### Phase D — standing invariants (ship alongside A–C, ~half session total)

New repo_lint checks (the registry pattern in `tools/repo_lint.py` makes each ~20–40 LOC): `catalog-coverage` · `routing-coverage` (warn) · `set-size-budget` (INDEX table vs actual, both layers from Phase B item 5) · `done-archival` (warn) · `index-two-way` (root .md files ↔ INDEX rows) · `changelog-cap` (warn) · `required-reading-phrase` (sibling-audit adoption: the phrase is only legal in WORKER_PROTOCOL.md/INDEX.md) · `stub-expiry` (stub with zero live citers → flag for archive). Wire the fast set into pre-commit beside dup-anchors. These checks would have caught essentially everything in §1's table before it compounded.

**End state**: worker-visible live tree ~1,344 KB → **~850–900 KB** (≤800 with the optional deep techniques compression); every-task set **163 → ~75 KB**; complete technique discovery in **10 KB (catalog) + routed by_symptom + an enriched context_pack**, instead of 348 KB of spelunking; zero § collisions by construction; zero grep pollution; lifecycle rules self-enforcing. Estimated total effort: **3–3.5 sessions**, each phase shippable independently between campaign waves (commit the working tree first — June-4 convention).

---

## 6. What NOT to do

- **No mass renumbering, no per-§ or per-family file sharding, no new ID scheme** (`T067_*`-style) for techniques.md — live prompts cite `#81`/`#67` bare *today*; the §-key is the citation API across ~1,100 refs. The catalog provides the discovery a split would buy, at ~0 migration cost. Revisit only on the §1b Phase-E trigger.
- **No third entry point** (a new START_HERE.md): entry-point proliferation is the documented June-4 drift mechanism (README/QUICKSTART/INDEX divergence caused bug #2). WORKER_PROTOCOL.md *is* the start-here — prompts already point at it first; fix its digest instead.
- **Don't compress the Aditya/Boladi/new-era blocks** (#106–#139) — they're already verdict-first, dense, and are the endgame fuel. The eviction targets are old-era stacks, meta-noise, and mis-homed ledgers.
- **Don't delete `_archive/`** or move it out of git — it's the rollback + provenance layer and it's working; just stop searching it by default (.rgignore).
- **Don't hand-maintain any new index.** If a surface can't be generated or lint-checked, don't build it — it will be 97 sections stale by the next audit.
- Don't drop a section title workers may grep — moved content leaves a one-line tombstone (existing convention, keep honoring it).

## 7. Open questions for the owner

1. **Catalog as file vs front-matter**: separate `techniques_catalog.md` (recommended: INDEX-routable, context_pack-includable, regenerable without touching the 348 KB file) vs regenerating the in-file verdict-index header. Both can coexist (file + 3-line pointer header).
2. **Deliverable-shaped root docs** (synth 45 KB, mlp 37 KB): apply the K/L/M extract-then-archive precedent fully (root doc becomes the ~5–12 KB extraction), or keep mid-size? Auditors split; precedent says extract-then-archive.
3. **.update/_integrated**: delete (protocol-as-written) or archive (provenance)? Either ends the leak.
4. **§-assignment during hot waves**: strict sorter-only (slows same-day integration) vs catalog-row-claim rule (keeps speed, lint-backed)? Plan assumes the latter.
5. Optional later: deep narrative compression of techniques §1–§42 era (~60–70 KB further) — worth a session only after A–D prove out.
6. **Phase-E trigger sign-off**: the sibling audit's family-page split stays parked unless techniques.md crosses ~450 KB or §-grep noise persists post-catalog — agree/adjust those trigger numbers, or kill Phase E outright.

## 8. If only 3 things happen

1. **Generated catalog + by_symptom backfill + context_pack slices + the prompt-authoring DISCOVERY step (Phase A items 1/2/5/7)** — fixes "can't find what we already figured out" at BOTH ends: the captain writing the prompt and the worker whose angle dead-ends; one session; permanent (lint-backed).
2. **traps table-ization + cost_floor eviction + SPEC exit (Phase B)** — halves what every worker pays every task (163→~75 KB); the single biggest recurring token saving.
3. **The mechanical lifecycle sweep (Phase C items 1–3): archive 275 KB of Done bodies, clear `_integrated/`, add `.rgignore`** — ~2 hours, −360 KB of grep-polluting duplicate text, restores the lifecycle contracts.

…and land the Phase-D lint checks with each, or this becomes audit #6's §2.

## 9. Crown jewels — refactor-survival addendum (new since June-4's §7 list, which remains in force)

Everything in `../Done/AUDIT_2026-06-04_context_bloat_refactor_plan.md` §7, **plus** the June 6–10 era: §39 MECHANISM + profile==credit law (incl. dynamics, task065) + REAL-INPUT-PROFILE gate-5 (task225/task360) · #75 variadic-Sum · #76 ESPRESSO→1×1-Conv · #77 F16 execution model (no native F16 Conv kernel; never-count-on-F16-Gemm) · #80-inverse grouped-ConvTranspose block expansion · #81 pad-floor crossover law + small-grid inversion · #82 free-output multi-term Einsum · #83 nibble-pack · #84 IG identity · #85 QLC int32-bias binarizer + BIAS-LEAK law · #86 occupancy-pack + bbox single-marginal · #87 selector-bank rank kill-test + FREE-OUTPUT SEPARABILITY trap · #88 row-byte flood (width≤8) · #89 einsum-final operand-order contract (L2 source-read) · #90 no-Mul colour decode · #91 fused linear-collapse+negpad conv · #92 uint8 signature + (Σw+1)-bin routing · #93 class+validity single-channel fold (+ off-grid-not-black trap) · #94 TopK k=N object extraction (float-only kernel note) · #95 eps-fractional fold (f16-safety bound + Clip-before-scatter) · #104 7-step linear-map procedure · #109 whole-one-hot separable einsum-final · Aditya #111–#127 + Boladi #128–#139 harvests (incl. #130 analytic-trig memory-0, #134 floor-discipline proof standard, #138 scorer-invisible expansion temporaries, #139 ScatterND tiny-patch) · ort_reality K2 source-read section + M1 opset-19–22 verdicts + GridSample/QLinearConv grader validations · traps §FAMILY-LABEL (3/20 batch-16 misnomers) + §K2 ACT-NOW PAIR · the catalog/lint tooling specs in this audit.

*Full per-layer evidence (file:line quotes, sampled A/B/C/D splits, extraction-fidelity checks) is preserved in the 5 auditor reports from this session; this file is the distilled plan of record. Coordinating-session verifications: section count (140), index freeze date, INDEX routing gaps, stub citations (re-verified after the initial hung-command error — see correction note), collision recurrences, lint registry, context_pack composition, growth attribution via `git cat-file -s` at 89903574/26f96669. Amended same day: §1b reconciliation after reading the sibling `AUDIT_2026-06-10_onnx_advanced_navigation_bloat.md` in full; §3b after reading `_Before writing prompt.md`, the Phase-3 template, and live batch prompts (Task130/Task042).*
