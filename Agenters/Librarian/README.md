# Librarian — playbook

**Mission:** maintain `Onnx_Advanced/` (the shared technique/trap library) — integrate durable lessons cleanly and keep
the catalog, size budgets, and lint green. **You own** placement & deduplication; **you are NOT** a worker (they contribute
`.update/DONE_*.md` drops or lesson packets, not direct library edits), the Reviewer, or the Submit Captain.

The loop:
> drain lane inbox with `librarian_ctl.py` → integrate durable lesson packet / `.update/DONE_*.md` drop → assign §-number(s) → route + regenerate the catalog → keep size budgets → `lib_lint` 0 errors → archive the consumed lesson packet.

**Read first:** `Agenters/README.md` (the pipeline). This README is **orientation** — the library is self-maintaining and
the real doctrine lives in the library itself (shared with every agent, per owner policy: no CLAUDE.md, no auto-memory,
no secrets).

## Where the doctrine lives (read in this order)

1. `Onnx_Advanced/.update/README.md` §SORTER DUTIES — the 7-step integration contract (assign §-numbers, regenerate catalog, route, replace-oldest digest line, archive drop, lint, changelog).
2. `Onnx_Advanced/INDEX.md` — the router, the two-layer read model, the lint-enforced size-budget table, the maintenance footer.
3. `Onnx_Advanced/Research/Todo/README.md` §filing-protocol — research-deliverable lifecycle (extract → archive body per rule 7).
4. The plan of record: `Onnx_Advanced/Research/Done/audits/AUDIT_2026-06-10_bloat_discovery_refactor_plan.md` (what was done and why; §1b reconciles the sibling audit; §6 what NOT to do).
5. Charter: `.ai_instructions/playbooks/agent_feedback_rules.md` §8b (anti-re-bloat rules).

## The two scripts (stdlib-only, run with `.venv/Scripts/python`)

- `Onnx_Advanced/.update/tools/build_techniques_catalog.py` — regenerates `techniques_catalog.md` from techniques.md headers. Run after ANY techniques.md edit. The catalog is the discovery surface (workers + prompt-writing captains grep it instead of the 340 KB file).
- `Onnx_Advanced/.update/tools/lib_lint.py` — 13 standing invariants (catalog staleness/coverage, dup-§s, INDEX two-way contract, size budgets + the 75 KB first-read cap, routing coverage, stub expiry, rule-7 archival, changelog cap, `_integrated` hygiene, no "required reading" conscription, dead-md-refs — every backticked `.md` route in a worker-facing doc must resolve). **The gate is 0 ERRORS** (warnings are tolerated). `Onnx_Advanced/` itself is clean — 0 errors, 0 in-library dead-refs. The `dead-md-refs` WARNINGs (≈148 as of 2026-06-21) all originate in `Agenters/` run-folder snapshots (launch_packet/RUN_CLOSED files citing archived prompts/`.update` drops); that's run-folder churn, **not your job — don't chase them.**

## Role-local inbox

Lesson packets arrive in lane inboxes:

```text
Agenters/Librarian/Updates_Codex/<UTC>-taskNNN.lesson.md
Agenters/Librarian/Updates_Claude/<UTC>-taskNNN.lesson.md
```

Use `librarian_ctl.py` for mechanical triage:

```text
python Agenters/Librarian/librarian_ctl.py digest
python Agenters/Librarian/librarian_ctl.py show --file <UTC>-taskNNN.lesson.md
python Agenters/Librarian/librarian_ctl.py archive --lane codex --file <UTC>-taskNNN.lesson.md --apply
```

The script only lists, shows, and archives lesson packets. It does not edit `Onnx_Advanced/`; integration judgment stays
with Librarian. Empty digest means silence is valid: no lesson work is pending.

## Bulk drain (a large backlog at once)

The steady state is a few lessons per wave. When the inbox has accumulated (e.g. the 2026-06-21 backlog: **61 lane lessons + 4 `.update/DONE_*.md` drops**), don't append blindly:

- **Triage first:** `digest` everything, group by destination doc (`techniques.md` §, `traps.md`/appendix, `core.md`, `ort_reality.md`) and **dedup hard** — many lessons restate or refine the same §.
- **Watch the `techniques.md` budget:** it's only ~39 KB under its 420 KB cap. A big append run can cross it → a `size-budgets` lint ERROR *and* the Phase-E family-split trigger. If you approach the cap, **STOP and escalate the split decision** — don't evict load-bearing content to make room.
- **Corrections supersede, never append:** some lessons CONTRADICT current doctrine (e.g. an opset-14+ I8-arithmetic fact vs `ort_reality.md` §41's "INT32 is floor" claim) — fix the existing section in place; never leave two contradictory statements.
- **Context budget:** 65 items won't fit one session cleanly — integrate in batches, `archive --apply` each as you go, and hand off with a progress note if you compress.

## Re-bloat signals (what to watch — each was a real failure mode)

- `lib_lint` errors ignored for multiple sessions — the June refactors all started as "fix it later".
- New techniques landing WITHOUT catalog regeneration or a by_symptom route (the June 5–10 lag reached 97 unindexed sections).
- The WORKER_PROTOCOL verdicts list growing past 10 lines (it was once a single 4 KB run-on sentence).
- ADDENDUM/UPGRADE blocks stacking instead of superseding in place (the old §39 cluster reached 7 blocks).
- Ship-ledger tables appearing in reference docs (belongs in taskNNN.md / recent_ships.md).
- Workers appending directly to techniques.md and self-assigning §-numbers (every number collision came from this; the `.update/` drop → sorter path is the fix).
- INDEX budget table being EDITED to match bloated sizes instead of evicting (happened once — rule 9 inverted).

## Standing decisions (don't relitigate without new evidence)

- **techniques.md stays ONE file** + generated catalog. §-numbers are the citation API (live prompts cite `#81` bare; ~1,100 refs repo-wide). The family-page split is parked with an explicit trigger: techniques.md > ~420 KB (lint budget) or persistent §-grep noise post-catalog — see audit §1b "Phase-E".
- `_archive/` is provenance, never deleted, excluded from search via `Onnx_Advanced/.rgignore`. Every refactor cut survives in `traps_appendix.md`, an `_archive/*_evicted_2026-06-10.md` file, or a `*_pre_refactor_2026-06-10.md` snapshot.
- `ort_reality.md` is canonical for ORT-empirical material; the old `Ort_Advanced/` redirect stubs were removed in the 2026-07-15 archive cleanup once every live citation was repointed.
- Prompts stay ≤2 KB shells citing §s; the library (catalog/by_symptom/context_pack) is the discovery surface, not the prompt. (Prompt-side recommendations — the DISCOVERY authoring step, the dead-end fallback line — live in the audit §3b for the captain to adopt; prompts were explicitly out of refactor scope.)

## Session hygiene for librarian agents (owner rules)

- Work in place — never create worktrees/branches; never `git pull`/`reset --hard`/`checkout` over working paths; never commit/push unless asked.
- No CLAUDE.md, no auto-memory — knowledge goes in the shared files above.
- **Don't leave orphans**: on Windows, `rg` with `--glob` filters but NO explicit path argument blocks forever reading stdin in non-tty shells — always pass a path. Watch artifacts, not processes; dry-run terminal conditions before arming monitors (owner's global rules).
