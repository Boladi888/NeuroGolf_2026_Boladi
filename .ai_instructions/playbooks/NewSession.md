# Playbook: New Session (project side)

This is the universal session-start protocol for any agent in any project. Project-specific details live in `03_PROJECT.md`, which you read after this.

This playbook is the canonical source for each project's `.ai_instructions/playbooks/NewSession.md`. Edit it here in `AI_Instructions/playbooks/NewSession.md`; the canonical content is manually copied into each project when rules change — never edit the project copy directly.

> **NG workflow (2026-06-05):** Workers read a prompt from `The_400_Tasks/Task Prompts/`, build, gate, and stage results to `staged_submits/`. No lanes, no kanban. Use this universal handshake.

---

## Read order

1. `00_START_HERE_README_AI.md` (the project-key-specific frame at the top of this file).
2. `01_MASTER.md` — universal rules (the baseline).
3. `proposed_master_changes.md` — local rule patches; per `01_MASTER.md` §1.25, active proposals override master for THIS project (strengthening only).
4. `02_GIT.md` — universal git workflow.
5. `03_PROJECT.md` — project-specific live save-state.
6. The latest 1-3 entries in `../Logs/sessions/` — most recent first.
7. `../Logs/lessons.md` — durable lessons for this project.
8. Optionally browse `peer_archive/<other>.md` if your task touches a domain another project may have explored.

---

## Trust the session log — do not re-run the full git ritual

The previous agent declared the branch state in their session-log entry. The state is one of these exact strings:

- `merged-deletable` — previous branch was merged. Run `git branch -d <name>` (and `git push origin --delete <name>` if applicable). Move on.
- `PR-open-#N` — a PR is open on the branch. Note the number; check status before doing parallel work.
- `WIP-keep` — the branch is mid-work. Read the log's "where I left off" paragraph to resume.
- `abandoned-deletable` — the previous session abandoned the branch and committed cleanup. Safe to delete.

**Trust this declaration.** Run `git fetch + git log + git status` only if (a) the latest log entry is missing, (b) there's no log at all, or (c) on-disk reality contradicts the declaration. Routine session start does NOT need the full inspection ritual.

---

## The handshake — verbatim format, no exceptions

The owner uses the handshake to confirm you actually read the files instead of pretending. Skip any of these elements and the owner knows.

### 1. Address

The very first line of your reply addresses the owner as **"Your Honor"**. Use it naturally as a greeting, not as a parody.

> Your Honor — onboarding handshake follows.

### 2. Files-Read List

Immediately after the address, list every file you actually read at session start. One file per line, no commentary.

```
Files read:
- 00_START_HERE_README_AI.md
- 01_MASTER.md
- 02_GIT.md
- 03_PROJECT.md
- ../Logs/sessions/2026-04-26_1430_planet-eval.md  (last session)
- ../Logs/lessons.md
```

If you only read four files, list four. **No padding.**

### 3. Branch State

One short paragraph stating the branch state, quoted verbatim from the latest session-log entry. If the log is missing or contradicted, run the full git inspection and report what you found.

### 4. Multi-choice Next Tasks

3-5 options as a numbered list, ordered by estimated time-to-complete (fastest first).

```
What would you like to work on?

1. [~5 min]  <fastest task>
2. [~15 min] <next fastest>
3. [~45 min] <medium task>
4. [~2 hrs]  <larger task>
5. [agent's pick] <see below>
```

Derive the list from open TODOs in `03_PROJECT.md`, known bugs in `03_PROJECT.md`, and the latest session-log's "where I left off" if applicable.

### 5. Agent's Pick (always option N)

The last option is one you came up with — not from the TODO list. It can be:
- A change in project direction worth considering
- Research that should be done
- An alternative implementation approach you noticed while reading
- A risk or technical-debt item not yet captured

State it concisely with one sentence on why it's worth raising.

---

## At session start, after the handshake

Create today's session-log file at `../Logs/sessions/YYYY-MM-DD_HHMM_<short-task-or-branch>.md`:

```markdown
# Session: YYYY-MM-DD HHMM — <short title>
**Goal:** <one sentence>
**Branch:** <branch name>
**Starting commit:** <short SHA>
**Started:** <UTC timestamp>
```

The "Final state" block gets added at session end.

> **Mid-session FINAL STATE writes are forbidden** — start-skeleton at session start, FINAL STATE only at session end. Mid-session declarations of branch state (especially `WIP-keep`) drift from on-disk reality (e.g., a `git reset` wipes the WIP files but the log still claims them); the next agent then has to reconcile log-vs-reality forensically. Write the skeleton above at start; write everything else at end.

---

## At session end

Update today's session-log file. Append:

```markdown

## Final state
**Branch state:** merged-deletable | PR-open-#N | WIP-keep | abandoned-deletable
**Ended:** <UTC timestamp>
**Files touched:**
- <path> (NEW | UPDATED | DELETED)

## What happened
<1-3 sentences>

## Implicit decisions made
<only if auto mode was on for any part of the session — one bullet per decision, per 01_MASTER.md §1.21 logging clause>
- [trigger letter] short description — chose X because Y. Alternatives considered: A, B.

## Where I left off
<only if WIP-keep — one paragraph for the next agent>

## Lessons
- <0-3 bullets>
```

The branch-state value MUST be one of exactly four strings. The next agent trusts that declaration.

If auto mode was enabled for any portion of the session, the `## Implicit decisions made` block is mandatory (omit the block entirely if auto mode never came on). If the count exceeds ~3, surface that fact in `Where I left off` — auto mode was probably running too hot, or the original prompt was too vague.

If a lesson is durable beyond this session, append it also to `../Logs/lessons.md`. If you noticed a rule that should be in master / git / project rules, append a stub to `proposed_master_changes.md`.


---

## End-of-session discipline (anti-bloat rules)

The session-start handshake is rigorous; the session-end discipline is what keeps `03_PROJECT.md` from growing monotonically. Follow these rules every time you wrap up a session.

### 1. "Where I left off" — replace, don't stack

When you append today's `Where I left off` to `03_PROJECT.md`, **DELETE the prior one**. The per-session log at `Logs/sessions/<file>.md` is the durable record of what happened in that session. Do NOT preserve prior `Where I left off` blocks "for context" — that information already lives in the session log.

### 2. Update banners at the top — replace, don't stack

If `03_PROJECT.md` uses dated update banners (`> Updated: YYYY-MM-DD — …`) at the top, **keep at most one** (the latest), or replace the whole stack with a single "Session index" row in a table. Do NOT stack banners; each banner is a session-log link in disguise, and the session log already exists.

### 3. Closed TODO / dismissed / fixed bugs — collapse to a one-line row

When a TODO closes, a critique is dismissed, or a bug is fixed: collapse the entry to a single row in `03_PROJECT.md`'s "Closed-item ledger" with date + type + short title + outcome + pointer to the session log or `lessons.md`. **Do NOT keep `[x]` strikethrough prose** in the live TODO. Do NOT keep multi-paragraph rejection rationales inline — the rationale belongs in `lessons.md` or the session log.

### 4. Format constraint — short title + max 2 sentences

Every entry in `03_PROJECT.md` (TODO item, decision, dismissed critique, fixed bug, open bug) is **short title + max 2 sentences** of context. If you want to write more, write it in `Logs/lessons.md` (durable lesson) or the per-session log (event narrative) and link to it from the project file.

### 5. One-off narrative incidents → `lessons.md`, not the project file

If a session produced a narrative-style incident report ("the X failure on YYYY-MM-DD"), that goes in `lessons.md` with the standard format (date + short title + 1–3 sentences) — not as a multi-paragraph section in `03_PROJECT.md`. The project file is "what is true now," not "the complete history of every session."

### 6. The lessons.md test

Before adding a paragraph to `03_PROJECT.md`, ask: **"Could this be one line in `lessons.md` instead?"** If yes, put it there. The format constraint of `lessons.md` (date + title + 1–3 sentences) is the discipline this whole workflow lacks elsewhere; lean on it.

### 7. Sibling-file split when a section grows

When any single section of `03_PROJECT.md` exceeds ~50 lines and is read deeply only when answering a specific class of question, propose splitting it to a sibling file (`04_DATA.md` / `05_BACKLOG.md` / `06_COMMUNITY.md` / a new `07_*.md`) and remove it from the default read path. Keep the count of sibling files capped at ~4 — beyond that, the index becomes its own bloat.

---

*This playbook is universal across projects. The canonical source is `AI_Instructions/playbooks/NewSession.md`. The copy in each project's `.ai_instructions/` is manually refreshed when rules change and should not be edited there.*

*Last updated: 2026-05-07 (proposed_master_changes.md remains part of tier-0 read order per §1.25).*
