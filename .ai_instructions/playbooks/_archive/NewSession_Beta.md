# Playbook: New Session — Beta lane

> **You are Beta.** Use this handshake instead of the universal `NewSession.md` when you are operating in the Beta lane. The universal handshake is fine but Beta's workflow has different read-order priorities and different write paths; this trimmed playbook saves you tokens and reduces the chance of touching Alpha-owned files.

## What Beta does

- Mining sweeps across the 400 tasks (`tools/mine_*.py`)
- Deep reads of competitor Kaggle notebooks
- Recipe drafts for Alpha (`Research/notes/recipe-<task>.md`)
- NAS forum scraping + cross-team workaround discovery
- Cost / blend-diff / cascade-risk analysis
- Ledger reconciliation (`The_400_Tasks/*.csv`, submission-candidate digests)
- Forwarding Alpha-handoff items via `[B->A]` tags in `CURRENT_STATE.md`

## What Beta does NOT do

- **NEVER** modify `build_submission.py`, `tests/test_detect.py`, `tools/dispatcher_audit.py`, or `tools/inspect_tasks.py` (including `SOLVED_TASKS`). Those are Alpha-owned. If you need to change them, file a `[B->A]` recipe and stop.
- **NEVER** open a PR that touches Alpha's files even "for consistency."
- **NEVER** run `kaggle competitions submit` without explicit owner approval — quota is coordinated.

## Read order (Beta-trimmed)

Tier 0 (mandatory at every session start):

1. `00_START_HERE_README_AI.md` — entry doc.
2. `01_MASTER.md` — universal rules. The clause that bites Beta most is §1.20 (Lane discipline) — make sure you have that one straight.
3. `proposed_master_changes.md` — local rule patches.
4. **`CURRENT_STATE.md`** at repo root — the one-file snapshot. Score, queue, lane assignments, don't-touch list, source-of-truth pointers.
5. `07_LANES.md` — Beta-specific file ownership.

Skip on first read (load only if a question demands it):
- `02_GIT.md` — Beta opens few PRs; read when one is imminent.
- `03_PROJECT.md` — read sections via `grep` only when CURRENT_STATE.md doesn't answer your question.
- `04_DATA.md`, `05_BACKLOG.md`, `06_COMMUNITY.md` — load on demand.

Latest session log:
- Filter `Logs/sessions/` by `beta-` prefix and read only the most recent 1–2.

## Handshake format

Same shape as `NewSession.md` — Your Honor + files-read + branch state + multi-choice — with the lane line required:

```
Your Honor — Beta onboarding handshake follows.

Files read:
- .ai_instructions/00_START_HERE_README_AI.md
- .ai_instructions/01_MASTER.md
- .ai_instructions/proposed_master_changes.md
- CURRENT_STATE.md
- .ai_instructions/07_LANES.md
- Logs/sessions/<latest beta- session>.md

Lane: Beta

Branch state: <verbatim from latest session log>

What would you like to work on?

1. [~15 min] <pick from "Active queue" or recipe-ready handoffs in CURRENT_STATE.md>
2. [~30 min] <next>
3. [~1 hr]   <next>
4. [~2 hr]   <next>
5. [agent's pick] <one item you came up with from grepping research notes>
```

Multi-choice options come from **CURRENT_STATE.md's Active queue + Recipe-ready section**, not from `03_PROJECT.md`'s Open TODO.

## Where Beta writes outputs

| Output | Path |
|---|---|
| Raw mining/probe dumps | `Logs/scratch/beta_<topic>_<date>/` *(gitignored — purge or promote at session end)* |
| Curated analytical CSV/MD | `Logs/<date>_<topic>.csv\|md` *(tracked)* |
| Curated research note / deep read | `Research/notes/<topic>-<date>.md` *(tracked)* |
| Recipe draft for Alpha | `Research/notes/recipe-<task>.md` *(tracked; marked `[B->A][Spark-ready]` when verified)* |
| NAS forum scrape summary | `Research/notes/nas-scrape-<date>.md` *(tracked)* |
| Session log | `Logs/sessions/<date>_<time>_beta-<task>.md` *(tracked)* |
| Lessons | prepend to `Logs/lessons.md` *(tracked)* |
| Updates to candidate ledgers | `The_400_Tasks/task_state.csv` + related — but **don't** edit `tools/inspect_tasks.py:SOLVED_TASKS` |
| Score / queue updates | `CURRENT_STATE.md` *(tracked — replace, don't stack)* |

## Session-end discipline (Beta-specific additions)

In addition to the universal session-end rules (see `playbooks/NewSession.md`):

- **Update `CURRENT_STATE.md`** if you discovered:
  - A new don't-touch task (public-negative isolated)
  - A new recipe ready for Alpha (`[B->A][Spark-ready]`)
  - A new entry in the active queue
- **Purge or promote `Logs/scratch/beta_*` dirs** — they're gitignored anyway, but a confused future agent will assume relevance if they linger. Promote analytical summaries; delete the raw dumps.
- **Don't open a PR** unless the work is collaborator-facing (recipe handoff, ledger refresh, scratch convention change, etc.). Tracked-file changes during a research session can sit on a branch awaiting Alpha-side movement.

## When to escalate to the universal handshake

If your task drifts into Alpha-territory (the user asks you to verify a build_submission.py change, or you find a bug in a shared file), stop, halt-immediately per `01_MASTER.md` §1.21(e), and ask the owner whether you should switch lanes or file the issue as `[B->A]`.

---

*This playbook is project-specific (NG, multi-track since 2026-05-02). For the universal handshake see `NewSession.md`. For Alpha see `NewSession_Alpha.md`.*
