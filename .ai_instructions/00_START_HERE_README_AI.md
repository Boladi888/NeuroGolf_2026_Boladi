# 00_START_HERE — NG

You are an AI agent starting a session in the project keyed `NG` (NeuroGolf). This is the first file you read.

> **AUTO MODE NOTE.** If the owner enables auto mode at any point this session, `01_MASTER.md` §1.21 (Auto Mode Discipline) is the load-bearing section — read it before §1.1. The decision-point checkpoints, the investigate-first vs halt-immediately split, the pivot rule, and the implicit-decision log all live there. Auto mode usually toggles ON mid-session (after the kit-check), not at session start, so this banner applies whether or not auto mode is currently on.

## Read order (tier-based)

The handshake declares which tier the agent loaded. Don't load Tier 2 unless the task or a question demands it.

### Tier 0 — always

1. `00_START_HERE_README_AI.md` (this file).
2. `01_MASTER.md` — universal rules.
3. `proposed_master_changes.md` — local rule patches; per `01_MASTER.md` §1.25, **Active proposals override master rules** for this project (strengthening only).
4. `02_GIT.md` — universal git workflow.
5. **`../CURRENT_STATE.md`** at repo root — single-file snapshot: score, queue, lane assignments, don't-touch list, source-of-truth pointers. **Read this BEFORE `03_PROJECT.md`** — it's the smallest, freshest doc and answers most "what's true right now" questions.
6. `03_PROJECT.md` — tech stack, architecture, implementation rules, open bugs. (Historical closed items moved to `03_PROJECT_HISTORY.md`; submission ledger moved to `../Logs/submissions_ledger.csv`.)

### Tier 1 — session-start when likely relevant

7. `playbooks/NewSession.md` — universal handshake. (Lane variants were retired 2026-06-05 with the lane system; the universal handshake is the only one.)
8. The latest 1–3 entries in `../Logs/sessions/` — most recent first.
9. `../Logs/lessons.md` — durable lessons accumulated for this project.

> **One-shot bootstrap:** `python tools/state.py` prints a ~10-line snapshot (score, SOLVED count, latest digest/probe/session pointers). Useful when starting a session.

### Tier 2 — load on demand only

9. `04_DATA.md` — data-analysis playbook. Open for "do we have data on X?" / "what predicts Y?" questions. (Optional file — may not exist for this project.)
10. `05_BACKLOG.md` — strategic backlog of experiments / future work tiered by priority. Open when picking the next experiment. (Optional.)
11. `06_COMMUNITY.md` — community-agent / external-implementation references. Open when porting an external pattern. (Optional.)
12. *(no 07_LANES.md — lane system retired 2026-06-05; see `playbooks/_archive/`)*
13. `peer_archive/<KEY>.md` — read-only mirror of another project's `03_PROJECT.md`. Open when the task touches a sibling project's domain.

**Declaring the tier in the handshake:** as part of the files-read list, list every Tier-0 + Tier-1 file you read by name; for Tier 2, state "Tier 2: none loaded" or list the specific siblings you opened.

## Files in this folder

- `00_START_HERE_README_AI.md` (this file) — entry doc + read order + end-of-session discipline.
- `01_MASTER.md` — universal rules. Source of truth: `AI_Instructions/README_AI_Master.md`.
- `02_GIT.md` — universal git workflow. Source of truth: `AI_Instructions/playbooks/Git.md`.
- `03_PROJECT.md` — project-specific live save-state. State now + pointers to durable record.
- `04_DATA.md` — (optional, on demand) data-analysis playbook companion.
- `05_BACKLOG.md` — (optional, on demand) strategic backlog of experiments.
- `06_COMMUNITY.md` — (optional, on demand) external / community-agent implementation references.
- *(07_LANES.md archived — lane system retired 2026-06-05)*
- `proposed_master_changes.md` — your channel for suggesting rule additions to the universal master/git playbook. **Append-only. Active proposals override master per §1.25.**
- `playbooks/NewSession.md` — handshake protocol + session-log conventions.
- `playbooks/Research.md` — research-workflow primer.
- `playbooks/handoffs/` — playbooks for producing portable briefs to outside agents.
- `peer_archive/` — read-only mirrors of other projects' `03_PROJECT.md` files.

## End-of-session discipline (anti-bloat rules)

The full 7-rule anti-bloat protocol lives in `playbooks/NewSession.md` ("End-of-session discipline" section). Read it once at first session-start and at every refresh. One-line summary for at-a-glance reference:

1. **`Where I left off` — replace, don't stack.** Delete the prior block when appending today's. Session log is the durable record.
2. **Update banners — replace, don't stack.** Keep at most one at the top of `03_PROJECT.md`.
3. **Closed TODO / dismissed / fixed bugs — collapse to one row.** Move to `Closed-item ledger` with a pointer; no `[x]` strikethrough prose.
4. **Format constraint — short title + max 2 sentences.** Longer narrative → `lessons.md` or session log.
5. **One-off narrative incidents → `lessons.md`.** Don't park multi-paragraph incident reports inside `03_PROJECT.md`.
6. **The lessons.md test.** Before adding a paragraph here, ask: "could this be one line in `lessons.md` instead?"
7. **Sibling-file split when a section grows.** ≥~50 lines + read-on-demand → split to `04_*.md` / `05_*.md` / etc. Cap siblings at ~4.

Full rationale + examples in `playbooks/NewSession.md`.

## Distribution model

This `.ai_instructions/` folder is a **manually-maintained snapshot** of the canonical files in `AI_Instructions/`. The owner copies updated rules in by hand when the canonical changes. There is no auto-bundle script.

If you spot a rule worth proposing for the universal master, append a stub to `proposed_master_changes.md` and the owner picks it up at the next overview.

*(Lane/multi-track section removed 2026-06-05 — NG runs single-worker prompt-based flow. Lane system archived in `playbooks/_archive/`.)*

---

*Now read `01_MASTER.md`.*
