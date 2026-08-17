# Playbook: Git Workflow (universal)

Universal git rules that apply in every project. Edit here; the canonical content is manually copied into each project's `02_GIT.md` when rules change.

The master rules (`README_AI_Master.md` at canonical; copied as `01_MASTER.md` in each project) cover the hard safety rules under §2 "Git & Version Control Safety" — those rules are authoritative. This playbook supplements with the workflow patterns the owner expects on top.

---

## 1. Branch + PR — never commit directly to master

1. **NEVER commit directly to `master` / `main`.** All changes go on a branch, then through a pull request.
2. **Create a branch** for your work BEFORE making any changes. Name it descriptively (`session-<short-task>`, `fix-<bug>`, `feat-<feature>`).
3. **Create a PR** when changes are ready. The owner reviews and merges.
4. **After merge, the owner must pull** to get changes in their local main worktree. Don't assume they know — remind them.

This rule applies in EVERY repo unless the owner explicitly opts out for a specific project (e.g., a personal sandbox where direct-to-main is fine).

### 1.1 One PR per session (default cadence)

The owner explicitly prefers **one thematic PR per session** over many small PRs, even where it slightly increases per-PR blast radius. Recent sessions where 5–6 small PRs landed in one session added review/merge/pull overhead without proportional benefit. The accepted trade-off: each PR is a bit larger and harder to revert in isolation, in exchange for the owner reviewing once per session instead of six times.

Small chores (single test, single helper extract, single docs nudge) bundle into the session's strategic PR or into one end-of-session "chore" PR. Don't fragment.

**Rebase before opening.** Master may have moved during the session — `git fetch origin && git rebase origin/master` before pushing the PR. A clean fast-forward avoids merge-conflict noise on the owner's review.


<div style="background-color: #fff3cd; border-left: 6px solid #ff8c00; padding: 12px; margin: 12px 0;">

🟧🟧🟧 **MULTI-TRACK ADDITION — ignore if your project is single-track (a single-track project can later become multi-track)** 🟧🟧🟧

**Multi-track projects.** When `03_PROJECT.md` declares parallel agent tracks (e.g. Alpha and Beta), the rule becomes **one thematic PR per lane per session**. Both lanes can land PRs in the same week — but each lane bundles its own session work and never bundles cross-lane changes into one PR. Coordination rules for multi-track projects are in §10.

🟧 **END addition**

</div>

**Note: the rule covers PR shape only.** Multiple experiments, multiple agent codename changes, and multiple validation runs per session are all fine — encouraged, in fact, since accumulated per-session data feeds the empirical-mining pipeline. The cap is on git artifacts (one PR / one submission) so the owner reviews once per session, not on experiment count.

### 1.2 `git_updates.md` — per-session PR-body scratch

Use a **session-scratch file at repo root named `git_updates.md`** to accumulate per-change rationale during the session. Each time you make a logically-distinct change, append a section explaining what + why. At session end, the accumulated content becomes the body of the bundled PR.

`git_updates.md` is gitignored — purely ephemeral. Delete it after PR submission or leave it; the next session overwrites/ignores it. The point is to **avoid losing the "why" between making a change and writing the PR description an hour later**.


<div style="background-color: #fff3cd; border-left: 6px solid #ff8c00; padding: 12px; margin: 12px 0;">

🟧🟧🟧 **ADDITION — NEW §1.3. Ignore if not relevant.** 🟧🟧🟧

### 1.3 Merge method — Rebase or plain Merge, NOT Squash

**Squash-merge is forbidden** for this repo. The default merge button on GitHub is "Rebase and merge" (or plain "Merge" if a merge commit is wanted). Reason: squash-merge collapses N commits into 1 on master, which destroys the per-commit patch IDs that `git rebase` uses to detect "this commit is already applied." Any session branch carrying commits beyond a squash-merged PR can no longer be rebased onto the new master cleanly — git replays the already-merged commits and conflicts on every binary file the squash also touched.

With Rebase/Merge merge methods, post-PR rebase detects per-commit equivalence and skips already-applied commits cleanly. Set the repo's PR settings accordingly (Squash disabled, Rebase preferred).

🟧 **END addition**

</div>

### 1.4 Never Stack PRs (No Exceptions)

A pull request's `base` is always `master`/`main`, never another open PR's branch. See `01_MASTER.md` §1.18 for the rationale and the recovery cost when this is violated. If the work needs to be split for review, sequence the PRs (open one, wait for merge, then open the next) rather than stacking them.

This applies even when the work is logically a "stack" of cleanup → smoke tests → speedup. Bundle into one PR with multiple commits, or sequence sequentially. Never stack.

## 2. Pull warnings

- **If a filename changed** in the merged PR (rename, move), the old file stays on disk after `git pull` and the new file appears alongside it. **Warn the owner** to verify only the new filename exists locally and delete the old one if needed.
- **If only file content changed** (no rename), a simple `git pull` updates everything automatically.
- **Always remind the owner to pull** after a PR is merged.

## 3. Branch state — declared, not inferred

The previous session's `Logs/sessions/<latest>.md` declares branch state in one of four exact strings: `merged-deletable | PR-open-#N | WIP-keep | abandoned-deletable`. Trust that declaration. Run the full `git fetch + log + status` only if the latest log entry is missing or contradicts on-disk reality.

When you start work, create a session-log file at `<project>/Logs/sessions/YYYY-MM-DD_HHMM_<task>.md` per the `00_START_HERE_README_AI.md` template. At session end, update it with the final branch state declaration.

### 3.1 Pull `main` BEFORE reading instruction or session-log files

A worktree's local `main` ref reflects the snapshot at worktree-creation time. Whenever PRs have merged between worktree creation and session start (which is most sessions), the worktree's `main` lags `origin/main` and the latest session log on disk may not actually be the latest one. Reading instruction files and session logs against a stale snapshot wastes tokens — they may have been rewritten or superseded in the unread commits.

**First two commands of any session, before any other reads:**

```
git fetch
git log HEAD..origin/main --oneline
```

If the second command produces output, run `git pull --ff-only` (or rebase if you're on a session branch already) BEFORE reading any of `01_MASTER.md`, `02_GIT.md`, `03_PROJECT.md`, or session logs. The instruction files themselves may have changed (rules added, sections restructured), and the prior session's "latest" log may have been superseded by an even-newer one in the unread commits.

The five-second cost of the fetch + log check beats re-reading every instruction file after discovering main has moved.


<div style="background-color: #fff3cd; border-left: 6px solid #ff8c00; padding: 12px; margin: 12px 0;">

🟧🟧🟧 **MULTI-TRACK ADDITION — ignore if your project is single-track (a single-track project can later become multi-track)** 🟧🟧🟧

**Multi-track projects:** include the lane in the filename — `YYYY-MM-DD_HHMM_<task>_<lane>.md` (e.g. `..._faf-fix_alpha.md`, `..._db-schema-v6_beta.md`). Branch-state declarations are per-lane; each lane's session log declares only its own branch state, never the other lane's.

🟧 **END addition**

</div>


## 4. Worktrees

Each git worktree has its own working directory. Edits in one worktree don't affect another. If you fix something (e.g., restore a missing `.gitignore`), check ALL worktrees, not just the one you're in.

`git worktree list` shows them. The main worktree is the one where the repo was originally cloned; everything else is auxiliary.

<div style="background-color: #fff3cd; border-left: 6px solid #ff8c00; padding: 12px; margin: 12px 0;">

🟧🟧🟧 **MULTI-TRACK ADDITION — ignore if your project is single-track (a single-track project can later become multi-track)** 🟧🟧🟧

**Multi-track projects** typically operate one worktree per lane off `master` (e.g., `worktrees/alpha/`, `worktrees/beta/`). The "main worktree" referenced by §5 #5 of this playbook and `01_MASTER.md` §2.4 #6 (the post-delete `git fetch --prune` rule) is still the original-clone worktree, not any per-lane auxiliary.

🟧 **END addition**

</div>


## 5. Destructive operations — think before acting

1. **Trace the full impact** of any `git revert`, `git reset --hard`, `git clean -fd`, force push, or rewrite-history operation BEFORE running it. Specifically: what files get created, modified, or DELETED on disk? What commits propagate to other branches via merge?
2. **Verify the result** after any git operation. Check `git status`, `git ls-tree`, AND on-disk state. Don't trust just one signal.
3. **The owner's VS Code must show ZERO untracked files at all times.** If you see untracked files after a git operation, your job isn't done.
4. **NEVER force-push to `master` / `main`** without explicit owner permission. Force-pushing to a feature branch is fine if you own it.
5. **After deleting a branch (local + remote), run `git fetch --prune` in the MAIN worktree** (not just the current one). Then tell the owner to reload VS Code.

## 6. Commit messages

- No `Co-Authored-By` lines. Ever.
- No AI / assistant / copilot / language-model references in any committed text.
- Concise and descriptive. Write as if the owner wrote them.
- Imperative mood: "Add foo handler" not "Added foo handler" or "Adds foo handler".
- First line ≤72 characters; longer body separated by a blank line if needed.

## 7. Public metadata caution

PR bodies, PR titles, commit messages, issue titles, issue bodies, branch names, release notes, and tag descriptions are **public metadata** on GitHub — even in private repos, they are visible to every collaborator and persist in edit history that cannot be fully scrubbed.

Gitignored files (the project's `03_PROJECT.md`, `.ai_instructions/` contents, `Logs/`, etc.) must NEVER be mentioned by name, described, or alluded to in any of those artifacts. Their existence and content must be invisible to anyone reading the repo's public surface.

Run a quick mental check before every `git commit -m`, `gh pr create`, `gh pr edit`, `gh issue create`: am I about to leak information about a gitignored file?

## 8. .gitignore is sacred (per-project)

Each project has its own `.gitignore` with its own conventions. Some are simple whitelist style (PinSub: ignore `*`, allow `!README.md`), some are mixed (PriceWraith: `/*` plus a whitelist plus an always-block list), and some shared-collaborator repos (e.g., OW) intentionally track `.gitignore` itself so the collaborator's clone has a consistent file-visibility surface — see master rules §2.1 rule 1 for the exception. The current convention is recorded in the project's `03_PROJECT.md`.

**Never edit `.gitignore` without explicit owner permission.** Restoring it from canonical content during a worktree fix is fine; redesigning it is not. Likewise, do not untrack a tracked `.gitignore` (or add `git add .gitignore` to a single-developer repo) without explicit owner permission and a stated plan.

If you see a tracked file that matches a gitignore rule (rare but happens — git tracks files added before the rule), don't auto-untrack it. Ask the owner what they want.

The `.ai_instructions/` and `Logs/` folders are dotted / unknown-name folders, automatically excluded by every project's existing whitelist or blanket-ignore rule. No `.gitignore` edit needed when bundles update.

### 8.1 Shared-private-collaborator repos — relaxed (blacklist) `.gitignore`

When the project is **a private repo shared with a small named set of trusted collaborators** (not a single-developer repo, not a public-release repo), the convention flips: instead of starting with `*` and whitelisting tracked files, **use a tiny blacklist that excludes only credentials, per-machine state, license-restricted third-party content, size-prohibitive assets, and session-ephemeral scratch.** Everything else is shared by default.

The project's `03_PROJECT.md` declares which convention is in use. Switching from whitelist to blacklist requires explicit owner permission and a one-time bulk-add of all the previously-hidden files (one PR's worth of churn).

Typical blacklist entries: `.claude/`, `.kaggle/`, `*.token`, `.env`, `.venv*/`, `__pycache__/`, `*.pyc`, `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`, plus project-specific size-prohibitive paths and the session-ephemeral `git_updates.md`.

## 9. Auto-pushed branches

Some agents (Claude Code in particular) auto-push session branches at session start. The owner may merge a PR between sessions and you don't know. **Before creating a new PR**, always check whether the branch has already been merged into master:

```
git fetch origin
git log origin/master..HEAD --oneline
```

If there are no commits ahead of master, the branch was already merged. Do NOT create a PR — tell the owner the branch is already merged and they just need `git pull`.

<div style="background-color: #fff3cd; border-left: 6px solid #ff8c00; padding: 12px; margin: 12px 0;">

🟧🟧🟧 **MULTI-TRACK ADDITION — ignore §10 if your project is single-track (a single-track project can later become multi-track)** 🟧🟧🟧

## 10. Parallel-track coordination (multi-track projects only)

If `03_PROJECT.md` declares parallel agent tracks (e.g., Alpha and Beta), the project's lane-assignments subsection lists file ownership and coordination rules. Read that section before starting work in any project that declares lanes — file ownership is project-specific.

**Cross-lane file edits require explicit owner authorization for the current session.** `01_MASTER.md` §1.1 / §1.2 ("don't touch what you weren't asked") and §1.20 (lane discipline) apply across lanes. Even with general permission to edit a file class, do not cross into another lane's owned files without session-scoped authorization.

**Shared-edit files** (typically `03_PROJECT.md`, `Logs/lessons.md`, `01_MASTER.md`, `02_GIT.md`, `.gitignore`) are edited by both lanes. Resolve rebase conflicts by section-level interleaving rather than whole-file overwrites — preserve the other lane's edits. When both lanes edit `03_PROJECT.md` in the same window, prefer adjacent-but-distinct sections.

**Parallel-rebase coordination.** If both lanes have PRs ready to push in the same window, the second lane to push rebases onto the first lane's *merged* state, NOT the still-open PR's branch. Rebasing onto an open PR's branch creates the §1.4-forbidden stacked-PR situation. The owner merges in submission order; the second lane pushes after the first merges, then the owner re-bases-and-merges the second.

**Don't both run shared-resource jobs concurrently.** Project-specific rules (in `03_PROJECT.md`'s lane-assignments subsection) typically restrict simultaneous use of CPU-heavy harnesses, shared databases, and rate-limited external APIs. When in doubt, the lane that started first finishes first; the second lane uses a different baseline / different DB file / different worker count, or waits.

🟧 **END addition**

</div>


## 11. Gitignored / untracked stores: merge to main worktree at handoff

Auto-worktrees and per-session sandboxes can accumulate gitignored or untracked data (SQL DBs, replay logs, sentinel directories, generated artifacts, ingest indexes, append-only ledgers, submission zips) that the owner's main working copy never sees. PR review, merge, and worktree cleanup do not propagate this data unless it is committed, copied, or archived. Master rule §1.22 establishes the principle (the session that creates the data owns the merge); this section is the protocol.

**At session end, for each per-worktree gitignored or untracked store the agent touched:**

1. **Identify the stores.** The project's `03_PROJECT.md` should declare them. Common shapes: `*.db` SQLite files, `tools/_runs/<id>/replays/*`, `INDEX.jsonl`-type ledgers, `_visible/` sentinel dirs, generated CSVs, `Logs/`, `submissions/`, profiler output, and ignored `data/` artifacts.
2. **Verify the worktree's copy is a strict superset of the main worktree's** before overwriting:
   - **SQLite DBs:** row-count check on every table — the worktree count must be ≥ the main count for each table. Use `sqlite3 <db> 'SELECT COUNT(*) FROM <table>'`.
   - **Replay-dir / file-tree stores:** file-set diff. The worktree's set must contain every filename in the main's set. Use `comm -23 <(ls main/...) <(ls worktree/...)` — should print nothing.
   - **Append-only ledgers:** line-count check. Worktree lines ≥ main lines.
3. **If verified superset → overwrite directly.** No backup needed. Backups (`.bak-<YYYYMMDD>-pre-merge`) just add disk clutter the owner cleans up later. Verification IS the safety net.
4. **If NOT a superset** (regression in any table, missing files, lower line count) → **STOP and surface to the owner.** Do NOT overwrite. Either write a merge script (e.g., a row-by-row UPSERT for SQLite) or document the gap loudly in the session log and leave both copies in place for the owner to reconcile.
5. **If the full set is too large or too noisy for git, preserve all of it in the main worktree under a local ignored artifact path** such as `local_artifacts/<date>-<worktree>/`, with a complete manifest and checksum. Do not cherry-pick "best" artifacts unless the owner explicitly asked for a curated export.
6. **Record the merge/archive in the session log's `## Final state` block** so the next agent reading the log knows what was synced. Format: `Non-tracked stores preserved: <path> (<count>, <checksum>)`.

**Project opt-out.** Per master §1.22, projects may declare in `03_PROJECT.md` that specific gitignored stores are cheap to regenerate (`node_modules/` after `npm install`, `.venv/` after `pip install`) and don't need merging. Default is "merge unless opted out."

**Why this matters.** A 24-hour autonomous session can ingest hundreds of MB into a worktree-local DB. Without this merge step, the owner returning to the main worktree sees only the pre-session state and either re-runs the ingest (wasting hours) or assumes the new data was lost. The cost asymmetry — seconds to verify and `cp` versus hours of re-work — is what makes this a hard rule rather than a soft preference.


---

*This playbook is the universal git workflow primer. Canonical source: `AI_Instructions/playbooks/Git.md`. Manually copied into each project as `.ai_instructions/02_GIT.md` when rules change.*

*Last updated: 2026-05-04 (added §11 Gitignored stores: merge to main worktree at handoff; per master §1.22).*
