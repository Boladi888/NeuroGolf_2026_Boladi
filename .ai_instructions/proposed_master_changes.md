> LIVE — edit freely. Append-only for new proposals.

---

# Proposed master rule changes — NG

Append-only file for capturing rule suggestions during project sessions. Reviewed and pruned during overview AI_Instructions sessions; accepted entries get applied to `AI_Instructions/README_AI_Master.md` or `AI_Instructions/playbooks/Git.md` and converted to one-line acceptance stubs here for traceability.

Format for each new entry:

```markdown
## YYYY-MM-DD — Short title
**Suggested location:** 02_GIT.md | 01_MASTER.md | 03_PROJECT.md (pick one)
**Reason:** What problem this solves; what it costs to NOT have this rule.
**Draft text:** The actual rule text, ready to drop into the source file with minimal editing.
**Source session:** ../Logs/sessions/<filename>.md
```

Rejected entries stay (annotated with `**Rejected:** <date> — <reason>`) so the next overview doesn't re-surface them.

---

## Active proposals

## 2026-05-10 — Post-merge rebase invariant for side-worktrees

**Suggested location:** 02_GIT.md §1.1 (extend "Rebase before opening")
**Reason:** Side-worktrees (Claude / codex) routinely produce multiple PRs in the same session. After a PR merges via rebase, the worktree's branch contains the pre-merge SHAs; master has post-merge SHAs with identical content. Opening the next PR from the same worktree without a fetch+rebase produces a silent `CONFLICTING` PR on GitHub even though no real divergence exists. Observed 2026-05-10 on PRs #60→#61 (resolved by manual rebase of `Logs/lessons.md` and `.ai_instructions/03_PROJECT.md`) and #61→#62 (would have hit the same false-conflict — caught by applying the proposed discipline). Cost per occurrence: ~30-60s recovery plus owner attention to authorize the force-push.
**Draft text:**

> **§1.1 Post-merge rebase invariant (addition).** When a PR you opened from a side-worktree (Claude / codex) merges via rebase, the local branch in that worktree contains stale SHAs — they differ from the just-merged master even though the patch content is identical. **Before making any further commit on that worktree** (whether on the same branch or a new branch off it), run `git fetch origin && git reset --hard origin/master` (if no local-only commits) OR `git rebase origin/master` (if local commits in flight). Skipping this is the most common cause of false-conflict PRs.
>
> Recommended cadence for side-worktrees: **one PR per worktree.** After the PR merges, either (a) reset the branch to fresh `origin/master` and reuse for the next PR's commits, or (b) treat the worktree as disposable and remove it. Long-running branches in side-worktrees accumulate stale-SHA divergence and produce false conflicts on every subsequent PR.

**Source session:** ../Logs/sessions/2026-05-10_1530_git-cleanup-and-data-layout-reorg.md, ../Logs/sessions/2026-05-10_1700_layered-reorg.md

---

## Accepted entries (one-line stubs for traceability)

(none yet)

---

*Last updated: 2026-05-10.*
