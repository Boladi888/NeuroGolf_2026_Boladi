# CLAUDE.md — Claude Code session ground rules (owner-set 2026-06-05; canonical: `.ai_instructions/playbooks/agent_feedback_rules.md` §9)

- **NEVER create a git worktree or branch on your own initiative** (no EnterWorktree, no `git worktree add`, no feature branches) — not even for "isolation" before code edits. Work in place in this main local dir. The owner works locally with uncommitted multi-agent WIP; stray trees/branches cost triage sessions. Only work inside a worktree if the owner explicitly spawned your session in one.
- **Never run `git pull`, `git reset --hard`, or `git checkout`/`git restore` over working paths, and never commit or push unless asked.** Uncommitted working-tree state IS live project work — losing it is worse than any git untidiness.
- **No Claude auto-memory for this project** — durable knowledge goes in the shared project docs (`.ai_instructions/playbooks/agent_feedback_rules.md` §8 has the routing table).
