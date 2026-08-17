# Handoff Playbooks

This folder holds playbooks for producing **portable briefs** — single self-contained markdown files an outside agent (folder-less or human reviewer) can pick up cold and work from.

## Currently

- `ResearchHandoff.md` — for handing research work to an agent without project-folder access.

## The 5-section skeleton

Every handoff playbook follows the same shape so receiving agents recognize the format across types:

1. **Header** — what this is, who you (the receiver) are, what's expected back.
2. **Project context** — relevant subset of `03_PROJECT.md` (description, tech stack, architecture/decisions; NOT TODOs / open bugs / where-I-left-off — those are noise for handoffs).
3. **The inputs** — the actual material (research GAPS, diff, error log, architecture doc, etc.).
4. **The workflow** — how the owner wants this kind of work done; usually a copy of the universal style guide for that work type (e.g., `playbooks/Research.md` for research handoffs).
5. **Output format** — what to send back, structured.

## Conventions

- **Default output location:** `<project>/<area>/_HANDOFF.md` where `<area>` is whichever folder hosts the relevant content (`Research/` for research, project root for code-review, etc.). The file is overwritten each time and gitignored.
- **Project-side generation only.** Handoffs are produced by the project-side agent that has full local context. There is no tooling that generates handoff files from outside the project — that would require cross-folder access the canonical-side workflow doesn't need.
- **Manual ingestion.** When the response comes back, the owner pastes it into a project session and asks the agent to file it per the playbook's "ingestion" section. No tooling for this yet.
- **Refusal conditions in every playbook.** Every handoff playbook lists explicit refusal cases — situations where the agent should NOT generate the brief and should tell the owner why.

## Adding a new handoff type

When a recurring pattern emerges that needs a folder-less agent (e.g., debugging a stuck bug, reviewing an architectural change), add a new playbook here. Match the 5-section skeleton. Define the ingestion path. Don't speculatively add types that haven't bit yet — the maintenance overhead grows quickly.

Future candidates (NOT built; waiting on use cases):
- `CodeReviewHandoff.md` — PR or diff review by an outside agent.
- `DebugHandoff.md` — diagnose a failure when the project agent is going in circles.
- `ArchitectureReviewHandoff.md` — second-opinion design feedback.

---

*Canonical source: `AI_Instructions/playbooks/handoffs/_README.md`. Manually copied into each project's `.ai_instructions/playbooks/handoffs/` when rules change.*

*Last updated: 2026-04-27.*
