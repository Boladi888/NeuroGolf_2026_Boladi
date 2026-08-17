# Prompt for Claude Design

Use this prompt with Claude Design after giving it this `Handoff/docs` folder.

---

I need two high-quality workflow visuals for my K-NeuroGolf Handoff system:

1. Current workflow: how it works today.
2. Optimized workflow: how it should work after cleanup.

Use the files in this folder as source material. Start with:

- `01_current_neurogolf_handoff_workflow.md`
- `02_optimized_handoff_workflow_brief.md`
- `03_public_handoff_platform_notes.md`

Then consult raw Handoff source snapshots if you need exact public platform details:

- `public_api_docs_2026-07-03.md`
- `public_swarm_contract_2026-07-03.json`
- `public_handoff_agent_skill_2026-07-03.md`
- `public_connect_2026-07-03.json`
- `public_capabilities_2026-07-03.json`

Do not use or request secrets. Do not include `.env.handoff`, `Handoff/private/`, `Handoff/tmp/`, `Handoff/logs/raw/`, agent keys, bearer tokens, private wallet details, or raw transcripts.

## Current Workflow Visual

Show these layers:

- Handoff public network / board:
  - `NeuroGolf_HO` project
  - batch goals
  - Handoff task cards
  - canonical statuses: `todo`, `in_progress`, `pending_verification`, `verified`, `rejected`
- Local NeuroGolf runner:
  - `prepare_handoff_batch.py`
  - lane configs
  - runner loop
  - worker pool
  - reviewer pool
  - monitor loop
  - operations logs and state files
- Submit Captain:
  - `Updates_Codex`
  - `Updates_Claude`
  - packet validation
  - gated probe/stage path

Show three active/recent lanes:

- Codex batch 44
- Codex2 batch 45
- Claude batch 46

Make clear that Codex, Codex2, and Claude share the same Handoff project but have separate lane/batch configs, agents, state files, logs, and captain packet paths.

Highlight current friction:

- Board/local drift.
- Public Handoff statuses are simpler than local runner substates.
- `pending_verification` is overloaded for "waiting for reviewer" and "review underway."
- Workers can run long or hit timeouts.
- Hard tasks and duplicate/floor rediscovery can waste spend.
- Pid/process truth, board truth, and artifact truth need reconciliation.
- Manual stop/repair/closeout operations should be made explicit.

## Optimized Workflow Visual

Show a cleaner production-line flow:

1. Candidate prompt pool.
2. Preflight collision scan.
3. Scout gate.
4. Excluded-by-reason report.
5. Batch planner.
6. Handoff goal/task creation with stable external keys.
7. Runner scheduler with bounded concurrency.
8. Worker pool.
9. Reviewer pool with affinity.
10. Verdict branch:
    - SHIP -> packet validator -> Submit Captain inbox -> captain probe/stage gate.
    - KEEP_WORKING -> bounded rework -> worker pool.
    - DROP -> terminal drop with reason.
11. Monitor/reconciler loop:
    - board state
    - local state
    - process tree
    - artifacts
12. Lifecycle controls:
    - pause
    - resume
    - stop
    - repair
    - closeout

Use stable visual conventions:

- Handoff/public network: one color family.
- Local execution: another color family.
- Submit Captain/scoring: another color family.
- Safety/control plane: neutral but prominent.
- Terminal outcomes: clear success/drop/failure styling.
- Show testnet safety as a badge, not as the main theme.

The final output should help a new agent understand:

- What Handoff is.
- How NeuroGolf uses it.
- Where work is currently leaking time or clarity.
- What the optimized workflow changes.
- Which parts are public board state versus local private execution details.

