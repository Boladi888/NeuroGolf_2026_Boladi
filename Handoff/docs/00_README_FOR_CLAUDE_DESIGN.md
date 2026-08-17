# Handoff Visualization Pack for Claude Design

Created: 2026-07-03

This folder is meant to give Claude Design enough context to visualize:

1. The current NeuroGolf Handoff workflow.
2. A cleaner optimized Handoff workflow.

Start with these files:

- `01_current_neurogolf_handoff_workflow.md` - the current local workflow, lanes, roles, state machine, live snapshot, pain points, and a Mermaid diagram.
- `02_optimized_handoff_workflow_brief.md` - a proposed target workflow with cleaner queues, scout gates, lifecycle controls, and visualization suggestions.
- `03_public_handoff_platform_notes.md` - distilled notes from the public Handoff website and API docs.
- `04_visualization_prompt_for_claude_design.md` - a ready-to-paste prompt for the design agent.

Raw public Handoff source snapshots are also included:

- `public_site_home_2026-07-03.html` from `https://handoff.socnet.lol/`
- `public_site_docs_page_2026-07-03.html` from `https://handoff.socnet.lol/docs`
- `public_api_docs_2026-07-03.md` from `https://handoff.socnet.lol/api/v1/docs`
- `public_swarm_contract_2026-07-03.json` from `https://handoff.socnet.lol/api/v1/docs/swarm`
- `public_handoff_agent_skill_2026-07-03.md` from `https://handoff.socnet.lol/skill`
- `public_connect_2026-07-03.json` from `https://handoff.socnet.lol/api/v1/connect`
- `public_capabilities_2026-07-03.json` from `https://handoff.socnet.lol/api/v1/capabilities`
- `public_payments_spec_2026-07-03.json` from `https://handoff.socnet.lol/api/v1/payments/spec`
- `public_contracts_info_2026-07-03.json` from `https://handoff.socnet.lol/api/v1/contracts/info`
- `public_webhooks_spec_2026-07-03.json` from `https://handoff.socnet.lol/api/v1/webhooks/spec`
- `public_source_manifest_2026-07-03.json` - fetch manifest with source URL, status, type, size, and fetch time.

Local source files worth reading in the repo:

- `Handoff/README.md`
- `Handoff/runner/README.md`
- `Handoff/config/public.json`
- `Handoff/config/runner_codex_p44.json`
- `Handoff/config/runner_codex2_p45.json`
- `Handoff/config/runner_claude_p46.json`
- `Handoff/logs/monitor_codex_batch44.latest.txt`
- `Handoff/logs/monitor_codex2_batch45.latest.txt`
- `Handoff/logs/monitor_claude_batch46.latest.txt`
- `Handoff/_proposals/combined_handoff_platform_feedback_2026-07-02.md`
- `Handoff/_proposals/codex_p43_handoff_feedback_2026-07-03.md`

Share boundary:

- Safe: this docs folder, public configs, monitor summaries, non-raw operations summaries.
- Do not share: `.env.handoff`, `Handoff/private/`, `Handoff/tmp/`, `Handoff/logs/raw/`, `Handoff/export/private/`, agent keys, bearer tokens, wallet/private credentials, or raw Codex/Claude transcripts.

