# NeuroGolf HandOff Workspace

This folder is the project-local control room for using HandOff as a personal workflow/kanban layer for K-NeuroGolf.

## Operating Rules

- Run on testnet only. The current project default is `base-sepolia`.
- Keep secrets out of this folder. Live credentials stay in the ignored root `.env.handoff`.
- Shareable evidence goes in `export/sanitized/` or the dated public API docs.
- Treat `logs/` as runtime-only scratch. Historical run logs were removed during
  retirement cleanup; the last retained supervisor pointer said no active
  HandOff runner/monitor or `ng_gate` process was found.
- Never share `private/`, `tmp/`, `logs/`, or `export/private/`.

## Current Identity

- HandOff MCP: `https://handoff.socnet.lol/mcp`
- Agent id: `boladi_a`
- Preferred settlement network: `base-sepolia` testnet

## Useful Commands

Verify the account, agent, and testnet posture:

```powershell
.\Handoff\scripts\Test-HandoffSetup.ps1
```

Re-apply testnet-only defaults:

```powershell
.\Handoff\scripts\Set-HandoffTestnet.ps1
```

Record a manual workflow event:

```powershell
.\Handoff\scripts\Write-HandoffEvent.ps1 -Kind note -Summary "Created initial NeuroGolf kanban plan"
```

Create a sanitized export for sharing:

```powershell
.\Handoff\scripts\Export-HandoffData.ps1
```

## Share Policy

Safe to share after review:

- `README.md`
- `config/public.json`
- `docs/public_*_2026-07-03.*`
- `export/sanitized/*.json`

Do not share:

- root `.env.handoff`
- `logs/`
- any `hot_...` token
- any `ak_...` key
- email/password
- private wallet details
- raw API captures
