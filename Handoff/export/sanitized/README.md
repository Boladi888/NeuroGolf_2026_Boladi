# Sanitized HandOff Exports

Generated snapshots for sharing go here.

Run:

```powershell
.\Handoff\scripts\Export-HandoffData.ps1
```

Before sharing, do a quick scan for accidental secrets:

```powershell
rg -n "hot_|ak_|password|secret|token|private key" Handoff\export\sanitized Handoff\logs
```
