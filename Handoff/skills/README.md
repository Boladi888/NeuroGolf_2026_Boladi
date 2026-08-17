# Handoff/skills/ — vendored skills (per-machine junctions)

This directory holds **skills the HandOff lanes consult at runtime**, mounted into the
repo as Windows directory junctions so every lane (Claude *and* Codex) can read them
via a stable in-repo path, without committing a drifting copy into git.

| Path | What it is | Single source of truth |
|---|---|---|
| `handoff-workflows/` | **Junction** → `..\..\..\Boladi_Skillz\HandOffz\handoff-workflows\` | `Boladi_Skillz/HandOffz/handoff-workflows/` |

- The junction contents are **gitignored** (see `.gitignore`); only this README is tracked.
- Edits belong in `Boladi_Skillz` — they reflect here immediately (it is the same folder).
- Prompts reference the skill by its in-repo path, e.g.
  `Handoff/skills/handoff-workflows/SKILL.md` and `.../references/<topic>.md`, so a lane
  that does not mount `Boladi_Skillz` can still read it.

## Recreate the junction (new machine / after a wipe)

```powershell
$src    = "<REPO_ROOT>\Handoff\skills"
$target = "<WORKSPACE_ROOT>\Boladi_Skillz\HandOffz\handoff-workflows"
if (Test-Path "$src\handoff-workflows") { Remove-Item "$src\handoff-workflows" -Force }
New-Item -ItemType Junction -Path "$src\handoff-workflows" -Target $target | Out-Null
```

`Remove-Item <junction> -Force` removes the **junction only**, never the target — but do
**not** pass `-Recurse` (that can recurse into and delete the real skill). See
`WHERE_DATA_LIVES.md`.
