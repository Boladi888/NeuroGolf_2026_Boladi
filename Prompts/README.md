# Prompts/ — Supervisor launch prompts

Archived supervisor operating contracts, one per **lane x backend**. They are kept as reusable examples for future multi-agent lane orchestration, not as proof that the old NeuroGolf runners are still live.

| file | lane | backend | stages to team? |
|---|---|---|---|
| `ALPHA Supervisor Claude-_promptz.txt` | ALPHA | Claude (`claude -p`) | yes - Part B drains all lanes to `Staged_Boladi` |
| `ALPHA Supervisor Codex-_promptz.txt`  | ALPHA | Codex (`codex exec`) | yes |
| `BETA Supervisor Claude-_promptz.txt`  | BETA  | Claude | no - feeds `Updates_Claude/Codex` for ALPHA |
| `BETA Supervisor Codex-_promptz.txt`   | BETA  | Codex  | no |

Each `_promptz.txt` is the **contract / fallback**. For a routine launch, paste the **short form** below plus a
one-line pointer; the Supervisor fills the rest from the contract and `prep_run.py`. Keeping concrete
inputs in the short form (and out of the pasted body) avoids leaking stale details — each prompt's
precedence clause resolves any conflict in favor of the prompt.

## Short-form launch request (preferred for routine launches)

```text
Apply Prompts/<ALPHA|BETA> Supervisor <Claude|Codex>-_promptz.txt. Use the prep_run.py fast path unless it fails.
prompt folders:
- The_400_Tasks/Task Prompts/<N>
run id seed: <lane>_p<N>_<backend>_cli
exclude: <taskNNN, ... | none>
worker concurrency: <N>
monitor: every 30 minutes
special overrides: <none | ...>
```

The Supervisor renders the run with:

```text
python Agenters/Orchestrator/prep_run.py --run-id <seed> --lane <Claude|Codex> \
  --prompt-folder "The_400_Tasks/Task Prompts/<N>" --worker-concurrency <N> [--exclude taskNNN] --bump
```

- `--bump` auto-picks the next free run id (avoids the `beta_p7b→l` seed-reuse drift from reusing a stale seed).
- Periodic check-in (read-only; audits only when terminal):
  `python Agenters/Orchestrator/runs_status.py pulse <run_id> --audit-if-terminal`
- Shared-probe check before any allowed probe: `python Agenters/Submit_Captain/captain_ctl.py probe-status`

The long numbered section of each `_promptz.txt` remains the authoritative contract — the short form just avoids
re-pasting it every launch.
