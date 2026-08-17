# Runner passthrough: Opus fast mode + per-task Fable override (owner 2026-07-10 "Opus fast + Fable on hard")

**Status:** SPEC — validated standalone, NOT yet landed. Land BETWEEN WAVES (no live lane), dry-run, then next batch.
**Why deferred:** `handoff_runner_claude.py` is launch-critical and its Python process re-spawns per poll (observed:
batch58 python PID re-spawned ~49 min into the run), so an in-place edit would load mid-batch — the README bars
hot-patching while a lane is live. Batch 58 was live when this was written.

## Owner decision
Supervisor prompt files (ALPHA+BETA, handoff+regular) + `SPAWN_PROMPT.md` / `prep_run.py` / `Agenters` READMEs already
updated to the doctrine: **default workers/reviewers to Opus fast mode; escalate known hard-floor / hardest-novel tasks
to Fable 5.** This proposal wires the runner so that doctrine can actually take effect. ("Fable Ultracode fast" is not a
real profile — fast mode is Opus-only; Fable is the heaviest/slowest model.)

## The change — `Handoff/runner/handoff_runner_claude.py`, `invoke_claude()` (currently lines ~423-429)

BEFORE:
```python
        cmd = [cl["executable"], "-p", "--output-format", "json", "--model", cl.get("model", "opus")]
        if cl.get("effort"):
            cmd += ["--effort", cl["effort"]]
        if cl.get("skip_permissions", True):
            cmd.append("--dangerously-skip-permissions")
        for d in cl.get("extra_dirs", []):
            cmd += ["--add-dir", str(d)]
```

AFTER:
```python
        # Per-task model override (owner 2026-07-10 "Opus fast + Fable on hard"): a task listed in
        # claude.model_overrides (task-id -> model) uses that model (e.g. "fable" for known hard-floor
        # tasks); everyone else uses claude.model. Absent the key => unchanged.
        model = (cl.get("model_overrides") or {}).get(task) or cl.get("model", "opus")
        cmd = [cl["executable"], "-p", "--output-format", "json", "--model", model]
        if cl.get("effort"):
            cmd += ["--effort", cl["effort"]]
        # Headless fast mode is OPUS-ONLY (~2.5x, identical quality, +cost). Enable via claude.fast_mode:true
        # (merged into any claude.settings dict). Skipped for non-opus models. Absent fast_mode/settings => no --settings.
        settings = dict(cl.get("settings") or {})
        if cl.get("fast_mode") and "opus" in str(model):
            settings["fastMode"] = True
        if settings:
            cmd += ["--settings", json.dumps(settings)]
        if cl.get("skip_permissions", True):
            cmd.append("--dangerously-skip-permissions")
        for d in cl.get("extra_dirs", []):
            cmd += ["--add-dir", str(d)]
```
`json` is already imported (line 17). This applies to BOTH workers and reviewers (both go through `invoke_claude`).

## New optional `claude` config keys (all default OFF -> fully backward-compatible)
- `fast_mode` (bool): when true AND model is an opus alias, append `--settings '{"fastMode":true}'`.
- `settings` (object): arbitrary extra `--settings` JSON; `fastMode` is merged in when `fast_mode` is set.
- `model_overrides` (object, task-id -> model string): e.g. `{"task017":"fable"}` for known hard-floor tasks.

## Validation (standalone, `scratchpad/validate_runner_builder.py` — ALL 5 PASS)
- A: batch58-style config (no new keys) => command **byte-identical** to today (no `--settings`). ← the safety guarantee
- B: `fast_mode:true` + opus => `--settings '{"fastMode": true}'`.
- C: `model_overrides{task017:fable}` + fast_mode => `--model fable`, **no** `--settings` (fast mode is opus-only).
- D: a non-overridden task under the same config => opus + fastMode.
- E: pre-existing `settings` dict is merged, not clobbered => `{"foo":1,"fastMode":true}`.

## Turning it on (per batch, after landing)
- Fast mode default: set `claude.fast_mode: true` in the rendered config (or have `prepare_handoff_batch.py` emit it by
  default — a 1-line change; confirm the +token-cost with owner before defaulting on, since it opposes the cost-default).
- Fable escalation: add known hard-floor tasks to `claude.model_overrides` (manually, or teach
  `prepare_handoff_batch.py` to populate from a hard-floor list).

## Apply-at-drain procedure
1. Confirm no live Claude lane: `runner_loop.claude.pid.json` -> `state:stopped`; no `handoff_runner_claude` python alive.
2. Apply the AFTER block above to `handoff_runner_claude.py`.
3. `python -m py_compile Handoff/runner/handoff_runner_claude.py`.
4. Dry-run with a throwaway config that sets `fast_mode:true` + a `model_overrides` entry:
   `... Start-HandoffRunner.ps1 -Once -DryRun -Lane Claude -ConfigPath <test>.json` and confirm the printed
   `DRY RUN claude ...` line shows `--model fable` for the override task and `--settings '{"fastMode":true}'` for opus.
5. Also dry-run the CURRENT batch config to confirm it still prints the unchanged opus/max command (no `--settings`).
6. Then the next batch can opt in.
