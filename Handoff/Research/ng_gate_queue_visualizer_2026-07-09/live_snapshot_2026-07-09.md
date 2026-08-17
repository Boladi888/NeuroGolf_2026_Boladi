# Live Snapshot - 2026-07-09

Snapshot window: approximately 18:00 to 18:04 America/New_York on 2026-07-09.

This was read-only except for creating this research folder. No active gate process was stopped or modified.

## Existing Status Tool

Command:

```powershell
python tools/ng_gate_status.py --json
```

Observed summary:

```json
{
  "throttle_dir": "C:\\Users\\Qx\\AppData\\Local\\Temp\\ng_throttle",
  "cpu_count": 24,
  "psutil": true,
  "normal_cap": 19,
  "normal_live": 1,
  "normal_free": 18,
  "heavy_cap": 1,
  "heavy_live": 1,
  "heavy_free": 0,
  "heavy_modes": ["ab", "battery", "synth", "truth"],
  "heavy_n_threshold": 20000,
  "slots": [
    {
      "kind": "slot",
      "pid": 3740,
      "label": "t295",
      "age_s": 2109.3,
      "alive": true,
      "lock": "slot_3740_8676.lock"
    }
  ],
  "heavy": [
    {
      "kind": "heavy",
      "pid": 3740,
      "label": "heavy t295",
      "age_s": 2109.5,
      "alive": true,
      "lock": "heavy_3740_8522.lock"
    }
  ]
}
```

Interpretation: `task295` was holding both the normal slot and the scarce heavy slot. Heavy capacity was fully occupied.

## Lockfiles

Directory:

`<TEMP_DIR>\ng_throttle`

Observed lock contents:

```text
3740 heavy t295 1783632509
3740 t295 1783632509
```

Interpretation: only `pid 3740` held active gate locks at this moment.

## Live Process Tree Evidence

Read-only process scan:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools/ng_gate.py|tools\\ng_gate.py' }
```

Condensed result:

```text
Pid    PPid   Name        AgeMin  HasLock      Children  Task  Mode
35960  32384  pwsh.exe    35.1    false        3740      295   truth
3740   35960  python.exe  35.1    heavy,slot             295   truth
21544  13352  pwsh.exe     2.2    false        9604      350   truth
9604   21544  python.exe   2.2    false        12896     350   truth
12896  9604   python.exe   2.2    false                  350   truth
```

Interpretation:

- `task295 truth --full` was active and holding the heavy slot.
- `task350 truth --full` had a live process tree but no lock yet. This is the live example of a waiting full gate that `ng_gate_status.py` cannot show as queued.
- The `task350` queue age could be approximated as about 2.2 minutes from process creation. That is not the same as a precise wait timestamp, but it is useful enough for a read-only first pass.

## Handoff Monitor Context

`Handoff/logs/monitor_codex2_batch57.latest.txt` reported:

```text
Counts: drop=15, review_parse_failed=7, ship_packeted=1, todo=2, worker_running=3
...
- task295: worker_running
- task301: worker_running
- task350: worker_running
...
Recent ops included repeated GATE_WAIT entries and task295 timeout/retry context.
```

`Handoff/logs/monitor_claude_batch56.latest.txt` reported:

```text
Counts: drop=7, ship_packeted=2, worker_running=1
...
- task366: worker_running
```

Interpretation: active lanes can be joined to the gate process view by task number, but the board can drift. The visualizer should label runner-local state separately from process-lock state.

## Rescue Queue Context

`Handoff/private/review_rescue_queue.jsonl` contains open rows for many strict-gain candidates whose full gates or reviewer verdicts were blocked or lost. Recent rows include:

```text
task037, task068, task099, task176, task200, task212, task300, task329, task350
```

Recent reasons repeatedly cite the heavy gate slot being busy, including `task295` holding the scarce slot.

Interpretation: the dashboard should show both live waiters and the rescue backlog. The rescue backlog is not the same as the live process queue.

## Safety Notes From Testing

- I did not run any `ng_gate.py` command.
- I did not call `--no-throttle`.
- I did not delete or reclaim any lockfile.
- I did not alter `Handoff/private/*.json`, logs, runner state, or active processes.
- The only write was this research packet under `Handoff/Research/`.

