# Implementation Notes - 2026-07-09

Implemented after review of the research packet.

## Accepted

- Added a read-only snapshot/watch tool: `tools/ng_gate_queue_snapshot.py`.
- Added a visible PowerShell launcher: `Handoff/scripts/Start-NGGateQueueWatch.ps1`.
- Added a root BAT for the user: `NG_Gate_Queue_Watch.bat`.
- Added a solid Tk GUI watcher: `tools/ng_gate_queue_window.py`.
- Added an automatic `ng_gate.py` queue-watch hook. On Windows, the first gate run opens one watcher if none is already alive.
- Kept explicit controls for the automatic watcher:
  - `--watch-queue` forces an open attempt for that run
  - `--no-watch-queue` skips it for that run
  - `NEUROGOLF_GATE_WATCH=0` disables auto-open for the environment
  - `NEUROGOLF_GATE_WATCH_INTERVAL=3`
- The watcher groups parent/child `ng_gate.py` process trees so PowerShell wrappers and venv launchers do not overcount as separate gates.
- The watcher joins active locks, live gate processes, runner state, and the rescue queue in one terminal view.

## Rejected For V1

- No browser dashboard. A plain visible terminal is lower-risk while active runs are live.
- Updated after user feedback: default `.bat` now uses a solid Tk GUI via `pythonw.exe` when available, because Windows Terminal transparency/acrylic profiles can make the terminal watcher look odd.
- No stop/kill buttons. The tool only observes.
- No throttle cap changes.
- No `--no-throttle` testing.
- No duplicate watcher storm. Auto-open is default now, but it uses a PID sentinel, process scan, and short launch lock so concurrent agents should converge on one watcher window.
- No throttle wait-record instrumentation yet. Queue age is currently based on process age for waiters; exact "entered acquire loop" timing can be added later if needed.

## Behavior

Manual watcher:

```powershell
.\NG_Gate_Queue_Watch.bat
```

Terminal fallback:

```powershell
.\NG_Gate_Queue_Watch.bat --terminal
```

One-off gate run with explicit watcher request:

```powershell
python tools/ng_gate.py truth --task 350 --cand path\to\candidate.onnx --full --watch-queue
```

Disable automatic watcher windows for this shell:

```powershell
$env:NEUROGOLF_GATE_WATCH = "0"
```

Force terminal watcher for auto-open:

```powershell
$env:NEUROGOLF_GATE_WATCH_TERMINAL = "1"
```

The auto hook writes a watcher PID sentinel in the throttle directory, scans for already-running watcher commands, and takes a short launch lock so many concurrent gates do not open many watcher windows. If the watcher is closed, the next gate run can open a new one unless disabled.

## Safety

- Snapshot reads `%TEMP%\ng_throttle\*.lock` directly and does not call lock reclaim/delete paths.
- Snapshot uses process metadata and JSON/log files only.
- Closing the watcher stops only the watcher.
- Closing an actual `ng_gate --new-window` gate still stops that gate as before.
