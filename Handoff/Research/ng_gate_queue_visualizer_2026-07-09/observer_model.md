# Proposed Observer Model

This model is intended for a read-only dashboard or terminal watch. It should be implemented before any throttle instrumentation or control buttons.

## Logical Objects

### Slot

Source: `%TEMP%\ng_throttle\*.lock`

Fields:

- `kind`: `heavy` or `slot`
- `pid`
- `label`: usually `tNNN` or `heavy tNNN`
- `task`: parsed from label when possible
- `lock_file`
- `lock_mtime`
- `lock_age_s`
- `pid_alive`

Important: read lockfiles directly. Avoid calling helper paths that reclaim/delete locks unless that side effect is deliberately reviewed.

### Process

Source: `Get-CimInstance Win32_Process`

Fields:

- `pid`
- `ppid`
- `name`
- `creation_time`
- `age_s`
- `executable_path`
- `command_line`
- parsed `mode`
- parsed `task`
- parsed `candidate_path`
- parsed `n`
- parsed `full`
- parsed `cwd` if available from context

### Logical Gate Run

Source: grouped process tree plus slots.

Why grouping matters:

- a PowerShell wrapper may contain the full `ng_gate.py` command;
- a venv launcher may also contain the same command;
- the real Python child may be the one that owns the lock;
- treating every matching process as a separate queue item would overcount.

Suggested grouping:

1. Find all command lines matching `tools/ng_gate.py` or `tools\\ng_gate.py`.
2. Build parent-child relationships across all processes.
3. For each root matching process, include descendants that share the same parsed task/mode/candidate.
4. If a child owns a lock, attach the lock to the group.
5. Collapse duplicate parent/child command lines into one logical gate run.

Fields:

- `group_id`
- `task`
- `mode`
- `candidate_path`
- `n`
- `is_full`
- `is_heavy_expected`
- `root_pid`
- `lock_pid`
- `process_pids`
- `started_at`
- `age_s`
- `slot_state`
- `runner_context`
- `orphan_flags`

## Classification

### active_heavy

Expected heavy and group owns a `heavy` lock.

Example from snapshot:

`task295 truth --full`, PID 3740, owns `heavy_3740_8522.lock`.

### active_light

Group owns a normal `slot` lock but not a `heavy` lock. This can include visible gates, capped truth/AB below the heavy threshold, and other light modes.

### waiting_heavy

Expected heavy and group has no heavy lock yet.

Example from snapshot:

`task350 truth --full`, process tree exists, no lockfile yet, waiting behind `task295`.

Queue age for this state is approximate unless wait-record instrumentation is added.

### waiting_normal

Group has no normal slot yet. This can happen if normal slots are full or the CPU guard blocks acquisition. It should be rare when `normal_cap` is large.

### orphan_suspect

This is a warning, not an action.

Set one or more flags:

- `no_runner_match`: task not present in any live runner state.
- `parent_gone`: parent PID not alive.
- `old_lock`: lock age exceeds a configurable threshold.
- `alive_but_low_cpu`: process alive, long age, very low CPU delta over several polls.
- `anonymous_worker`: `python -` or `python -u -` process tied to Handoff raw logs, not a clean `ng_gate.py` command.
- `state_disagrees`: runner says task is terminal but process is still alive.

Do not auto-kill based on these flags.

## Handoff Context Join

Join by task number, but display source confidence:

- `process`: live command line and PID.
- `lock`: active throttle slot.
- `runner_state`: local `Handoff/private/runner_state_*.json`.
- `monitor_latest`: monitor summary text.
- `rescue_queue`: open rescue row.

Because board state can drift, the visualizer should prefer local process and runner state for "what is happening now", while showing board drift as context.

## Terminal Display Sketch

```text
NG Gate Queue Watch  2026-07-09 18:04:12  refresh=3s

HEAVY SLOT 1/1 used
task  mode   age    pid   owner          candidate
295   truth  35m    3740  Codex2 worker  task295_v14_cp20_waring.onnx

WAITING FULL GATES
task  mode   wait   pids              owner             reason
350   truth  2m     21544>9604>12896  Codex2 reviewer   waiting for heavy slot

LIGHT GATES
(none)

RESCUE QUEUE OPEN
oldest  task277  age 1d+  missing full_ab, full_truth, reviewer_verdict
recent  task350  age 7m   missing truth-full, ab-full, reviewer_verdict

ORPHAN SUSPECTS
(none confirmed; no action taken)
```

## Polling

Default: every 2 to 5 seconds.

Avoid sub-second polling. The process and file reads are cheap, but there is no need to add noise.

## Output Formats

Recommended snapshot JSON keys:

```json
{
  "generated_at": "2026-07-09T22:04:12Z",
  "throttle": {
    "dir": "...",
    "normal_cap": 19,
    "normal_live": 1,
    "heavy_cap": 1,
    "heavy_live": 1
  },
  "runs": [],
  "rescue_queue": {
    "open_count": 0,
    "oldest_open": [],
    "recent_open": []
  },
  "orphan_suspects": []
}
```

This lets a terminal watch, future web UI, or automation consume the same read-only core.

