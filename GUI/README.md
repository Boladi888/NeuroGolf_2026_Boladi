# NeuroGolf Lab

[ Python 3.10+ ] [ FastAPI ] [ React ] [ ONNX ] [ ARC-AGI ] [ Local-only ]

NeuroGolf Lab is a browser-based ARC task viewer and visual ONNX graph editor,
integrated with the surrounding K-NeuroGolf project. It validates graphs with
the **exact official competition semantics** (one-hot `[1,10,30,30]` encoding,
`> 0.0` threshold decode, train + test + arc-gen), scores them with the
project's official scorer (`tools/_score_v2.py`), compares against the live
best in `task_state.csv`, and saves passing artifacts into
`The_400_Tasks/Task_by_Task/<task>/prototypes/`.

Everything runs on localhost. There are no external uploads.

## Quick Start

```bat
:: Windows - double-click start.bat (or run start.ps1)
start.bat
```

This opens the server in its own visible console window titled
`NeuroGolf Lab :8082`. **Closing that window stops the server** — the server
runs in-process in that window, and the process additionally binds itself to a
Windows JobObject with `KILL_ON_JOB_CLOSE`, so any helper processes are reaped
by the kernel no matter how the window dies (close button, taskkill, logoff).
No orphaned processes.

```bash
# bash / Git Bash: run in the current terminal instead
bash start.sh
```

All starters build the web client on first run and serve at:

```text
http://127.0.0.1:8082
```

Manual equivalent:

```bash
cd client && npm install && npm run build && cd ..
python launch.py            # or: python launch.py --port 9000
```

## Doors Maze (game mode — the default view)

The app opens in **Doors Maze**: solve a task by walking a path of intuitive
"doors" (crop, flip, recolor, tile…) from the input picture to the goal
picture. Every door applies to **all train examples at once** — that's what
makes a sequence a *rule*. The meter shows % matching cells; promising doors
glow (1-step lookahead); the filmstrip rewinds/branches; winning paths verify
against the full official sets and export either as a **real scored ONNX
model** (auto-compile through the competition gate) or as a **replayable
recipe JSON** in `Task_by_Task/<task>/prototypes/` for the agent loop to golf.
Tutorials and pre-found warm leads live in `doors_campaign.json`. Design and
roadmap: `DOORS_DESIGN.md`. The classic graph editor is one click away
("← Lab mode").

## Project Integration

The server auto-detects the K-NeuroGolf project root (the parent repo) or uses
`NEUROGOLF_PROJECT_ROOT` from `.env`. Integration points:

| What | Where | Access |
|---|---|---|
| Official scorer | `tools/_score_v2.py` (memory + params -> points) | read-only |
| Live best baseline | `The_400_Tasks/task_state.csv` | read-only |
| Current best models | `submissions/submission.zip` (all 401) | read-only |
| Exported artifacts | `The_400_Tasks/Task_by_Task/<task>/prototypes/` | **additive only** |

Artifacts are written as `taskNNN_gui_<UTC>.onnx` plus a `.json` sidecar with
the graph, validation report, and score. Existing files are never modified, so
concurrent agent sessions are not disturbed. Without a project root the app
still works standalone; exports land in `GUI/exports/`.

## Representations

The real competition encoder is one-hot: input `[1,10,30,30]` F32, output
decoded by `(out > 0.0)`. The GUI supports two graph modes (MODE panel):

- **Scalar (default)** — build on color grids `[1,1,30,30]`. Export wraps the
  graph in an automatic bridge: `ArgMax` decode on the way in, `Equal` against
  a 0..9 palette on the way out, masked by the input active region so padding
  cells stay all-zero. Correct for same-size tasks; the bridge costs roughly
  24k memory + 912 params, which the score panel reports honestly.
- **One-hot (raw)** — build directly on `[1,10,30,30]`. No bridge, no overhead;
  required for grid-resizing tasks and competitive golfing.

Exported models always have I/O tensors named exactly `input` / `output` (the
official verifier hard-codes both).

## Web GUI Workflow

1. Select a task; inspect train/test/extra examples.
2. Optionally load the current best model (`BEST TEMPLATE` -> Load Graph) straight
   from `submissions/submission.zip` for inspection.
3. Build a graph; `Run` previews the output grid on the displayed example.
4. `Validate` — competition-exact check over train + test + arc-gen, plus an
   official score and delta vs. the live best. Nothing is written.
5. `Export ONNX` — same checks; on success the artifact is saved into
   `Task_by_Task/<task>/prototypes/`.

## Headless Agent Workflow

```bash
# Validate + score without saving:
python scripts/agent_export.py --task task276 --color-remap '{"6":2}' --validate-only

# Full export into Task_by_Task/<task>/prototypes/:
python scripts/agent_export.py --task task010 --graph graph.json

# Options: --representation scalar|onehot, --bridge-mask input|none,
#          --sets train  (quick iteration; default train,test,arc-gen)
```

`graph.json` may contain either a full export payload or `nodes`/`edges`.
Validation always runs against the official task JSON on the server side;
client-supplied training pairs are only used for the scalar Run preview.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/compile` | compile only, report bytes + I/O summary |
| `POST /api/run` | run on one grid (scalar or one-hot models) |
| `POST /api/validate` | competition-exact validation + official score, no write |
| `POST /api/export` | validate + score + save artifact (additive) |
| `GET /api/project-status` | project root / scorer / baseline availability |
| `GET /api/task-baseline/{task}` | live best cost/points from task_state.csv |
| `GET /api/best-graph/{task}` | current best ONNX as editable GUI graph |
| `GET /api/best-onnx/{task}` / `GET /api/best-zip` | download current best |
| `GET /best/manifest.json` | manifest of the canonical submission zip |

## Supported Nodes

Core:

```text
Input Output Constant Cast Identity
Equal Greater Less GreaterOrEqual LessOrEqual Not And Or Xor
Add Sub Mul Div Mod Min Max Sum Relu Abs Neg Floor Clip Sign Sqrt
ReduceSum ReduceMax ReduceMin ArgMax Where
```

Spatial and coordinate:

```text
RowIndex ColIndex Slice Pad Concat Transpose Tile Resize Conv
```

All tensors are static. Scalar canvas is `[1,1,30,30]`; one-hot is `[1,10,30,30]`.

## Development Checks

```bash
python -m pytest -q
cd client && npm run build
```

## Scoring Reference

```text
points = max(1, 25 - ln(memory + params))    # natural log; incorrect = 0
```

See `Onnx_Advanced/core.md` in the parent project for the full validator
rejection rules — the export gate mirrors them (`competition_prechecks`).
