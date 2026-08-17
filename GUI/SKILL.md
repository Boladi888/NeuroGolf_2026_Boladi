---
name: neurogolf-lab
description: Use NeuroGolf Lab to solve ARC/NeuroGolf tasks through the visual graph workflow or headless agent export. Covers task inspection, graph JSON, competition-exact validation, official scoring, and local artifact export into Task_by_Task prototypes.
---

# NeuroGolf Lab

Use this skill when working in the NeuroGolf Lab app (`GUI/`) to build,
validate, score, or export ARC/NeuroGolf ONNX candidates.

## Rules

- Use the GUI workflow or `scripts/agent_export.py`; do not hand-write ONNX
  except for diagnostics.
- `/api/export` is the gate: compile -> official validation (one-hot encoding,
  `> 0.0` decode, train + test + arc-gen) -> official score -> local save into
  `The_400_Tasks/Task_by_Task/<task>/prototypes/`. No external uploads.
- A candidate is not successful unless `validation.ready` is true; a candidate
  is not *useful* unless `deltaPoints` is positive vs. the live best.
- The parent project is read-only except those additive prototype writes.
- Do not print or commit `.env`, logs, archives, `client/dist/`, `node_modules/`.

## Workflow

1. Start the backend (`start.ps1` / `start.sh`, serves http://127.0.0.1:8082).
2. Inspect the task; check the live baseline (`GET /api/task-baseline/<task>`).
3. Optionally load the current best graph (`GET /api/best-graph/<task>`, pulled
   from `submissions/submission.zip`) for inspection.
4. Propose the smallest graph recipe; iterate with `POST /api/validate`
   (`"exampleSets": ["train"]` for speed, full sets before export).
5. Export with `POST /api/export`; report nodes, edges, validation counts,
   score, deltaPoints, and the artifact path.

## Representations

- `scalar` (default): color grids `[1,1,30,30]`, auto-bridged to the one-hot
  competition encoding at export. Same-size tasks only; bridge overhead is
  ~24k memory + ~912 params.
- `onehot`: raw `[1,10,30,30]` competition encoding. Needed for resizing tasks
  and competitive golfing.

## Useful Node Families

Pixel logic:

```text
Constant Equal Greater Less GreaterOrEqual LessOrEqual Not And Or Xor
Add Sub Mul Div Mod Min Max Sum Where Cast Clip Sign Abs Neg Floor Sqrt Relu
```

Global/axis logic:

```text
ReduceSum ReduceMax ReduceMin ArgMax RowIndex ColIndex
```

Spatial logic:

```text
Slice Pad Concat Transpose Tile Resize Conv
```

## Scoring

```text
points = max(1, 25 - ln(memory + params))   # incorrect networks score 0
```

The export response includes the official score and the `task_state.csv`
baseline. Promotion decisions beyond prototypes/ (candidate zips, Kaggle
submission) belong to the main project workflow, not this app.
