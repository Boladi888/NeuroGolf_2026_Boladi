# NeuroGolf Lab Agent Notes

Use this app as an ARC/NeuroGolf solving workbench. It is integrated with the
parent K-NeuroGolf project and is local-only (no external uploads).

## Primary Workflow

1. Inspect task JSON from `data/competitions/neurogolf-2026/`. The
   `client/public/tasks/` copies are ignored/regenerable, and the FastAPI
   server falls back to project data when the local `data/` junction exists.
2. Form a transformation hypothesis.
3. Build a graph using the same node JSON shape the GUI emits.
4. Test with `POST /api/validate` (no write), ship with `POST /api/export`.
5. Validation is the **official competition gate**: one-hot `[1,10,30,30]`
   encoding, `> 0.0` threshold decode, train + test + arc-gen must all pass,
   plus the official validator prechecks (I/O named `input`/`output`, no banned
   ops, static dims, file-size limit).
6. Passing exports are scored with `tools/_score_v2.py` and saved to
   `The_400_Tasks/Task_by_Task/<task>/prototypes/taskNNN_gui_<UTC>.onnx`
   (+ `.json` sidecar). Additive only — existing files are never touched.

## Representations

- `"representation": "scalar"` (default): graph computes on color grids
  `[1,1,30,30]`; export wraps it in an ArgMax-decode / palette-encode bridge
  (~24k memory + ~912 params overhead, reported honestly by the score).
  Padding is masked by the input active region (`"bridgeMask": "input"`), so
  it is correct for same-size tasks only.
- `"representation": "onehot"`: graph computes directly on `[1,10,30,30]`.
  Required for grid-resizing tasks and for competitive costs.

## Cost Discipline

The response always includes `score` (memory/params/cost/points) and
`baseline` (live best from `task_state.csv`) with `deltaPoints`. A negative
delta means your graph is worse than the current best — do not promote it.
Check `The_400_Tasks/Task_by_Task/taskNNN.md` before spending effort.

## Headless Export

```bash
python scripts/agent_export.py --task task276 --color-remap '{"6":2}' --validate-only
python scripts/agent_export.py --task task010 --graph graph.json
python scripts/agent_export.py --task task010 --graph graph.json --sets train   # fast iteration
```

The helper posts to the same `/api/validate` / `/api/export` endpoints as the
GUI. Server-side validation always uses the official task JSON.

## Safety

- Do not bypass the compiler by writing ONNX directly except for diagnostics.
- The parent project is read-only except the additive prototypes/ writes above.
  Never modify `task_state.csv`, `submissions/`, or other agents' files.
- Do not print `.env` values; do not commit secrets, logs, build output,
  archives, or dependency folders.
- Run `python -m pytest -q` and `cd client && npm run build` before commits when
  the local environment and data junction are available.
