# Prompt template — taskNNN drive-to-floor (≤2 KB shell, Phase-3 2026-06-05)

**Protocol**: read `Onnx_Advanced/WORKER_PROTOCOL.md` FIRST; it owns gates, staging, no-submit, and library routing.

**Library DB**: before large docs run `python Onnx_Advanced/library_db/lib_query.py status` + `python Onnx_Advanced/library_db/lib_query.py context NNN`; use `search`/`op` for surprises; stale => rebuild or Markdown.

```
# taskNNN -> drive to floor (<one-line angle>)
**Target: <X or proven floor>** — team-best ≈ <C> (re-measure: extract_task.py NNN + _score_v2; baseline against the TEAM-BEST graph, not the home cost).

## Assets (Path B — read in THIS order; existing graph LAST)
GUI/previews/taskNNN_preview.png · GUI/client/public/tasks/taskNNN.json · external/ARC-GEN/tasks/task_<uuid>.py (confirm visible pairs) · Task_by_Task/taskNNN.md · Onnx_Advanced/INDEX.md · existing/staged ONNX (only AFTER the scratch plan).

## Visual-first + scratch plan (Path B — before any build code)
Visual-first: rule / output-size relation / object-color roles / ≥2 ARC-GEN falsifications. Then 2-3 from-scratch formulations with `memory+params` estimates (dominant intermediate shape+dtype, initializers, final-output family); pick the cheapest credible one and name the charged tensors it avoids. Anti-anchor: do NOT inherit the live graph's representation.

## Direction
<decoded rule or decode-first instruction; task-specific §-pointers (e.g. techniques #44/#47, traps §CLASS-CHECK); known dead-ends from taskNNN.md>

## Recon
extract_task.py NNN · largest_intermediate.py --top 20 · officials data/competitions/neurogolf-2026/taskNNN.json · history taskNNN.md (READ FIRST) · corpus.build(NNN, want=200000)

## Protocol
Gates/staging/deliverables per WORKER_PROTOCOL.md. DO NOT submit — captain ships.
```

Authoring rules: `_Before writing prompt.md` + batch registration in `current_strategy.md`. Old full template: `_archive_templates/`.
