# Onnx_Advanced/transforms/ — reusable, tested transform helpers

The p43/p44 win-rate proposals kept hitting the same wall: the wins came from a handful of
*repeated* structural motifs, but each lived as a one-off prototype script, so correctness depended
on worker memory and every follow-on task re-derived the same graph surgery. This directory is the
fix: each motif becomes **one importable, output-preserving (or explicitly gated) function with a
fixture pair and a test**. A worker reuses the proven helper by name before writing a new builder.

## Convention (every helper must follow this)

1. **Signature.** `def <verb>(model: onnx.ModelProto, **opts) -> tuple[onnx.ModelProto, dict]` — mutate
   a copy or in place, and return `(model, report)` where `report` names what changed (bytes/params
   saved, tensors touched). Pure graph surgery — no file I/O in the function body.
2. **Output-preserving or explicitly gated.** Prefer transforms that are correct *by construction*
   (e.g. `share_initializer`: identical bytes ⇒ identical values). If a transform can change behavior
   (dtype narrowing, crop), it must say so in its docstring and the caller must gate it (ab / truth).
3. **Fixture pair.** `fixtures/<name>_live.onnx` + `fixtures/<name>_cand.onnx` (or a builder in the
   test that constructs them) — the real or synthetic before/after the transform reproduces.
4. **Test.** `test_<name>.py` asserts: (a) the static cost / params drop, (b) ORT parity vs the
   original on random input (skipped cleanly if onnxruntime is absent), and (c) at least one negative
   (a case the transform must NOT touch). Run: `.venv\Scripts\python.exe Onnx_Advanced\transforms\test_<name>.py`.
5. **Hard-record link.** When a transform is derived from a task, cite the task's `hard_records/` route
   (the narrative `.md` convention under `The_400_Tasks/Task_by_Task/taskNNN/hard_records/`).

## Helper registry

| Helper | Status | Motif / origin | Safety |
|---|---|---|---|
| `share_initializer.share_initializers` | **implemented + tested** | merge byte-identical initializers (task260 two shifted power tables; task323 mask reuse) | output-preserving by construction |
| `delete_dead_sentinel` | TODO (add on demand) | drop an unused residue/sentinel initializer/slot (task394) | preserving if truly unreferenced |
| `pad_axes_shave` | TODO | opset Pad axes byte-shave (task134, task259) | ab-gate |
| `dtype_narrow` | TODO | narrow an initializer/tensor dtype (F32→F16/U8) | ab/truth-gate (behavior can change) |
| `final_qlinear_decoder` | TODO | remove an F16 final-decoder (task201) | truth-gate |
| `shared_mask_zero_fold` | TODO | mask reuse + zero-fold (task323) | preserving |
| `flat_board_gather_fold` | TODO | delete gather/materialization (task308) | truth-gate |
| `bit_class_lut_remap` | TODO | LUT remap (task345) | truth-gate |
| `decode_tail_narrow` | TODO | shave an F16 decode tail (task273) | ab/truth-gate |

Add a TODO helper only when a real task needs it (don't speculatively build the whole table) — but
when you do, follow the convention above so the next worker inherits a tested tool, not a script.

Related existing tools (script form, not the library API): `tools/dedup_onnx_inits.py`,
`tools/const_to_init.py`, `tools/peephole_rewrite.py`, `tools/ng_graph_cost_diff.py` (names the
charged structure a transform should target).
