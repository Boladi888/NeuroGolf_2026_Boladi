# tools/agent_helpers/ — diagnostic scripts for ONNX work

Ready-to-use scripts for common operations. Run any of these directly with the venv Python; they print results to stdout.

## Scripts

| Script | What it does | Example |
|---|---|---|
| `extract_task.py` | Extract `taskNNN.onnx` from `submissions/submission.zip` to a temp file | `python tools/agent_helpers/extract_task.py 110` |
| `inspect_graph.py` | Print full graph: nodes, initializers, value_info, flags | `python tools/agent_helpers/inspect_graph.py path/to/model.onnx` |
| `largest_intermediate.py` | Print top-N intermediates by memory bytes | `python tools/agent_helpers/largest_intermediate.py model.onnx [N=10]` |
| `diff_graphs.py` | Structural diff between two ONNXes (nodes added/removed/changed, initializers) | `python tools/agent_helpers/diff_graphs.py before.onnx after.onnx` |
| `dump_intermediates.py` | Run model on input grid, dump every intermediate tensor (debug verify divergence) | `python tools/agent_helpers/dump_intermediates.py model.onnx --tid 110 --pair-idx 0` |
| `find_op.py` | Grep nodes by op_type or name regex | `python tools/agent_helpers/find_op.py model.onnx ReduceSum` |
| `score_compare.py` | Side-by-side cost comparison of two ONNXes | `python tools/agent_helpers/score_compare.py a.onnx b.onnx` |
| `regenerate_graph_diagnostics.py` | Rebuild the `<!-- AUTO-GRAPH-DIAG -->` sections in every `Task_by_Task/taskNNN.md` | `python tools/agent_helpers/regenerate_graph_diagnostics.py` (or specific tasks: `... task110 task117`) |

## Read-first

Every Task_by_Task/taskNNN.md has an auto-generated `## Graph Diagnostic` section at the bottom (look for `<!-- BEGIN AUTO-GRAPH-DIAG -->`). It contains:

- Cost / memory / params / opset
- Quick-look flags (has_Conv, has_MatMul, first_node, last_node, ends_with_Cast_to_F32, first_node_is_Slice, etc.)
- Op-type histogram
- Top initializers by params
- Top 5 largest intermediates by bytes
- Suggested techniques (auto-derived from flags)
- Full node sequence (collapsed by default)

Use that diagnostic FIRST. Most "inspect this task" questions are answered there. Only run these scripts when you need something the diagnostic doesn't cover.

## Regeneration

When the canonical submission.zip changes, regenerate ALL diagnostics:

```
.venv/Scripts/python tools/agent_helpers/regenerate_graph_diagnostics.py
```

Takes ~5-10 minutes for 401 tasks. The script replaces only the marker-delimited section; manual notes in the `.md` are preserved.

## All scripts use the same venv

```
.venv/Scripts/python <script_path>
```

Set `PYTHONIOENCODING=utf-8` if you see encoding errors on Windows.
