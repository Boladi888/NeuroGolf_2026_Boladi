# Deep-read: `neurogolf-logic-driven-ensembe.ipynb` ("Safe Blend")

Source: `Research/primary_sources/neurogolf-logic-driven-ensembe.ipynb` (4 cells, ~10 KB total). A conservative blender notebook. Self-described principle: **"do not touch a working model; only repair what is necessary for a valid submission."**

## Pipeline (single code cell, ~9 KB)

1. Load a "top submission" zip (`konbu17/neurogolf-2026-4959-61-4-28-update-version`) byte-for-byte. Never modify.
2. Scan all other notebook-input zips for candidates.
3. For each candidate ONNX: try byte-for-byte; if `input`/`output` tensor names diverge, run the I/O-rename repair (below). Reject if neither path produces a runnable model.
4. For each task 0..399: pick candidate IF strictly cheaper than top AND passes a random-input runtime smoke test; ELSE keep top byte-for-byte; ELSE write a 1×1-identity-Conv fallback (cost 10⁸).
5. Emit `submission.zip` + `submission.csv`.

## The reusable piece — I/O-rename repair

The only non-trivial logic is `safe_load_model` (~50 lines). When a candidate ONNX's input or output tensor name isn't literally `"input"` / `"output"` (validator-required), it:

- Iterates `graph.input[0]` and `graph.output[0]`, builds a `rename_map` from old → new.
- Renames `graph.input[0].name` and `graph.output[0].name`.
- Walks every `graph.node`, every `node.input[i]` and `node.output[i]`, replaces names from the map.
- Walks `graph.value_info` and `graph.initializer`, renames where applicable.
- Sets `model.ir_version = 8` (bumps to the current floor; harmless to any opset ≥ 10 graph).
- Re-validates by constructing a fresh `ort.InferenceSession` from the new bytes; returns `(ok, fixed_bytes, cost_estimate)`.

This is the entire defensive repair surface. It's specifically a fix for graphs whose internal naming deviates from the validator's hardcoded `input`/`output` strings (verified by `nu.run_network` per `Research/A4_grader_replication.md`).

## Applicability to us

- **Right now: zero.** `Research/primary_sources/00_INDEX.md` records that all 400 of our `submissions/taskNNN.onnx` use the correct `input`/`output` names (verified 2026-05-02). We don't ingest external graphs in our pipeline.
- **If we ever start blending** (e.g., using ashok205's 148 net-new ONNXs as fallback for tasks we don't have): the rename logic is exactly the reusable bit. Worth pulling into `tools/onnx_repair.py` at that point.
- **No new pattern ideas** here. The notebook contributes blender plumbing, not algorithmic content.

## Mini-recipe for future `tools/onnx_repair.py` (deferred until needed)

```python
def repair_io_naming(raw_bytes):
    """Try byte-for-byte; if input/output names diverge from validator standard,
    rename in-place and re-emit. Returns (ok, fixed_bytes_or_None, reason).
    """
    # 1. Try as-is
    try:
        sess = ort.InferenceSession(raw_bytes, providers=['CPUExecutionProvider'])
        if sess.get_inputs()[0].name == 'input' and sess.get_outputs()[0].name == 'output':
            return True, raw_bytes, 'as-is'
    except Exception:
        pass
    # 2. Try rename pass
    model = onnx.load_model_from_string(raw_bytes)
    g = model.graph
    rename_map = {}
    if g.input[0].name != 'input':  rename_map[g.input[0].name] = 'input'
    if g.output[0].name != 'output': rename_map[g.output[0].name] = 'output'
    if not rename_map:
        return False, None, 'naming-already-correct-but-fails-load'
    g.input[0].name = 'input'
    g.output[0].name = 'output'
    for node in g.node:
        for i, x in enumerate(node.input):  node.input[i] = rename_map.get(x, x)
        for i, x in enumerate(node.output): node.output[i] = rename_map.get(x, x)
    for vi in g.value_info:
        if vi.name in rename_map: vi.name = rename_map[vi.name]
    for init in g.initializer:
        if init.name in rename_map: init.name = rename_map[init.name]
    model.ir_version = 8
    fixed = model.SerializeToString()
    try:
        sess = ort.InferenceSession(fixed, providers=['CPUExecutionProvider'])
        if sess.get_inputs()[0].name == 'input' and sess.get_outputs()[0].name == 'output':
            return True, fixed, 'renamed'
    except Exception as e:
        return False, None, f'post-rename-fail: {e}'
    return False, None, 'rename-incomplete'
```

Cross-check on our 13 SOLVED_TASKS would be a no-op (everything passes byte-for-byte). The function is dormant insurance for the day we want to ingest external graphs.

---

*Filed 2026-05-02 by Beta lane in the followup batch. Source: `Research/primary_sources/neurogolf-logic-driven-ensembe.ipynb`. No `[Alpha]`-tagged TODOs surfaced — the notebook contributes blender mechanics, not pattern ideas.*
