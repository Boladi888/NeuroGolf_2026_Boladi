# task384 — F16-output trick (drop final Cast)

| Field | Value |
|---|---|
| Task | task384 |
| Pattern | F16-output trick (final Cast(F16, F32) drop) — techniques.md §3 |
| Cost reduction | -36,900 (114,111 → 77,211) |

## The pattern this exemplifies

Some NeuroGolf ONNXes end with:

```
... → Concat → output_f16 [F16, [1,10,30,30]] = 18,000 bytes intermediate
      Cast(output_f16, F32) → output [F32, [1,10,30,30]] = excluded as output
```

The `output_f16` is a COUNTED F16 intermediate (18,000 bytes). The final F32 `output` is EXCLUDED from memory accounting (model outputs are skipped).

Drop the final Cast → Concat's output IS the model output → the F16 intermediate disappears (becomes the excluded output).

## Before

```
... compute chain → Concat(...) → output_f16 [F16, [1,10,30,30]] (18,000 bytes intermediate)
                    Cast(output_f16, to=FLOAT) → output [F32, [1,10,30,30]] (excluded)
```

## After

```
... compute chain → Concat(...) → output [F16, [1,10,30,30]] (excluded — renamed)
graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16
```

Concat's output IS the model output. F16 dtype.

## Surgery (3 lines)

```python
m = onnx.load_from_string(live_bytes)
g = m.graph
new_nodes = []
for n in g.node:
    if n.op_type == "Cast" and n.output and n.output[0] == "output":
        continue                                      # drop the final Cast
    if n.op_type == "Concat" and n.output and n.output[0] == "output_f16":
        n.output[0] = "output"                        # rename to be the model output
    new_nodes.append(n)
g.ClearField("node"); g.node.extend(new_nodes)
g.output[0].type.tensor_type.elem_type = 10           # FLOAT16 = TensorProto.FLOAT16
```

## Why it's safe

`run_network` does `(out > 0.0).astype(float)`. F16 representation of {0.0, 1.0} is exact; `> 0.0` produces the same {True, False} mask as F32. Then `.astype(float)` makes {0.0, 1.0} F64. Same as F32 path under the decoder.

Verified on task092, task117, task384 via `verify_against_live.py` 100% M/M.

## When to apply

Detect with:
```python
m = onnx.load(...)
last_node = m.graph.node[-1]
if last_node.op_type == "Cast":
    # check it casts from F16 to F32 and feeds the output
    if any(a.name == "to" and a.i == TensorProto.FLOAT for a in last_node.attribute):
        # previous node should be Concat with output named output_f16 or similar
        ...
```

The `final_cast_f16_to_f32_advisory` pattern (`tools/spread_patterns/pass_final_cast_f16_to_f32_advisory.py`) automates the detection.

## Generalizes to

`techniques.md` §3 (F16-output trick). Apply to any task whose graph ends with `Concat → F16_tensor → Cast(F32) → output`.

Stacks naturally with crop (see `task117_crop15_f16.md`).

## Critical: task384's wildcard history

A DIFFERENT task384 candidate was used earlier this session as a wildcard (FAIL_7/266 visible verify) and regressed -13.35 LB on Kaggle. **That was a different candidate** (`task384_ch4_cursor_v1`), not this `task384_f16drop` one. The current `f16drop` candidate verifies 100% M/M — fully safe.

This is why staged candidate identity matters: always confirm WHICH candidate file you're submitting.

## Reproducibility

The trick is fully mechanical. Detection and surgery are both automatable. The advisory pattern flags candidates; manual application or a future transform pattern would close the loop.

For new tasks: open the graph, check the last node. If `Cast → output`, check its source. If the source is a Concat with an F16 output, apply the surgery. Verify byte-exact.

## Cross-ref

- `task117_crop15_f16.md` — F16-output trick stacked with crop-to-active-grid.
- `task055_per_channel.md` — F16 placement when there are multiple Cast positions to choose.
- `techniques.md` §3 — the full technique description.
