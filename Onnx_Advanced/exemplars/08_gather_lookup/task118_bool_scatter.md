# task118 — bool ScatterND

| Field | Value |
|---|---|
| Task | task118 |
| Cost reduction | -38,341 |
| Pattern | ScatterND with BOOL input (opset 11+) |

## Pattern

task118 uses ScatterND to fill regions: given index positions and a fill value, place the fill value into a tensor at those positions.

The standard implementation uses F32 inputs. The rewrite uses BOOL — same semantics, 4× smaller intermediate.

```
# Before (F32)
result = ScatterND(
    data=zeros [1, 10, H, W] F32 = 36,000 bytes,
    indices=[positions] INT64,
    updates=[1.0] F32 = 4 bytes per update
)

# After (BOOL)
result = ScatterND(
    data=falses [1, 10, H, W] BOOL = 9,000 bytes,
    indices=[positions] INT64,
    updates=[True] BOOL = 1 byte per update
)
```

The data tensor (the output canvas being scattered into) is 4× smaller in BOOL.

## Why it works

ScatterND is dtype-preserving. The `data` and `updates` must share dtype, but BOOL is supported under opset 11+.

For NeuroGolf where output values are 0/1, BOOL representation is exact. The output is BOOL [1, 10, H, W] which under `(out > 0.0)` decode produces the same {0, 1} mask as the F32 version.

## Generalizes to

Apply ScatterND with BOOL inputs when:
- Output is a 0/1 mask.
- Model is opset 11+.
- Downstream consumers (if any) accept BOOL.

Even better: if your task uses ScatterND to BUILD an output mask AND that output mask is the final answer (graph output), the F16-output trick generalizes — declare graph output as BOOL.

## Cross-ref

- `01_whole_graph_rewrites/task376_input_to_bool.md` — whole-graph BOOL propagation. ScatterND with BOOL inputs is part of this family.
- `02_bool_first/task273_drop_pad_cast_bool.md` — Pad with BOOL input (companion op for ScatterND).
- `ort_reality.md` — ScatterND requires opset 11+.

## Reproducibility

Inspect your graph for ScatterND ops. If the data tensor is F32 0/1 and downstream is bool-tolerant, switch the data and updates to BOOL. Verify byte-exact.
