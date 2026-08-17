# task022 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task022.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `neurogolf-v2-task022-marker-neighborhood`
- **Graph name**: `task022_marker_neighborhood`
- **Opset**: 12
- **Nodes**: 361
- **Initializers**: 40

## Algorithm hints (from naming)

- anchor/marker-pattern (silent-fail-trap family)

## Op frequency

| Op | Count |
|---|---:|
| `Greater` | 81 |
| `Mul` | 72 |
| `ReduceSum` | 72 |
| `Where` | 72 |
| `Gather` | 27 |
| `Pad` | 9 |
| `Sum` | 9 |
| `Cast` | 9 |
| `Slice` | 8 |
| `Sub` | 1 |
| `Concat` | 1 |

## Initializer names (first 30)

```
idx11
pad_to_30
zero
valid3
idx1
idx2
idx3
idx4
idx5
idx6
idx7
idx8
idx9
cell_0_0
cell_0_1
cell_0_2
cell_1_0
cell_1_1
cell_1_2
cell_2_0
cell_2_1
cell_2_2
marker_shift_-1_-1_starts
marker_shift_-1_-1_ends
marker_shift_-1_-1_axes
marker_shift_-1_-1_pads
marker_shift_-1_0_starts
marker_shift_-1_0_pads
marker_shift_-1_1_ends
marker_shift_-1_1_pads
```

## First 12 nodes (algorithm entry point)

```
    0: Gather         {'axis': 1} (input, idx1) -> ch1
    1: Gather         {'axis': 2} (ch1, idx11) -> ch1_rows
    2: Gather         {'axis': 3} (ch1_rows, idx11) -> ch1_11
    3: Gather         {'axis': 1} (input, idx2) -> ch2
    4: Gather         {'axis': 2} (ch2, idx11) -> ch2_rows
    5: Gather         {'axis': 3} (ch2_rows, idx11) -> ch2_11
    6: Gather         {'axis': 1} (input, idx3) -> ch3
    7: Gather         {'axis': 2} (ch3, idx11) -> ch3_rows
    8: Gather         {'axis': 3} (ch3_rows, idx11) -> ch3_11
    9: Gather         {'axis': 1} (input, idx4) -> ch4
   10: Gather         {'axis': 2} (ch4, idx11) -> ch4_rows
   11: Gather         {'axis': 3} (ch4_rows, idx11) -> ch4_11
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 22
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
