# task369 — LIVE ALGORITHM

> **Hidden-bad task.** Reading this BEFORE writing a prototype is
> mandatory. The live ONNX scores well on Kaggle's hidden test set —
> any replacement must produce identical outputs on hidden inputs,
> which means matching the live's algorithm exactly. Visible + arc-gen
> verify is necessary but NOT sufficient (see Logs/lessons.md
> 2026-05-11 evening entry).

## Provenance

- **Source**: live `task369.onnx` in active submission zip
  (`submission.zip`)
- **Producer**: `(anon)`
- **Graph name**: `task369`
- **Opset**: 10
- **Nodes**: 24
- **Initializers**: 11

## Op frequency

| Op | Count |
|---|---:|
| `Mul` | 6 |
| `Sub` | 6 |
| `Relu` | 3 |
| `Slice` | 2 |
| `Conv` | 2 |
| `Min` | 2 |
| `Add` | 1 |
| `Concat` | 1 |
| `Pad` | 1 |

## Initializer names (first 30)

```
w_k4_ch0
one_f
two_f
zero_10
zeros_4_10
slice_chain_starts_1
slice_chain_ends_1
slice_chain_axes_1
slice_chain_steps_1
slice_chain_starts_2
slice_chain_ends_2
```

## First 12 nodes (algorithm entry point)

```
    0: Slice          (input, slice_chain_starts_1, slice_chain_ends_1, slice_chain) -> ch5
    1: Slice          (input, slice_chain_starts_2, slice_chain_ends_2, slice_chain) -> ch0
    2: Conv           {'pads': [1, 1, 1, 1]} (ch0, w_k4_ch0) -> adj0_raw
    3: Mul            (ch0, adj0_raw) -> adj0
    4: Sub            (adj0, one_f) -> adj0_m1
    5: Sub            (adj0, two_f) -> adj0_m2
    6: Relu           (adj0_m1) -> relu_adj0_m1
    7: Relu           (adj0_m2) -> relu_adj0_m2
    8: Sub            (relu_adj0_m1, relu_adj0_m2) -> is_adj0_ge2
    9: Relu           (adj0) -> relu_adj0
   10: Min            (relu_adj0, one_f) -> adj0_capped
   11: Sub            (one_f, adj0_capped) -> is_adj0_eq0
```

## How to validate a prototype against this

```
python tools/verify_against_live.py \
  --proto path/to/prototype.onnx \
  --tid 369
```

**The output should be `Prototype vs LIVE : N/N` exactly.** Even one
diverging pair on visible+arc-gen means the algorithm is different and
will likely be hidden-bad. And remember: matching on visible is
NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet
was -13.39 LB on hidden). For the strongest signal, the prototype
should follow the algorithm DERIVED FROM THIS FILE, not from visible
examples.
