# cellular_automata.md — CA quick-reference backing store

> Actionable content already extracted to `techniques.md`: unroll cost law + corpus-measured iteration floors → §37; flood kits → #47 (row-packed u32), #88 (u8 row-byte flood), #126 (iterative MaxPool flood). This file keeps only the kernel quick-ref, decision table, one priced worked example, and morphology one-liners.

## Neighbor-counting kernel quick-ref

Dtype notes (ORT 1.24 CPU, opset ≤ 18): `Conv` is float-only (`b_f = 4`); ai.onnx `Where` rejects bool X/Y on opset 9–18 — mux bools as `Or(And(c, a), And(Not(c), b))`; `ReduceSum`/`ReduceMax` reject bool until opset 20. Neighbor counts stay exact in float32 (0/1 masks, sums ≤ 9). Score conventions in every row: `param_elements` = unique initializer elements (a kernel shared by many nodes counts once); `intermediate_memory` = one materialized output per node; shapes `[1,1,H,W]`; `b_b ≈ 1`, `b_i64 = 8`.

Winning skeleton: **bool/int mask → `Cast` to float once → 3×3 `Conv` with explicit `pads=[1,1,1,1]`** (zero padding free on the Conv itself — no `Pad` node). One skeleton covers 4-neighbor, 8-neighbor, and 9-cell sums by swapping kernel values. Multi-channel: grouped Conv needs a `[C,1,3,3]` weight that physically duplicates the kernel (`9C` params) — prefer separate Convs reusing **one** shared kernel initializer.

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| 4-connectivity neighbor sum | `Cast`, `Conv` | 6 | `tensor(bool)` preferred; `tensor(float)` if already float | `tensor(float)` | 9 | `2HW·b_f` on bool input; `HW·b_f` if input already float | Kernel = `[[0,1,0],[1,0,1],[0,1,0]]`; center excluded |
| 8-connectivity neighbor sum | `Cast`, `Conv` | 6 | `tensor(bool)` preferred; `tensor(float)` if already float | `tensor(float)` | 9 | `2HW·b_f` on bool input; `HW·b_f` if input already float | Kernel = `[[1,1,1],[1,0,1],[1,1,1]]`; center excluded |
| 9-cell sum including self | `Cast`, `Conv` | 6 | `tensor(bool)` preferred; `tensor(float)` if already float | `tensor(float)` | 9 | `2HW·b_f` on bool input; `HW·b_f` if input already float | Kernel = all ones |
| Same-color-neighbor count for color `C` | `Equal`, `Cast`, `Conv` | 7 if `x` is int32/int64; 11 if `x` is uint8/float | `tensor(int32)` or `tensor(int64)` palette board | `tensor(float)` | 10 | `HW·b_b + 2HW·b_f` | One scalar color constant plus one shared 3×3 kernel |

## Worked example: Conway's Life one step (priced)

B3/S23 in pure boolean ops — `new_alive = (neighbors == 3) | ((neighbors == 2) & alive)` — so no `Where` needed (good: ai.onnx `Where` excludes bool X/Y on opset 9–18).

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| Conway Life one step | `Cast`, `Conv`, `Equal`, `Equal`, `And`, `Or` | 11 | `tensor(bool)` | `tensor(bool)` | 11 | `2HW·b_f + 4HW·b_b` | Params are one shared 8-neighbor kernel plus scalar constants `2` and `3` |

For `[1,1,30,30]`: `1800·b_f + 3600·b_b + 11`; with `b_f=4`, `b_b≈1` that is `7200 + 3600 + 11 = 10,811` — the intermediate-memory term alone is **10,800 bytes**. Exact compares against 2 and 3 are safe on float32 because neighbor sums are tiny integers.

```python
from onnx import helper, TensorProto

k8 = helper.make_node(
    "Constant", [], ["k8"],
    value=helper.make_tensor(
        "k8_t", TensorProto.FLOAT, [1, 1, 3, 3],
        [1, 1, 1,
         1, 0, 1,
         1, 1, 1],
    ),
)
two = helper.make_node(
    "Constant", [], ["two"],
    value=helper.make_tensor("two_t", TensorProto.FLOAT, [], [2.0]),
)
three = helper.make_node(
    "Constant", [], ["three"],
    value=helper.make_tensor("three_t", TensorProto.FLOAT, [], [3.0]),
)

alive_f = helper.make_node("Cast", ["alive"], ["alive_f"], to=TensorProto.FLOAT)
nbrs = helper.make_node("Conv", ["alive_f", "k8"], ["nbrs"], pads=[1, 1, 1, 1], strides=[1, 1])
eq3 = helper.make_node("Equal", ["nbrs", "three"], ["eq3"])
eq2 = helper.make_node("Equal", ["nbrs", "two"], ["eq2"])
stay = helper.make_node("And", ["eq2", "alive"], ["stay"])
new_alive = helper.make_node("Or", ["eq3", "stay"], ["new_alive"])
```

## Majority vote / color propagation (cost table)

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| Mode among 8 neighbors via per-color counters + `ArgMax` | `Equal × K`, `Cast × K`, `Conv × K`, `Concat`, `ArgMax` | 12 | `tensor(int32)` or `tensor(int64)` palette board | `tensor(int64)` | `K + 9` with one shared kernel; for `K=10`, **19** | `KHW·b_b + 3KHW·b_f + HW·b_i64` | For `K=10` and float32 path: `138HW` bytes; explicit tie-break to lowest color via `select_last_index=0` |

`K=10`, `[1,1,30,30]`: **124,200** intermediate bytes + **19** params (all ten Conv branches reuse one 8-neighbor kernel). Grouped-Conv variant: same activation cost but **100** params (`9K + K`) — the grouped weight stores ten kernel copies. Tie-break to smallest color index needs opset 12 `ArgMax(select_last_index=0)`.

## Multi-step unrolling

Unroll cost law + corpus-measured iteration floors: techniques.md §37.

## Boundary handling one-liners

- Outside = dead/zero: `pads=[1,1,1,1]` on the `Conv` itself — free, no `Pad` node, no extra bytes.
- Reflect: `Pad(mode='reflect') → Cast(if needed) → Conv(pads=0)`; extra score vs zero-pad bool path = `(H+2)(W+2)·(b_b + b_f) - HW·b_f` ≈ **+1,520 bytes**/step at 30×30. Reflection reuses interior edge values — changes corner/edge counts, not an infinite-board approximation.
- Torus: no `Pad(mode='wrap')` in opset ≤ 18 (only `constant`/`reflect`/`edge`) — wraparound needs explicit Slice+Concat.

## Erosion / dilation one-liners

Binary morphology = CA neighborhood logic: 3×3 dilation = neighborhood OR = `MaxPool`; 3×3 erosion = neighborhood AND = `Neg → MaxPool → Neg` (ONNX has no `MinPool`). int8 is the cheapest CPU path: `MaxPool` takes int8/uint8 from opset 12, `Neg` needs a signed dtype.

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| Dilation via `MaxPool` | `MaxPool` or `Cast`, `MaxPool` if source is bool | 12 for int8 path | `tensor(int8)` preferred for score; `tensor(bool)` requires `Cast` | same as input path | 0 | `HW·b_i8` on int8 input; `2HW·b_i8` if starting from bool and casting | Use `kernel_shape=[3,3]`, `pads=[1,1,1,1]`, `strides=[1,1]` |
| Erosion via `-MaxPool(-x)` | `Neg`, `MaxPool`, `Neg` or `Cast`, `Neg`, `MaxPool`, `Neg` | 12 for int8 path | `tensor(int8)` preferred for score; `tensor(bool)` requires `Cast` | same as input path | 0 | `3HW·b_i8` on int8 input; `4HW·b_i8` if starting from bool and casting | ONNX has no `MinPool`; this is the standard min-via-negation trick |

Recipes: `dilate3 = MaxPool(x_i8, kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1])`; `erode3 = Neg(MaxPool(Neg(x_i8), same attrs))`.

## Decision table

Score model: `intermediate_memory_bytes + param_elements`, node count ignored, shared initializers counted once.

| Rule type | Recommended pattern | Why it wins under this score | Main caveat |
|---|---|---|---|
| Binary threshold rule over 4-neighbors | `Cast → Conv` with 4-neighbor kernel | Minimal legal ORT CPU path; only one 3×3 kernel initializer | Float Conv path on CPU; bool must be cast first |
| Binary threshold rule over 8-neighbors | `Cast → Conv` with 8-neighbor kernel | Same as above, and reusable for Life-like rules | Boundary semantics depend on zero vs reflect |
| Life-like B/S rule | Shared 8-neighbor kernel + boolean comparisons + `And`/`Or` | Avoids `Where`; tiny shared params; exact rule mapping | Use bool state tensor to keep logic cheap |
| Same-color contagion / affinity | `Equal(x,c) → Cast → Conv` per color of interest | Cheapest exact same-color count on a palette board | One branch per color |
| Majority / mode propagation | `Equal×K → Cast×K → Conv×K → Concat → ArgMax`, with **one shared kernel** | Separate Conv branches beat grouped Conv because weight sharing keeps params at `K+9`, not `9K+K` | Tie policy must be explicit; use opset 12 `ArgMax` if you care |
| Binary dilation | `MaxPool` on int8 mask | Zero params and cheaper than float Conv | Cast once if source is bool |
| Binary erosion | `Neg → MaxPool → Neg` on int8 mask | Cheapest exact min-neighborhood path without unsupported ops | Requires signed dtype for `Neg` |

---

Evicted long-form (scoring-assumption prose, full unrolling analysis, boundary/morphology prose, extra code blocks, checklist): `_archive/cellular_automata_evicted_2026-06-10.md`
