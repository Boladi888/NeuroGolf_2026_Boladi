# cellular_automata.md

## Neighbor-counting kernels

Assumption used throughout: `param_elements` counts **unique initializer elements** in the graph, so one kernel tensor reused by many nodes is counted once; `intermediate_memory_bytes` counts one materialized output per node in the unrolled graph. That is a graph-score assumption, not a runtime-peak-memory claim. ONNX Runtime can reduce runtime peak via memory-pattern and memory-reuse optimizations, and operator/type support is execution-provider-specific; below I use the conservative ORT 1.24 **CPUExecutionProvider** view, where `Conv` is float-only, `Where` in the ai.onnx domain does not accept `tensor(bool)` for X/Y on opset 9–18, and `ReduceSum`/`ReduceMax` do not accept bool until opset 20. ONNX `And`/`Or`/`Not` are boolean logical ops, so the bool caveat in the card is the right default. Inference note: the “unique initializer” counting rule follows from ONNX graphs exposing graph-level initializers consumed by nodes.

Let `b_f = bytes_per_element(float)` on the Conv path, `b_b = bytes_per_element(bool)`, and all shapes be `[1,1,H,W]` unless noted. For ORT 1.24 CPU, the practical Conv path is `tensor(float)`, so in concrete examples `b_f = 4`. All memory formulas below are inferred from one full output tensor per node.

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| 4-connectivity neighbor sum | `Cast`, `Conv` | 6 | `tensor(bool)` preferred; `tensor(float)` if already float | `tensor(float)` | 9 | `2HW·b_f` on bool input; `HW·b_f` if input already float | Kernel = `[[0,1,0],[1,0,1],[0,1,0]]`; center excluded |
| 8-connectivity neighbor sum | `Cast`, `Conv` | 6 | `tensor(bool)` preferred; `tensor(float)` if already float | `tensor(float)` | 9 | `2HW·b_f` on bool input; `HW·b_f` if input already float | Kernel = `[[1,1,1],[1,0,1],[1,1,1]]`; center excluded |
| 9-cell sum including self | `Cast`, `Conv` | 6 | `tensor(bool)` preferred; `tensor(float)` if already float | `tensor(float)` | 9 | `2HW·b_f` on bool input; `HW·b_f` if input already float | Kernel = all ones |
| Same-color-neighbor count for color `C` | `Equal`, `Cast`, `Conv` | 7 if `x` is int32/int64; 11 if `x` is uint8/float | `tensor(int32)` or `tensor(int64)` palette board | `tensor(float)` | 10 | `HW·b_b + 2HW·b_f` | One scalar color constant plus one shared 3×3 kernel |

These rows come directly from ONNX `Conv`, `Equal`, and `Cast` semantics plus ORT 1.24 CPU kernel/type support. `Conv` uses weights shaped `[M, C/group, kH, kW]`, so if you ever apply the same neighbor-count kernel independently to `C` state channels, the grouped extension is straightforward: multiply the 3×3 kernel elements and the float activations by `C`. The output counts remain exact on float32 for these kernels because the mask values are 0/1 and the sums are only 0–9; that exactness claim is an inference from the arithmetic, not a published ONNX guarantee.

Winning pattern for generic neighborhood counting: **bool/int mask → Cast to float once → 3×3 Conv with explicit `pads=[1,1,1,1]`**. It is the fewest-op legal path on ORT CPU under opset ≤ 18, and the same skeleton covers 4-neighbor, 8-neighbor, and 9-cell sums by swapping the kernel values.

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
x_f = helper.make_node("Cast", ["x_bool"], ["x_f"], to=TensorProto.FLOAT)
n8 = helper.make_node(
    "Conv", ["x_f", "k8"], ["n8"],
    pads=[1, 1, 1, 1], strides=[1, 1],
)
```

## Conway's Life one step

Conway’s Game of Life on the 8-neighbor Moore neighborhood is the standard B3/S23 rule: a dead cell is born with exactly 3 live neighbors, and a live cell survives with 2 or 3 live neighbors. In boolean algebra, the one-step update is exactly `new_alive = (neighbors == 3) | ((neighbors == 2) & alive)`, so no `Where` is needed for the canonical rule. That is ideal on ORT CPU because ai.onnx `Where` still excludes bool X/Y on opset 9–18.

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| Conway Life one step | `Cast`, `Conv`, `Equal`, `Equal`, `And`, `Or` | 11 | `tensor(bool)` | `tensor(bool)` | 11 | `2HW·b_f + 4HW·b_b` | Params are one shared 8-neighbor kernel plus scalar constants `2` and `3` |

For `[1,1,30,30]`, the symbolic score contribution is `1800·b_f + 3600·b_b + 11`. With the practical CPU path `b_f=4` and the common scorer convention `b_b≈1`, that is `7200 + 3600 + 11 = 10,811`; the intermediate-memory term alone is **10,800 bytes**. If your scorer assigns a different byte width to bool tensors, only the `3600·b_b` term changes. The “exact compare against 2 and 3” step is safe on float32 here because neighbor sums are tiny integers; that exactness is an inference.

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

For other binary CA rules that genuinely need a boolean mux, use `Or(And(c, a), And(Not(c), b))` instead of `Where(c, a, b)` on bool tensors. That rewrite matches the card’s caveat and stays within ONNX boolean op typing plus ORT CPU’s ai.onnx `Where` restrictions.

## Color propagation and majority vote

Assumption for this section: ties are resolved by the **smallest color index**. The cleanest way to make that explicit is `ArgMax(select_last_index=0)` at opset 12. If you leave tie behavior implicit at opset 11, you lose that explicit contract.

The requested baseline is: for each color `c in {0,…,K-1}`, compute `Equal(x, c) → Cast → Conv(8-neighbor kernel)`; concatenate the `K` counters on the channel axis; then `ArgMax(axis=1, keepdims=1)`. Under **this score**, that baseline is better than the tempting grouped-Conv rewrite, because the ten separate Conv nodes can all reuse **one** 9-element kernel initializer, while grouped Conv needs a `[K,1,3,3]` weight tensor containing `9K` kernel elements. Extra nodes are free in this score; duplicated weight elements are not. That winner/loser comparison is an inference from ONNX graph initializers plus `Conv` weight/group rules.

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| Mode among 8 neighbors via per-color counters + `ArgMax` | `Equal × K`, `Cast × K`, `Conv × K`, `Concat`, `ArgMax` | 12 | `tensor(int32)` or `tensor(int64)` palette board | `tensor(int64)` | `K + 9` with one shared kernel; for `K=10`, **19** | `KHW·b_b + 3KHW·b_f + HW·b_i64` | For `K=10` and float32 path: `138HW` bytes; explicit tie-break to lowest color via `select_last_index=0` |

For the common `K=10`, `[1,1,30,30]` case, the step cost is `124,200` intermediate bytes under the usual `b_f=4`, `b_b≈1`, `b_i64=8` convention, plus **19** unique parameter elements if all ten Conv nodes reuse one shared 8-neighbor kernel. A grouped-Conv variant has essentially the same activation cost but **100** parameter elements (`9K + K`) because the grouped weight tensor must physically store ten copies of the same 3×3 kernel. A streaming “running best color” alternative can reduce runtime peak in some executors, but under the present graph score it is usually worse because it replaces a free-ish `Concat+ArgMax` finish with repeated comparisons and updates, and it still cannot use `Where(bool,bool,bool)` on ORT CPU.

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

# One branch; repeat for colors 0..9 with color constants color_0 ... color_9.
eq0 = helper.make_node("Equal", ["x", "color_0"], ["m0"])
m0f = helper.make_node("Cast", ["m0"], ["m0f"], to=TensorProto.FLOAT)
c0 = helper.make_node("Conv", ["m0f", "k8"], ["c0"], pads=[1, 1, 1, 1], strides=[1, 1])

# After building c0 ... c9:
stack = helper.make_node("Concat", ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"], ["counts"], axis=1)
new_color = helper.make_node("ArgMax", ["counts"], ["new_color"], axis=1, keepdims=1, select_last_index=0)
```

## Multi-step unrolling cost analysis

I interpret “compression” here as **deduplicating repeated kernels/scalars by reusing graph initializers**. Under that interpretation, the key scaling law is simple:

- Shared-initializer build: `Score(N) = P_shared + N·A_step`
- Naive duplicated-constants build: `Score(N) = N·(P_step + A_step)`

where `A_step` is the per-step intermediate-memory term and `P_shared` is the unique initializer count for the whole repeated rule. This follows from ONNX graphs having graph-level initializers and from ORT executing the same typed nodes repeatedly once the graph is unrolled. Runtime peak can still be lower than the graph score because ORT supports memory-pattern and memory-reuse optimizations.

For the two important rules above:

- **Life**: `A_life = 2HW·b_f + 4HW·b_b`, `P_life = 11`
- **Mode propagation with palette size `K`**: `A_mode(K) = KHW·b_b + 3KHW·b_f + HW·b_i64`, `P_mode(K) = K + 9` if the 3×3 kernel is shared across branches

At `H=W=30`, with `b_f=4`, `b_b≈1`, `b_i64=8`:
- `A_life = 10,800`
- `A_mode(10) = 124,200`

| Iterations | Per-iter cost (bytes + params) | Cumulative cost | When prohibitive |
|---:|---|---|---|
| 1 | `A_step + P_shared` | `A_step + P_shared` | Baseline |
| 2 | `A_step` after the first step if initializers are shared; otherwise `A_step + P_step` | `2A_step + P_shared` | Extra unrolling already dominates parameter-dedup savings whenever `A_step >> P_step` |
| 4 | same as above | `4A_step + P_shared` | For `30×30`, majority-vote is already ~0.47 MiB; Life is ~42 KiB |
| 8 | same as above | `8A_step + P_shared` | For `30×30`, majority-vote is ~0.95 MiB; Life is ~84 KiB |
| 16 | same as above | `16A_step + P_shared` | For `30×30`, majority-vote is ~1.90 MiB; Life is ~169 KiB |
| 32 | same as above | `32A_step + P_shared` | For `30×30`, majority-vote is ~3.79 MiB; Life is ~338 KiB |
| 64 | same as above | `64A_step + P_shared` | For `30×30`, majority-vote is ~7.58 MiB; Life is ~676 KiB |

Two practical thresholds fall out immediately. First, **“unrolling cost ≥ savings from compression” happens almost immediately**. For Life at `30×30`, the second extra step adds `10,800` bytes but only risks duplicating `11` parameter elements if you were naive; for 10-color majority vote it adds `124,200` bytes against only `19` shared parameters. In other words, once `N ≥ 2`, activation growth dwarfs constant-dedup savings on any nontrivial board. Second, if you use a simple graph-score budget heuristic `M_budget`, then `N_max ≈ floor((M_budget - P_shared)/A_step)`. With a `1 MiB` budget heuristic, `N_max ≈ 97` for Life on `30×30`, but only `≈ 8` for 10-color majority vote. Those budget breakpoints are engineering inferences rather than published ORT limits.

## Boundary handling

For fixed-size same-shape updates, **zero padding is the default winner** when the rule semantics mean “outside the board is dead/zero.” You get it for free by setting `pads=[1,1,1,1]` on the `Conv` itself. There is no extra pad tensor, so the bool-input neighbor-count path stays at `2HW·b_f` intermediate bytes for Q1-style kernels.

Reflective boundaries are semantically different and cost more. ONNX `Pad` supports `mode='reflect'`, but under opset ≤ 18 on ORT CPU the cheapest legal CA path is usually `Pad(bool/int/float) → Cast(if needed) → Conv(pads=0)`. On a bool-input board, the reflected 8-neighbor counter costs `(H+2)(W+2)·b_b + (H+2)(W+2)·b_f + HW·b_f`, so compared with the zero-pad bool path the extra score term is `(H+2)(W+2)·(b_b + b_f) - HW·b_f`. At `30×30` with `b_f=4` and `b_b≈1`, that is an extra **1,520 bytes** per step. Semantically, reflection reuses interior edge values and therefore changes corner and edge counts; it is not an approximation to an infinite board.

One useful boundary note for CA engineers: if you really want **toroidal wraparound**, `Pad(mode='wrap')` is outside the allowed opset range here. In opset ≤ 18 you only have `constant`, `reflect`, and `edge`, so torus logic would need explicit slicing/concatenation rather than `Pad`.

## Erosion and dilation

For binary masks, morphology is just CA neighborhood logic in another costume: 3×3 dilation is neighborhood OR, and 3×3 erosion is neighborhood AND. On ORT 1.24 CPU, `MaxPool` supports `int8`/`uint8` from opset 12 onward, and `Neg` supports signed numeric types including `int8`, so **int8 is the cheapest binary morphology path** on CPU. That is better for this score than bouncing everything through float unless you are already in a float Conv-heavy pipeline. The “OR/AND equivalence” is an inference from max/min over {0,1}.

| Pattern | Ops | Opset min | Input dtype | Output dtype | Param elements | Intermediate memory | Notes |
|---|---|---:|---|---|---:|---|---|
| Dilation via `MaxPool` | `MaxPool` or `Cast`, `MaxPool` if source is bool | 12 for int8 path | `tensor(int8)` preferred for score; `tensor(bool)` requires `Cast` | same as input path | 0 | `HW·b_i8` on int8 input; `2HW·b_i8` if starting from bool and casting | Use `kernel_shape=[3,3]`, `pads=[1,1,1,1]`, `strides=[1,1]` |
| Erosion via `-MaxPool(-x)` | `Neg`, `MaxPool`, `Neg` or `Cast`, `Neg`, `MaxPool`, `Neg` | 12 for int8 path | `tensor(int8)` preferred for score; `tensor(bool)` requires `Cast` | same as input path | 0 | `3HW·b_i8` on int8 input; `4HW·b_i8` if starting from bool and casting | ONNX has no `MinPool`; this is the standard min-via-negation trick |

```python
from onnx import helper

dilate3 = helper.make_node(
    "MaxPool", ["x_i8"], ["dilated"],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
)
```

```python
from onnx import helper

neg1 = helper.make_node("Neg", ["x_i8"], ["nx"])
pool = helper.make_node(
    "MaxPool", ["nx"], ["mx"],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
)
erode3 = helper.make_node("Neg", ["mx"], ["eroded"])
```

## Decision table

The recommendations below are for the score in this card: `intermediate_memory_bytes + param_elements`, with node count ignored and shared initializers counted once.

| Rule type | Recommended pattern | Why it wins under this score | Main caveat |
|---|---|---|---|
| Binary threshold rule over 4-neighbors | `Cast → Conv` with 4-neighbor kernel | Minimal legal ORT CPU path; only one 3×3 kernel initializer | Float Conv path on CPU; bool must be cast first |
| Binary threshold rule over 8-neighbors | `Cast → Conv` with 8-neighbor kernel | Same as above, and reusable for Life-like rules | Boundary semantics depend on zero vs reflect |
| Life-like B/S rule | Shared 8-neighbor kernel + boolean comparisons + `And`/`Or` | Avoids `Where`; tiny shared params; exact rule mapping | Use bool state tensor to keep logic cheap  |
| Same-color contagion / affinity | `Equal(x,c) → Cast → Conv` per color of interest | Cheapest exact same-color count on a palette board | One branch per color |
| Majority / mode propagation | `Equal×K → Cast×K → Conv×K → Concat → ArgMax`, with **one shared kernel** | Separate Conv branches beat grouped Conv because weight sharing keeps params at `K+9`, not `9K+K` | Tie policy must be explicit; use opset 12 `ArgMax` if you care  |
| Binary dilation | `MaxPool` on int8 mask | Zero params and cheaper than float Conv | Cast once if source is bool |
| Binary erosion | `Neg → MaxPool → Neg` on int8 mask | Cheapest exact min-neighborhood path without unsupported ops | Requires signed dtype for `Neg` |

## Verification checklist

- [x] All six research questions were answered, in the required order.
- [x] Boolean caveats were respected: `And`/`Or`/`Not` treated as bool ops; no `Where(bool,bool,bool)` pattern was used; `ReduceSum`/`ReduceMax` on bool were treated as cast-first on ORT 1.24 CPU.
- [x] Cost formulas were expressed symbolically in `H`, `W`, and per-dtype bytes-per-element symbols.
- [x] Multi-step unrolling included explicit per-step formulas and shared-vs-naive cumulative formulas.
- [x] A rule-type decision table covered more than four rule families.
- [x] `onnx.helper.make_node` snippets were included only for the winning patterns, with kernel constants inline.
- [x] Conway’s Life rule was cited, and ONNX/ORT operator behavior was grounded in official specs and ORT kernel/type documentation. 