# ONNX Advanced Mini-Library For Web Workers

This is the compact replacement for the local `Onnx_Advanced` library. Use it
when you do not have repo access.

## Scoring And Accounting

NeuroGolf cost is:

```text
cost = memory_bytes + params
points = max(1.0, 25.0 - ln(max(1.0, cost)))
```

MACs/FLOPs do not matter for the current objective.

`params` is the count of unique initializer elements. It is dtype-blind:

- `[1,10,30,30]` F32 initializer = 9000 params;
- `[1,10,30,30]` BOOL initializer = 9000 params;
- sharing one initializer by name across multiple nodes counts once.

`memory` is the byte size of materialized intermediate tensors after strict
shape inference and runtime profiling. The graph tensors named exactly `input`
and `output` are skipped. This is the biggest golf rule:

- make expensive rendering happen in the final node named `output`;
- do not build a big tensor and then pass it through another node.

Clearing `value_info` does not hide memory. The scorer re-inferrs real
intermediate tensors.

## Fixed IO And Decode

Submitted models use:

- input name: `input`;
- output name: `output`;
- input shape: `[1,10,30,30]` F32 one-hot;
- output shape: `[1,10,30,30]`;
- all dimensions static, no symbolic dims.

The output decoder thresholds each channel with `> 0.0`. The output values do
not need to be exact 0/1 if their sign pattern is correct. BOOL output can be
valid when it has the right shape and sign semantics.

## Byte Table

Common intermediate memory sizes:

| Shape | F32 | F16 | U8/BOOL |
| --- | ---: | ---: | ---: |
| `[1,10,30,30]` | 36000 | 18000 | 9000 |
| `[1,1,30,30]` | 3600 | 1800 | 900 |
| `[1,10,10,10]` | 4000 | 2000 | 1000 |
| `[1,1,10,10]` | 400 | 200 | 100 |
| `[1,2,10,10]` | 800 | 400 | 200 |
| scalar `[]` | 4 | 2 | 1 |

For Task 374, beating 1365 means every 100-byte plane matters.

## Fatal Or Banned Routes

Avoid:

- `Loop`, `Scan`, `If`, subgraphs, functions, custom domains;
- `NonZero`, `Unique`, `Compress` and other dynamic-shape filters;
- symbolic dimensions or missing static shapes;
- sparse initializers as op inputs;
- extra graph inputs or outputs;
- initializer names colliding with `input` or `output`;
- packaging-only tricks.

Dynamic graphs can price lower locally and higher under real profiling. Static
graphs are much safer.

## Free-Output Families

If a large tensor is the graph `output`, its memory is not counted. Useful final
nodes include:

- `Equal(canvas, channel_ids)` as output;
- `Where(cond, paint, input)` as output when overlaying a change onto input;
- `Conv`, `ConvInteger`, `QLinearConv`, `Einsum`, `Pad`, `Gather`, `Slice`, or
  `Transpose` directly into output, when shape and semantics fit.

Safety checks:

- out-of-active-grid channels must be non-positive or all false;
- padding/empty cells often need an out-of-range sentinel such as color 10;
- one scalar sign cannot encode many mutually exclusive classes under `>0`;
- a free-output Conv with positive off-center taps can leak outside the active
  region unless bias/masks suppress it.

## Equal-Tail And Color-Index Canvas

For a normal one-hot output, a common cheap tail is:

```text
single-channel U8 color-index canvas [1,1,H,W]
Equal(canvas, channel_ids [1,10,1,1]) -> output
```

The `Equal` result is free if it is the graph output. The charged part is the
single-channel canvas:

- full 30x30 canvas = 900 bytes;
- 10x10 canvas = 100 bytes, but padding/expansion may add more;
- a `[1,10,h,w]` BOOL intermediate costs `10*h*w` bytes, so price it against the
  single-channel canvas.

Empty cells that should decode to no channel must use sentinel value 10, not
color 0, unless color 0 is truly the desired output.

## QLinearConv And ConvInteger

Quantized ops are useful because their lanes can stay U8/INT8:

- params still count as element count, not bytes;
- scales and zero-points cost params too, so share them when legal;
- `QLinearConv` requires U8/INT8 input. Casting the whole F32 input frame to U8
  costs 9000 bytes and usually loses;
- use QLinearConv after you already have a narrow crop/mask/descriptor;
- `ConvInteger` can be a cheap final decoder because the output can be free.

Task 374's current 1365 family uses two U8 feature planes and a final
ConvInteger decoder. A one-plane linear ConvInteger decoder cannot represent the
four needed classes: background, longest, second-longest, remaining.

## Descriptor-First Thinking

Before building per-cell planes, ask whether the rule can be represented as:

- row vector plus column vector;
- small scalar descriptors;
- marginal reductions over rows/columns;
- separable row x column products;
- late final-output expansion.

For Task 374, row and column segment descriptors are more promising than early
30x30 canvases.

## Useful Primitive Patterns

- `Gather(input, perm, axis=1)` is the floor for a simple color permutation.
- `Slice`/`Gather` directly to `output` can be zero-memory.
- `ReduceMax` / `ReduceSum` over small cropped masks can create row/column
  descriptors.
- `ArgMax` is useful for "first/last/maximum position" when tie policy is known.
- `Where(cond, a, b)` is the universal branchless if-else and is usually cheaper
  than mask-multiply chains.
- `Equal`, `Greater`, `Less` produce BOOL masks; check dtype support before
  assuming bool arithmetic works.
- `Einsum` can remove materialized projection tensors when it directly computes
  a marginal or final separable output, but large multi-operand contractions may
  be slow or risky.

## Task 374 Price Awareness

The bundled 1365 graph is approximately:

- `gray_crop_f` F32 `[1,1,10,10]`: 400 bytes;
- `gray_u8`: 100 bytes;
- horizontal detector lanes: about 200 bytes;
- selected and masked two-feature planes: 400 bytes;
- row/column length and feature descriptors: about 210 bytes;
- params: 55.

This makes the current family tight. To beat it, a new idea probably needs one
of these:

- remove or fuse a counted 100- or 200-byte plane;
- delete at least one scalar or initializer while preserving correctness;
- replace the two-feature final decoder with a different final-output primitive;
- avoid `gray_crop_f` or delay the F32 input-read tax;
- derive a smaller row/column descriptor pipeline from the rule.

Do not let those bullets over-constrain you. They are price hints, not a proof.

## Floor Doctrine

"Floor" means one of two things:

- strong forced-component proof: every correct formulation needs the charged
  object;
- tried-path evidence: a particular architecture or lever failed.

Most old floor notes are tried-path evidence. Treat them as maps, not stop
signs. A better Task 374 answer may require changing the final decoder or
output convention rather than shaving the current graph.
