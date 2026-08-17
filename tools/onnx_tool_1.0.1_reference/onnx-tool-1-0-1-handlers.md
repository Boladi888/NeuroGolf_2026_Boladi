# Source: onnx-tool 1.0.1 op handlers and cost accounting

**Verification source**: `/usr/local/lib/python3.12/dist-packages/onnx_tool/node.py` from `pip install onnx-tool==1.0.1`. Line numbers are from this file.

## MAC unit constants (top of node.py)

```python
ADD_MACS = 1
MUL_MACS = 1
CMP_MACS = 1
RESIZE_LINEAR_MACS = 4
EXP_MACS = 32
DIV_MACS = 4   # refers to vrcp14ps
```

This means all element-wise integer or float arithmetic counts as 1 op/element. There is **no operator-class discount** for AND, OR, XOR, NOT vs Mul or Add.

## Default Node profile (line 212)

```python
class Node():
    def profile(self, intensors, outtensors):
        return [0, 0]
```

Any op without an override returns 0 macs and 0 params. **This is what gives `Gather`, `Slice`, `Reshape`, `Transpose`, `Cast`, `Identity` their zero-MAC accounting.**

## NpMathBase.profile (line 247–264)

```python
class NpMathBase(Node):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS         # default
        self.ratio = max(1, len(self.input) - 1)
    def profile(self, intensors, outtensors):
        outshape = outtensors[0].get_shape()
        macs = volume(outshape) * self.ratio * self.op_mac
        return [macs, 0]
```

So for a `Mod`, `Add`, `Sub`, `Mul`, `Min`, `Max`, `Abs`, etc. on a `[1,10,30,30]` tensor, macs = 9000 × 1 × 1 = **9,000**.

## Per-op `op_mac` overrides

Grep on `op_mac` in node.py gives:
- `MinNode.op_mac = CMP_MACS`        (line 286)
- `MaxNode.op_mac = CMP_MACS`        (line 296 area)
- `NegNode.op_mac = CMP_MACS`        (line 308)
- `DivNode.op_mac = DIV_MACS`        (line 318) — **4× Mul**
- `ModNode.op_mac = ADD_MACS`        (line 335)
- `MulNode.op_mac = MUL_MACS`        (line 346)
- `AbsNode.op_mac = CMP_MACS`        (line 357)
- `ExpNode.op_mac = EXP_MACS`        (line 375) — **32× Mul**
- `SoftmaxNode.op_mac = EXP_MACS + DIV_MACS = 36`  (line 387)
- `SqrtNode.op_mac = SQRT_MACS`      (line 448)
- `PowNode.op_mac = POW_MACS`        (line 459)
- `SinNode.op_mac = SIN_MACS`        (line 470)
- `CosNode.op_mac = COS_MACS`        (line 481)

**Practical takeaways for J.01 / J.16**:
- `Mod` and `Mul` are equally cheap (1 op/element).
- `Div` is 4× more expensive than `Mod` for the same effect — always prefer `Mod` for periodicity.
- `Exp`, `Softmax`, `Sigmoid` are 32–36× a `Mul`, never use them in NeuroGolf.

## Memory accounting

In `node.py` `Node` class and Graph object, intermediate `value_info` entries' byte size (= elem_size × product(shape)) is summed. Per the brief: "memory metric is custom: sum of `value_info` intermediate-activation byte sizes + declared IO (excluding `'input'`/`'output'` themselves)".

**Implication**: every intermediate tensor adds 4 × volume bytes (for float32) or 1 × volume (for int8) etc. A simple Cast→Op→Cast chain on `[1,10,30,30]` adds ≥ 36,000 + 36,000 = 72,000 bytes to the cost (input and output are excluded).

Verified empirically: my `Cast(int8)+Cast(float)` two-node graph profiles as macs=0, memory=45,000 B (= 9,000 for the int8 intermediate + 36,000 for the float output, with the float output dropped by the validator → real contribution 9,000 B).

## Bitwise / BitShift handlers — **MISSING**

```bash
$ grep -in "bitwise\|bitshift" node.py graph.py model.py
# → no matches
```

When loading a graph that contains `BitwiseAnd`, onnx-tool emits:

```
UserWarning: node BitwiseAnd is not registered for profiling, return 0 Macs and 0 params as default. Use NODEPROFILER_REGISTRY to register your profiler for this node.
```

And then `shape_infer` raises:

```
NotImplementedError: this Node BitwiseAnd-... has no value_infer
```

**This means a graph containing any `Bitwise*` or `BitShift` op cannot be fully profiled by onnx-tool 1.0.1.** Whether the live Kaggle validator handles this differently is unknown to me — verify before relying on it.

## `params` is in element count, not bytes

A length-900 `int32` Gather initializer reports `params = 900`, not `params = 3600`. Same for `int64` (still 900 elem). So **the dtype of an initializer affects `memory` (via the Gather node's accounting) but not `params`**. The cheapest perm representation is therefore int32 — same `params`, lower `memory`.

## Constant folding (`{'constant_folding': True}`)

The flag triggers a fold pass that replaces any node whose every input is a constant (initializer or already-folded) by a new initializer. Source: `onnx_tool/data/ConstantFolding.md` in upstream:
> Reported reductions on real models: 7,689 → 3,343 nodes; 1,139 → 110 nodes.

Verified locally: a `Range(0, 900, 1)` node feeding into `Gather` folded down to a single `idx` initializer of shape `(900,)`, with no Range node remaining in the saved graph.

## Aggregate API used by the validator

```python
mdl = onnx_tool.Model(path, {'constant_folding': True})
mdl.graph.shape_infer({'input': np.zeros((1,10,30,30), dtype=np.float32)})
mdl.graph.profile()
mdl.graph.print_node_map()  # tabular per-node and Total row
```

The `Total` row's `Forward_MACs`, `Memory`, and `Params` columns correspond to `macs`, `memory`, `params` in the score formula. Subtract the size of the model's declared `output` tensor (4 × 9,000 = 36,000 B for float32 `[1,10,30,30]`) from the reported `Memory` to get the validator's effective intermediate-memory contribution.
