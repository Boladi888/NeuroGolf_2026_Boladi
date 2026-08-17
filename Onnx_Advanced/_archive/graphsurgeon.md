# graphsurgeon.md — NVIDIA's onnx-graphsurgeon as cleaner alternative to raw protobuf

`onnx-graphsurgeon` (often imported as `gs`) is NVIDIA's library for editing ONNX graphs at a higher level than raw `onnx.helper` protobuf manipulation. It's designed for TensorRT users but works fine for any ONNX surgery task.

`pip install onnx-graphsurgeon` (no extra C++ deps).

For NeuroGolf's pattern-pass approach (`tools/spread_patterns/*.py`), this library could simplify many of the rewrite scripts.

---

## The mental model difference

**Raw `onnx.helper`** treats a graph as protobuf — you manipulate lists of `NodeProto`, `TensorProto`, etc. Cross-references are by string name. Easy to leave dangling references on delete.

**`onnx-graphsurgeon`** treats a graph as an object graph — nodes have actual `Tensor` references for inputs/outputs. Cross-references are live Python objects. Cleanup is automatic when you do `graph.cleanup()`.

Example: deleting a node.

```python
# Raw protobuf:
graph.node.remove(node)
# But every downstream node still has the deleted node's output name as input — silent breakage.
# You have to manually find downstream consumers and rewire.

# graphsurgeon:
node.outputs.clear()   # disconnect outputs
graph.cleanup()        # removes unreachable nodes and dead refs
```

`cleanup()` finds dead branches, removes them, and re-numbers everything. **The biggest source of NeuroGolf rewrite bugs (dangling references after surgery) goes away.**

---

## Core API surface

```python
import onnx
import onnx_graphsurgeon as gs

# Load
graph = gs.import_onnx(onnx.load("model.onnx"))

# Inspect
for node in graph.nodes:
    print(node.op, node.name, [t.name for t in node.inputs], [t.name for t in node.outputs])

for t in graph.tensors().values():  # dict of name -> Tensor
    print(t.name, t.shape, t.dtype)

# Modify: insert a new node
new_input = gs.Variable(name="new_in", dtype=np.float32, shape=(1, 10, 30, 30))
new_const = gs.Constant(name="const", values=np.zeros((1,), dtype=np.float32))
new_out = gs.Variable(name="new_out", dtype=np.float32, shape=(1, 10, 30, 30))
new_node = gs.Node(op="Add", inputs=[new_input, new_const], outputs=[new_out], name="my_add")
graph.nodes.append(new_node)

# Modify: rewire
some_consumer.inputs[0] = new_out   # changes consumer to read from new node

# Cleanup
graph.cleanup().toposort()

# Save
onnx.save(gs.export_onnx(graph), "modified.onnx")
```

---

## Key classes

### `Tensor` (base for Variable / Constant)

```python
class Tensor:
    name: str
    dtype: numpy.dtype  
    shape: tuple
    inputs: List[Node]   # nodes that produce this tensor (usually len 1)
    outputs: List[Node]  # nodes that consume this tensor
```

- `Variable`: tensor produced by a node (or graph input).
- `Constant`: tensor with stored values (becomes an initializer at export).

### `Node`

```python
class Node:
    op: str              # e.g. "Add", "MatMul"
    name: str
    inputs: List[Tensor]
    outputs: List[Tensor]
    attrs: OrderedDict   # attribute name → value
```

Modifying `node.inputs[i] = some_tensor` automatically updates the tensor's `outputs` list. Bidirectional consistency.

### `Graph`

```python
class Graph:
    nodes: List[Node]
    inputs: List[Tensor]   # graph inputs
    outputs: List[Tensor]  # graph outputs
    name: str
    opset: int
    
    def tensors(self) -> Dict[str, Tensor]: ...
    def cleanup(self, remove_unused_node_outputs=True) -> Graph: ...
    def toposort(self) -> Graph: ...
    def fold_constants(self) -> Graph: ...   # like onnxsim's constant folding
```

`cleanup` + `toposort` + `fold_constants` together do a lot of what onnxsim does.

---

## NeuroGolf pattern-pass example

A typical `pass_X.py` in `tools/spread_patterns/` does:
1. Load model.
2. Find a structural pattern (e.g., `Cast → Pad`).
3. Rewrite (e.g., drop the Cast).
4. Save.

With graphsurgeon:

```python
import onnx
import onnx_graphsurgeon as gs

def apply_drop_cast_before_pad(input_path: str, output_path: str) -> bool:
    graph = gs.import_onnx(onnx.load(input_path))
    
    rewrites = 0
    for node in graph.nodes:
        if node.op != "Pad":
            continue
        pad_data_input = node.inputs[0]
        producer = pad_data_input.inputs[0] if pad_data_input.inputs else None
        if producer is None or producer.op != "Cast":
            continue
        if producer.attrs.get("to") != onnx.TensorProto.FLOAT:
            continue
        if producer.inputs[0].dtype != np.bool_:
            continue
        # The cast: BOOL → F32, feeds Pad. We can wire Pad directly to the BOOL tensor.
        node.inputs[0] = producer.inputs[0]
        rewrites += 1
    
    if rewrites == 0:
        return False
    
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    return True
```

vs the raw `onnx.helper` equivalent which would need:
- Manual producer-consumer map building.
- Manual node deletion from `graph.node` list.
- Manual `value_info` clearing.
- Manual reference-name rewiring on every consumer.

**graphsurgeon: ~15 lines. Raw protobuf: 40+ lines.**

---

## When NOT to use it

- For **read-only** inspection or score computation, raw `onnx.load` + protobuf walks are fine — graphsurgeon adds dependency for no benefit.
- For **single-attribute changes** (e.g., set `mode="constant"` on an existing Pad), raw protobuf is shorter: `node.attribute[i].s = b"constant"`.
- For projects that need **no extra dependencies** (worth noting if NeuroGolf's competition environment is locked).

---

## Verification: does graphsurgeon round-trip cleanly?

```python
import onnx
import onnx_graphsurgeon as gs

original = onnx.load("model.onnx")
graph = gs.import_onnx(original)
exported = gs.export_onnx(graph)
onnx.save(exported, "roundtrip.onnx")

# Compare:
loaded_back = onnx.load("roundtrip.onnx")
# Compare protobuf field-by-field — they should be equivalent (modulo node ordering after toposort).
```

[NEEDS EMPIRICAL] Specific things to verify on a NeuroGolf model:
- Are all `value_info` entries preserved?
- Is the opset_import preserved exactly?
- Does the round-tripped model load in ORT identically (same `score_network` result)?

---

## What graphsurgeon does NOT solve

- It doesn't add NeuroGolf-specific scoring awareness. You still need to think in params/memory/MAC terms.
- It doesn't do automatic algebraic-identity rewrites — those are project-specific patterns.
- It's NOT a replacement for `onnxsim` (which does constant-folding more aggressively). Use both.

---

## Recommendation for NeuroGolf

If you're writing a NEW pattern-pass script that involves more than ~3 node-insert / delete / rewire operations, **start with graphsurgeon**. The reduced bug surface is worth the dependency.

For existing passes that work, don't migrate just to migrate.

---

## TODO — verify empirically

1. `graphsurgeon` round-trip equivalence on a NeuroGolf-style model with BOOL ops and stale `value_info`.
2. `graphsurgeon.fold_constants()` vs `onnxsim` — which catches more for NeuroGolf-shape graphs?
3. Whether `gs.Constant.values` (numpy array) → exported initializer preserves dtype exactly (esp BOOL/UINT8 round-trip).
4. Compatibility with NeuroGolf's `_EXCLUDED_OP_TYPES` validator — graphsurgeon doesn't enforce ONNX schema, so you could accidentally produce banned ops.
