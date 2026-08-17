# ort_errors.md — error message → meaning → fix

When ORT or the validator rejects a model, the error message is often cryptic. This table is the lookup. Add to it when you debug a new one (drop a `DONE_*.md` in `.update/`).

## How to triage an unfamiliar error

1. **Run `onnx.checker.check_model(model, full_check=True)`** first. Many errors surface here before ORT, with clearer messages.
2. **Run `onnx.shape_inference.infer_shapes(model)`** next. If this raises, the issue is a shape/dtype mismatch.
3. **Try `onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])`** as a final load-time check.
4. If steps 1-3 all pass and the model still fails at score time, the issue is at execution (usually an op-specific dtype mismatch or unsupported op).

## Error table

| Error fragment | Likely cause | Fix |
|---|---|---|
| `"two nodes with same node name"` | Multiple emitted nodes share `name` field. Common when batch-generating with `helper.make_node("Foo", name=f"{x.name or 'fallback'}")` and multiple x's have empty names. | Use a counter: `def _uniq(p): counter[0] += 1; return f"{p}_{counter[0]}"`. Every node gets a unique name. |
| `"Type Error: Type X of output arg (name) does not match expected type Y"` | The graph output's declared dtype doesn't match the dtype produced by the node that emits it. Common after dtype-changing surgery (BOOL/UINT8 Pad rewrites). | Update `graph.output[0].type.tensor_type.elem_type` to match the producing node's output dtype. |
| `"[ShapeInferenceError] Incompatible dimensions"` | Stale `value_info` cached shapes from before graph surgery. ORT loads the cached shapes and they conflict with the new graph structure. | `graph.ClearField("value_info")` before save. (Or `model = onnx.shape_inference.infer_shapes(model)` if you want fresh inferred shapes.) |
| `"[ShapeInferenceError] (op: Op_Type)"` with no other detail | The named op's spec violation — usually wrong input count, wrong dtype, or wrong attribute. | Look up the op in `onnx.defs.get_schema("Op_Type")`. Check input/output counts and dtype constraints. |
| `"Op (Pad) [ShapeInferenceError]"` opset 10 | `pads` attribute length is wrong, or wrong layout. | `pads` is `INTS` of length `2 × rank`. Layout: `[x1_begin, x2_begin, ..., xN_begin, x1_end, x2_end, ..., xN_end]`. NOT interleaved. |
| `"Op (Equal) [ShapeInferenceError]"` opset 10 | Equal received float inputs. Opset 10 Equal accepts BOOL/INT32/INT64 only. | Cast inputs to INT32 first or bump to opset 11+ for float/numeric Equal. See `ort_quirks.md §4` for the shared-Cast trick to net positive bytes. |
| `"Invalid model. opset_import for domain '' is null"` | The model has no `opset_import` for the default domain. | Always include `opset_imports=[helper.make_opsetid("", N)]` in `helper.make_model`. |
| `"Op (X) is not a registered function/op"` | Op name typo, or op exists in newer opset than declared. | Verify op name spelling. If the op exists in a newer opset, bump opset_import. If banned (LOOP/SCAN/NONZERO/UNIQUE/SCRIPT/FUNCTION/COMPRESS) — the validator rejects regardless of opset. |
| `"Constant node ... has no value attribute"` | A Constant node missing its `value`/`value_floats`/`value_ints`/`value_strings`/`sparse_value` attribute. | Provide one. Or, preferred: move the value into `graph.initializer` (no node-overhead). See `techniques.md` §18 (const_to_init). |
| `"Cannot find node input 'foo'"` | A node references a tensor name that isn't an input, initializer, or earlier node's output. | Reorder nodes so producers come before consumers. Or fix the name — common typo source. |
| `"Cannot find node input 'foo'"` after deletion | You deleted a producer but downstream consumers still reference its output. | Rewire consumers' input names before/during the delete. Build a producer→consumer map first; mutate atomically. |
| `"Initializer name conflicts with input name"` | An initializer shares a name with the model input or output. NeuroGolf validator's `calculate_memory` returns None (model rejected). | Rename the initializer. |
| ORT_DISABLE_ALL OneHot error | OneHot index/depth dtype mismatch, most often INT32 indices with INT64 depth. | Make `indices` and scalar `depth` the same dtype (INT32+INT32 or INT64+INT64). See `ort_quirks.md §3`. |
| `"InferenceSession failed: [ONNXRuntimeError] FAIL : Node (X) Op (Where) [TypeInferenceError]"` | `Where(bool, bool, bool)` — ORT does not support BOOL data inputs. | Rewrite as `Or(And(c, a), And(Not(c), b))`. See `ort_quirks.md §2`. |
| `"InferenceSession failed: [ONNXRuntimeError] FAIL"` (generic, no detail) | Many possible causes; ORT didn't surface specifics. | (1) Run `onnx.checker.check_model(model, full_check=True)`. (2) Run `onnx.shape_inference.infer_shapes(model)`. (3) Check every initializer is referenced. (4) Check no banned ops. (5) Check opset_import is set. |
| `"check_model: ValidationError"` (with detail) | Schema violation in your construction. | Read the detail. Almost always tells you the exact node and constraint. |
| `"Sparse tensor with no values"` | Sparse initializer's values are empty. | Provide values in `init.sparse_tensor.values` and indices in `init.sparse_tensor.indices`. |
| `"dimensions: size <= 0"` | A `dim_value <= 0` in some tensor's shape. Validator rejects. | Fix the shape. Or, if symbolic, ensure all dims are concrete `dim_value > 0` (no `dim_param`). |
| Score returns `None` after model loads | One of the validator's rejection paths triggered AFTER load: e.g., more than 1 input, name collision i/o ↔ init, subgraphs, functions, non-default opset domain, scalar tensor type issues. | See `core.md` "Validator rejection criteria" — work through the list. |

## When to suspect a Kaggle-specific issue (not ORT)

If the candidate scores fine locally (`score_network_v2` returns a number) but Kaggle returns ERROR for that task:

- **Cross-task memory cascade**: see `core.md` "Cross-task memory cascade bug". A high-memory ONNX in slot N can ERROR slot N+1. Local validators cannot detect this.
- **Kaggle scoring formula drift**: rare but happens (rewrite on 2026-05-04). Check forum thread before debugging further.

## Diagnostic snippets (copy-paste)

```python
# Full pre-flight check
import onnx, onnxruntime as ort
model = onnx.load("candidate.onnx")
onnx.checker.check_model(model, full_check=True)
model = onnx.shape_inference.infer_shapes(model)
sess = ort.InferenceSession("candidate.onnx", providers=["CPUExecutionProvider"])
print(f"Loaded OK. Inputs: {[i.name for i in sess.get_inputs()]}")
```

```python
# List all initializer names + shapes + dtypes
for init in onnx.load("x.onnx").graph.initializer:
    print(f"{init.name:<40} shape={list(init.dims)} dtype={init.data_type}")
```

```python
# List all node op_types in order
for n in onnx.load("x.onnx").graph.node:
    print(f"{n.op_type:<15} in={list(n.input)} out={list(n.output)}")
```

```python
# Show where shape inference fails
try:
    onnx.shape_inference.infer_shapes(onnx.load("x.onnx"), strict_mode=True)
except onnx.shape_inference.InferenceError as e:
    print(f"Shape inference broken: {e}")
```
