# onnx_utilities.md — lesser-known onnx package utilities

The `onnx` Python package has more than just `load`/`save`/`helper`. These submodules can save significant code for splicing, extracting, and converting models.

---

## `onnx.compose` — splice models together

Provides functions to merge two ONNX models by connecting outputs of one to inputs of another.

```python
import onnx
from onnx import compose

model_a = onnx.load("front_half.onnx")
model_b = onnx.load("back_half.onnx")

# Merge: connect model_a's "out_x" to model_b's "in_x"
merged = compose.merge_models(
    model_a, model_b,
    io_map=[("out_x", "in_x")],      # list of (a_output, b_input) pairs
    prefix1="m1_", prefix2="m2_",    # rename internal nodes/inits to avoid collisions
)
onnx.save(merged, "merged.onnx")
```

### Other functions

- `compose.merge_graphs(graph_a, graph_b, io_map, ...)` — same but works on `GraphProto` directly.
- `compose.add_prefix(model, prefix)` — namespace all node/init/io names. Use before merging to avoid collisions.
- `compose.expand_out_dim(model, dim_idx, ...)` — insert a new dim at position `dim_idx` in inputs/outputs.

### NeuroGolf use cases

**1. Compose template + permutation:**

If you have a "computation skeleton" model and N variants of a "color permutation" model, you can keep them as separate small `.onnx` files and compose at build time:

```python
skeleton = onnx.load("compute_core.onnx")
perm = onnx.load(f"perm_{task_id}.onnx")
final = compose.merge_models(skeleton, perm, io_map=[("compute_out", "perm_in")])
```

Useful if you're iterating on permutation logic separately from the core algorithm.

**2. Splice in a preprocessing crop:**

```python
crop_model = build_crop_to_active_region()  # small graph that slices the active grid
core = onnx.load("solution_core.onnx")
final = compose.merge_models(crop_model, core, io_map=[("cropped_out", "input")])
```

### Gotchas

- The connected output dtype/shape MUST match the connected input. Use `expand_out_dim` or insert Cast/Reshape between if needed.
- `prefix1`/`prefix2` are CRITICAL when models share name spaces — without them, init name collisions silently break the merge.
- `compose.merge_models` deep-copies. For large models, memory cost is 2× temporarily.

---

## `onnx.utils.extract_model` — carve out a subgraph

Extract a sub-model from a larger ONNX model given input/output tensor names.

```python
from onnx.utils import extract_model

extract_model(
    input_path="full_model.onnx",
    output_path="subgraph.onnx",
    input_names=["intermediate_tensor_a", "intermediate_tensor_b"],
    output_names=["intermediate_tensor_c"],
)
```

This walks the graph backward from `output_names` until it hits `input_names`, then materializes the slice as a standalone model.

### NeuroGolf use cases

**1. Bisect a failing model:**

If a model fails at ORT load but you don't know which node, extract progressively smaller subgraphs to isolate the bad op:

```python
extract_model("failing.onnx", "test_first_half.onnx",
              input_names=["input"], output_names=["mid_tensor"])
# If "test_first_half.onnx" loads, the bad node is in the second half.
```

**2. Reuse a costly sub-computation:**

If you computed a complex mask in model A and want to use it in model B's pipeline tests, extract just the mask-computing subgraph and benchmark it standalone.

### Gotchas

- The input/output tensor names must already exist in the graph. Run `onnx.shape_inference.infer_shapes` first to ensure intermediates have value_info.
- Initializers used by the extracted subgraph are copied; unused initializers are dropped.
- The extracted model has its own opset_import (inherited from source).
- **Doesn't validate semantic correctness** — extracting a non-contiguous slice produces a broken model.

---

## `onnx.version_converter` — upgrade/downgrade opset

Programmatically convert a model from one opset to another. Useful when you want to use opset-N ops in a previously opset-M graph.

```python
from onnx import version_converter

model = onnx.load("opset10.onnx")
upgraded = version_converter.convert_version(model, target_version=13)
onnx.save(upgraded, "opset13.onnx")
```

### What it handles automatically

- `Squeeze` / `Unsqueeze` / `Reduce*` opset 12→13 schema break (axes attr → input).
- `Pad` opset 10→11 (attr → input form).
- `Clip` opset 6→11 (min/max attr → input).
- Most "input-vs-attribute" rearrangements.

### What it does NOT handle

- Opset bumps that introduce semantically new ops (Trilu opset 14, etc.) — these are introductions, not conversions.
- ORT-specific behaviors. The converter is spec-based; ORT may still reject the result.
- Custom validation rules (e.g., NeuroGolf's banned op list).

### NeuroGolf use cases

**1. Bump a legacy opset-10 model to opset 11 for Pad-input-form + UINT8 support:**

```python
model = version_converter.convert_version(onnx.load("legacy.onnx"), 11)
# Now Pad uses input form; you can add UINT8 Pad rewrites.
```

**2. Bump to opset 14 for Trilu / F16 CumSum:**

```python
model = version_converter.convert_version(onnx.load("graph.onnx"), 14)
# Now Trilu is available; you can rewrite stored tri-masks into Trilu generations.
```

### Gotchas

- The converter may produce a "valid spec but doesn't load in ORT" model. Always verify-load after conversion.
- DOWNGRADING (e.g., 13 → 10) often loses information (e.g., F32 Equal becomes invalid). Avoid unless necessary.
- `target_version` is the OPSET version, not IR version. IR version stays the same.

---

## `onnx.helper` — beyond the basics

Most people know `make_node`, `make_graph`, `make_model`, `make_tensor_value_info`. Less-used:

### `make_attribute`

Explicit attribute construction (vs. passing kwargs to `make_node`):

```python
attr = helper.make_attribute("equation", "bij,bjk->bik")
node = helper.make_node("Einsum", ["A", "B"], ["C"], name="ein")
node.attribute.append(attr)
```

Equivalent to:
```python
node = helper.make_node("Einsum", ["A", "B"], ["C"], name="ein", equation="bij,bjk->bik")
```

The explicit form is useful when programmatically generating attributes from variables.

### `make_empty_tensor_value_info`

Tensor value_info with no shape/dtype info — for cases where you want to declare a tensor name without committing to its shape:

```python
vi = helper.make_empty_tensor_value_info("dynamic_tensor")
```

Rarely useful for NeuroGolf (everything has fixed shape).

### `printable_graph(graph)`

Returns a human-readable string representation of the graph. Useful for debugging:

```python
print(helper.printable_graph(model.graph))
```

Output format is like:
```
graph {
  Add(input_a, input_b) -> add_out  
  Cast(add_out) -> cast_out
  ...
}
```

Much more scannable than `print(model.graph)` (which dumps raw protobuf text).

### `make_function`

Builds a `FunctionProto`. **NeuroGolf BANS functions** (`graph.functions` must be empty per validator). Don't use this — included only so you know it exists.

---

## `onnx.checker.check_model` — beyond `full_check=False`

```python
import onnx
onnx.checker.check_model(model, full_check=True)
```

The `full_check=True` flag also runs shape inference and validates intermediates. Catches errors that the default check misses.

For NeuroGolf, run this AFTER every graph mutation pass to catch shape mismatches early:

```python
def safe_save(model, path):
    onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
```

---

## `onnx.shape_inference` — programmatic shape inference

```python
import onnx.shape_inference

# In-place inference: returns a new model with value_info filled in
inferred = onnx.shape_inference.infer_shapes(model)

# Strict mode: raises on inconsistency instead of warning
inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)

# Data-prop mode: also propagates constant values where possible
inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
```

### NeuroGolf use cases

**1. Validate shape consistency after surgery:**

```python
try:
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
except onnx.shape_inference.InferenceError as e:
    print(f"Shape mismatch after surgery: {e}")
```

**2. Discover dtype propagation:**

If you rewire a node from F32 to BOOL, run `infer_shapes` to see if downstream tensors' types update correctly.

### Gotchas

- `value_info` populated by shape inference COUNTS TOWARD MEMORY in the NeuroGolf scorer. After inferring, clear what you don't need:
  ```python
  inferred = onnx.shape_inference.infer_shapes(model)
  needed_names = {...}  # names you want validated
  inferred.graph.value_info[:] = [vi for vi in inferred.graph.value_info if vi.name in needed_names]
  ```
- Strict mode catches issues but may be too aggressive for partially-built graphs.

---

## `onnx.numpy_helper` — beyond `from_array` / `to_array`

### `numpy_helper.bfloat16_to_float32` and `float32_to_bfloat16`

For BF16 inits (rare in NeuroGolf):

```python
from onnx import numpy_helper
bf16_arr = numpy_helper.float32_to_bfloat16(np.array([1.0, 2.0, 3.0]))
```

### `numpy_helper.to_dict`

Convert a `MapProto` (key-value pairs) to a Python dict. Rarely used in NeuroGolf.

---

## Combining utilities: "carve, modify, reassemble"

```python
import onnx
from onnx import compose
from onnx.utils import extract_model

# 1. Extract the "compute core" from a known-good model
extract_model("full.onnx", "core.onnx",
              input_names=["preprocessed_in"], output_names=["raw_out"])

# 2. Build a small postprocessor in a separate model
postproc = build_postprocessor()  # your function

# 3. Compose
core = onnx.load("core.onnx")
final = compose.merge_models(core, postproc, io_map=[("raw_out", "postproc_in")])

# 4. Validate
onnx.checker.check_model(final, full_check=True)
onnx.save(final, "final.onnx")
```

This pattern is overkill for single-task surgery but useful if you're building a pipeline-of-models that you iterate on independently.

---

## Cross-references

- `Ort_Advanced/ort_errors.md` for what `check_model` errors mean.
- `Onnx_basics_worth_reviewing.md` for the `value_info` memory cost discussion.

## TODO — verify empirically

1. Does `version_converter.convert_version` produce ORT-loadable models for NeuroGolf-style graphs at all common bump targets (10→11, 11→13, 13→14)?
2. Does `extract_model` preserve all required value_info for ORT to load the extracted subgraph?
3. Does `compose.merge_models` produce models that score correctly in NeuroGolf's scorer (with proper opset handling)?
