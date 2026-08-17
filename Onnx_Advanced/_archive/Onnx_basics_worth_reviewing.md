# Onnx_basics_worth_reviewing.md — scoring mechanics

- `Constant` node with `value` attribute → counts toward `params` via the element count of `attr.t.dims`. Sparse `sparse_value` attribute counts `attr.sparse_tensor.values.dims` (stored values, NOT logical dense shape).
- `graph.initializer` → counts toward `params` via `init.dims`. `graph.sparse_initializer` counts `sparse_init.values.dims` (stored values).
- Both also count toward the file size limit (1.44 MiB binary).
- Prefer `initializer` over `Constant` nodes — same params cost, but no node-overhead in the graph.
- `calculate_memory` RE-INFERS `value_info` via `onnx.shape_inference.infer_shapes(model, strict_mode=True)` before counting. **Real intermediates are counted even if `graph.value_info` is empty in the saved model.** Clearing stale `value_info` before save is required only to avoid load-time conflicts, NOT to hide memory.
- Each value_info entry contributes `num_elements × dtype.itemsize` to the scorer's `memory` count. Scalars cost `dtype.itemsize` bytes (`BOOL []` = 1 B, `F16 []` = 2 B, `F32 []` = 4 B, `INT64 []` = 8 B) — NOT always 4 bytes.
- Declaring EXTRA value_info (dead entries / aliases) for tensors that aren't real intermediates IS wasted bytes. Real inferable intermediates are counted regardless.
