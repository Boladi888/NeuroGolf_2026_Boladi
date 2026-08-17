# task182 hard record: sparse Conv weight to cut the 250-param hash kernel

Live bar: **4460** (Boladi, sha `b2c1867a...`, 2026-07-11). Cost model (verified in
`neurogolf_utils.py`): **memory = SUM of intermediate bytes** (dtype matters), **params =
initializer ELEMENT COUNT** (dtype-free; `calculate_params` sums `math.prod(init.dims)`).

Attempted route: the hash Conv weight `v_w12 [1,10,5,5]` = **250 params but only 25 nonzero**
(all in channel 1; the other 9 channels are zero to consume the 10-channel one-hot input directly
and avoid a 3600 B blue-plane extraction). `calculate_params` counts `sparse_initializer` and
`Constant(sparse_value)` by `math.prod(values.dims)` = **nonzero count**, so a 25-nnz sparse form
would score 25 params (save 225 -> ~4235).

Failed gates (two independent kills):
1. `onnx.checker.check_model(full_check=True)` -> `[ShapeInferenceError] (Conv): W ... has
   unsupported type: sparse_tensor(float)`. ONNX Conv rejects a sparse_tensor weight. And
   `calculate_memory` calls `check_model(full_check=True)` -> would return None regardless.
2. `sanitize_model` (used by the profiler AND the grader) renames node inputs but NOT
   `graph.sparse_initializer` names, so the Conv input `v_w12` -> `safe_name_36` no longer
   resolves: `ng_gate visible` -> `InvalidArgument: Node input 'safe_name_36' is not a graph
   input, initializer, or output of a previous node`.

`Constant`-node-with-`sparse_value` is also dead: its output tensor `[1,10,5,5]` FLOAT would be
counted as **1000 B of intermediate MEMORY** (node outputs are charged), far worse than the 225
params it saves.

Why dead: no sparse representation survives the checker + sanitize + memory-charging path. The
250-element dense hash kernel is FORCED (its 225 zeros buy avoidance of a 3600 B blue plane;
params are cheap element-count, memory is not). Do not re-attempt sparse initializers / sparse
Constants on this (or likely any) graph under this grader.
