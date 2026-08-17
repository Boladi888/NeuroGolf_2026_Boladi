# task165 — sparse literal-table params are scorer-dead

- Live/team-best bar: **1,641** (`memory=1,538`, `params=103`), SHA-256 `6faf9093ffe11c0fbe8583054205b9ee4ddc36b6f7c4d6c31222055a8d5b43a9`.
- Bar source: `python tools/ng_current_bar.py --task 165 --json` → authoritative manifest cost 1,641.
- Attempted route: convert zero-heavy initializers to sparse tensors: `row_powers` dense 30 → 18 nnz, `rowcap_u8` dense 30 → 17 nnz, `nonbg_selector` dense 10 → 9 nnz.
- Prototype: `The_400_Tasks/Task_by_Task/task165/prototypes/task165_sparse_main_tables.onnx`, SHA-256 `1e1a3915a06587b6c7d3c33201fd64c58e85ff442d184a5c81de7c824eb80346`.
- Exact failed gates:
  - `row_powers` sparse checker: `Einsum ... Rank of input 2 (0) does not match the equation indices (2)`.
  - `rowcap_u8` sparse checker: `Greater ... unsupported type: sparse_tensor(uint8)`.
  - visible on combined sparse prototype: `[visible] ERROR: Invalid model. Node input ... is not a graph input, initializer, or output of a previous node.`
- Why dead: sparse initializers are not legal dense op inputs for this scorer/shape-inference path; the sanitizer also does not preserve sparse initializer references. This matches `traps.md` sparse-op warning.
- One remaining hypothesis: only revisit sparse if using an op/schema that explicitly consumes sparse tensors and survives `score_network_v2`; `Einsum`/`Greater`/current literal-table sparse conversion is dead.
