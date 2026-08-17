# multi_axis_indexing.md — GatherND, ScatterND, GatherElements, ScatterElements

These four ops handle indexed read/write patterns that don't fit `Gather` / `Scatter` (which operate along a single axis).

For NeuroGolf they replace `Reshape → flatten → Gather → Reshape` patterns and enable per-position lookups directly.

---

## The four-way distinction

| Op | Index meaning | Output shape rule |
|---|---|---|
| `Gather(data, indices, axis=k)` | Pick whole slices along `axis=k` | `data.shape[:k] + indices.shape + data.shape[k+1:]` |
| `GatherElements(data, indices, axis=k)` | Per-position lookup along `axis=k`; indices shape ≡ output shape | `indices.shape` (same as data shape except along axis k) |
| `GatherND(data, indices, batch_dims=0)` | Multi-axis index; each row of `indices` is a coordinate | `indices.shape[:-1] + data.shape[indices.shape[-1]:]` |
| `ScatterElements(data, indices, updates, axis=k)` | Per-position write along `axis=k`; mirrors `GatherElements` | `data.shape` |
| `ScatterND(data, indices, updates)` | Multi-axis write; mirrors `GatherND` | `data.shape` |

### The clarifying example

Take `data = [[1,2,3],[4,5,6]]` (shape `[2,3]`).

**`Gather(data, [1, 0], axis=0)`**: pick rows 1 and 0 → `[[4,5,6],[1,2,3]]` (shape `[2,3]`).

**`GatherElements(data, [[0,1,2],[2,1,0]], axis=1)`**: per-position column pick →
- row 0: pick cols [0,1,2] → [1,2,3]
- row 1: pick cols [2,1,0] → [6,5,4]
→ `[[1,2,3],[6,5,4]]` (shape `[2,3]`).

**`GatherND(data, [[0,1],[1,2]])`**: each row is (i,j) coordinate →
- coord (0,1) → 2
- coord (1,2) → 6
→ `[2, 6]` (shape `[2]`).

The three are not interchangeable. Pick the one whose semantics match your need.

---

## GatherND **[opset 11; batch_dims attr opset 12+]**

```
GatherND(data, indices, batch_dims=0)
data: any tensor of rank R
indices: INT64 tensor of rank Q; last dim length K ≤ R - batch_dims
output: rank Q-1 + (R - K - batch_dims)
```

Each "row" of `indices` (along its last axis) is a K-dimensional coordinate into `data`. Output replaces the indexed coordinates with a slice of the remaining axes.

### Common shapes

- `data: [N, M]`, `indices: [P, 2]` → `output: [P]` (P scalars from P coordinate pairs).
- `data: [N, M]`, `indices: [P, 1]` → `output: [P, M]` (P rows, like `Gather(axis=0)`).
- `data: [1, 10, 30, 30]`, `indices: [P, 4]` → `output: [P]` (P scalars from 4-D coords).
- `data: [1, 10, 30, 30]`, `indices: [P, 2]` → `output: [P, 30, 30]` (P [H,W] planes if you index batch+channel).

### NeuroGolf use cases

**1. Sparse-cell lookup without flattening:**

You have specific (h, w) cells you want from input. Naive: `Reshape([900]) → Gather(flat_indices) → Reshape`. With GatherND:

```python
# Want the 5 cells at positions [(0,5), (3,12), (10,10), (20,25), (29,0)]
indices = np.array([[0,0,0,5],[0,0,3,12],[0,0,10,10],[0,0,20,25],[0,0,29,0]], dtype=np.int64)
# But wait — every NeuroGolf channel needs the same cells, so:
indices = np.array([[0,5],[3,12],[10,10],[20,25],[29,0]], dtype=np.int64)
# data: [1, 10, 30, 30], we want all 10 channels at each of 5 (h,w) coords.
# Reshape data to [10, 30, 30] (drop batch), then GatherND with indices [P, 2] → output [P, 10]... 
# Actually batch_dims=2 lets us index just the last 2 axes per spatial pair. See spec.
```

Cost: indices `[5, 2]` INT64 = 80 bytes. Output `[10, 5]` = 50 F32 = 200 bytes. vs storing a 900-elem mask = 900 bytes.

**Wins when you need ≤ ~50 specific cells. Otherwise stored mask is cheaper.**

**2. `batch_dims=k` for partially-batched indexing:**

If you want different per-channel cells: `data: [1, 10, 30, 30]`, `indices: [1, 10, P, 2]`, `batch_dims=2` → output `[1, 10, P]`. Per-channel sparse lookup with no per-channel index duplication.

### Cost analysis

- Index tensor cost: P × K × 8 bytes (INT64).
- For P × K product up to ~100, GatherND beats a stored 900-byte mask.
- Sparser the access pattern, the more GatherND wins.

### Gotchas

- `indices` is INT64. Float-cast indices fail.
- `batch_dims` is opset 12+ — bump opset_import if you need it.
- ORT supports GatherND but [NEEDS EMPIRICAL] — verify with the exact `batch_dims` you use.
- Output is a flat list of slices. To restore a 2-D spatial shape, you need a subsequent Reshape (which means you knew the shape statically — defeats some of the savings).

---

## ScatterND **[opset 11; reduction attr opset 16+]**

```
ScatterND(data, indices, updates, reduction="none")
attributes:
  reduction: "none" (default), "add", "mul" (opset 16+: also "min", "max")
data: any tensor of rank R
indices: INT64 tensor of rank Q; last dim K ≤ R
updates: shape = indices.shape[:-1] + data.shape[K:]
output: same shape as data
```

Writes `updates` into `data` at the coordinates given by `indices`. With `reduction="add"`, accumulates instead of overwriting.

### Already in use

task118 used bool ScatterND (-38,341 bytes). The reduction modes (especially `"add"`) are underexplored.

### NeuroGolf use cases (reduction modes)

**1. Sparse stamping with accumulation:**

Place 10 small patterns at different cells, accumulating where they overlap:

```python
# data: zero canvas [1, 10, 30, 30]
# indices: [N, 4] coordinates for N stamps
# updates: [N] (or [N, ...] for non-scalar stamps)
ScatterND(zero_canvas, stamp_coords, stamp_values, reduction="add")
```

Avoids building a separate sum-of-stamps pipeline.

**2. Sparse mask construction:**

Set specific cells to True without storing a full mask:

```python
zero_bool = ConstantOfShape([1, 10, 30, 30], value=False)  # ~33 bytes
true_scalar = ConstantOfShape([N], value=True)  # ~17 bytes
mask = ScatterND(zero_bool, coords, true_scalar, reduction="none")
# Total stored: coords (N×4×8 bytes) + ~50 bytes overhead
# vs storing the full BOOL mask: 9000 bytes
# Wins for N < ~280
```

### Gotchas

- `reduction="add"`/`"mul"` are opset 16+. NeuroGolf typically uses opset 10–13 — bump if needed.
- [NEEDS EMPIRICAL] Verify reduction modes work under ORT_DISABLE_ALL.
- `updates` shape MUST be `indices.shape[:-1] + data.shape[K:]`. Mismatch fails at load.

---

## GatherElements / ScatterElements **[opset 11]**

```
GatherElements(data, indices, axis=0)
data: any tensor
indices: INT64, same RANK as data, same shape except possibly along `axis`
output: same shape as indices
```

Per-position lookup. Element-by-element along `axis`, the index tells which position in `data` to pick.

### Distinction from Gather

`Gather(data, indices, axis=k)` picks WHOLE slices using `indices`. Output rank can differ from input.

`GatherElements(data, indices, axis=k)` picks per-position elements. Output rank = input rank.

### NeuroGolf use case

**Per-row permutations:**

You have data `[10, 30]` and want to permute the 30 columns differently in each row. Stored permutation: 300 INT64 = 2400 bytes. Output: 1200 F32 = 4800 bytes.

```python
# permutation indices [10, 30] - each row has its own column ordering
out = GatherElements(data, permutation_indices, axis=1)
```

Without GatherElements you'd need to OneHot the indices and MatMul — much more expensive.

### Gotchas

- `indices` rank MUST equal `data` rank. Common error: forgetting to broadcast indices to full rank.
- Indices along non-`axis` dimensions are interpreted positionally (must match data shape).
- ORT supports GatherElements but [NEEDS EMPIRICAL] — verify dtypes.

---

## ScatterElements **[opset 11; reduction opset 16+]**

Mirror of GatherElements. Writes per-position.

```
ScatterElements(data, indices, updates, axis=0, reduction="none")
```

Used in task387 (see `.update/DONE_2026-05-23T031500Z_task387_colorgrid_scatterelements.md`).

Same reduction modes as ScatterND. Same opset 16+ gotcha for `"add"`/`"mul"`/`"min"`/`"max"`.

---

## Comparison cheat sheet

```
Need to pick whole rows/cols from one axis      → Gather
Need to pick per-position values along one axis → GatherElements
Need to pick scalars from multi-axis coords     → GatherND
Need to write to whole rows/cols                → Scatter (opset 11 deprecated for ScatterElements)
Need per-position writes along one axis         → ScatterElements
Need multi-axis sparse writes                   → ScatterND
```

---

## Cross-references

- `exemplars/08_gather_lookup/` — the existing Gather-family exemplars.
- `.update/DONE_2026-05-23T031500Z_task387_colorgrid_scatterelements.md` — task387 used ScatterElements.
- `exemplars/01_whole_graph_rewrites/task382_cumsum_gather.md` — CumSum + Gather pattern that might be replaceable by GatherND in some cases.

## TODO — verify empirically

1. ORT support matrix for each: data dtypes accepted by each Gather/Scatter variant under ORT_DISABLE_ALL.
2. `reduction="add"`/`"mul"`/`"min"`/`"max"` on ScatterND/ScatterElements at opset 16+.
3. `batch_dims` parameter on GatherND.
4. MAC cost per `onnx-tool` — all four are zero-MAC index operations.
