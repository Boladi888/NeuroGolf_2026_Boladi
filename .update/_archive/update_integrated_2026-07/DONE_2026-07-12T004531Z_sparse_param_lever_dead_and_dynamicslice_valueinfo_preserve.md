# Transferable insight — the params lever has a HARD dead-end (sparse-initializer / Constant-sparse to cut a zero-heavy Conv kernel is rejected by checker + sanitize + memory-charging), AND a graph with a data-dependent Slice must PRESERVE its value_info or scoring returns None

**Source**: task182 (`bo_ng_cld_worker1`, Claude lane, 2026-07-12). Live team-best **4,460**
(Boladi, sha `b2c1867a…`, conv shape-hash paradigm). Staged **4,425** PATH_A_BYTE_EQUAL
(ab --n 2000 2267/2267 byte_equal=True; visible 267/267; truth 2000/2000; profile static==profile==4,425;
ng_safety WARN = inherited QLinearConv only). A real −35 win via the params/coord lever after the two
dominant tensors (F32 16×16 shape-hash 1024 + BOOL 30×30 output mask 900) were proven forced.

Sibling of `DONE_2026-07-11T221500Z_valueinfo_orphan…` (params lever exists on memory-floor tasks) and
`DONE_2026-07-10T202012Z_argmax_indexplane_is_int64_costtrap…` (ArgMax output is INT64). This packet adds
the **failure modes** of the params lever those two packets invite you to pull.

## 1. Sparse-initializer / Constant(sparse_value) to cut a zero-heavy kernel is DEAD under this grader
The bait (real and tempting): `calculate_params` counts `sparse_initializer` AND `Constant`'s
`sparse_value`/`value_ints`/… by `math.prod(values.dims)` = **nonzero count**. A hash/select Conv weight
like `[1,10,5,5]` with 9 zero channels is **250 params but 25 nonzero** — sparse would score 25 (−225).
Three independent kills, all verified on task182:
- **onnx.checker rejects it**: `check_model(full_check=True)` (which `calculate_memory` runs first, L196)
  → `[ShapeInferenceError] (Conv): W … has unsupported type: sparse_tensor(float)`. ONNX Conv/MatMul do
  not accept a sparse tensor operand. So the scorer returns None before it ever counts params.
- **`sanitize_model` breaks the name link**: it renames graph.initializer + all node inputs via
  `get_safe_name`, but **never iterates `graph.sparse_initializer`**. The Conv input `v_w12`→`safe_name_N`
  no longer resolves to the (un-renamed) sparse init → `InvalidArgument: Node input 'safe_name_N' is not a
  graph input, initializer, or output of a previous node` at session load. (The real Kaggle grader uses the
  same `sanitize_model`, so this is not just a local-tool artifact.)
- **`Constant(sparse_value)` is self-defeating**: its OUTPUT tensor is a node output, so `calculate_memory`
  charges the DENSE materialized size as intermediate MEMORY (`[1,10,5,5]` F32 = 1000 B) — far more than
  the 225 params it saves.
**Rule of thumb:** the params lever is real (dedup/dtype-free element count), but you cannot cheat the
element count with sparsity. Zero-heavy kernel channels that exist to consume a wide input directly (and
avoid materializing a big sliced plane) are FORCED at full element count. Don't spend a probe on sparse.

## 2. A data-dependent Slice forces you to PRESERVE value_info (mirror of the orphan-prune rule)
`calculate_memory` re-runs `onnx.shape_inference.infer_shapes(model, strict_mode=True)` and returns None
if ANY tensor has `dim_value <= 0` or a `dim_param`. A **dynamic crop** — `Slice(input, starts, ends)`
where starts/ends are computed from `ArgMax` (frame localization, object crop, etc.) — has an output shape
that strict inference CANNOT derive from scratch (it yields `dim_param unk__k` for the slice output and
everything downstream → score None). The live graph only scores because its **stored value_info carries the
concrete `[1,2,5,5]`**, and strict inference PRESERVES existing concrete dims rather than overwriting them.
Therefore, when you surgically edit such a graph, **do NOT `del g.value_info[:]` and re-infer** (the classic
cleanup): patch only the changed tensors' dtype/shape in place and ADD value_info for any new tensors.
This is the exact mirror of the orphan-prune rule in `DONE_…221500Z`: after node DELETION prune orphans,
but after a dtype/attr edit on a dynamic-Slice graph you must KEEP the stored dynamic-shape value_info.
Verify with `check_model(full_check=True)` then `profile_real_cost.py` (None ⇒ you stripped a needed shape).

## 3. Bonus floor note: "TopK/compaction" prompt framing ≠ a compaction-able task
task182's prompt titled it "TopK compaction, floor_est ~1200". The actual rule is a framed-shape-match
RECOLOR (recolor every blue sprite congruent to a framed key). A static ONNX graph **cannot emit a
variable-length anchor list** (no true compaction/gather-of-K), so it must render a dense 30×30 recolor
mask (900 B forced) and compute a per-position exact-match score (F32 16×16 shape-hash 1024 B forced —
exact match on the FREE F32 one-hot input via Conv; a UINT8/QLinearConv match needs a ≥3600 B blue-plane
Einsum). Forced 1924 alone ⇒ ~1200 is unreachable. When a SWEEP/STRUCTURAL prompt's `floor_est` assumes a
selection/compaction shape, first check whether the OUTPUT is dense (per-pixel edit) — if so the "compact
K winners" framing is a mis-model and the output mask + per-position score set the real floor.
