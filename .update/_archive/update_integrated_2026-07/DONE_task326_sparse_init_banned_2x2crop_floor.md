# Transferable insight — sparse initializers CANNOT cost-golf params (scorer's full_check rejects them); + fixed-crop floor is a length-N dense zeroing mask

**Source**: task326 batch 53 (`bo_ng_cld_worker3`, 2026-07-07). LIVE 30 (Aditya, sha `5e386d31…1778c`) confirmed at forced floor; no improvement. Perfect-25 rumor disproven.

**Task**: `output = input[0:2,0:2]` — fixed top-left 2×2 crop (output always 2×2; 266/266 arc-gen). Encoded `[1,10,30,30]` target is one-hot in top-left 2×2, exactly zero elsewhere.

## The BIG reusable finding: sparse-initializer param-golf is DEAD

The cost model has a tempting hole and a hard gate that closes it:
- **`calculate_params` counts a sparse initializer by NNZ only** — `math.prod(sparse_init.values.dims)` = number of nonzeros, ignoring the (large) dense shape and the index tensor. A dense `[30]` mask `[1,1,0×28]` = 30 params; the *same* tensor as a sparse init (2 nonzeros) = **2 params**. Looks like free 15× savings.
- **BUT `calculate_memory`'s FIRST line is `onnx.checker.check_model(model, full_check=True)`**, which does shape inference and **rejects a sparse tensor feeding any op**:
  - `Mul`  → `(op_type:Mul): B typestr: T, has unsupported type: sparse_tensor(float)`
  - `Einsum` → `(op_type:Einsum): Rank of input 1 (0) does not match the equation indices` (sparse shape not propagated).
  - → `calculate_memory` raises → `score_network` returns `None` → **INVALID submission** (cost fails, not just unchanged).
- **ORT itself runs sparse initializers fine** (materializes to dense at load) — so a naive local ORT test *passes* and looks shippable. The kill is purely in the scorer's `check_model(full_check=True)` path. Measured: `_score_v2` on the sparse candidate → `None`.

**Rule of thumb**: **never** try to cut params with `sparse_initializer` — it doesn't survive `score_network`. Applies to EVERY task, not just crops. (Re-open only if the scorer ever drops `full_check` or onnx adds sparse op-input support.)

## Fixed-rectangle-crop floor (forced-component argument)

For a fixed crop to a top-left `k×k` on an `N×N` (=30) canvas where the target is one-hot inside and **exactly zero** outside:
- A **multiplicative spatial gate is FORCED** — inputs have content outside the crop (up to 12×12 here), and the grader compares the full `[1,10,30,30]` with `np.array_equal` (no bbox/trim). Identity/Transpose/Flip/Rotate all give `0/266` (nonzero where target is zero).
- The gate `M[r,c]=1 iff r<k and c<k` is rank-1 separable, `M=v vᵀ`, `v=[1×k,0×(N-k)]`, **length N** (fixed by the Einsum contraction with the N-wide axis).
- Cheapest legal carrier of `v`: a **dense `[N]` init = N params, 0 memory**, via `Einsum("nchw,h,w->nchw", X, v, v)` (ONE reused mask → N params, not 2N). Team-best = exactly this at **cost N=30**.
- **Everything else loses**: sparse (invalid, above); `Slice[0:k,0:k]+Pad` charges the `[1,10,k,k]` F32 crop intermediate (160 B here, cost 174); generating `v` (Constant[k]+Pad→[N]) charges the `[N]` F32 intermediate (120 B). Charge asymmetry (same as DONE_task262): **params = element count (dtype-free); memory = bytes** — a length-30 F32 intermediate (120 B) beats 30 params only downward, never up.

⇒ **cost = N (canvas width) is the forced floor for a fixed top-left k×k crop.** Don't grind sub-N.

**Applies to**: Cropping / fixed-region-select tasks (Object_Based; 92 similar to task326). Catalog #119 free-output Einsum micro-floors. Guard pairs with DONE_task262 (single zero-intermediate Einsum > Slice-to-shrink-params when output is full-canvas) and DONE_task129 (always render full-canvas; compact output → visible 0).
