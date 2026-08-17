# task326 — WALL: perfect-25 ≤1 disproven; cost 30 is the forced floor

**Live bar**: cost **30** (manifest, authoritative), owner Aditya, sha `5e386d31e01e3f7cbf3fd025115ebf3d82c20522e73941921053e5cb8bb1778c`.
`ng_current_bar --task 326` → live_cost 30. (task326.md AUTO `STATUS: 166` is STALE — pre-dates Aditya's cost-30 Einsum; do not trust it.)

**Rule (verified 266/266 arc-gen + 3 train + 1 test)**: `output = input[0:2, 0:2]` — a fixed
top-left 2×2 crop. Output grid is ALWAYS 2×2. Encoded target `[1,10,30,30]` is one-hot in the
top-left 2×2 and **exactly zero** everywhere else.

## Dead routes (do not re-explore)

1. **Perfect-25 zero-param family (Identity / Transpose / Flip / Rotate) — DEAD.**
   Programmatic sweep on encoded tensors, all `0/266`:
   identity, transpose, flipud, fliplr, rot90, rot180, rot270 → each 0/266.
   `ng_gate visible` on Identity onnx: **0/266 FAIL**.
   *Why forced*: inputs are up to 12×12 with content OUTSIDE the 2×2, so the input tensor is
   nonzero where the target is zero. The grader compares the FULL `[1,10,30,30]` with
   `np.array_equal` (no bbox/trim — see `tools/ng_harness/modes.py:mode_visible` +
   `encode.encode_grid`). No value-preserving whole-canvas op can zero the region outside 2×2.
   A multiplicative spatial gate is FORCED. The "perfect 25 / cost≤1" premise in the prompt is wrong.

2. **Sparse-initializer cost-golf — DEAD (rejected by the real scorer).**
   `calculate_params` counts a sparse initializer by `prod(values.dims)` = NNZ only, so a sparse
   `[30]` mask with 2 nonzeros would price **params=2 → cost 2**. BUT `calculate_memory`'s FIRST line
   is `onnx.checker.check_model(model, full_check=True)`, which raises on a sparse tensor feeding an op:
   `Mul`  → `B typestr: T, has unsupported type: sparse_tensor(float)`;
   `Einsum` → `Rank of input 1 (0) does not match the equation indices (1)`.
   → `score_network` returns `None` (INVALID submission). ORT *runs* sparse inits fine, but the
   scorer's full_check kills them. Confirmed: `_score_v2` on the sparse candidate → `None`.
   (Prototype kept: `prototypes/task326_sparse.onnx`.) **This is general — no task can cost-golf via sparse inits.**

3. **Slice[0:2,0:2] + Pad — DEAD (worse: 174).**
   The minimal materialized crop `[1,10,2,2]` F32 is a charged intermediate = **160 B memory**;
   + 14 params (Slice starts/ends/axes + Pad pads) = **cost 174** ≫ 30. Measured. Memory alone (160) already loses.

4. **Generated mask (Constant[2] + Pad→[30])** — the padded `[30]` F32 vector is a charged
   intermediate = 120 B memory → cost ≥124. Loses. INT8 seed forces a Cast (F32 [30] intermediate, 120 B) or casts input (destroys one-hot). Loses.

## Forced-floor argument for cost 30

Any correct net computes `X ⊙ M`, `M[r,c]=1 iff r<2 and c<2` (rank-1 separable, `M=v vᵀ`, `v=[1,1,0×28]`, length 30 — length fixed by the 30-wide canvas axis for the Einsum contraction). Cheapest legal carrier of `v`:
- dense init `[30]` = **30 params, 0 memory** (team-best `Einsum nchw,h,w->nchw`, ONE reused mask).
- sparse init (2) → invalid (route 2). generated/sliced (route 3/4) → ≥120 B memory.

⇒ **30 params / 0 memory / cost 30 is the forced floor. Team-best already sits on it. WALL — no sub-30 legal formulation exists.**

`profile_real_cost --task 326 --cand Team_Best_Onnx/task326.onnx --arcgen 266` → **static=30 profile=30** (memory 0 + params 30). No credit-gap.

## One remaining hypothesis (only if the cost model changes)
If `check_model(full_check=True)` were ever dropped from `calculate_memory` (or sparse tensors gained
op-input support in the onnx checker), the sparse-`[30]`-mask Einsum would price **cost 2**. Re-test route 2
before closing if the scorer's sanitize/check path changes.
