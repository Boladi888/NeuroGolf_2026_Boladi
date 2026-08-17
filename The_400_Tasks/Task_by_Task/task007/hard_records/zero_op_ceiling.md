# Dead route — perfect-25 <=1 zero-op ceiling (ZeroopsweepCeilingAudit)

- **Live bar:** cost **78** (manifest, owner Boladi), live sha256 `25e729678c62f5f7472cd86857ada940dd8dde5a6e5a734b79e227ca5ba31089`.
- **Date:** 2026-07-07 · worker `bo_ng_cld_worker1` (Claude lane).
- **I/O contract:** `input [1,10,30,30] FLOAT` (one-hot, 10 colors × 30×30 canvas) → `output [1,10,30,30] FLOAT`.

## Attempted route
Prompt hypothesis: a legal zero-param (`<=1`) direct-output op — Identity, all
Transpose perms, or an attr-only final-output op with no initializer payload —
that reproduces the ARC output at cost `<=1` ("perfect-25 ceiling").

Only genuinely payload-free ops qualify: **Identity** and **Transpose** (perm is
an attribute, no initializer). Flips/rotations need `Slice` initializers
(negative steps) = payload, so they cannot be `<=1` and are excluded by
definition (prompt Dead-ends).

## Exact failed gate
| candidate | score_v2 (macs,mem,params) | visible |
|---|---|---|
| `task007_zop_identity.onnx` (Identity) | (0, 0, 0) | **0/266 FAIL** |
| `task007_zop_transpose_hw.onnx` (Transpose perm=[0,1,3,2]) | (0, 0, 0) | **0/266 FAIL** |

One visible miss = certain hidden 0 (traps §VISIBLE-FAIL). No truth/ab run — dead on visible.

## Why it's dead (forced-component argument — this bars ALL `<=1` routes, not just these two)
Generator `task_05269061.py`: output fills **every** cell with `colors[(r+c)%3]`;
input shows each color **only on the selected diagonals** `diags` (rest black),
and `diags` is **chosen randomly per example**.

1. Input ⊊ output (strict subset of filled cells), so `output != input` → Identity out.
2. Both input and output are transpose-symmetric in (r,c), so `Transpose(input)=input!=output` → Transpose out. Flips/rotations likewise only permute a partially-filled grid and cannot fill the empty diagonals.
3. **General bar:** correctness requires broadcasting each recovered color from its
   *variable* source diagonal to *all* same-residue diagonals. That source index is
   **data-dependent** (it depends on `diags`, which varies per input). Any single
   attr-only permutation op is a **fixed, data-independent** geometric map and
   cannot perform a data-dependent broadcast. Therefore no zero-param permutation
   op can be correct — regardless of which perm/flip/rotation is chosen.
4. A correct route is forced to carry payload: (a) a strided per-residue color
   recovery (`Slice` steps / basis tensor) **and** (b) a re-tiling that broadcasts
   each color across its residue class (basis / index tensor). Both are stored
   payload `> 1`. The minimal known realization folds both into a single
   free-output `Einsum` whose basis tensors ARE that payload = **v11's 78**.

## Verdict
**WALL** on the `<=1` perfect-25 ceiling. It is forced-infeasible, not merely
un-found. The 78 rung (v11 `direct_trig_factored`, analytic floor re-confirmed
2026-06-10 & 2026-06-13) stands as the working best. No candidate below 78 was
produced; nothing staged.

## One remaining hypothesis (for a future batch, NOT the <=1 ceiling)
If anyone revisits sub-78: the only untried lever is whether the free-output
`Einsum` basis `[2,30]` can be expressed with a smaller *counted* footprint via a
representation the scorer charges differently (e.g. a dtype whose 46 zero-pad
entries pack cheaper), without materializing a `>=240 B` intermediate. Prior Tile
factorization (v13) already failed both on cost (240 B intermediate) and
correctness (nonzero outside the 7×7 window). This is a scorer-accounting probe,
not a zero-op route.
