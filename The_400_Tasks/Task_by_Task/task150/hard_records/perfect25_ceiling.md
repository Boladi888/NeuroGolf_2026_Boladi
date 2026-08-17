# task150 dead route — perfect-25 (cost≤1) ceiling  [WALL]

**Date:** 2026-07-07 · worker `bo_ng_cld_worker2` (Claude lane) · HflipCeilingAudit
**Live bar at audit:** cost **158**, sha256 `f7b9a9bc8ebf4522236b117321fa49c34a483e02e6aa789f7c437a46039ecdf3`
(manifest, owner Aditya; `Team_Best_Onnx/task150.onnx`; static=profile=158 via
`profile_real_cost.py --task 150 --arcgen 266`).

## Rule
Horizontal flip within the active square region: `out[r,c]=in[r,N-1-c]` for `c<N`,
padding (`c≥N`) stays zero. N∈{3..9} varies per input (ARC-GEN `size=randint(3,9)`).

## Live 158 graph (the reachable floor)
`ReduceL2(input)→Cast(INT32)→Sub(N,arange[1..30])→Gather(axis=3)`.
Charge: `mirror_indices[30]INT32 = 120 B` + `size_f[]F32 4` + `size_i32[]INT32 4`
= 128 mem + `arange[30] = 30 params` = **158**. (ReduceL2 fuses ReduceSum→Sqrt,
so N is one op ⇒ only 2 scalars, vs fliph_v2's 3 → that is the 162→158 delta.)

## Attempted route (prompt target): cost≤1 direct-output perfect-25
The only near-zero-payload way to "reverse columns" is a **full-axis Slice, step=-1**
(the symmetry_and_affine H-flip winner). Built as `prototypes/probe_slice_reverse.onnx`.

## Exact failed gate
- `_score_v2`: memory=0 params=4 **cost=4** (points 23.6137) — would be near-perfect-25.
- **`ng_gate visible --task 150` → `0/266 (0.00%) FAIL`.** Certain hidden 0 (traps §VISIBLE-FAIL).
- Self-test vs correct flip-within-region: 0/500 byte-exact.

## Why it's dead (forced-component argument)
1. The active N-block is **top-left-anchored** (cols [0,N-1]); a full-axis reverse
   relocates it to cols [30-N, 29] — wrong position, padding semantics broken.
2. Keeping the block at [0,N-1] forces the map `c→N-1-c` for `c<N`: an **N-dependent
   permutation**. N is a runtime property ⇒ the permutation cannot be a static
   initializer ⇒ it must be a computed `[30]` index intermediate.
3. Gather (only free-output permutation op) requires INT32/INT64 index; INT8/UINT8/INT16
   **rejected at graph-validation** (re-confirmed `probe_gather_idxdtype.py`). INT32×[30] = **120 B forced**
   (output width fixed 30 ⇒ index length 30 forced).
4. The `-c` ramp base `arange[1..30]` = **30 params forced** (Range loses: [30] output charged + dyn-start).
5. Building N = 1 Reduce scalar (F32, forced from F32 input) + 1 Cast scalar (INT32 for the Sub) = **8 B forced**.
6. **ReverseSequence** (the only variable-length in-region reverse primitive) — ORT kernel
   asserts `time_axis<2` (`reverse_sequence.h:21`); our column axis is 3. Reaching it needs a
   36,000 B reshape/transpose AND a `[batch]=[300]` INT32 sequence_lens (1,200 B). Strictly worse.
   Re-confirmed dead (`probe_reverseseq.py`).

⇒ **Floor = 120 (index) + 30 (arange) + 8 (scalars) = 158.** A cost≤1 graph would need a
zero-payload variable-in-region reverse op; none exists in the ORT/opset≤18 kernel set.

## Remaining hypothesis
None credible. 158 is the proven single-Gather floor and is already live. Any tie (rebuild of
the ReduceL2 form) scores 0 by dedup and must not be staged. **WALL — no sub-158 candidate.**
