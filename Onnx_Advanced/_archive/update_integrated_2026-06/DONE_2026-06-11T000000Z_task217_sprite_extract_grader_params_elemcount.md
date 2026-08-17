# task217 rework — two transferable findings (Worker-2b, 2026-06-11)

Context: REWORK of the task217 "fractal2" dissolve candidate. Drove cost
2668 -> 2178 (-490B) applying two named levers. Builder:
`The_400_Tasks/Task_by_Task/task217/prototypes/build_task217_dissolve_u8_v5.py`.
Staged: `staged_submits/task217_dissolve_int32sprite_cost2178/`. All gates byte-equal
(visible 266/266, ab --full 200266/200266, truth --full 200000/200000 100%,
adversarial 4788/4788).

## Finding 1 (the one worth promoting): grader counts `params` by ELEMENT COUNT, not bytes

This contradicts the natural assumption (and several manifests, incl. task217's own
prior manifest which lists init "bytes"). Measured directly:

- Switching the two kron Gather index initializers `block_idx`/`tile_idx` from
  INT64[9] to INT32[9] changed my static raw-byte dump from 371B -> 299B of init
  bytes, but the grader's reported `params` stayed at **66, unchanged**, and total
  cost stayed identical.
- Conclusion: `score_network`'s `params` term = sum of initializer ELEMENT counts
  (dtype-agnostic). Casting an INITIALIZER to a narrower int dtype does NOT lower
  params cost. (It is still harmless, and the worker SHOULD still narrow them for
  correctness/clarity, but do not budget a saving for it.)
- Where the INT32 lever DOES pay: **intermediate** tensors (the `memory` term, which
  is byte-accurate, max(static,runtime) per tensor). I64[9]=72B -> I32[3]=12B on the
  4 shift-index intermediates was a real `memory` saving.

So the doctrine refinement: **"INT32 indices save bytes" applies to INTERMEDIATE
index tensors (memory term, byte-priced), NOT to index INITIALIZERS (params term,
element-priced).** techniques.md #74 / ort_reality.md note that Gather accepts INT32
indices; they should add this params-vs-memory caveat so workers don't over-claim a
saving on initializer dtype narrowing. I did not find this distinction stated in
ort_reality.md, scoring_internals.md, or cost_floor_reference.md — please confirm and
add it (a quick grep of scoring_internals.md "max(static,runtime)" mechanism block is
the right home; it already explains the memory term but not the params element-count
rule).

## Finding 2: sprite-extract collapses a kron/fractal expansion chain

For an outer-product / kron self-product fractal `out[3R+r,3C+c]=sprite[R,C] AND
sprite[r,c]`, the shift Gathers that locate the sprite only ever read its small KxK
top-left. So build the [1,1,K,K] sprite ONCE (K*K bytes) and expand to the full grid
via two block/tile index Gathers (block_idx = i//K, tile_idx = i%K) which stay in
0..K-1 and therefore index validly into the KxK sprite. This shrank every
pre-expansion BOOL carrier from [1,1,9,9]=81B toward 9-27B; only the two final 9x9
kron operands + the AND result are forced full-grid. This composes with techniques.md
#82's outer-product fractal kit and #80 (D2S block stamp) — worth a one-line cross-ref
that "carry the SPRITE, not the expanded plane; expand last."

## Floor note for task217
2178 is the dissolve floor for this formulation: forced = pre_pad 810 (10-ch output
one-hot) + ch0_crop 324 (F32 input touch) + two F16 [1,1,9,9] detector copies 324 +
three forced-9x9 BOOL kron operands 243. Killing one F16 detector copy re-introduces
an F32 [1,1,9,9]=324 or extra ReduceMin nodes that cost more than the 162 saved.
