# Transferable insights — D4/K-orientation template-match tasks: the K-channel match map is a forced cost floor, reduced detection empirically fails, and box shape is derivable from the local bounding box

**Source**: task191 (`bo_ng_cld_worker2`, 2026-07-11). LIVE **10,235** (Sho, sha `e6eae9fe…`), a
QLinearConv INT8 match+stamp graph. Re-audited at the 10,235 bar; **WALL confirmed** (no stageable
candidate below it). Generator `task_7df24a62` = D4 variable-template match + box-stamp. Bench
scripts: `Task_by_Task/task191/prototypes/bench_reduced_detection.py`, `bench_exact_localbbox.py`,
`dump_sho_kernels.py`.

## Insight 1 — the K-orientation match map `[1,K,H,W]` U8 is an EXACT-detection floor (do not chase "pack it")
For a task that matches a template in K distinct D4 orientations at every anchor (task191: K=8,
H=W=23 → 4,232 B = 41% of cost), the K-channel match map is **information-theoretically forced** for
byte-exact output. Measured on 40,000 cached truth pairs:
- **Single perfect hash of the window + membership in the K oriented-hashes**: 14.33% of pairs WRONG,
  all false-NEGATIVES. Root cause: each orientation matches a `t_o×w_o` FOOTPRINT and IGNORES cells
  outside it (generator loops `range(tall)×range(wide)` only). A full-window hash is too strict and
  rejects true matches that have stray yellow just outside a footprint. Footprint-specific hashing =
  per-orientation = back to K channels.
- **Single D4-symmetrized ±1 correlation, threshold at peak**: 70.88% WRONG, millions of
  false-POSITIVES (cross-orientation partial correlations reach the match peak).

Why no linear trick works: a window matching "any of K orientations" is an OR of K exact template
matches. A linear conv cannot OR templates; `<K` linear channels cannot isolate K points in
`{0,1}^(kernel)`. Cost model charges the SUM of every intermediate tensor (`neurogolf_utils.py:266`,
`input`/`output` free), Loop/Scan banned (`_EXCLUDED_OP_TYPES`), so the K orientations must all
materialize simultaneously as one `[1,K,H,W]` U8 tensor. Also: an "exact hash" needs enough bits to
be injective (≥ log2(#windows)), which pushes the plane to INT32/multi-U8-channel and costs ≥ the
K-channel map anyway. **Do not re-explore Klein/coset-symmetrized kernels, positional-hash+Equal
membership, or bit-packing to beat this map — all reduce to one of the two failures above.**

This generalizes the existing task191 don't-do ("do NOT bit-pack the 8 orientations") with the
measured failure rates and the footprint-specificity root cause. `floor_est`-style heuristics that
price only "a UINT8 carrier + output canvas" (task191 prompt claimed ~1,200 vs real ~8,835 forced)
systematically UNDER-price K-orientation exact-match tasks — treat such floor hints as wrong here.

## Insight 2 — box/stamp shape == LOCAL bounding box of the matched footprint (orientation label not needed for painting)
When the stamp paints "bbox + 1-cell border of the matched template", and the template touches all
edges of its bbox (task191 generator line 88 guarantees this), then at EVERY match the painted box
shape equals the local yellow bounding-box aspect `(t,w)` — verified **20000/20000**. So the paint
geometry is derivable from local pixels WITHOUT knowing which orientation fired. BUT on task191 this
did NOT lower cost: OR-ing the K-channel map to a single match plane ADDS H·W bytes on top of the
already-forced K-channel map, and stamping the K-channel map directly (as LIVE does) is cheaper.
File this as an algorithm fact that MAY help a task whose stamp shape is otherwise expensive to
carry per-orientation — check whether a single match-plane + local-bbox paint beats a K-channel
stamp before assuming you need per-orientation stamp masks.

## Insight 3 (meta) — the per-tensor charged audit (task005 lens) is worth running even on a K-channel WALL, but calibrate: runtime kernel-bank duplicates are DATA-DEPENDENT, not static
Applied `DONE_2026-07-11T203244Z_…pertensor_audit_breaks_detection_wall` to task191. Dumping the
runtime-built kernel banks (`dump_sho_kernels.py`) showed the 8 match kernels / 8 stamp masks
collapse to 4 (or 2) distinct for SYMMETRIC patterns — tempting to Concat-reuse. **But this is
data-dependent**: asymmetric patterns (32% of the corpus need all 8 distinct orientations) force the
static graph to build all 8. So runtime duplicate-detection does NOT license a static Concat-reuse
shave. Also checked the "shrink 5×5 match kernel → 3×3" lever (patterns are ≤3×3): DEAD because the
INT8 ±1 kernel's support is genuinely 5×5 (pattern not top-left + a full row/col of value-1
reference cells), so 3×3 drops load-bearing weights and misaligns the two chained QLinearConvs.
Lesson: run the audit, but when the "scaffolding" is a RUNTIME-BUILT kernel bank, verify redundancy
is STATIC (holds for all inputs) before pricing a reuse — dump the tensor on an ASYMMETRIC input,
not a symmetric one.

## Insight 4 (ORT fact) — `ReduceSum` rejects UINT8, so the "ones-vector MatMul → ReduceSum" param lever is BLOCKED for U8 projections
Applying the task077 param-lever ("replace an all-ones projection vector with a param-free reduce")
to task191's two blue row/col `QLinearMatMul(s23_u8, ones[23])` projections (46 params) FAILED:
ORT errors `Type Error: Type 'tensor(uint8)' of input parameter (x) of operator (ReduceSum) … is
invalid`. `ReduceSum` (like `AveragePool`, lib note #5) has no UINT8 kernel at any opset. Casting
the U8 plane to I32 first to enable ReduceSum adds a `[1,1,23,23]` I32 = 2,116 B intermediate — far
exceeds the 46-param saving. So on a U8-quantized graph, a `QLinearMatMul`-with-ones projection is
NOT reducible to a param-free `ReduceSum`; the ones vector stays. (The sum itself is safe:
`QLinearMatMul` saturates at 255 but the max encoded row/col sum over 200k pairs was 178 < 256.)
Reshape-sharing the two ones vectors (`onesr[1,23]`↔`onesc[23,1]`) is break-even (U8: 23 params →
23 memory B); for I32 same-value inits (`ar5r`/`ar5c`) it's WORSE (−5 params, +20 memory B).
Corollary for the params-lever sweep: check the tensor DTYPE before pricing a reduce/dedup swap —
U8 blocks ReduceSum, and moving an element from an initializer (1 param) to a computed tensor costs
`itemsize` memory bytes (4× worse for I32).

## Sorter note: ort_reality.md (dtype-support facts, near AveragePool-rejects-U8) — add: `ReduceSum` rejects UINT8 at all opsets; the ones-vector-MatMul→ReduceSum param lever is blocked for U8 projections (Cast-to-I32 tax exceeds the param saving).
## Sorter note: traps.md — extend the task191/K-orientation floor entry (or add near §VISIBLE-FAIL) with Insight 1's measured reduced-detection failure rates (14.3% FN hash / 70.9% FP D4-sum on 40k) and the footprint-specificity root cause; note floor_est under-prices K-orientation exact-match tasks.
## Sorter note: Algorithm_Library/algorithm_patterns.md — add Insight 2 "box shape == local bbox of matched footprint (100%); doesn't reduce cost when a K-channel match map is already forced".
## Sorter note: techniques.md (per-tensor-audit rule from task005) — extend with Insight 3 caveat: runtime-built kernel-bank duplicates are data-dependent; verify static redundancy on an asymmetric input before pricing Concat-reuse.
