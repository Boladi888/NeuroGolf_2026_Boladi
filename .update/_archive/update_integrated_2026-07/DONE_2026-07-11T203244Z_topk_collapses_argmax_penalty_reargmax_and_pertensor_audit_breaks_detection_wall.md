# Transferable insight — `TopK(x,k=2)` collapses the "ArgMax → penalize-top → ArgMax-again for 2nd place" idiom BYTE-EQUAL; and a per-tensor charged audit can break a "detection irreducible" WALL

**Source**: task005 (`bo_ng_cld_worker1`, 2026-07-11). Audited LIVE **2852** (Aditya, sha
`d38cdf04…`), byte-equal base 2843. **NEW WIN**: byte-equal **2727** (−125 vs LIVE, +0.0448 pts),
staged `task005_topk_src_second_cost2727` (sha `485cd9e5…`). Complements library **#156**
(tiny-K scalar selector: chained `Greater`/`Where` beats `Concat→ArgMax→Gather`) — same
"rank-k selection subgraph is over-built" family, different collapse op (`TopK` here).

## Technique 1 — the two-argmax "second place" idiom is a byte-equal `TopK`
A very common detection pattern for "densest colour + fallback colour":
`source = ArgMax(scores)` → `mask = Equal(ids, source)` → `mask_f = Cast(mask)` →
`penalty = mask_f * BIG_NEG` → `second_scores = scores + penalty` → `second = ArgMax(second_scores)`.
When `BIG_NEG` **fully evicts** the top element (here `-10000`, dwarfs any count), `second` is
exactly the 2nd-largest score. So the ENTIRE block == **`TopK(scores, k=2, largest=1, sorted=1)`**
→ `Split(num_outputs=2)`: `idx[0]=source`, `idx[1]=second`. This DELETES 4 charged intermediates
(`mask` BOOL[N], `mask_f`/`penalty`/`second_scores` F32[N]) + the `ids` and `BIG_NEG` initializers;
adds back only TopK `vals[k]`+`idx[k]` and `k=[2]`. On task005 (N=10): **−125 B** (mem −115,
params −10). Generalises to k>2 (rank-k via `Split`/`Slice` of the TopK idx).

**Byte-equality is exact, not approximate** — two facts to check:
1. The penalty must push already-picked elements strictly below ALL remaining (a "remove", not a
   "nudge"). Verify the constant's magnitude vs the score range.
2. **Tie-break must match.** ORT `TopK(largest=1)` returns equal values lower-index-first; ONNX
   `ArgMax` default (`select_last_index=0`) also picks the lowest index. They agree. If the LIVE
   ArgMax uses `select_last_index=1`, they DON'T — use TopK on negated input or re-check. Validate
   by building the ONNX and diffing vs LIVE on random tie-forcing grids (I used 4000) + `ab --n 2000`.
`TopK` is NOT a banned op (`_EXCLUDED_OP_TYPES` = LOOP/SCAN/NONZERO/UNIQUE/SCRIPT/FUNCTION/COMPRESS).

## Technique 2 (meta) — a per-tensor charged audit breaks "detection irreducible" WALLs
task005 had a documented **WALL at 2852** (2 workers + a reviewer) declaring "detection
irreducible." It was reducible by −125. The WALL reasoned about the BIG tensors (paint/canvas/
GridSample) and the ALGORITHM (validation net-positive) but never enumerated every node-output
against the exact cost model. Do this: `calculate_memory` charges EVERY node-output/value_info
tensor at `elements × itemsize` (summed, not peak), FREE only for tensors literally named
`input`/`output`; `calculate_params` counts initializer/Constant **element count** (dtype-free).
So every ArgMax/Cast/Mul/Equal in a detection block is a charged tensor — enumerate the top-40 with
`largest_intermediate.py` and ask of each: *is this the minimal op set / dtype / element count for
its exact output?* The "selection scaffolding" (masks, penalties, re-scored copies) is where
byte-equal shaves hide, because a WALL author fixates on the forced OUTPUT tensors.

## Dead-route note (bicubic GridSample forbids the argmax→contraction shortcut)
Tempting shave: replace `ArgMax(sampled_onehot, axis=channel)` (INT64, charged) with a cheap
contraction `sampled · [0..9]` (F32, smaller). **DEAD when GridSample mode=bicubic** (or any
non-nearest): the sampled column is NOT a clean one-hot (task005: clean only 1192/2128 cols),
so `argmax != round(channel-contraction)` (169/2128 diverge) → not byte-equal. Only valid if the
sample points land exactly on cell centres AND mode collapses to nearest. Check the GridSample
`mode` attr before pricing this.

## Rule of thumb
When a task's LIVE graph has a **"top-1 + top-2 / densest + fallback"** colour/label selector built
as ArgMax→mask→penalty→ArgMax, try the one-line `TopK(scores,2)→Split` swap and prove byte-equal
(tie-break + full-eviction). And before trusting ANY "detection irreducible / WALL" verdict, run the
per-tensor charged audit — masks/penalties/re-scored [N] copies are frequently collapsible even when
the paint/canvas/sample tensors are genuinely forced.
