# task382 — GatherElements broadcast-index cost trap (2026-06-11)

## What was discovered

Attempted to apply #93 scalar-select to task382 v22 (cost 6,362). The approach:
replace the existing two-step `shift_bank_i32 [10,20] Gather → [10,20] bank → Gather → [20,20]`
with a single GatherElements using a broadcast index map:
`col_iota[1,20] - Reshape(shift, [20,1]) + SHIFT_PAD → [20,20] INT32 → Gather([28] padded edge) → [20,20] UINT8`.

Result: cost went from 6,362 to **11,031** (+73% worse).

## Why it's bad

The [20,20] INT32 index tensor costs 4 bytes/element × 400 elements = **1,600B** as a charged
intermediate. The eliminated shift_bank_i32 only saved 200 params (200B in param accounting,
dtype-blind) plus the [10,20] UINT8 bank intermediate = 200B runtime memory. Net delta: +1,400B.

Additionally, the Transpose+Where for h/v orientation (each [20,20] = 400B) cannot be eliminated
unless you pre-select the index orientation — but doing so requires EITHER:
- Two [20,20] INT32 index maps (one h, one v) + Where = 3×1,600B = 4,800B, OR
- One [20,20] INT32 index map + one Transpose = 2×1,600B = 3,200B

Both are dramatically worse than the current v22 structure.

## The deeper principle

**#93 GatherElements with broadcast index is ONLY beneficial when the index tensor is SMALLER
than the data it replaces.** For INT32 indices:
- INT32 costs 4B/element
- UINT8 data costs 1B/element

So a [N,N] INT32 index costs 4× more than a [N,N] UINT8 result. For N=20: 1,600B vs 400B.

The technique works well when:
1. The output is assembled from a FEW cells (scalars), so the index is tiny (task253)
2. The index can be expressed in BOOL (1B/elem) — but GatherElements requires INT32/INT64
3. N is small enough that 4× overhead is acceptable

For N=20 with a full [20,20] output, the broadcast index approach is ALWAYS worse than the
current two-step bank approach (which uses UINT8 intermediate = 1B/elem).

## #93 APPLICABILITY BOUNDARY EXTENSION

The existing `techniques.md §93` APPLICABILITY BOUNDARY note says:
> "#93-select pays ONLY when each branch result is small (scalars / a few cells, assembled AFTER
> the select — as in task253)"

This should be extended with the **index cost law**:
> GatherElements with a dynamically-computed [N,M] INT32 index map costs 4N×M bytes —
> this dominates the savings from eliminating an [N,M] UINT8/BOOL data tensor whenever
> N×M > ~50 elements (INT32:UINT8 = 4:1 cost ratio). For task382 N×M = 400: always worse.
> Use GatherElements only when the index can be precomputed (static initializer) or is
> very small (< ~50 elements).

## Where encountered

- task382, v23 builder: `The_400_Tasks/Task_by_Task/task382/prototypes/build_task382_crop20/build_task382_where_v23_scalarselect.py`
- Empirical cost measurement: static=11,031 (memory=10,836, params=195) vs v22 6,362

## What's in the library

`techniques.md §93` has a boundary note but doesn't quantify the INT32 index cost law.
`ort_reality.md` mentions GatherElements works at INT32 but doesn't price it.

## Sorter note: techniques.md §93 APPLICABILITY BOUNDARY — extend with "index cost law" paragraph from above
