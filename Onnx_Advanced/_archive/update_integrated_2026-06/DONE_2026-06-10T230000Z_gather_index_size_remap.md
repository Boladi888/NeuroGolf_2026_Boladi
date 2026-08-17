# Gather-index size-conditional remap (task298, 2026-06-10)

## What was discovered

For concentric-ring remap tasks where the spatial mapping depends on object size (6×6 vs 8×8), the cheapest architecture avoids the `And/Or` boolean mask cascade entirely. Use a precomputed INT32 flat index array (one per size variant) and `Where(is_size, idx_A, idx_B)` to select the active index, then `Gather` from a 4-element pixel stack.

**Architecture**: `Gather(px_stack [1,10,1,4], src_flat [64] INT32, axis=3) → [1,10,1,64] F16 → Reshape [1,10,8,8]`

Where:
- `px_stack` = Concat([px0_f16, px1_f16, px2_f16, zero_f16], axis=3) — 4 source vectors
- `src_flat` = Where([1] BOOL, idx_6 [64] INT32, idx_8 [64] INT32) — size-selected index
- `idx_*` = precomputed table: 0=px0, 1=px1, 2=px2, 3=zero for each of 64 cells

**Size detection**: check corner cell `inner[0,:,0,7]` — non-zero iff 8×8 object. `ReduceMax(corner_cell, axes=[1,2,3]) → Cast(BOOL)`. Far cheaper than Conv all-ones approach.

**Why this beats the boolean mask cascade**:
- Old approach (v3): 6×64B BOOL masks + 6×And intermediates + 3×Or intermediates + 3×Where F16 = 576 + 3840 = 4416B memory, 384 BOOL params
- New approach (v5): 2×64 INT32 indices (128 params) + src_flat 256B + px_stack 80B + Gather 1280B + Reshape 1280B = 2896B memory, 128 INT32 params
- Net: save ~1520B memory + 256 params

**Important correctness note**: For concentric-ring tasks, perm = [ring_k → source_idx] maps ring indices to source diagonal pixels. For both 6×6 and 8×8, the perm is IDENTICAL ([2,0,1,2]). The ONLY difference is that 6×6 has no ring3 and cells OUTSIDE [0:6,0:6] are all-zeros. Ring3 of 8×8 (center 4 cells at {(3,3),(3,4),(4,3),(4,4)}) maps to perm[3]=2 (px2), but the same cells in 6×6 are ring2 → perm[2]=1 (px1). Do NOT assume "outermost→background=zero" — the outermost ring DOES use the standard perm (ring0→perm[0]=2=px2).

## Where encountered

- Task: 298 (ring remap, ≤8×8 nested squares)
- Build script: `The_400_Tasks/Task_by_Task/task298/prototypes/build_task298_ring_remap_v5.py`
- Staged: `staged_submits/task298_ring_remap_gather_cost5850/`
- Cost: 5850 (vs LIVE 11290) — 48% reduction. 100% visible + 100% truth (200k).

## What's missing from the library

§108 (concentric ring collapse) describes the Chebyshev predicate approach but doesn't mention the `Where(BOOL, idx_A, idx_B) → Gather` pattern for size-conditional remaps. This pattern generalizes to any task where:
1. The output mapping is a SIZE-DEPENDENT Gather from a small set of source vectors
2. Two static index tables can be precomputed (one per size variant)
3. A single corner/marker cell can detect the size

The key trick: `Where(BOOL, INT32_A [N], INT32_B [N])` selects between two Gather index arrays in ONE op. The result feeds directly into `Gather(stacked_sources, indices, axis)` without materializing intermediate [1,10,8,8] tensors.

Also worth noting: INT32 indices (4 bytes) vs INT64 (8 bytes) halves the src_flat intermediate memory (256B vs 512B). Gather supports INT32 indices per ort_reality.md.
