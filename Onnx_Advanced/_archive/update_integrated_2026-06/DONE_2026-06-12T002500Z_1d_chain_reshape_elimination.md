# 1D chain shape trick: eliminate [1,1,N] → [N] reshape by making reshape target [N] from the start

## What you discovered

When a graph has a processing chain that starts with `Reshape([1,1,H,W] → [1,1,N])` and later ends with `Reshape([1,1,K] → [K])` (where K < N, after Gather), you can eliminate the second Reshape and its intermediate by changing the first Reshape target from `[1,1,N]` to `[N]` (1D). All subsequent operations naturally produce 1D tensors, and any Gather(data[N], indices[K]) directly produces [K] without needing an extra reshape.

**Concrete example (task264 v19 → v20):**
- Before: `Reshape(pool16[1,1,5,5], to_flat=[1,1,25]) → combined[1,1,25]` → `TopK(axis=2)` → ops produce `[1,1,9]` tensors → `Gather(bins[1,1,9], Bd_idx[8], axis=2) → colors_raw3[1,1,8]` → `Reshape(to_8=[8]) → colors_rawf[8]`
- After: `Reshape(pool16[1,1,5,5], to_25=[25]) → combined[25]` → `TopK(axis=0)` → ops produce `[9]` tensors → `Gather(bins[9], Bd_idx[8], axis=0) → colors_rawf[8]` (direct, no reshape needed)

**Savings on task264:**
- Eliminated: `colors_raw3 F16 [1,1,8] = 16 bytes` (intermediate)
- Eliminated: `to_8 INT64 [1] = 1 param` (Reshape initializer)
- Reduced: `to_flat INT64 [3] → to_25 INT64 [1]` = 2 params saved
- Net: -16 bytes memory, -3 params = **-19 total cost** (2312 → 2293)

## Where you encountered it

- Task: task264
- Builder: `The_400_Tasks/Task_by_Task/task264/prototypes/build_task264_v20.py`
- Staged: `staged_submits/task264_v20_1d_chain_cost2293/`

## Why it matters

The pattern `[1,1,N] → Reshape → [K] (K < N)` at the end of a chain always has a wasted intermediate. If the chain starts by reshaping a pooled tensor to [1,1,N], changing that to [N] eliminates the entire 3D tensor lineage and avoids the final 1D reshape. The savings are small (16 B + 3 params here) but real. Applies to any graph where:
1. A 4D tensor is flattened to [1,1,N] (rank-3)  
2. Then TopK/similar extracts K elements as [1,1,K] (rank-3)
3. Then a Gather produces [1,1,K2] (rank-3) from the K results
4. Then a final Reshape drops to 1D [K2]

If you flatten to [N] at step 1 instead, all subsequent shapes are 1D and step 4's Reshape disappears.

## What's missing from the library

Not sure if techniques.md has this explicitly. The spirit is related to "Do not squeeze trailing singleton dims just to make shapes prettier" in #147, but the inverse — **do NOT keep leading singleton dims** in reshape targets when a later reshape to 1D is inevitable. The gain is the intermediate shape tensor that's double-counted.

Check #147 first — if it covers this, just extend with the measurement. Otherwise add as a note.

## Sorter note: techniques.md #147 — extend with "1D chain shortcut: when your reshape target for flattening is [1,1,N] and you later reshape again to 1D, flatten directly to [N] from the start; eliminates the intermediate [1,1,K] Gather output and its reshape initializer"
