# Free-output one-hot fold: the active-region size decides whether it wins

**Where:** task046 (worker-046, 2026-06-17). Builder: `The_400_Tasks/Task_by_Task/task046/prototypes/build_task046_freeout_probe.py`. Current best: `staged_submits/task046_dualdilated_read_cost2339/` (v24, cost 2339).

## What I discovered (a quantitative refinement of technique #67 FREE-OUTPUT)

The #67 free-output catalog says: "make the expensive op BE the output; the 10-channel one-hot explosion never exists" (task148 2880→900, task287). This is true ONLY when the active (non-padded) region of the output is LARGE relative to the channel count. I measured the opposite case on task046 and it REGRESSES.

The task046 output is the BOOL one-hot `[1,10,30,30]`, but its active region is only **3×16 = 48 cells** (height-3 snake, W_out≤16). Producing this output needs exactly ONE billed pre-output tensor (the free-output node absorbs only the LAST op). There are two legal final orderings of {channel-expand 1→10, spatial-pad active→30×30}:

- **channel-small ordering (current/v24):** `Equal(carrier[1,1,3,16], ids[1,10,1,1]) → oh_b[1,10,3,16]` (BILLED **480 B**) → `Pad == output` (free). 
- **spatial-full ordering (the #67 swap):** `Pad(carrier[1,1,3,16], sentinel) → CGfull[1,1,30,30]` (BILLED **900 B**) → `Equal == output` (free).

Measured: v24 = 2339 (mem 2211); the swap = **2760** (mem 2631), byte-equal vs LIVE (367/367, divergences=0 — functionally correct, just bigger). The swap LOSES by +421 B.

## The law

The billed pre-output tensor is `min(channels × active_cells, 1 × full_grid_cells)`.
- When `active_cells ≈ full_grid` (task148/287: full 30×30 active) → `1×900 < 10×900` → free-output swap WINS (carrier is the small one).
- When `active_cells ≪ full_grid` (task046: 48 active) → `10×48 = 480 < 1×900` → KEEP the channel-small one-hot, put the spatial Pad as the free output (v24 already does this).

**Decision rule before reaching for #67 to dissolve a one-hot output:** compare `C × active_cells` vs `full_grid_cells`. Fold to free-output only if the one-channel spatial carrier is the smaller of the two. For small-active-region one-hots, the materialized channel-small one-hot at the active size IS the floor — and the structure that already bills it while making the spatial Pad the free output is optimal.

## What's missing from the library

#67 lists the wins but not the failure boundary. It should carry this active-region discriminator so workers don't burn a rung re-deriving the task046 wall. Also relevant to #93's "output via gather-BOOL" and OneHot-axis-1: both still bill a `[1,C,active]` intermediate, so they don't beat the channel-small Equal either. The byte-equal correctness of the swap (it passes the gold gate, just costs more) is a good reminder that "gate-clean" ≠ "cheaper".
