# task225 — quad-scatter free-output dissolve fails on ROLE non-separability (floor re-confirmed at 918)

## What I discovered

For a **4-color quadrant SCATTER** (one 2×2 block `[[a,b],[c,d]]` whose four cells each
paint a diagonally-adjacent 2×2 patch at ±2 offset), the "free-output Einsum from the
input one-hot to dissolve the 10-channel onehot intermediate" thesis **fails for a
structural reason distinct from rank**: the scatter is **role-non-separable**.

Which diagonal a given color paints depends on the **joint (r,c) block detection**, not on
the source cell alone. A single input cell is role-ambiguous: cell (2,2) is the
a-corner for block (2,2), the b-corner for block (2,1), the c-corner for (1,2), and the
d-corner for (1,1). So a fixed source→dest placement tensor `Place[(sr,sc)->(R,C)]` is
ill-defined. Conditioning on the detected config `g[9]` makes it well-defined but the
placement tensor becomes `Place[9,6,6,6,6] = 11664` params (dense; the 4 active source
slices per config don't compactify the stored tensor), or it factors into the
separable K/R/C terms that **already lost** (the task225 v40-v43 family: 1665–5791,
all byte-equal + profile==static but worse than the 360 B onehot6).

This is a different failure mode from task183 (which failed because the cyan scatter
*mask* was rank>1). Here the per-role *placement* could even be low-rank, but the
**role assignment itself** (binding a color cell to one of 4 diagonals) is the
non-separable factor, because it requires the 2×2 full-block detection.

## Also refuted (same task, this session)

- **Channel-preserving rearrangement** (output one-hot = channel-independent spatial
  rearrangement of the free input one-hot, so colors flow as channels and no 10-ch
  rebuild is needed): bottoms out at a `[1,10,6,6]` U8 color-plane intermediate = 360 B
  (== the onehot6 it tries to replace) AND needs a 6×6 F32 input slice (1440 B). No win.
  General lesson: "let the color ride the input channels" still forces a C-channel
  spatial intermediate at the active resolution; it only helps if that intermediate is
  the FREE graph output, which the role-isolation step prevents here.
- **Paint-kernel shrink**: the QLinearConv canvas kernel `[1,4,6,6]=144` is provably
  minimal. Conv geometry: with a single-nonzero detection grid, kernel index
  `kr = pad-1-dR` is position-independent, so kernel size = **offset span**, not pad.
  The row/col offset span is {-2,-1,0,1,2,3} (block r..r+1 + up-patch r-2..r-1 + down
  r+2..r+3) = 6 distinct → 6×6 kernel forced, symmetric OR asymmetric pads. No 4×4/5×5.
- **Read-basis shrink**: `S30x4=120` is minimal (4-wide window over the 30-axis forced
  by input shape; params = element COUNT, so F16 doesn't reduce it).
- **channel-0 drop**: onehot6 must be the full `[1,10,6,6]`; LIVE has ch0=True in the
  active-grid background cells (the 24-border pads to all-False zeros), so a [1,9,6,6]
  variant would need a separate ch0 plane + Concat → nets zero.

## Where

- task: 225 (QuadScatterFreeOutputColorOperand)
- floor: `staged_submits/task225_inputop_cidx_cost918/task225.onnx` (v47, Boladi),
  builder `The_400_Tasks/Task_by_Task/task225/prototypes/build_task225_v47_cidx4_qconv.py`
- re-verified this session: visible 265/265, truth 200000/200000, profile==static==918,
  byte-equal to LIVE. Aditya manifest team-best = 1033 (so 918 already beats it).

## Why it matters

Saves a future worker from re-walking the free-output dissolve on quad-scatter tasks.
The decision rule to add to the library: **before attempting a free-output color-from-input
Einsum on a scatter, check whether the source→dest ROLE is a function of a single source
cell or of a joint multi-cell detection.** If role binding needs joint detection (block /
shape / orientation), the placement tensor is dense and the dissolve loses to a compacted
small-grid onehot — this is a companion to the existing #116 crossover and the task183
rank gate (#110). The existing
`DONE_2026-06-22T120000Z_task183_dissolve_rank_floor.md` covers the rank variant; this
adds the **role-binding non-separability** variant.

## What's missing from the library

`by_symptom.md` routes "WHOLE one-hot output could be a separable product" to #109/#119
but does not warn about role-binding non-separability in multi-color scatters. The
#110 rank gate covers mask rank; it should be extended (or a sibling added) with: "for
multi-source-color scatters, also test whether the color→placement ROLE is determined by
a single source cell or a joint detection — joint detection ⇒ dense placement ⇒ dissolve
loses, independent of mask rank."
