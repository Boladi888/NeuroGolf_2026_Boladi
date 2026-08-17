# task035 — U8 scan rebuild: canvas-Equal #81 crossover FLIPS in U8 + dynamic-bbox 4-ray floor

**Where:** task035 (cyan-pool ray-projection rule, generator `task_1f642eb9`).
Builder `The_400_Tasks/Task_by_Task/task035/prototypes/build_u8.py`; staged
`staged_submits/task035_u8scanfold_cost5335/`; floor proof
`prototypes/floor_proof.py` (3,500 pairs). Rung went 7,155 (F16) → 5,335 (U8),
all gates clean (visible 266/266, truth-full 200000/200000, ab-full byte-equal,
ORT 0/500, profile==static).

## 1. The #81 one-hot-vs-canvas crossover FLIPS to canvas's favor once the index plane is U8

The standard #81 pad-floor law says for a 10×10 active output the `[1,10,10,10]`
one-hot (1,000B BOOL) sits at/above the `[1,1,30,30]` canvas (900B) crossover, and
the canvas path "loses" because it needs an extra +100B u8 cast — netting the same
1,000B. **That accounting assumed the working plane was F16.** Once the whole middle
is already U8 (see #2), the painted index plane is ALREADY U8, so the canvas path
costs ONLY the 900B padded `[1,1,30,30]` U8 — the +100B cast is already paid/absent —
and `Equal(canvas, labels[1,10,1,1])` IS the `[1,10,30,30]` BOOL graph output
(uncounted). Net: **canvas 900B < one-hot 1,000B**, saving 100B AND removing the
one-hot intermediate node. Pad value = 10 (a 0–9 sentinel) makes off-grid cells
all-channels-0 for free. So: **the #81 crossover is dtype-dependent — re-evaluate it
AFTER casting the producer plane to U8; the canvas wins at 10×10 in U8.**

## Sorter note: extend the #81 pad-floor-crossover entry (techniques.md / cost_floor_reference.md) — add "the 10×10 one-hot-vs-canvas crossover FLIPS to canvas (900B U8 + free Equal-as-output) once the index plane is already U8; the +100B cast that made them tie is gone." (verified task035, 2026-06-11)

## 2. U8-everything for color-index scans: ~halves a F16 directional-scan middle

12 F16 [1,1,10,10] working planes (idx/cval/4 proj/paints) carry only color indices
0–9 — lossless in U8 (1B vs 2B). U8 MaxPool is opset-12+ ORT-confirmed; `Where(BOOL,
U8, U8)` runs on ORT CPU (only the **INT8** Where data branch is NOT_IMPLEMENTED —
U8 is fine, verified 0/500). Boundary masks via the AND-NOT/Greater fuse
(`Greater(cyan_u8, shifted_cyan_u8)` = `cyan AND NOT neighbor_cyan` in one op).
Cast the conv decode output to U8 BEFORE the col-slice (300B idx_row_u8 replaces a
400B F32 idx_f32). Combined U8 conversion saved 1,820B on this task.

## Sorter note: by_symptom.md directional-scan route — add "if the scan carries only color indices 0–9, run cval/proj/paint planes in U8 (1B); U8 MaxPool opset12+, Where(BOOL,U8,U8) OK on ORT CPU (INT8 data branch is the one that's NOT_IMPLEMENTED), AND-NOT boundary via Greater(cyan_u8,shifted)." (already partly in ort_reality §59/§522/§527; cross-link the scan-middle case)

## 3. Single all-sides Pad → N Slices beats N per-direction Pads for neighbor shifts

For a 4-neighbor (or k-shift) stencil, Pad the source by 1 on ALL sides ONCE
(`[1,1,12,12]` = 144B) then take 4 windowed Slices, instead of 4 separate
direction-specific Pads (4×110B = 440B intermediates). Saved ~296B here.

## Sorter note: tiny_architectures.md / by_symptom neighbor-shift — add "k directional 1-cell shifts: one all-sides Pad + k Slices (1 pad intermediate) beats k per-dir Pads (k pad intermediates)."

## 4. DROP recorded: 3102 (Aditya) is below the dynamic-bbox 4-ray projection floor

This rule (border pixel projects color onto nearest pool-edge cell, clamped into a
dynamic cyan-rectangle bbox) provably FORCES 4 directional projections: a 3,500-pair
generator scan shows all 4 pool edges (top/bottom/left/right) recolored in ~3,000
pairs each, and up to 8 distinct recolor-colors per pair (per-cell varying paint, so
#292 broadcast-overlay is inapplicable). Index-space I/O alone is forced to
decode 1,200B (single-axis conv crop) + canvas 900B = 2,100B, and the 4-ray scan
middle adds ≥~1,700B even idealized → architectural floor ≈3,800B+ > 3,102. The U8
rung sits exactly at the realized floor (5,044B mem). **Team already holds 3,102;
a tie/worse scores nothing → DROP.** Open question for the library: how was 3,102
reached on a dynamic-bbox 4-ray rule? (Either a fundamentally different formulation,
or 3,102 is a different/looser instance.) No action needed unless a 3,102-class
recipe surfaces.

## Sorter note: no library rule change for #4 — this is a per-task floor verdict (task035.md already holds the proof). Optionally note in a "dynamic-bbox multi-ray projection" floor-ledger row: 4 forced directional scans + index decode + canvas ≈ 3,800B+ floor.
