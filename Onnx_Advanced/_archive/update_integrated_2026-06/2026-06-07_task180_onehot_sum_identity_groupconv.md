# task180: one-hot-sum identity (IG) + groups-halved placer-Conv (2026-06-07)

**Ship class**: params-only single-node (#68-ADDENDUM) — third member: **180@210** joins
215@110, 252@280. LIVE 9,688 → 210 (46×), byte-equal, memory ZERO.

Two generalizable moves:

## 1. The one-hot-sum identity (IG) — a data-derived constant for placer-Convs

`IG(t) = Σ_ic in(ic, t)` is **1 at any in-grid cell, 0 outside** (encoder one-hot). Any Conv
reading the raw input gets free per-tap "in-grid indicator" columns (weight 1 on ALL input
channels at that tap). Uses:
- **bg channel without bias/ones-init**: ch0 = −IG(t00) + IG(t01) + IG(t10) + IG(t11) − 2Σmasks
  = +2−2Σm where all taps in-grid, ≤ 0 at every partial-tap position. (Positive bias is
  UNSAFE in placer convs — fires outside the active block; IG replaces it.)
- **block confinement**: the pair −IG(t_a) + IG(t_b) is 0 exactly where both taps are in-grid,
  ≤ −1·(reads) elsewhere — kills ghost fires of single-tap mask channels off the target block,
  removing data-purity assumptions entirely (task180 v1: byte-equal to LIVE on ANY one-hot
  8×8 input, zero purity assumptions, W [10,10,2,2] = 400).

## 2. Groups-halving under generator-certified purity

When the generator certifies per-region color purity ("region q holds only {0, color_q}"),
"region occupied" ⇔ `1 − in0(cell)` — the bg channel substitutes for ALL color channels in
masks. That collapses cross-channel reads into the group containing in0: task180 v2 =
Conv(groups=2, W[10,5,2,2]+B[10]) = 210. Negative biases (−3/−6) are outside-safe (all-zero
taps ⇒ output = bias ≤ 0). Check the groups partition BEFORE assuming: contiguous-half
partition must give every output channel its needed reads (g=5/10 died on cross-group needs).

## DOC-FIX FLAG: scoring_internals A.2 "sub-byte packing" is stale

A.2 (~line 99) claims params are "bytes of the declared dtype × element count — dtype-aware"
and ranks uint4/uint8/F16 weight packing as top levers. That describes the RETIRED onnx-tool
counter. The canonical scorer (`neurogolf_utils.calculate_params`, core.md) counts
**elements, dtype-blind** — confirmed by the task055 Range-vs-init probe (zero_param_
generators.md correction). Sub-byte packing buys file bytes only, never params. A.2 should be
annotated on next library pass (task180's sub-210 hunt burned ~30 min re-disproving it).

## Dilated-kernel quadrant reads

`Conv(kernel 2×2, dilations=[4,4], pads=[0,0,4,4])` reads the four 4×4-quadrant cells
(r,c),(r,c+4),(r+4,c),(r+4,c+4) per output cell AND places the result block top-left with
the canvas grown back to 30×30 by free end-pads — overlay/zip/combine-quadrant rules need
NO crop, NO concat, NO placer stage. Generalizes: block size b ⇒ dilations=[b,b],
pads=[0,0,30−2b+... (pad to keep 30)]. Floor proof template in task180.md.
