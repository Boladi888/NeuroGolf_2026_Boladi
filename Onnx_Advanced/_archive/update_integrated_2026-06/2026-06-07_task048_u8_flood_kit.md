# task048 u8 flood kit — 6 reusable idea-classes (2026-06-07, 8,831 → 1,153/865/793 staged ladder)

Context: task048 = "are the two 2x2 red boxes connected through {red,cyan} cells?" → 1x1
answer. LIVE = 12-step 4-conn F32 Conv/Greater/Where flood (576 B/step). Final ladder:
**v5_k12 1,131** (200k-truth 100%, dual-truth-certified more-correct-than-LIVE), v5_k7 771,
**v5_k6 699 (sub-700, 7.9% of LIVE)**, + v3_k12 2,149 staged as the boring-static sibling.
All static, profile==score_v2 exact. Builders in `task048/prototypes/`.
v5's extra tricks (all probed): ScatterElements(zeros, idx, val) one-hot seed beats
Equal+Cast+Mul (−16); the last flood round can skip its mask when the consumer reads
mask-subset bits; a 2-target hit-test (`highbit | highbit>>1` = box-B's bottom row)
trades +2 B for −1 round at the visible floor.

## 0. ROW-BYTE bit-packed flood — #47's crossover FLIPS at narrow grids (the headline)

At grid width ≤ 8, one u8 holds a whole ROW (bit c = col c): planes become [1,1,8,1]
lanes at **8 B/tensor** and the #47 economics invert — adopt for ANY iterated morphology
on ≤8-wide grids (u16 for ≤16, u32 for ≤32 — but only ≤8 gets boundary-free shifts):
- **Pack for FREE inside the extraction conv**: `Conv(input, W[2,5,1,8], group=2,
  pads=[0,0,-22,-22])` — 1x8 kernel with taps 2^c emits packed row-bytes [1,2,8,1]
  directly (negpad crop + channel-select + bit-pack in ONE node; group=2 halves W to
  80p). Values ≤255 ⇒ F32→U8 Cast exact.
- **8-conn round = 72 B**: `ew = x | x<<1 | x>>1` (u8 overflow IS the row boundary at
  width 8 — no edge masks); `d8 = ew | pad_down(ew) | pad_up(ew)` (negative-Pad lane
  shifts, probed); `reach' = d8 & pass`. 9 tensors x 8 B vs 128 B/step unpacked u8.
- **No unpack stage**: read single cells as bits — `GatherElements(lane, row_idx) →
  BitShift >> col → And 1`. A bbox/count consumer would need unpack; a hit-test doesn't.
- **Bit tricks on ORT u8 (all probed)**: lowest set bit = `x & (0−x)` (Sub wraps);
  highest set bit = 3-step smear `x|=x>>1; x|=x>>2; x|=x>>4; x−(x>>1)`; BitwiseOr/And
  at opset 18; BitShift amounts as runtime tensors broadcast fine.
- ArgMax trap: binarize lanes (`Min(lane,1)`) before ArgMax-ing for first/last nonzero
  row — raw bytes rank by VALUE not by occupancy.

## 1. Min-masked u8 MaxPool flood — 128 B/step binary flood (extends cellular_automata.md)

`reach' = Min(MaxPool(reach, 3x3, pads=1), mask_u8)` — TWO 64-B tensors per step on an
8x8 grid. MaxPool(u8) and Min(u8) are both ORT-CPU registered (probed). **Min against a
NON-BINARY mask {0,1,2,...} keeps reach binary** (pool of binary is binary; min(1, m≥1)=1),
so the mask can double as a value-coded class plane — no Greater/Where/binarize per step,
no QLinearConv quant plumbing, no saturation analysis. vs LIVE-style Conv+Greater+Where
F32 = 4.5x cheaper per step. Caveat: Mul-masking instead of Min breaks binariness when the
mask has values >1.

## 2. Outer-product one-hot seed from a flat index — 99 B vs OneHot's 256 B

To turn `idx = ArgMax(flat [1,64])` into a [1,1,8,8] one-hot plane: do NOT OneHot(64).
`r = BitShift(idx_u8, 3, RIGHT); c = BitwiseAnd(idx_u8, 7)` then
`Equal(iota_r [1,1,8,1] u8, r) -> Cast u8` x2 and `Mul` the broadcast pair.
Rank-4-init vs rank-2-runtime-scalar broadcast Equal is legal (probed). 99 B + 20p vs
OneHot-F16→Cast→Reshape 256 B (OneHot u8-values is NOT_IMPLEMENTED in ORT; F16 is the
cheapest working values dtype). Power-of-2 grid sides make the decompose 2 bitwise ops.

## 3. ArgMax value-priority — delete the target-class mask tensor

When ArgMax must find the first/last cell of class A on a plane that also holds class B,
weight the extraction Conv so A's code > B's code (`W{chA:2, chB:1}`): ArgMax (and
select_last_index=1) then locates class-A cells on the RAW value plane — the dedicated
binary class-A mask tensor (Equal/BitwiseAnd + Cast, 64-128 B) disappears. Works whenever
≥1 class-A cell is guaranteed.

## 4. Generator-invariant connectivity relaxation (GENERATOR-INVARIANT doctrine instance)

ARC-GEN task_239be575 line 69 RESAMPLES until `path_exists(4conn) == path_exists(8conn)`
— ambiguous topologies are excluded from train/eval/hidden by construction. So the model
may use whichever connectivity is CHEAPER-TO-ITERATE: 8-conn (plain 3x3 MaxPool, max
k-need 12) instead of the rule's literal 4-conn (cross kernel, max k-need 18 — 6 more
steps = +768 B). Grep any connectivity/flood task's generator for equality-constrained
resample loops before paying for the literal metric. (Officials bypass the loop — gate
on them separately; here officials max-need was 7 (lastcell) / 6 (count).)

Bonus theorem (count-stage deletion): for two disjoint ordered 2x2 boxes, the row-major
FIRST and LAST target-class cells always lie in DIFFERENT boxes ⇒ "flood from first
reaches last" ⇔ inter-box connectivity — one GatherElements (71 B) replaces the
count-and-threshold stage (132 B). Corpus-reproven 0/200k.
