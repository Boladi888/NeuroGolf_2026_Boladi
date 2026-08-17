# task048 — archived cost-analysis history

Moved out of `task048.md` 2026-06-22 (w-048) to keep the live file under the 20 KB
manual-content budget. This is the superseded-rung structural detail (v4 baseline
through w03 Path-B price check). The live file keeps the Rule, Attempts table, the
current v8 floor proof, the w-048 lever-by-lever floor proof, and the Don't-do log.

## v5 bitpack structure (the substrate all later rungs inherit)

v5 bitpack structure (memory ≈148 + 72·K − 8 + params 119; opset 18) — **one u8 byte per
row** (8-wide grid; bit c == column c, LSB = col 0). v5 = v4 plus: ScatterElements seed
(zeros-init + scatter lowbit at am_f — kills Equal/Cast/Mul + iota8, −16), last round
emits unmasked d8 (hit-test reads red bits, red ⊆ pass, −8), and the **2-cell box-B
hit-test**: bmask = highbit | highbit>>1 = BOTH bottom cells of box B (B is rightmost in
the last red row by definition of "last"; left-neighbor-of-last is red on 200,000/200,000
corpus). Officials max k-need drops 7 → 6 (measured over all 270); corpus max stays 12.

### v4 baseline structure
- pack 96+8: `Conv(input, W[2,5,1,8], group=2, pads=[0,0,-22,-22])` → [1,2,8,1] F32 64
  (group0 reads ch0-4 with taps 2^c on local ch2 = reds-packed; group1 reads ch5-9, local
  ch8 = cyans-packed — **extraction, crop AND bit-packing fused into ONE conv**; W = 80p,
  the group split halves it from [2,10,1,8]=160p) → Cast(U8) 16 → Split(axis=1,
  num_outputs attr = 0 params) 8+8 → pass = BitwiseOr(reds, cyans) 8 (bits disjoint:
  one color per cell ⇒ Or == union).
- index pair 24: nz = Min(reds,1) 8 (binarize first — ArgMax on raw bytes would pick the
  largest BYTE = rightmost-heavy row, not the first red row) → ArgMax(axis=2, keepdims)
  ×2 (first / select_last_index) 8+8 — **no [1,64] flatten tensor at all**.
- seed 27: row_oh = Equal(iota8 i64 [1,1,8,1], am_f)→Cast 8+8; fbyte =
  GatherElements(reds, am_f, axis=2) 1; lowbit = `fbyte & (0−fbyte)` 2 (u8 wrap ⇒ two's
  complement; = 2^first_col); seed = Mul(row_oh, lowbit) 8.
- flood 72/round (9 tensors × 8 B on [1,1,8,1] u8 row-byte lanes):
  `ew = x | x<<1 | x>>1` (2 BitShifts + 2 Ors — **u8 overflow IS the row boundary**, no
  edge masks needed at width 8); `d8 = ew | pad_down(ew) | pad_up(ew)` (2 negative-Pad
  lane shifts + 2 Ors); `reach' = d8 & pass` (BitwiseAnd). 3x3-box dilate == 8-conn.
- decision 23: lbyte = GatherElements(reds, am_l) 1; highbit via 3-step smear
  (`x|=x>>1; x|=x>>2; x|=x>>4; high = x−(x>>1)`) 8; rlast = GatherElements(reach_K, am_l)
  1; hit = rlast & highbit 1 (∈ {0, 2^c}); Cast-F16 2 + Sub(0.5) 2 = margin (2^c ≤ 128
  F16-exact). Exact-last-cell test, NOT any-red-of-row (boxes can share rows — see
  Don't-do). First/last-in-different-boxes theorem unchanged (corpus-reproven 0/200k).
- output 0: placer Conv unchanged (#78; scalar margin input, no bias ⇒ no boundary leak;
  empty-cells property validated full-mask on 20k corpus + 270 officials per rung).

K-need facts (8-conn lastcell test, corpus-measured): max = 12 (ONE pair caps the exact
rung); officials max = 7 (visible floor). v3 (plain-op MaxPool/Min, memory 570 + 128·K +
43p) kept as the boring-static sibling — its per-stage breakdown is in git history.

## v6 bmask shortcut (2026-06-12)
bmask = smo3 - (smo3>>2) replaces v5's 4-op chain (smh, highbit, hb2, bmask). Proof:
smo3 = 2^(n+1)−1; smo3−(smo3>>2) = 2^n+2^(n-1) = highbit|highbit>>1. Verified all 256 u8
values. Saves 2 intermediate tensors (smh, highbit) = 2 bytes memory. c2 param already
present (reused from smear). v6 k6 cost = 697. Byte-equal to v5 k6.

Floor reasoning: the 9-op round is the or-of-5 minimum (self+E+W + 2 lane pads; BitwiseOr
is binary-only — no variadic; Max(bytes) ≠ Or). The #47 "crossover at K=12 for 30-wide
u32 lanes" verdict FLIPS at 8-wide grids: rows fit single bytes, E/W shifts are
boundary-free, pack fuses into the extraction conv, and NO unpack exists (decision reads
one bit). K=12 capped by one corpus pair; K=7 = visible floor.

## v7 Pad axes shortcut (2026-06-13)
ONNX Pad opset 18 has an optional `axes` input that specifies which axes to pad. With
axes=[2] (height only), the pads vector shrinks from 8 elements [0,0,H_beg,0,0,0,H_end,0]
to 2 elements [H_beg,H_end]. INT64 required by ORT (Tind=INT32 accepted by spec but ORT
rejects it). Savings: 16 params → 5 params = -11 params. Byte-equal to v6 on 200,270
inputs. New cost 686 = 578 memory + 108 params.

## w03 Path-B price check (2026-06-21): non-flood classes all > 686
No non-flood replacement priced below 686. A plain #126 8x8 U8 MaxPool flood needs at least
3 charged 64B planes per step (`MaxPool`, mask/clip, accumulate) = 192B/round, so 6
visible-floor rounds already cost >=1,152B before extraction/seed/decision; row-byte #88 is
72B/round. #33/#34 CC labeling is worse: even on 8x8, F16 label propagation is >=128B planes
and >=512B/iteration, and histogram/equality machinery adds INT64 label indexes. #122
MatMulInteger does one-step projection/counting, not transitive closure; a 64-cell reach
vector produces INT32[64] =256B per step, and any 64x64 reachability/adjacency
materialization is >=4,096 params or bytes before masking. Forced-component wall for the
routed idea class, not a general theorem for all future algorithms (#134 discipline).

## v8 full-box rework (2026-06-21): the live floor, cost 642
r03's missed same-flood formulation holds in ONNX. Seed side computes
`low2 = lowbit | (lowbit << 1)`, scatters it at `am_f` as `seed_top`, shifts that row down
with `pdn_2/axes_2`, and ORs into `reach0`, seeding the whole first 2x2 box. Flood runs k=5,
so `d84` is the final unmasked expansion and the old `reach5` mask plus final round are gone.
Target side keeps `bmask = highbit | highbit>>1`, gathers `d84` at `am_l`, gathers the row
above via `Pad(d84, pdn_2, axes_2)`, ORs those reads, then applies `bmask`. Cost is
642 = 534 memory + 108 params; full real profile is exact. Truth-full is 199,434/200,000,
inside the filtered-clean band and better than v7's 198,749/200,000, but not byte-equal, so
stage only as `RISK_SINGLE_PROBE` and keep v7 cost 686 as the safe floor.
