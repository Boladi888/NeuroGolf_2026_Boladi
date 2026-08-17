# task279 — archived cost-analysis history

Superseded by the QLinear erosion-flood floor proof in `task279.md` (team-best 5,057, 2026-07-02). Retained for provenance: rowcol_border v6 (7,871) + v1/v2/v3 flood-canvas breakdowns.

---
**(historical) CURRENT FLOOR (v6, 7,871 — 2026-06-30).** The v3 flood-canvas analysis below is for
a DIFFERENT (more expensive, 100%-truth) formulation. The shipping path is the
`rowcol_border` family (byte-equal to LIVE at 99.930% truth). v6 intermediate budget:
- `slice_1` F32 [1,1,16,16] = **1,024 — FORCED.** Input is F32; `Slice` inherits F32 dtype.
  Smallest F32 plane covering the 16×16 active region (verified max grid). No Slice+Cast
  fusion; Cast-first ([1,10,30,30]→U8 = 9,000) or wider slices are strictly larger.
- `pad_26` BOOL [1,1,30,30] = **900 — FORCED.** The `Where` condition must broadcast against
  `input [1,10,30,30]`; 16≠30 forbids a 16×16 condition, so a 30×30 single plane (BOOL = min
  dtype) is mandatory. Equal-before-Pad makes this the ONLY 30×30 tensor.
- 18 × U8 [1,1,16,16] = 4,608 + small axial tensors (208/208/192/192) = 800 — the flood/CC
  scaffold. **Empirically minimal under byte-equal-to-LIVE** (falsifications 2026-06-30):
  - 2 flood steps (vs 3) → visible 245/266 (92.11%) FAIL.
  - single-kernel CC (mp5-only → 264/266; 7×7 → 208/266) FAIL — `mp5 + v(2×1)` both needed.
  - no border seed (axial marks alone) → visible 219/266 (82.33%) FAIL.
  All visible-fail ⇒ certain hidden 0 (VISIBLE-FAIL axiom). 3 flood + mp5+v + frame+axial
  border seed is the byte-equal minimum.
- params 283 (the 256-elem `border_frame` precompute; nets −32 vs v5's m1 Max — the last
  data-independent fusion available).
**WALL at 7,871.** Both structural floors (1,024 F32 slice + 900 BOOL Where cond) are forced
by the F32 one-hot input contract and the 30×30 `Where` broadcast. The scaffold is at its
byte-equality floor (3 independent reductions falsified above). Further cuts would need a
sub-16 working grid (data-dependent → cost fiction, traps §GRADER-GAP) or fewer flood/CC
steps (breaks byte-equal-to-LIVE). v6 is the floor of the `rowcol_border` family.

LIVE cost 50,462 dominated by:
- 22 F32 [1,1,16,16] flood intermediates = 22×1024 = 22,528 bytes
- Final Concat [1,10,16,16]F32 = 10,240 bytes
- Other nodes = ~17,000 bytes

v1 achieves 25,273 via:
- F16 intermediates: halves 1024→512 bytes per flood step
- Canvas+Equal output: eliminates 10,240-byte Concat, replaces with 256-byte U8 canvas + 900-byte padded canvas
- Same algorithmic structure as LIVE (11 8-conn MaxPool c9-flood + 4 4-conn Conv CC)

**v2 floor breakdown (15,816 mem + 44 params)** — UINT8 (1 B/elem) is the dtype floor for masked flood (MaxPool has no BOOL kernel, and BOOL is 1 B anyway):
- Channel extraction (2 F32 Slices + 2 Casts→u8): 2,048 + 512 = 2,560 (irreducible: input is F32)
- Seeding (Add + Sub + Pad-18×18-fill-1 + MaxPool-valid + Mul): 256+256+324+256+256 = 1,348
- Border flood 11 × (MaxPool 256 + Mul 256) = 5,632 (irreducible: 11 steps data-required)
- enc_9 Sub: 256
- CC flood 4 × (MaxPool[3,1] + MaxPool[1,3] + Max + Mul) = 4 × 1,024 = 4,096 (4-conn without Conv)
- Arithmetic canvas (Mul·7 + Add + Mul·9 + Add): 1,024
- Pad→30×30: 900 (the canonical 900-B single-channel canvas floor)

**v3 floor breakdown (14,024 mem + 43 params)**:
- extraction 2,560 (2 F32 slices + 2 casts — F32 input format floor)
- seed 1,348 (Add+Sub+Pad18+MaxPool+Mul — padding-mask seeding is mandatory)
- border flood 11 × 512 = 5,632 (**corpus depth histogram: {1:2028, 2:34673, 3:72899, 4:58300, 5:24630, 6:6233, 7:1000, 8:196, 9:32, 10:8, 11:1}** — the tail mandates 11)
- enc9 256 · CC 2,560 (8-conn ring step + 2 quartets, officials-mandated) · canvas 768 · pad30 900

**<10k / <5k verdict (2026-06-04, after generator analysis): PROVABLY UNREACHABLE without breaking correctness.**
- Scaffolding floor = extraction 2,560 + seed 1,348 + enc9 256 + CC 2,560 + canvas 768 + pad 900 = **8,392** before ANY border-flood step.
- <10k allows ≤3 flood steps (8,392 + 3×512 = 9,928). Depth ≤3 covers only **55%** of the corpus even WITH erode-4×4 seed enrichment (+1,280 B) — a <10k variant breaks ~45% of generator outputs. Dead.
- Seed-enrichment (erode-4×4 all-9 windows, sound because interiors ≤3×3) does NOT cut the worst case: corpus tail still needs 11 (depth-9: 32 pairs, 10: 8, 11: 1). Cutting to 8 steps breaks 237/200k (0.12%) vs LIVE to save 256 B — rejected as -EV.
- <5k is below the scaffolding floor alone.

**8k-and-below analysis**: NOT achievable. Bit-packing the grid into uint32 lanes (BitShift/BitwiseOr, opset 18) was priced: ~10 ops × 64-72 B ≈ 650 B/step — WORSE than MaxPool+Mul's 512 B/step, because MaxPool does the whole 2D dilation in one op while bit-ops need ~10 (op-count × tensor-size is what matters). Masked-separable CC (V-then-H half-steps) fails straight corridors needing 4 same-axis moves. No alternative O(few-step) algorithm found:
- Degree-2 closed-loop detection: 87% error (enclosing components have branches/tails — NOT pure rings)
- 4-directional CumMax scan ("has 1 in all 4 dirs"): 10% accuracy (fails for multi-component grids)
- Transition parity scan (count c9→c1 transitions from 4 directions, AND): 89% error — fundamental: two isolated c1-cells in same column from different components give same parity as a real ring; parity cannot test CONNECTIVITY
- Morphological thinning (remove deg-1 endpoints) + parity: 12% error — still fails for disconnected-component false positives
- Aggressive thinning (remove deg≠2) + parity: 87% error — destroys enclosing rings that have branches (common in this task)
- Alternative seeding (MaxPool(bseed)&c9 as seed): 69% error — expands seeds incorrectly, marks non-enclosed c9 as enclosed

**PROVED: flood-fill is the MINIMUM algorithm for this task.** All scan/parity/topology approaches fail because they cannot test topological connectivity without iterative operators. The 24,243 cost is the achievable floor.
