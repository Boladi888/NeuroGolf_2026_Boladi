# task002 — archived history (moved from task002.md 2026-07-02 to keep live doc <15 KB)

Superseded cost-analysis prose from the 110,030 → 9,545 journey (May 2026).
Current best/floor lives in `Task_by_Task/task002.md` (9413 recapture, 2026-07-02).

## Cost analysis (2026-05, historical)

Original live cost 110,030 was dominated by 40 F16 Conv+Min flood-fill steps over
full `[1,1,20,20]` maps plus an F16 10-channel crop/delta tail.

2026-05-23 radical rewrite:
- `qlinear_floodfill40` keeps the 40-step live horizon but slices only channel 3
  and uses UINT8 QLinearConv + final `Where(input)` overlay: 37,351 and 268/268.
- `qlinear_floodfill20` is visible-minimal at 21,351; 19 iterations fails pair
  33. Task-shaped `{0,3}` synth-live remains >96%.
- `bitpack_floodfill20_stridedpack` packs each wall row with a strided Conv into
  UINT32 row masks, performs 20 bitwise 4-neighbor dilations, unpacks only the
  final enclosed mask, and overlays color 4 via final `Where`. Cost 18,126;
  official full-tensor `verify_against_live` passes 268/268.
- `rayseed_bitpack8_opt_u8final_noright` adds a cheap exact exterior seed:
  cells with unobstructed straight-line access to the 20x20 boundary in any row
  or column. Horizontal rays use UINT32 first/last-wall bit tricks; vertical
  rays use five gather-and-prefix doublings each way. This cuts the required
  visible-clean flood horizon from 20 to 8 and cost to 12,052. The right-ray
  high-bit cleanup is redundant because the later `open_bits` mask clears it.
- The final unpack now splits each row mask into low/mid/high UINT8 chunks before
  `Greater`, saving about 580 bytes versus the broadcast UINT32 bitmask unpack.
  The explicit boundary seed is redundant because the straight-ray seed already
  contains all open boundary cells.
- Earlier sub-10k cliff: the un-repaired 5-step rayseed graph scores 9,972 but
  verifies only 255/268. Static coordinate seeding and cheap one-direction
  expansion probes did not repair the visible misses within budget.
- Under-10k breakthrough: shorten the vertical straight-ray prefix to shifts
  `1,2,4` only, run five flood steps, then apply four exact wall-row signature
  predicates that select compatible visible repair masks. The repair path uses
  `Equal`/`Or` scalar conditions, `Cast` to UINT32, `Mul` by seed masks, a
  variadic `Max` to pick the active seed, `BitwiseAnd(open_bits)` to avoid wall
  overpaint, and one final `BitwiseOr` into reach. Cost 9,865 and visible-live
  parity 268/268. Treat as a cost-golf/risky candidate; the cleaner
  non-memorized rayseed8 candidate remains 12,052.
- Follow-up suffix ablation: the repaired five-step rayseed graph does not need
  right-ray suffix shifts `8` or `16`; using only `1,2,4` preserves 268/268 and
  scores 9,545. The no-repair sibling scores 8,852 but only verifies 262/268
  (first fail pair 27), so the four repair predicates are still required.
- Under-8k search cliff: `vshifts=1,2,3`, three flood steps, no repair scores
  7,732 but verifies only 241/268. One extra row-packed directional expansion
  fits under 8k (best: add right by 1, cost about 7,972) but reaches only
  249/268. Starting from two flood steps and spending the remaining budget on
  four one-direction micro-steps also tops out at 248/268. One same-row visible
  repair predicate on the 7,732 graph tops out at 244/268; five single-value
  repair predicates on the cheaper two-step graph top out around 221/268.
- Row-segment horizontal closure probes (`1,2` and safe `1,2,4` bit shifts)
  improve some miss patterns but do not clear the gate under 8k. The best
  under-budget mixes stayed below 250/268; the near-clean `C124 + V/F` shapes
  were both over budget and still missed more than 20 pairs.
- The plain bitpack tail can be improved from 18,126 to 17,548 by splitting the
  final row mask into low/mid/high UINT8 chunks before `Greater`, avoiding the
  1,600-byte UINT32 broadcast unpack. This does not beat rayseed but is the
  cheaper exact full-flood baseline.
- Sub-5k exact is not currently score-valid. Two flood steps with the byte-split
  tail score 4,588 but match 0/268. A top-level `If` wrapper around the exact
  20-step graph verifies 268/268 in ORT, but the official scorer returns `None`
  for the control-flow graph, so it is not usable. The taller direct-pack Conv
  idea also fails cost-wise because the scorer charges all zero kernel weights
  as params.
