# task302 — WALL: stretch fold (ch5_u8 + fill8 → one output-writing Conv) to 1300-1450

- **Live bar**: 1574 (Vamshi, sha256 `366237800121a80b62e22e725646ec0d3c7a6a56a5ec3dd71a57d350e6498528`, manifest.csv + Tasks sheet, updated 2026-07-02T02:35:15Z).
- **Current best (staged, holds)**: 1573 — `task302_regionfillrewrite_cost1573/task302.onnx`, sha256 `f39b26ef41e9d40cb7cfd1ecb511d289e57af66377cbb0e188ef87669b21937e`, static=profile=1573 (mem 1216 + par 357). −1 recapture, visible 266/266, full truth 200000/200000, ab byte_equal (per its manifest).
- **Verified by**: bo_ng_cld_worker3 (Claude lane), batch 42 slot 12, 2026-07-03.

## Representation (the 1573 graph)
`Slice(ch5,F32)[576] → Cast(u8)[144] → QLinearConv frame5(5×5 hollow)[64] → QLinearConv fill3(3×3,pad3)[144] → Concat[ch5_u8,fill8][288] → QLinearConv final(w[10,2,3,5], b[10], pads[1,2,19,20])[output,excl]`
Memory 1216 = 576+144+64+144+288. Params 357 = final_w 300 + frame5_w 25 + fill3_w 9 + final_b 10 + ch5_starts/ends 4+4 + 5 scalars.

## Named lever FALSIFIED — fold ch5_u8+fill8 into 1 channel (would save ~150 par + 144 mem → ~1279)
Independent forced-component proof (from the actual `final_w` bytes, this session):
- A linear single-channel encoding `e = a·ch5_u8 + b·fill8` forces the final conv's two channel weight planes to be **proportional** (ratio b/a fixed at every tap) and forbids any tap that responds to only one channel.
- Actual `final_w[:,1]/final_w[:,0]` has **49 distinct ratios** over 71 both-nonzero taps (min −32, max 127, std 16.1).
- **7 single-channel-only taps**: 4 with ch5-weight=0 & fill8-weight≠0, 3 with fill8-weight=0 & ch5-weight≠0.
- ⇒ No linear 1-channel encoding can reproduce this classifier (#40 independence wall). Confirms the Codex MILP existence-level result: `encoded 2*ch5+fill8 3×5 → outside-vs-8 patch conflict`, `5×5 → channel-0 MILP infeasible`.

## Other stretch levers, all walled
| Lever | Target | Result |
|---|---|---|
| Shrink final 3×5 kernel losslessly | −120 par | DEAD: no all-zero kernel row/col; every spatial pos has 9-10/20 nonzeros (dense). |
| Kill F32 `ch5` slice (576 = 47% of mem) | −432 mem | DEAD: F32 input ⇒ Slice stays F32 ⇒ Cast to u8. Slice-first (576+144) is already optimal vs cast-first full input (9000+144). No ONNX op crops+quantizes in one. |
| Split Concat → 2 convs + Add | −288 mem | DEAD: per-conv requant clamps u8 differently (final_b up to −1280) ⇒ not byte-equal; also creates two [1,10,30,30] intermediates (9000 ea). |
| Merge morphology frame5+fill3 → 1 qconv | fewer nodes/mem | DEAD (Codex): single 5×5/7×7 → patch conflict (same patch → color 8 & 0 / None & 5). |
| Sparse final_w to cut params | −par | DEAD (Codex): QLinearConv rejects sparse weights (ShapeInferenceError/TypeInferenceError). |
| Param micro-shaves | −few B | None left: bias must stay len-10 (traps §8h sub-declared-bias UB); 5 scalars all distinct+used; ch5_starts/ends already 4+4 min after axes removal (the 1573 shave). |

## Verdict
Stretch 1300-1450 is a **WALL** for this detector-stack representation. Memory floor 1216 (F32 crop 576 forced, 2-channel Concat 288 forced by rank-independence, u8 grids minimal) + params floor 357 (dense 3×5×2 classifier, all reduction routes rejected) = 1573. The −1 recapture is the credited win. A new win would need a representation that reads channel 5 without a 12×12 F32 intermediate AND classifies with a rank-1 (single-channel) feature — the byte evidence says the classifier is genuinely rank-2, so this is not a micro-shave gap.
