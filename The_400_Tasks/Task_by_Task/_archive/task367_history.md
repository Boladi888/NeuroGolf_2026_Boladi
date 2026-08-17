# task367 — archived history

Full prose for superseded rungs, moved out of `task367.md` per WORKER_PROTOCOL staging rule (manual content > 20 KB). The current best (v4, 19,334) and the current floor proof live in `task367.md`. These are kept for provenance only.

## Superseded attempt prose

**2026-06-04 | ✅ v1 exact detector-row trim — staged, byte-equal** | PASS — updated `build_task367_v1.py` to trim the folded direct-25 detector output from `[1,25,20,20]` to `[1,25,18,20]` with detector pads `[0,2,4,6]` and stamp pads `[5,4,2,0]`. Gates: local builder visible 266/266 and synth-reference 868/1000; `ng_gate truth --full` 200000/200000; `ng_gate ab` 2266/2266 byte_equal=True. Staged at `.hermes/board/staged_submits/task367_v1_qconv_rowtrim_t_acacec0d_on_47225_cost19465/`; no submit. Under-10k probes rejected: 18x19 uniform width trim cost 19,015 but truth only 198894/200000 and A/B 2257/2266; historical 4,209 rowhash/MPH visible lookup is still hidden-bad (`truth --full` 261/200000); shape pruning cannot clear 99% because every 1..5 by 1..5 size is common in the 200k corpus. | Cost: 19,465 (memory 15,500 + params 3,965).

**2026-06-04 | ✅ v1 threshold-folded structural direct-25 QConv — staged, byte-equal** | PASS — `build_task367_v1.py` keeps the structural direct-25 detector/stamp rule but folds the per-channel detector thresholds into QLinearConv INT32 bias, folds `side = 1 - zero` into the zero-channel detector weights, and uses QConv asymmetric padding instead of materialized padded feature planes. Gates: `ng_gate visible` 266/266; `ng_gate truth --full` 200000/200000; `ng_gate ab` 2266/2266 byte_equal=True. Staged at `.hermes/board/staged_submits/task367_v1_qconv_biasfold_t_8bfc6d83_on_47225_cost20465/`; no submit. | Cost: 20,465 (memory 16,500 + params 3,965).

**2026-05-24T20:35Z | ✅ SHIPPED structural_direct25_qconv_v2 — Kaggle → 6083.81 (+14.24 LB)** | v2 post-MPH-failure recovery; 887/1000 synth (88.7%); 266/266 vs expected + 266/266 vs LIVE; 19 nodes 2 QLinearConv (NO MPH); EXACT theoretical max | Cost: 47225. | submit-captain batch ship 2026-05-24T20:35Z.

**2026-05-24T04:11Z | ⚠️ THIRD non-confirm of unlock pattern — Kaggle → 6055.01 (+0.00 LB)** | cost 49772 -> 4209 (12x reduction); 266/266 vs LIVE + 266/266 vs expected BUT no task-shaped synth % reported AND used MPH (displaced perfect hash) which is lookup-family in disguise. v1 prompt did not require synth gate; v2 will. Cost: 4209. | submit-captain ship 2026-05-24T04:11Z.

- 2026-05-19 | worker-7 crop-to-20x20 | SKIPPED — cost regression. 266/266 verify passed but memory 45046→55426 (+10380), params 6406→5398 (−1008). Net cost 51452→60824 (+9372). Wrapper overhead (+32k for input_cropped+output_active at 20×20) exceeds spatial savings from 11 (30,30) intermediates shrinking to (20,20) (saves ~24k).
- 2026-05-22 | row-hash + packed-pair + qmatmul label rewrite | PASS — `verify_against_live` 266/266 vs expected and vs LIVE. Replaced full-cell hash with 20 row-count hash, modulo-bucket selector, packed four rects + active dims into two INT64 words/row, fill via uint8 `QLinearMatMul`, padded one-channel label before final BOOL `Equal`. | 7,181 (memory 6,084 + params 1,097).
- 2026-05-23 | row-hash Conv/Where + displaced perfect hash | PASS — `verify_against_live` 266/266 vs expected and vs LIVE. Replaced the 20x20 F32 color-5 crop with a 1x30 Conv row-sum lane, removed active-dim/label-canvas reconstruction via `Where(fill_mask, color4, input)`, replaced bucket compare/ArgMax with displaced perfect hash (`B=74`, table size `271`). | 4,209 (memory 3,184 + params 1,025). NOTE: hidden-bad lookup family — `truth --full` 261/200000. Dead end, not a structural target.
- 2026-05-24 | independent first-principles reference + task-shaped synth | PASS — raw `task367.json` numpy reference is 266/266 visible. Synth construction gate 859/1000. | files: `prototypes/task367_rule_reference.py`, `prototypes/task367_synthetic_gate.py`. NOTE 2026-06-24: this reference is BUGGY vs true ARC-GEN ground truth (over-fills connector-bounded false pockets) — see task367.md Don't-do log.
- 2026-05-24 | structural rectangle-pocket UINT8 QConv ONNX | PASS but not cost-competitive — 266/266 vs expected/LIVE. ONNX-vs-reference synth 1000/1000. Signed INT8 edge-penalty kernels folded edge/active-boundary checks into one detector; cost 159,542 probe → 95,343. | 95,343 (memory 90,500 + params 4,843).
- 2026-05-24 | structural row-bit ONNX | PASS but worse unvectorized — same rule as 20-bit row integers; 266/266; synth 1000/1000. Params 2,065 but 1,944 tiny row-vector node outputs dominate memory. | 161,961 (memory 159,896 + params 2,065).
- 2026-05-24 | structural direct-25 signed QConv ONNX | PASS under 50K — one 25-channel signed INT8 detector scores all (h,w) candidates (+4 evidence, -3 edge-continuation), one UINT8 QConv stamps interiors. 266/266 all modes; synth 887/1000. | 47,225 (memory 41,668 + params 5,557).

## Superseded floor proofs / cost-path detail

### 2026-06-05 v1 activation proof (338-style exact margins) — superseded as params floor by v4

Memory breakdown for the v1 row-trim graph (static graph, _score_v2 exact):

| Intermediate | Shape | Dtype | B |
|---|---|---|---|
| candidate_u8 | [1,25,18,20] | UINT8 | 9,000 |
| ch0_20 | [1,1,20,20] | F32 | 1,600 |
| ch5_20 | [1,1,20,20] | F32 | 1,600 |
| fill30 | [1,1,30,30] | BOOL | 900 |
| v1_features | [1,2,20,20] | UINT8 | 800 |
| zero_u8 | [1,1,20,20] | UINT8 | 400 |
| five_u8 | [1,1,20,20] | UINT8 | 400 |
| stamp_score | [1,1,20,20] | UINT8 | 400 |
| fill20 | [1,1,20,20] | BOOL | 400 |
| **Total memory** | | | **15,500** |

v1 params: K_detector [25,2,7,9] INT8 = 3,150; K_stamp [1,25,6,5] UINT8 = 750; bias [25] INT32 = 25; scalars/indices = 40. Total 3,965 → v1 cost 19,465. v4 trims K_stamp dead row + dup scalars + Pad-18 → params 3,834 → 19,334.

**§A — candidate_u8 = 9,000 B is the architecture floor**: 25 channels because each (h,w) size has distinct threshold 4×(h·w+2h+2w)−4 (16..176); no single linear threshold separates valid from invalid across all 25; all 25 present in 200k (rarest 5×5: 8,518). Spatial minimum 18×20 (18×19 fails full-truth). Dtype UINT8. Packing to 4 B/pos needs 25 Greater (9,000) + ≥49 Mul/Add (+70,560) — 9× worse. F16 doubles to 18,000. Composed kernel: non-linearity irreplaceable.

**§B — F32 slice intermediates = 3,200 B unavoidable**: Cast before Slice needs [1,10,30,30] UINT8 = 9,000 B (6× worse). #46 Einsum cannot crop spatial dims. §41 negpad exhausted.

### 2026-06-05 composed-kernel attempt (v2) — cost 6,855 but 0/266 visible, route closed

Folding detector [25,2,7,9] + stamp [1,25,6,5] into one composed kernel [1,2,12,13]: `K_comp[0,in_ch,DR+5,DC+6] += K_stamp[0,h,sr,sc] × K_det_mod[h,in_ch,dr,dc]`, pads [5,6,6,6]. Would eliminate the 9,000 B candidate → ~6,855 B. Measured visible 0/266, synth 0/500. Why: the detector's UINT8 saturation (clamp negatives) is the essential non-linearity between stages; without it, ~96 invalid channel×stamp positions each contribute ~−50 raw, overwhelming the valid channel's ~+20, netting ≈ −4,800 → clamped to 0 → no fill. Structurally irreplaceable in any single-pass linear composition.

### Cost path (historical)

- Live extracted: 49,772 (memory 43,366 + params 6,406).
- task367_rowcol_packed_bool: 20,800 (row+col hash, packed INT32 rects, INT8/BOOL).
- task367_rowhash_label_bool_v12_bucket12: 7,181 (memory 6,084 + params 1,097).
- task367_rowhash_conv_where_v13: 4,684 (memory 3,304 + params 1,380).
- task367_rowhash_conv_where_mph_v14: 4,209 (memory 3,184 + params 1,025) — HIDDEN-BAD lookup family (+0.00 LB, truth 261/200000). Dead end.
- structural direct25 QConv: 47,225 → v1 fold 20,465 → row-trim 19,465 → v4 param-trim 19,334.

## Archived Don't-do entries (moved 2026-07-12 by worker3, task-md-size trim — about the RETIRED 25-channel / crop-wrapper graphs, superseded twice by Sho 15,082 then Vamshi 13,926)
- Composed-kernel approach (folding detector [25,2,7,9] + stamp [1,25,6,5] into [1,2,12,13]): built+tested 2026-06-05, visible 0/266. UINT8 saturation between the two QLinearConv stages is non-linear; without it ~96 invalid channel×stamp positions each contribute ~−50 raw, overwhelming the valid channel's ~+20. Linear composition cannot reproduce the clamp.
- F16-Conv for the (retired 25-ch) detector: candidate [1,25,18,20] as F16 = 18,000 B vs UINT8 9,000 B → cost rose to ~28,465. (General principle: F16 doubles detector-tensor bytes; still applies to any detector.)
- crop-to-20 wrapper (on the old 30×30 graph): base cost 51,452 (memory 45,046); wrapper adds two (1,10,20,20) intermediates (+32k) vs ~24k spatial saving → net regression. N/A now — the live graph already crops to 20×20 inside the Slice.
- Detector shape pruning for a 99%-full-truth candidate: the 200k corpus uses every 1..5×1..5 pocket shape frequently; dropping only 5×5 misses 8,518 pairs.

## More archived Don't-do entries (moved 2026-07-12 by worker3 — retired non-flood/lookup approaches)
- 18x19 uniform detector-width trim: visible-clean at 19,015 but truth --full 198894/200000 and A/B 2257/2266 byte_equal=False. (About the retired 25-ch graph.)
- rowhash/MPH under-10k line (`task367_rowhash_conv_where_mph_v14`): visible 266/266 but truth --full 261/200000 — HIDDEN-BAD lookup family, not a real solution.
- Single local linear-stencil probing: 7x7 windows have conflicting labels; 11x11 removes conflicts but a linear separator still leaves thousands of errors. (And trained distillation is prompt-forbidden.)
