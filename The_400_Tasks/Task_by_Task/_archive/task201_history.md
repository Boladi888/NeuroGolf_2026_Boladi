# task201 — archived history (superseded prose, moved 2026-07-06)

Moved out of `task201.md` to keep manual content under the size cap. This is the full
Attempts log and Cost analysis for the **obsolete 171,221 → 4,718 → 9,738 → 35,577-era**
graphs. The CURRENT best (1773, byte-equal) and its forced-floor/WALL proof live in
`task201.md` (## Current best & floor) and `task201/hard_records/floor_forced_components.md`.

## Attempts (historical)

_One row per attempt. Format: `YYYY-MM-DD | approach | result | cost | session log`._
- 2026-05-16 | `t_6b4a362b` / worker2 | `task201_v1_broadcast_col_masks_on_5727_75_20260516` reviewer-approved for submit-captain as tiny-EV rebase candidate | cost 171,221 -> 170,597; expected +0.003651 local pts; 266/266 raw/LIVE match; preserved old-base package must be rebased onto current 5727.97 best | Kanban `t_6b4a362b`; reviewer approval comment on source card
- 2026-05-16 | `t_50ea359f` / worker2 | `task201_v2_reduced_mask_sums_on_5736_01_20260516` preserved for review; current-base package changes only `task201.onnx` | cost 171,221 -> 137,733 (delta -33,488); expected +0.217638 local pts; raw/LIVE 266/266 and adversarial parity 6650/6650; candidate ONNX SHA `FFCF34062256233803D3255157552B98AAAABB5062542F58B80D5FE641301F3F`; package SHA `55AE2C7BFEF25083246CED5B2DE6D25C7F342FCBDEB2FCD8537156CA1A81EEBA` | preserved at `<NOGIT_ROOT>/orchestrator/preserved/candidates/task201_v2_reduced_mask_sums_on_5736_01_20260516/`
- 2026-05-19 | worker-7 `task201_v3_reduce_sum_reorder_t_52155951_on_5806_87` | Eliminate v156 (1,10,13,13) intermediate: replace Mul(v134,v154)+ReduceSum([0,1,2]) with ReduceSum(v134,[2])+Mul(v154)+ReduceSum([0,1]). Builds on v2 (137,733). 266/266 verify. Cost 171,221→132,013 (−39,208 B from canonical, −5,720 B from v2). Staged: `.hermes/board/staged_submits/task201_v3_reduce_sum_reorder_t_52155951_on_5806_87/` |
- 2026-05-30 | worker-7 **crop_v1 (slice-before-cast)** | LIVE is ALREADY cropped to 13×13 (node1 Slice); the prompt's "−85% full crop" premise is WRONG (it misread the op-histogram Squeeze:16/Mul:15 as 30×30 tensors). Only full-canvas intermediate is node0 `Cast(input→F16)[1,10,30,30]=18,000`. Move the Slice **before** the Cast → `Slice(input F32)[1,10,13,13]=6,760` + `Cast→F16=3,380`. Provable identity (Cast∘Slice = Slice∘Cast). Cost **84,895→73,655** (−11,240). 266/266 byte-equal vs LIVE-in-submission.zip; ARC-GEN 1500/1500=100%. `prototypes/build_task201_crop_v1.py` |
- 2026-05-30 | worker-7 **crop_v2b (UINT8 narrow, opset 14)** | All 18 `[1,10,13,13]` planes are one-hot 0/1 (shifts via Gather, masks via Mul/Sub, selects via Where, OR via ReduceMax). Bump opset 10→14 (convert Squeeze/Unsqueeze/ReduceSum/Pad axes/pads attr→input), narrow the planes + 0/1 masks F16(2B)→UINT8(1B); keep one F16 cast of v134 for the ReduceSum branch; output dtype→UINT8 (decodes via `>0`). Provably exact (UINT8 & F16 both represent {0,1}; Mul=AND, Sub(v26,v26·mask)≥0 no underflow, ReduceMax=OR). Cost **73,655→45,053** (mem 44,679, params 374), pts 14.2844. 266/266 byte-equal; ARC-GEN 3000/3000=100%. |
- 2026-05-30 | worker-7 **crop_v3 (narrow + 3 byte-equal fusions) — RECOMMENDED** | v2b + (a) **v88/v89 elim**: `v90=ReduceMax(v26·not_v84,ax1)=not_v84·ReduceMax(v26,ax1)` (channel-reduce first → drop two `[1,10,13,13]`); (b) **index-select fold**: `Where(scalar v135,Gather(v134,v21),Gather(v134,v22))` then `Where(scalar v171,·,v134)` → `Gather(v134, Where(v171,Where(v135,v21,v22),id13))` (drop v172/v173/v174); (c) **3-gather compose**: ax2/ax3 gathers commute + two ax3 perms compose (idx3=idx2[v14]) → drop v175/v176 standalone. Cost **45,053→35,577** (mem 35,190, params 387; orphan F16 `v6` dropped after Greaters→`v6_u8`), **pts 14.5205**. Gates: verify 266/266 byte-equal; ARC-GEN **5000/5000=100%**; official run_network 266/266; adversarial 6650/6650=100% (8 modes). **Predicted +0.8697 LB** (raw log_e(84895/35577)×1.0, single-task wildcard). SHA256 `DD47859F9B2CA1950581F9C1691AA6B9AACA33F77850A743ABD6AD5278A92A01`, 7,431 B. Reproduce: `.venv/Scripts/python.exe prototypes/build_task201_crop_v2.py narrow3` → `task201_crop_v3.onnx`. |
- 2026-06-05 | **decoded-rule v1 full rewrite** | `prototypes/build_task201_v1.py` implements the frame/template overlay rule directly: 13x13 color-index decode via asymmetric Conv crop, static bbox/index math, fixed-size Gather template read, U8 single-channel canvas, final BOOL `Equal`. Cost **35,577→9,738** (mem 9,447, params 291), points **14.5205→15.8162**. Gates: `ng_gate visible --task 201 --cand ...` **266/266**; official `run_network` **266/266**; `verify_against_live` **266/266 proto vs expected / LIVE / proto-vs-LIVE**; `ng_gate truth --full` **200000/200000**; `ng_gate ab --full` **200266/200266, byte_equal=True, divergences=0**. ONNX SHA256 `E160093D2022E0825C01D58494F45FF0C41ACE23A3EFC49EA3FD709738574098`, size 9,329 B. Staged under `.hermes/board/staged_submits/task201_v1_decoded_overlay_9738/`; do not submit from worker session. |
- 2026-06-05 | **decoded-rule v2 cropped-output floor push** | `prototypes/build_task201_v2_cropout.py` keeps the v1 rule but renders only the maximum normalized crop `7x8`, expands to BOOL one-hot, and pads that one-hot to 30x30. Added micro-cuts: reuse `template_col_vals` for orientation, reuse `idx8_m1` in flipped source indices, final-active sentinel, border-mask-only frame overlay, 1-D bbox/template reductions, U8 ArgMax casts, and `GreaterOrEqual` outside-frame tests. Cost **35,577→4,718** (mem 4,439, params 279), points **14.5205→16.5409**. Gates: visible **266/266**; `truth --full` **200000/200000**; `ab --full` **200266/200266, byte_equal=True**; `verify_against_live` **266/266 proto vs expected / LIVE / proto-vs-LIVE**. ONNX SHA256 `53E4F601FDB7BF33EA79B182CA02D2F55A661B2FDCF46F1D103D735F37191A6A`. Staged under `.hermes/board/staged_submits/task201_v2_cropout_t_53E4F601_on_35577_cost4718/`. |

> **Later (batch 50, 2026-07-03/06):** team-best rebased to the einsum-reduction pipeline
> (2004), then QLinearConv-final (1786), then row-code decode narrow (**1773**, current). See
> `task201.md`.

## Cost analysis (historical — 4,718 / 9,738 / 35,577 era)

**Measured 2026-06-05 (`task201_v2_cropout_boolpad_opset20_edge_upperclamp_tvals23_int32.onnx`).**
The cropped-output rewrite is the current valid floor: cost **4,718** = memory **4,439** +
params **279**. It preserves the v1 decoded rule, but all output assembly stays at `7x8`
until the final one-hot Pad. Largest counted tensors:

| Item | Bytes | Role |
|---|---:|---|
| `color_rows` F32 `[1,1,13,30]` | 1,560 | Asymmetric Conv color decode; row-crops without a full 30x30 color grid. |
| `color13_f32` `[1,1,13,13]` | 676 | Static column slice of the decoded color grid. |
| `onehot_o` BOOL `[1,10,7,8]` | 560 | Final cropped one-hot before graph-output Pad. |
| `color13` / `is4` / `outside_frame` / `template_vals` `[1,1,13,13]` | 676 total | Minimal single-channel frame/template detection surface. |
| `gather_rows` + `template_read` | 147 | Fixed Gather template read over the 13x13 source. |
| 7x8 U8/BOOL render intermediates | ~280 | Active mask, canvas, frame color, final sentinel pass. |

Floor note: the remaining large anchors are the color decode (`color_rows` + `color13_f32`),
the required cropped one-hot expansion, and the full 13x13 outside-frame/template masks. Further
local graph surgery is now small; a sub-1k result likely needs a fundamentally different way to
avoid decoded-grid materialization, not just another mask fusion.

**Measured 2026-06-05 (`task201_v1.onnx`).** The decoded-rule rewrite hits the prompt target:
cost **9,738** = memory **9,447** + params **291**. The rule no longer carries any
`[1,10,13,13]` working planes. It decodes the input to a single-channel color-index grid and
keeps the final 10-channel expansion as the graph output.

Largest counted tensors:

| Item | Bytes | Role |
|---|---:|---|
| `color_rows` F32 `[1,1,13,30]` | 1,560 | Asymmetric Conv color decode; crops rows for free with a `[1,10,18,1]` kernel. |
| `canvas30` U8 `[1,1,30,30]` | 900 | Single-channel output canvas padded with sentinel 10 before free `Equal` expansion. |
| `color13_f32` `[1,1,13,13]` | 676 | Static column slice of the decoded color grid. |
| 26-ish BOOL/U8 `[1,1,13,13]` masks/canvases | ~4,394 | Frame/template masks, fixed Gather template read, and output assembly. |
| INT64 index vectors `[13]` | ~832 | Static-shape Gather row/col indices; avoids dynamic Slice rejection. |

Floor note: obvious remaining savings are small mask-fusion and index-plumbing reductions.
The hard anchors are the color source (~2.2k incl Conv+slice+cast), the 900-B output canvas,
the fixed template read, and the dependent bbox/orientation masks. A sub-10k build is proven
by measurement; a sub-4k claim would need a new way to avoid most per-cell mask materialization,
not just local graph surgery.

**Measured 2026-05-30 (worker-7).** The LIVE graph (cost 84,895) was ALREADY cropped to 13×13
at node 1 (`Slice(starts=[0,0],ends=[13,13],axes=[2,3])`). The cost is NOT from full-canvas
30×30 ops — it is dominated by the **18 distinct `[1,10,13,13] F16` one-hot planes (3,380 B each
= 60,840)** plus the single full-canvas `Cast(input→F16)[1,10,30,30]=18,000`. Confirmed by a
per-tensor replica of `calculate_memory` (`prototypes/inspect_task201.py`): my breakdown sums to
the official 84,587 mem exactly.

The rule (846bdb03): 4 color-4 corner markers define a box; colored shapes get placed inside.
The algorithm = frame/corner detection (ArgMax index machinery on tiny [13]/[1,13] tensors) +
two candidate one-hot placements built by 2-D `Gather` shifts of `v24`/`v26` + an orientation
score (ReduceSum of `v134` weighted by column-region masks `v1..v4`) + a final `Where` composite.

### v3 floor decomposition (cost 35,577 — at ≈1.0× the Path-A floor)
| Item | Bytes | Why irreducible (Path A) |
|---|---:|---|
| `_input_sliced` F32 `[1,10,13,13]` | 6,760 | input is F32; Slice can't change dtype. Slice-then-cast (6,760+1,690) beats cast-then-slice (9,000+1,690). |
| `v134_f16` cast `[1,10,13,13]` | 3,380 | `ReduceSum` rejects U8/BOOL at all opsets; F16 is cheapest legal (INT32 = 6,760). Sums ≤169 exact in F16. |
| 12× UINT8 planes `[1,10,13,13]` | 20,280 | 1 B/elem is dtype-minimal for 0/1. 12 = LIVE's irreducible single-consumer dataflow stages (v24,v26,v85,v86,v87,v132,v133,v134,v152,w176,v177,v181). |
| 2× F16 `[1,10,1,13]` (v134_h/hw) | 520 | column-sum scoring; sums>1 need F16. |
| ~11× U8/BOOL `[1,1,13,13]` masks | 1,859 | 1 B/elem; per-cell region masks + channel-max. |
| 11× INT64 `[13]` index machinery | 1,144 | ArgMax→INT64; could be INT32 (−~570) but touches the rule-critical index logic — deliberately NOT narrowed (risk ≫ 0.016 LB). |
| misc scalars / [13] / [1,13] | ~1,247 | ArgMax detection + scoring scalars. |
| params | 387 | index/axes/mask initializers (mostly the new axes-input + pads INT64 inits from the opset bump). |

**Floor reached (for the Path-A 35,577 graph).** Every `[1,10,13,13]` is now either UINT8
(minimal for 0/1), the one F16 forced by ReduceSum, or the F32 input slice forced by the F32
input. The only sub-floor path is a **color-INDEX `[1,1,13,13]` representation** — a Path-B rule
rewrite, out of scope for that byte-equal Path-A task. (Superseded: the batch-50 einsum-reduction
pipeline reached 1773 by a different route.)
