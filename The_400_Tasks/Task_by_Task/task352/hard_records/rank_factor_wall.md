# task352 — GroupedConvRankFactor: WALL @ 460 (forced floor)

- **Live bar:** 460 (manifest, authoritative), sha256 `ff0694c98a21d92ccd938f8f42dd04b5de7e79b93d7c877df49b8e82591ee83a`, owner Aditya, 2026-06-10.
- **Verdict:** WALL. 460 is the forced floor under the `memory + params` cost model. The prompt's ~300 target rests on a params-only mental model that this cost function does not use.

## LIVE structure (decoded)
Single grouped Conv, `group=2`, `W[10,5,3,3]`=450 + `b[10]`=10 = **460**. pads=1.
- `out[0] = in[0] − nbr8(in[2]) − 0.5`
- `out[1] = 8·in[0] + 9·in[1] + nbr8(in[2]) − 8.5`
- `out[2..9] = in[2..9]` (identity pass-through)
- `nbr8(x)` = 8-neighbor sum = box3×3(x) − x. The "2 non-separable kernels" are the two rank-2 rings in out0/out1; both are the same `ring = ones3×3 − center`.

## Why the prompt's rank-factor route cannot win (two independent failures)

**1. Params count ELEMENTS, not nonzeros or rank** (`neurogolf_utils.calculate_params`, line 309: `math.prod(init.dims)`).
"27 of 450 nonzero" is irrelevant — a dense `[10,5,3,3]` weight always costs 450 params. ONNX Conv has no sparse-weight form. The `[10,5,3,3]` shape is FORCED: producing 10 output channels with `group=2` (`C/group=5`) gives weight `[10,5,3,3]`. group must be an equal divisor of 10 (∈{1,2,5,10}); the cross-channel coupling {out0,out1,out2}←{in0,in1,in2} needs a ≥3-channel group, so only group=2 qualifies (group=5→2ch, group=10→depthwise both break the coupling). group=1 is worse ([10,10,3,3]=900). ⇒ single-Conv weight floor = 450, +bias[10] (the −0.5/−8.5 constants force a length-10 bias) = **460**.

**2. Any multi-node decomposition incurs ≥900 B of charged intermediate memory** (`calculate_memory`, line 234: only tensors literally named `input`/`output` are exempt; all other node outputs charged `num_elements × itemsize`).
Every spatial intermediate is `[1,C,30,30]`, min 900 elements (C=1). Even as BOOL/INT8 that is **900 B > 460** — one intermediate alone blows the whole budget. Separable-splitting the box filter (row-conv ∘ col-conv) or factoring rank-1 kernels ADDS nodes ⇒ ADDS intermediates ⇒ strictly worse.

## Empirical confirmation (gates run)
| candidate | nodes | params | memory | profile cost | verdict |
|---|---|---:|---:|---:|---|
| LIVE (grouped Conv) | 1 | 460 | 0 | **460** | bar |
| cand_v1 (Split+box-Conv+arith+Concat) | 13 | 17 | 64,800 | 64,817 | LOSE (memory) |
| cand_2node (Conv → Add-bias) | 2 | 460 | 36,000 | 36,460 | LOSE (1 F32 intermediate) |

- `ng_gate visible --task 352 --cand cand_v1.onnx` → `266/266 (100.00%) PASS` (function is correct; byte-equal to LIVE on 40 random ARC grids, max abs diff 0). Correctness was never the problem — cost is.
- `profile_real_cost --arcgen 266`: LIVE=460 (mem 0 + params 460); cand_v1=64,817; cand_2node=36,460.

## Remaining hypothesis
None viable. To beat 460 you would need a SINGLE node (0 charged intermediates) with <460 weight+bias elements that reproduces per-pixel 8-neighbor aggregation + cross-channel mix + additive bias. No ONNX op below Conv[10,5,3,3]+bias[10] does this. **460 is the floor. Do not re-explore rank factorization / separable split / channel-drop for this task.**
