# task087 — rot180 perfect-25 ceiling audit: WALL at 5

**Date:** 2026-07-07 · worker `bo_ng_cld_worker2` (Claude lane)

## Live bar at audit time
- **live_cost = 5** (authoritative: manifest, owner Aditya)
- live sha256 = `1271ff5e7d4160c264c13690ccfe8081f64fe7cbbe588828c0b4054d129ad7d9`
- live graph = `Team_Best_Onnx/task087.onnx` = **single RoiAlign node**, rois F32[1,4] (4 params) + batch_indices INT64[1] (1 param) = **cost 5** (memory 0 + params 5). Verified `profile_real_cost --arcgen 266` → static=5 profile=5. Visible 266/266 PASS.

## Rule (confirmed)
`rot180` (reverse spatial axes 2 & 3) on ALL 266 train+test+arc-gen pairs — `np.rot90(i,2)==o` for every pair, 0 mismatches. This equals a full-canvas reverse of the [1,10,30,30] one-hot tensor (RoiAlign live best proves canvas-reverse ≡ grid-rot180 here).

## Prompt hypothesis tested
floor_est ≈ 1 (perfect-25, cost≤1 no-payload direct-output). **Confidence was MED. REFUTED by this audit.**

## The forced-floor argument (why cost 5 is the floor, not 1)
Cost = `memory + params`. Two forcings:

1. **Single-node is forced.** Any reversal split across ≥2 nodes materializes at least one full-canvas `[1,10,30,30]` F32 intermediate = **36,240 B billed memory**. Empirically confirmed: `Range + Gather(axis2) + Gather(axis3)` (3 scalar params, computed reversed indices, zero stored index payload) profiled at **cost 36,243** (memory 36,240). So the entire reversal must happen inside ONE node whose sole output is the graph output (memory 0). This also kills ReverseSequence (reverses ONE axis per node → 2 nodes → billed intermediate) and any AffineGrid+GridSample / MatMul-anti-identity / Einsum-permutation decomposition.

2. **Reversal is never a zero-payload default.** No ONNX op reverses spatial axes as its default behavior; every single-node reversal carries a mandatory geometry tensor:
   | single-node op | payload | cost | status |
   |---|---|---:|---|
   | RoiAlign | rois F32[1,4]=4 + batch_indices INT64[1]=1 | **5** | LIVE / floor |
   | MaxRoiPool | rois F32[1,5]=5 (batch idx folded in) | 5 | tie, not a win |
   | Slice steps=-1, axes [2,3] | starts/ends/axes/steps each [2] = 8 | **8** | built + profiled = 8, loses |
   | Conv / MatMul / Einsum flip | anti-identity 30×30 ≥ 900 params | ≥900 | loses |
   | GridSample (grid input) | grid [1,30,30,2] = 1800 | ≥1800 | loses |

   Minimum over all single-node reversals = **5**. Both mandatory RoiAlign inputs are irreducible (num_rois=1 is minimal; batch_indices is required). Slice reversing 2 axes cannot go below 8 (4 index tensors × len 2).

## Verdict
**WALL at cost 5.** rot180 is NOT a perfect-25 (cost≤1) task — it has a hard geometry-payload floor. RoiAlign is already sitting on the forced floor. No stageable candidate (ties do not stage; nothing prices below 5).

## Gates run (cheapest-first; losers not carried to truth/ab per ladder)
- `profile_real_cost` live RoiAlign → 5 (baseline). visible 266/266 PASS.
- `profile_real_cost` Range+2Gather → **36,243** (memory 36,240) — DEAD.
- `profile_real_cost` Slice steps=-1 → **8** — loses to 5.
- No truth/ab/battery run: every alt candidate priced ≥ live 5, so a `--full` gate would be pure burn (protocol gate ladder).

## One remaining hypothesis (for the next batch, low confidence)
The only way below 5 would be an ONNX op that reverses BOTH spatial axes in a single node with ≤4 params of payload. None exists in opset ≤ 21. If a future opset adds a payload-free `Flip`/reverse attribute (e.g. a Slice `steps` attribute rather than input, or a Reverse op), rot180 would drop toward cost 1. Until then, 5 is the floor. Do not re-explore multi-node reversals (memory-billed) or ≥8-param Slice.
