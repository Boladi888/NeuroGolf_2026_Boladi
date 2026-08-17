# task307 — sub-4 WALL (seed-swap ConvTranspose floor = 4)

- **Worker:** bo_ng_cld_worker3 (Claude lane), batch 53 audit pass, 2026-07-07.
- **Live bar at audit:** cost **4**, sha256 `31a84ecda537ea6c1cacaacfebe59e7e1fec88750d622469cd21fb302801eb23`,
  owner Vamshi, authoritative source = manifest (`ng_current_bar.py --task 307 --json`).
  Verified: `profile_real_cost` static=4 profile=4; `ng_gate visible` 266/266 PASS.

## What the live cost-4 graph is (documents the trick — supersedes the stale "floor 40" in task307.md)
Single `ConvTranspose`, opset 11, with the tensors **swapped**:
- `seed=[1,1,2,2]` all-ones (4 params) is fed as **X (data)**.
- runtime `input=[1,10,30,30]` is fed as **W (weight)** — runtime input is NOT counted.
- attrs `dilations=[2,2] strides=[1,1] pads=[0,0,30,30]`.
The 2×2 dilated seed convolved (transpose) against the input-as-kernel writes each
input pixel into a 2×2 same-color block. Only counted bytes = the 2×2 seed = **4 params**,
memory 0. This is the #80-inverse block-expansion primitive done as a params-only final.
It already broke the old cost-40 grouped-ConvTranspose floor recorded in task307.md.

## Sub-4 attempts (all dead)
| # | route | cost | gate | verdict |
|---|---|---:|---|---|
| 1 | seed with a corner zeroed (`[1,1,1,0]`) | 4 (zeros still count) | visible **0/266** | every 2×2 block needs all 4 taps = pixel value; drop one ⇒ wrong. No param saving anyway. |
| 2 | 2-param single-axis seed `[1,1,2,1]` | 2 | shape-inference error (dim3 29≠30) + visible **0/266** | a 2×1 footprint can't span the 30-wide output nor tile 2×2; single-axis only. |
| 3 | 1-param seed `[1,1,1,1]` | 1 | (structural) | no duplication — identity-ish, cannot make a 2×2 block. |
| 4 | separable 2× = two ConvTransposes (2×1 then 1×2 seeds) | 2+2 params **+ materialized intermediate** | — | intermediate `[1,10,30,~]` F32 ≈ 18k memory ≫ 4. Worse. |
| 5 | Resize x2 + Slice/crop; DepthToSpace/SpaceToDepth repack; Tile | — | — | all need a non-output crop/repack plane, memory ≥9,000 (task307.md don't-do). |
| 6 | GridSample fixed grid | 1,800 | — | dead (task307.md). |
| 7 | perfect-25 identity / Transpose-perm / flip / rotate (prompt rumor) | ≤1 | — | disproven by the task rule: this is genuine 2× **tiling**, not a shape-only permutation. Pre-scan found no whole-grid identity/transpose match. |

## Forced-component argument (why 4 is the floor, not a give-up)
2×2 block expansion is a linear map sending each input pixel to a 2×2 all-ones block.
The cheapest counted encoding of that block is a single 2×2 initializer = **4 elements**.
Params count elements (dtype-blind, zeros included), so:
- product-4 shape `[.,.,2,2]` is the ONLY 2D footprint that tiles 2×2 → 4 params minimum;
- product ≤3 shapes (1×1, 2×1, 1×2) cannot form a 2×2 footprint (verified 0/266 / shape error);
- separable splitting trades the 4-param tensor for 2+2 params **plus** a materialized plane (worse);
- every non-ConvTranspose family that avoids a 4-tap seed forces either ≥40 params or a ≥900 B plane.

**Verdict: WALL at cost 4.** No scoreable static graph tiles 2×2 for <4 counted bytes.
One remaining (low-confidence) hypothesis if ever revisited: a runtime-derived `ConstantOfShape`
seed whose shape tensor is itself uncounted — but the shape input `[1,1,2,2]` is a 4-element
INT64 initializer (same 4 params), so it does not beat 4. Do not re-explore without a new
scoring law.
