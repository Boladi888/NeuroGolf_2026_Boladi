# task140 — dead route: perfect-25 ceiling (cost ≤ 1) for rot180

**Live bar:** cost **5** (manifest, owner Aditya, sha256 `d4830ee432562703ce0e624377faf58bf72b5433e6b4616e8268be0d20a86dc8`, updated 2026-06-19).
LIVE = single **RoiAlign** node: `rois` F32[1,4]=4 + `batch_indices` INT64[1]=1 → params 5, memory 0. static=5, profile=5.

**Attempted route:** express rot180-of-3×3-at-top-left as a cost ≤1 (perfect-25) direct-output graph, per the CeilingAudit prompt.

**Exact failed gates (2026-07-07, bo_ng_cld_worker3):**
- `probe_identity.onnx` (Identity, cost **0**): `ng_gate visible --task 140` → **0/265 (0.00%) FAIL**. The transform is a genuine reflection (`output != input` on every real pair), so the cost-0 perfect-25 candidate is a certain hidden 0.
- `probe_slice_rev.onnx` (whole-canvas Slice steps=[-1,-1], cost **8**): `ng_gate visible --task 140` → **0/265 (0.00%) FAIL**. A payload-free whole-canvas reversal places the 3×3 content at the *bottom-right* corner; the target keeps it top-left. Sub-region placement is mandatory.
- Both are visible-zeros → per traps §VISIBLE-FAIL, no truth/ab run.

**Why it's dead (forced floor at 5):**
Grader `points = max(1, 25 − log(max(1, memory+params)))` → perfect-25 needs `memory+params ≤ 1`, i.e. an attribute-only (0-payload) graph. `calculate_params` counts graph initializers **and Constant-node tensor values** (constant-folding enabled), so there is no free-payload trick.
rot180-of-a-sub-region is a *positional index-reversal* within `[0:3,0:3]`, zeros elsewhere. No attribute-only ONNX op reverses index order within an axis: Transpose only swaps axes (A[j,i], not A[2−i,2−j]); DepthToSpace/SpaceToDepth are monotonic block reshuffles; Reshape/Identity/Flatten/Squeeze don't reflect. So a 0-param graph is impossible.
Every op that *can* express the reversal carries a counted payload:
- RoiAlign — `rois[1,4]` + `batch_indices[1]` = **5** (both inputs required by the op signature; cannot drop to 4).
- MaxRoiPool — `rois[1,5]` = **5** (batch index folded into rois; still 5).
- Slice-reverse (→[1,10,3,3]) + Pad (→[1,10,30,30]) = **≥16** (starts/ends/axes/steps 8 + pads 8).

The region-resample floor is 5 and it is already shipped. **Verdict: WALL at 5. Perfect-25 (≤1) is infeasible for task140.**

**One remaining hypothesis (thin):** an opset/op that reflects a sub-region using only attributes (0 payload) — none exists in the current allowed op set. If ONNX ever adds a `Flip`/`Reverse`-with-axis-attribute op (no index tensor), revisit; until then do not re-explore.
