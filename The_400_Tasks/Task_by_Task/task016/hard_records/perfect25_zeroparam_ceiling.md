# DEAD ROUTE — task016 perfect-25 / cost≤1 zero-param ceiling (ZeroopsweepCeilingAudit)

- **Live bar:** cost **10**, sha `fb83a8e988f586009bf17279b7aedb194e3492a8e6f357be6d5e9ed903615733`
  (source: manifest, authoritative, not stale; owner Vamshi). Verified via
  `tools/ng_current_bar.py --task 16 --json` on 2026-07-07.
- **Route attempted:** the prompt's hypothesized "perfect-25" family — a no-payload
  direct-output single node (Identity / Transpose perms / attr-only op) that prices ≤ 1.
- **Exact failed gate:** `ng_gate visible --task 16` = **0/267 (FAIL)** for every zero-param
  candidate:

  | candidate | op | cost | visible |
  |---|---|---:|---|
  | `zp_identity.onnx` | Identity | 0 | **0/267** |
  | `zp_transpose_0123.onnx` | Transpose perm=[0,1,2,3] | 0 | **0/267** |
  | `zp_transpose_0132.onnx` | Transpose perm=[0,1,3,2] | 0 | **0/267** |
  | `zp_slice_reverse.onnx` | Slice axis1 steps=-1 | 4 | **0/267** |

  (builder: `prototypes/zeroparam_sweep.py`; cost via `_score_v2.score_network_v2`.)
  No truth/ab run — a visible miss is a certain hidden 0 (traps §VISIBLE-FAIL).

- **Why it's dead (forced-component argument):** task016 is ARC `0d3d703e`, a color
  permutation. The one-hot encoder puts color `c` on channel `c`, so the rule is an
  **arbitrary permutation WITHIN axis 1**, idx `[0,5,6,4,3,1,2,7,9,8]` = involution
  `(1 5)(2 6)(3 4)`, fixing `{0,7}`. A cost≤1 direct-output op cannot realize this:
  - **Identity** leaves channels in place (0/267).
  - **Transpose** permutes *whole axes*; any shape-preserving perm leaves axis-1 order
    untouched — it can never reorder *within* an axis (0/267).
  - **Slice steps=-1** only realizes the channel *reversal* idx `[9…0]` (cost 4); task016's
    idx is a generic 8-channel involution, not a reversal (0/267).
  - No other zero-payload ONNX op reorders arbitrary elements within an axis. Arbitrary
    within-axis reordering **requires** an explicit index/weight tensor. The minimal such
    node is `Gather idx[10]` = **10 params**. Sparse/sub-INT32 indices either score `None`
    (unshippable) or add ≥40 mem; every data-path alt (Split/Slice/Reshape/MatMul/Conv/
    Scatter) materializes ≥36000 B or ≥100 params — see the 15-candidate PROVEN floor
    table in `task016.md`.

- **Verdict:** WALL. The perfect-25 ceiling audit is a rumor; task016 is NOT in the
  zero-public-weight family. **Floor = 10, already team-best.** No candidate strictly
  below 10 exists. Do not re-explore the zero-param / perfect-25 family for task016.

- **One remaining hypothesis (already killed in the floor table):** none. The cost-4
  full-axis Slice reversal is the only sub-10 recolor single-node op and it is idx-specific
  (reversal only). task016's non-reversal perm forecloses it.
