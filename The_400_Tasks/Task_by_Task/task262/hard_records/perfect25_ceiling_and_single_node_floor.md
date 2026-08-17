# task262 hard record — perfect-25 ceiling audit + single-node floor (2026-07-07, bo_ng_cld_worker2)

Live bar at audit: **cost 80**, sha256 `560f5b67f2de5fd3bec4eedb4681f9b536d993fa9e1567c05af6ada329014095`
(manifest, owner Vamshi, 2026-07-02). Source: `tools/ng_current_bar.py --task 262 --json`.

## Rule (derived from JSON, all 4 train + 1 test verified)
Input 3×3: each row has exactly one `5` (rest `0`). The COLUMN of the `5` maps to an
output color that fills the entire row:  **col0→2, col1→4, col2→3** (non-monotonic).
Officials I/O = one-hot `[1,10,30,30]`, 3×3 at top-left, background color 0 elsewhere.

## DEAD ROUTE 1 — perfect-25 cost≤1 (zero-param direct-output). INFEASIBLE.
Forced-component argument: output alphabet {2,3,4} is **disjoint** from input alphabet
{0,5}. Identity/Transpose/Flip/Rotate/Reshape only reorder existing values — they can
never synthesize colors 2/3/4. Producing a new channel activation requires a payload
initializer (color-map tensor or learned weights). Additionally the map is position-
dependent and non-monotonic (0→2,1→4,2→3), so 3 arbitrary col→color assignments must be
encoded. ⇒ any correct graph carries ≥1 payload initializer ⇒ cost > 1. Not a perfect-25.
Kill rule fired: no candidate is visible-correct at cost≤1. Did NOT grind.

## DEAD ROUTE 2 — beat 80 via Slice + Einsum (2-node). NET LOSS.
Grader **requires the full 30×30 output canvas**: a `[1,10,3,3]` output model scored
`visible 0/11 FAIL` (empirically confirmed — small-output family is dead).
Consequence: a 30-wide tensor is forced in the final op. `input` (30-wide broadcast
source) is a graph INPUT → not charged. But any 2nd node emits an intermediate; the
smallest one carrying per-row column-position over 30 rows × 3 cols = 90 F32 = **360 B**,
which alone exceeds 80. Measured:
- `task262_slice_einsum.onnx` (Slice ch5+cols0:3 → Einsum `bkrj,jo,blrc->borc`, M[3,10]):
  visible 11/11 PASS, but profile **396** (memory 360 + params 36). LOSS vs 80.
Slicing rows too shrinks the intermediate to 36 B but forces a 3×3 output → visible 0/11.

## FLOOR — single-node Einsum = 80 (the live best is at floor).
Zero-intermediate ⇒ single node. Distinguishing 3 column-classes → 3 output channels
needs rank ≥ 2 (rank-1 ⇒ output = scalar·V ⇒ argmax depends only on sign(scalar) ⇒ ≤2
classes). Rank-2 over the 30-position column axis forces Q[30,2]=60; rank-2 → 10 output
channels forces V[2,10]=20 ⇒ **80**. Alternatives are worse: Conv weight [10,10,1,3]=300;
explicit channel-selector Einsum `bkrj,k,jp,po,blrc→borc` = sel[10]+Q[30,2]+V[2,10]=90.
The live `task262_pairwise_rank2` graph already sits exactly at this 80 floor.

## Remaining hypothesis (unexhausted)
A single-node op that reads absolute column position with fewer than 60 params — none
found; the 30-length column axis is intrinsic to any weight indexed by column, and
slicing to shorten it reintroduces a charged intermediate. Treat 80 as the practical
floor pending a novel single-node contraction.

## VERDICT: WALL. No sub-80 candidate. Nothing staged (no tie/loss staged).
