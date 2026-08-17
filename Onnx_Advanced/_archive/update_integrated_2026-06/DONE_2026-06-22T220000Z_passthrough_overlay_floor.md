# Passthrough-overlay tasks have a 2-plane floor that the #141 bit-AND trick can't reach

**Where:** task151 (full-H-line ∩ full-V-line → stamp a 3×3 yellow ring at the crossing,
center kept, everything else passthrough). Builders: `The_400_Tasks/Task_by_Task/task151/prototypes/build_task151_coord_route.py` (+ a maximally-fused sibling). Team-best = no-bias `Conv[10,10,3,3]` = 900, mem 0.

**What I discovered (transferable cost law for "passthrough + local overlay" rules):**

When the rule is `output = input passthrough EXCEPT a small region gets recolored`, the
tempting coordinate/free-output route (`Where(region_mask, color_vec[1,10,1,1], input) -> output`)
has TWO unavoidable charged intermediates, even though the final `Where` output is the
scorer-skipped `output` node:

1. **A region-detection plane.** To find the region from a one-hot input you need a
   per-cell channel reduction (slice ch0, or ReduceSum over channels) → a `[1,1,30,30]`
   plane = **3,600 B F32** (or 900 B if you can keep it BOOL, but the line-marginal
   needs the float/int counts).
2. **The `Where` condition is ONE 2-D tensor.** A 3×3 BOX is the AND of a row-band
   `[1,1,30,1]` and a col-band `[1,1,1,30]`; their AND result is `[1,1,30,30]` BOOL =
   **900 B**. You cannot keep the condition 1-D — a single 1-D condition paints whole
   rows/cols, not a box.

So the floor of the overlay route is **≥ 3,600 + 900 = 4,500 B**, which loses badly to a
0-memory dense Conv whose single output IS the graph output. Measured task151:
fused coord route = **7,713**, full = 12,213, both 266/266 correct. Conv = **900**.

**Key non-transfer to flag (this is the subtle part):** task047's #141 axis-bitset
`BitwiseAnd(rowbits[1,1,30,1], colbits[1,1,1,30]) -> output` hits 862 on the SAME
2-line geometry — so it's natural to assume task151 can reuse it. It can't. #141 produces
a **pure separable outer product** with NO input passthrough; every output cell is a
function of (row_code, col_code) only. task151 must reproduce all 10 input channels
everywhere outside the ring, and a bit-AND of two 1-D code vectors cannot reproduce an
arbitrary 10-channel input plane. **#141 wins only when the output is fully determined by
(row,col); the moment you need input passthrough, you're back to a charged detection plane
+ a charged 2-D overlay condition, and a 0-memory dense emitter (Conv) is the floor.**

**What's missing from the library:** #141 (techniques.md:4902) and #142 describe the
free-output separable/dissolve wins but do NOT state the passthrough exclusion. Suggest
extending #141/#142 (or by_symptom) with: "free-output separable applies only when the
output is a pure (row,col[,channel]) function; passthrough-of-input forces a detection
plane + 2-D overlay condition (≥4,500 B for a 30×30 box overlay), at which point a
0-memory dense Conv/output-node emitter is the floor." This is the exact trap that makes
task151 a confirmed 900-WALL despite looking like task047.
