# task286 — WALL verdict (2026-07-02, bo_ng_cld_worker2, Claude lane)

**Bar:** team-best **21558** (owner Vamshi, manifest sha `406cc1e321bd…`,
updated 2026-07-02T02:35:15Z). Profiled locally: `static==profile==21558`
(fully static, 0% gap). No candidate produced below it → **nothing staged.**

## What changed since the local task doc was written
The home task286.md / Boladi mirror / library DB all describe the OLD
53,464-byte MaxPool/Min BFS flood and cite a documented floor of
"flood depth × per-step (geodesic) + I/O ≈ 23,210" with "sub-20k not supported".
**That floor is obsolete.** Vamshi shipped a fundamentally different
**bit-parallel column-packed flood** at 21558 — below the old 23,210 "floor".

## Vamshi's graph (the thing to beat), reverse-engineered
- 4091 nodes, opset 18, fully static. Input `[1,10,30,30]` F32 → output
  `[1,10,30,30]` UINT32 (dtype 12).
- Columns of the 25-wide active grid are **bit-packed into UINT32 scalars**
  (bit r = row r). 3 color roles extracted by Einsum (`h_pass_c`=open/0,
  `h_seed_c`=seeds, `h_cyan_c`=barrier/8) split low/high 15 rows then OR-merged.
- Flood = repeated passes of: per-column **segmented log-fill** vertical
  (Hillis-Steele, shifts **{1,2,4}** each direction, using run-masks
  `o&(o<<1)` etc. that are pure-open-derived and **hoisted/computed once**),
  then a horizontal merge with neighbor columns. ~9 effective passes.
- Output assembled via checkerboard parity (`h_rowpar/h_colpar/h_parmix`),
  ArgMax label select, ScatterND into `[10,30,1]`, final broadcast to
  `[1,10,30,30]` (the output tensor is **NOT charged** — only intermediates are).

## Cost model (measured)
`cost = Σ(intermediate activation bytes) + Σ(initializer element counts)`
= **20912 + 646 = 21558**. Graph inputs and the final output tensor are NOT
charged. Breakdown of the 20912:
- **16,248 B (78%)** = ~4020 scalar `[1,1]` UINT32 bit-ops (BitwiseAnd 1436,
  BitwiseOr 1337, BitShift 1244) — the flood core.
- ~4,664 B = I/O tail (ScatterND 1200, Einsum 800, Cast 720, Concat 632,
  Unsqueeze 480, 19×[1,30] masks, Gather 276).

## Rule (verified)
4-connected maze reachability: fill `0`-cells reachable from the 2 seed cells
through non-barrier (barrier = color `8`) cells; paint a `(r+c)`-parity
checkerboard of the two seed colors. A numpy 4-conn BFS reproduces the
ARC-GEN ground-truth changed-cells **exactly** (100% on sampled corpus).
→ transitive closure is genuinely required; **no non-flood shortcut exists.**

## Gates on the team-best (reference targets any candidate must match)
- `ng_gate visible --task 286` → **265/265 (100%)** (old LIVE was only 257/265;
  Vamshi's rule is strictly more correct).
- `ng_gate truth --n 5000` → **5000/5000 (100.000%)**.
- `profile_real_cost --arcgen 266` → static=21558 profile=21558.

## Levers tried and why they failed
1. **CSE / common-subexpression cleanup** — custom exact structural CSE pass
   (`prototypes/cse_pass.py`): **0 nodes removed.** The graph is already
   optimizer-clean; LEFT vs RIGHT shifts and the evolving flood state make the
   ~4000 ops genuinely distinct. No free bytes.
2. **Vectorizing scalars → `[1,W]` tensors** — byte-neutral. Cost = Σ bytes =
   ops×width; scalar `[1,1]`×25 columns == one `[1,25]` op. No win.
3. **Three independent from-scratch flood schedules** (`prototypes/flood_core_sim2.py`,
   correct wall-safe segmented fill, serpentine Gauss-Seidel horizontal,
   never=0/500 on corpus, best-case 1 vfill/pass):

   | schedule | shifts | max passes | flood bytes | vs Vamshi 16,248 |
   |---|---|---|---|---|
   | partial-log | {1,2,4} | 20 | **42,800** | 2.6× worse |
   | mid-log | {1,2,4,8} | 17 | 47,100 | 2.9× worse |
   | full-log | {1,2,4,8,16} | 17 | 57,700 | 3.6× worse |

   **The full-log "fewer passes" hypothesis is FALSIFIED.** Deeper vertical
   shifts reduce pass count only marginally (20→17) because the pass count is
   bounded by the number of **horizontal direction-reversals** in the maze
   geodesic (~up to 20 on a 500-grid sample), not by vertical fill depth. The
   extra shift cost per pass ({1,2,4}=21 → {1,2,4,8,16}=33 ops/col) more than
   offsets it. Vamshi's ~9 passes vs my naive serpentine's ~20 means his
   horizontal wavefront is ~2× more efficient than straightforward serpentine
   (an unrecovered trick — likely horizontal log-doubling or a bidirectional
   scheme), on top of pruned late-pass shifts.
4. **Cutting Vamshi's pass count** — rejected: he tuned to truth 100% and the
   task's own D\*-pinning history (hidden needs the exact maximum depth) makes
   trimming a certain hidden-0 risk with no local way to prove safety.
5. **I/O tail (~4664 B)** — not independently reducible without rebuilding the
   whole flood; any rebuild inherits the ≥16,248 B core, so total > 21558.

## Forced-component floor argument
The credited cost is Σ intermediate bytes. The flood requires transitive
closure over a wall-blocked 25×25 maze whose hidden geodesic depth is pinned
near-maximal. Any wavefront representation charges (bytes/step × steps); a
correct wall-safe bit-parallel step needs the segmented run-masks + shift/and/or
per column, and the number of steps is lower-bounded by the maze's
direction-reversal count. Every independent formulation I built lands the flood
core at ≥42,800 B — 2.6× above Vamshi's 16,248 B — even before the shared
~4,664 B tail and 646 params. Vamshi's specific horizontal-propagation scheme
is materially tighter than a from-scratch bit-parallel flood, and no cleanup
lever (CSE, vectorize, dtype) applies. **Verdict: WALL at 21558 for this
worker's toolkit.** Re-attack would require recovering and then out-optimizing
Vamshi's horizontal wavefront trick.

## Artifacts (task-local)
- `prototypes/vamshi_ref/task286_teambest_21558.onnx` (copy of team-best, sha
  `406cc1e321bd08ccce766912914f9e49b997ceb3ba68c1cc767d5fd71345ff89`)
- `prototypes/vamshi_ref/full_nodes.txt` (full node dump)
- `prototypes/cse_pass.py` (exact structural CSE — 0 removals)
- `prototypes/flood_core_sim.py`, `flood_core_sim2.py` (from-scratch flood
  op-count experiments)
