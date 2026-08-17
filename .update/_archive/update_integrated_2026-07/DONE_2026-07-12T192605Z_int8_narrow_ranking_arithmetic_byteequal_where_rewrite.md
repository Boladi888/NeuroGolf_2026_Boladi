# Transferable insight — narrow an INT32 integer "ranking/code" chain to **INT8** for a byte-equal shave; `Where` has no INT8 kernel so rewrite it as `b+cond*(a-b)`; prove no-overflow with a full-corpus range audit

**Source**: task096 (`bo_ng_cld_worker1`, 2026-07-12). LIVE **5537** (Sho, sha `a05a9164…`), a
2026-07-10 WALL declared it floored ("only a forbidden rung-epsilon shave remains"). **NEW WIN**:
byte-equal **5013** (−524, +0.099 pts), staged `task096_int8rank_cost5013` (sha `2e1aff30…`).
Complements the 2026-07-11 task005 "per-tensor charged audit breaks a WALL" lesson — same family
(the WALL priced the FORCED tensors but never dtype-audited the selection/ranking scaffolding).

## The lever — INT32[k] integer arithmetic is 4× over-charged
`calculate_memory` charges every node-output at `elements × itemsize`. A ranking/decision subgraph
(spans, extents, radius codes, offsets, "ambiguous" branches) is usually built in **INT32** out of
habit — but its values are tiny integers. INT8 costs **1 B/elem vs 4**. On task096 the ranking was
~46 `INT32[5]` tensors (~920 B); narrowing to `INT8[5]` (5 B each) saved ~690 B gross.

## ORT op×dtype reality at opset 14/18 (probed, CPU EP)
- **INT8 native + integer-exact**: `Add Sub Mul Div Mod Max Min Equal Less`. `Div`=C-truncation
  (`45//8=5`), `Mod` exact — identical to INT32 when operands & results fit `[-128,127]`.
- **INT8 has NO kernel**: `Where` (and INT16 has no `Max`/`Min`; F16 `Div` returns floats & F16
  `Mod` errors "fmod must be true"). So a mixed chain can't be one dtype.
- **Fix for `Where`**: `Where(cond,A,B)` on INT8 → `B + Cast(cond,INT8)*(A−B)` = `Add(Mul(Sub(A,B),
  Cast(cond)))`. Integer-exact, no overflow if `|A−B|` and the product fit INT8. Cost: +3 tiny
  tensors/Where; still a net win vs the 15 B/tensor saved. (`Where 17→8`, `Add/Sub/Mul +9` each.)
- **Boundaries stay INT32**: `Gather`/`ScatterElements` indices and `Slice` starts/ends require
  int32/int64 — keep those (`a11`,`a_col`,`radius_idx`,`starts2`) INT32 and Cast INT8→INT32 at the
  edge (one cast ≈ 20 B, cheap vs the chain saving). `ArgMax` outputs are forced INT64; Cast them
  straight to INT8.

## The mandatory proof — a full-corpus range audit (overflow is a silent hidden-data trap)
INT8 is byte-exact **only if no value ever leaves `[-128,127]`**. Visible/`ab --n 2000` can PASS
while a rare hidden sample overflows → wrap → divergence (a classic false-confirm). So:
1. Expose every narrowed tensor as an output on the **INT32 LIVE** graph, run the **full cached
   corpus** (`corpus.iter_tensors(task, 200000)` — it's already built for shipped tasks), record
   global min/max. Confirm `max|·| < 128` **with margin**, and prove the tight bounds analytically
   (task096: `col_span≤19` (ArgMax over a 19-axis) & offset clamp `≤10` ⇒ ambiguous codes ≤116;
   selected `code_i32≤45`). 2. Run LIVE-vs-cand **byte-equal over all 200k** (pure-integer transform
   ⇒ any divergence == an overflow): got **200000/200000, 0 div**. This is stronger than the `ab
   --n 2000` ship gate and is what makes a dtype-narrow trustworthy on hidden data.

## Mechanics that bit me (do these)
- **Preserve the original `value_info`** — only patch dtype on converted tensors; do NOT
  `del value_info[:]` and re-infer, or a data-dependent `Slice` (e.g. a dynamic-offset LUT crop)
  loses its hard-coded shape → symbolic dim → `calculate_memory` silently returns `None`.
- After rewiring shared int constants to INT8 dups, **prune orphaned INT32 initializers** (reclaims
  those params). Give each new node a unique `.name` (renaming a node's output but not its `.name`
  → "two nodes with same name" load error).

## Rule of thumb
When a LIVE graph has an INT32 decision/ranking chain of small-magnitude integers, price an INT8
narrow: keep index/`Slice`-bound tensors INT32, rewrite `Where` as `b+cond*(a−b)`, keep `Div`/`Mod`
native, and **prove `max|value|<128` over the full corpus + 200k byte-equal** before staging. A
"floored / only rung-epsilon left" WALL that never dtype-audited the ranking scaffolding is often
wrong by hundreds of bytes.
