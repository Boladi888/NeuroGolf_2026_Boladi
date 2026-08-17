# J.14 — Z‑order (Morton) curve mapping

## Verdict
**Partially viable** — the cheapest possible 2‑D→1‑D linearisation is a length‑900 Gather initializer, which is essentially free (3 600 B params at int32) and gives downstream Slice nodes (also cost 0) over physically‑local windows. The catch is that "physically‑local along Morton" ≠ "physically‑local in 2‑D" near every quadrant boundary, so it only beats Conv on row/column/quadrant‑structured tasks.

## Actionable ONNX design rules
- **Morton index initializer for 30×30:**
  - Build at compile time: for each `(r,c)` with `0 ≤ r,c < 30`, set `morton[r*30+c]` = bit‑interleave of `r` and `c` (5 bits each ⇒ 10‑bit code, range 0–1023). Discard nothing — we keep all 900 entries; the discontinuities of the curve at non‑power‑of‑two boundaries are absorbed because we only ever use the table as a permutation, not as a recursive iterator.
  - **Byte cost:** length‑900 `int32` = **3 600 B params**; `int64` = 7 200 B params. Either is < 0.5 % of the 1 509 949 B model cap.
- **Inverse table:** an additional length‑900 int32 `inverse_morton` (3 600 B) lets us round‑trip back from Morton order to (row,col) at zero MAC cost.
- **Cost of one full 2D→1D→2D round trip on `[1,10,30,30]`:**
  - `Reshape [1,10,30,30] → [1,10,900]` (free, shape‑only).
  - `Gather(axis=2, indices=morton)` (free per locked rule; output `[1,10,900]` ⇒ 36 000 B activation memory).
  - …downstream Slice/Conv1D/Add work on the 1‑D form…
  - `Gather(axis=2, indices=inverse_morton)` (free).
  - `Reshape [1,10,900] → [1,10,30,30]` (free).
  - **Round‑trip overhead:** 0 MAC + 36 000 B activation per intermediate tensor + 7 200 B params (two tables, int32). This is < 1/10 of a Conv 1×1.
- **Where it pays off:** tasks whose answer is "fill row" / "fill column" / "fill quadrant" become 1‑D Slice operations. A row‑fill on 2‑D requires a Conv 1×1 with channel‑dependent kernel (90 100); on Morton‑linearised it can be a single Slice + Concat which is free. **Best case savings: a full Conv 1×1 (90 100) per row/column rule.**
- **Where it does NOT pay off:** 4‑neighbour Conv tasks. Two cells that are vertically adjacent in 2‑D can be far apart in Morton order — for the 30×30 case the largest Morton "jump" between vertical neighbours is up to ~32. A Conv 3×3 in 2‑D has no equivalent 1‑D Conv along Morton.
- **Non‑power‑of‑two:** 30 is not a power of 2; standard practice (`plavin/morton`) is to embed in 32×32 and skip out‑of‑range indices. Because we pre‑compute the table off‑line, the curve's "skip" structure is invisible to ONNX — we just store a permutation of `range(900)`. **This does NOT break anything** and does NOT require padding the activation tensor to 32×32.

## Literature survey & project‑mapping commentary
- **Hilbert flatten** (`scispace.com/.../hilbert-flattening.pdf`) reports Hilbert > Morton > Zigzag for 1‑D CNN locality preservation on real images. Hilbert tables for non‑power‑of‑two sizes are also possible (the same `plavin/morton` repo provides 3‑D variants). For the NeuroGolf budget the table cost is identical (900 ints) so **prefer Hilbert over Morton when the task is a generic local 2‑D rule**; prefer Morton only when its quadrant structure aligns with the task (e.g., recursive sub‑grid rules).
- **ARC families with potential Morton wins:**
  - Row‑fill / column‑fill / "extend the longest line" — straight 1‑D Slice/Concat after Morton.
  - Quadrant‑structured tasks (e.g., "the answer is the quadrant with the most blue cells") — Morton's natural Z‑shape lines up with quadrant boundaries.
  - Diagonal patterns — Morton's interleave makes diagonals into clusters of bits; not as clean as anti‑diagonal Slice in raw 2‑D.
- **Sparse coding / cache‑locality** literature (Wikipedia Z‑order, Morton order) shows that tile‑aligned data layouts use Morton internally; this gives free reuse of cache structure but is irrelevant to onnx‑tool's static measurement.
- **Neural cellular automata** are a natural neighbour: they evaluate local update rules over a 2‑D grid; if expressed in Morton coordinates they become 1‑D update rules on a Cayley‑style sequence. For ARC this is overkill (NCA's many iterations would explode `params`), but a *single* Morton‑linearised step can subsume multiple Conv 1×1 layers.

## Discrepancies with COVERED list
None. The 30‑not‑power‑of‑2 issue is handled at table‑construction time off‑graph and never appears in the ONNX model.

## Recommended next experiment
Build a minimal ONNX graph with `Reshape → Gather(length‑900 int32 morton initializer) → Slice → Gather(inverse_morton) → Reshape` applied to flattened input; measure macs+memory+params via onnx_tool 1.0.1; compare to a Conv 3×3 baseline (810 900) on the same row‑fill task. Hypothesis: cost ≈ 7 200 B params + ~72 000 B activation memory + 0 MACs ≪ 810 900.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from j14-morton-zorder.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.14 — Z-order (Morton) curve mapping

## Summary

The Z-order or Morton curve, due to Guy Macdonald Morton (IBM technical report, 1966), interleaves the bits of a (row, col) pair to produce a 1-D index that preserves a useful (though not optimal) amount of 2-D spatial locality. It is foundational in cache-aware algorithms, octree/quadtree indexing, GPU texture layouts, and database spatial joins. For NeuroGolf the proposal is to use a Morton permutation to convert 2-D adjacency tasks into 1-D sequence tasks that can be solved with cheaper 1-D operations.

## Project mapping

This is the most promising of the five topics, but the win is smaller and more conditional than first appears. Cost analysis on `[1,10,30,30]` with `onnx-tool 1.0.1` (verified by my probes):

| Construct | macs | memory (B) | params | total cost | pts |
|---|---:|---:|---:|---:|---:|
| Identity | 0 | 0 | 0 | **0** | 25.000 |
| Transpose H↔W | 0 | 0 | 0 | **0** | 25.000 |
| `Reshape→Gather(900,int32)→Reshape` (one-way Morton permute) | 0 | ~75,624 | 907 | **~76,531** | 14.484 |
| `Reshape→Gather→Gather→Reshape` (Morton roundtrip) | 0 | ~115,224 | 1,807 | **~117,031** | 13.330 |
| Conv 1×1 baseline | 90,000 | 0 | 100 | **90,100** | 13.591 |
| Conv 3×3 baseline | 810,000 | 0 | 900 | **810,900** | 11.394 |

**Key empirical findings:**

1. **A pure Morton round-trip costs ~117,031** (worse than Conv 1×1) because the two intermediate `[1,10,900]` tensors are 36,000 B each. There is no point doing a round-trip unless work happens between the two Gathers.
2. **A one-way Morton permute (Reshape→Gather→Reshape) costs ~76,531**, beating Conv 1×1 (90,100) by ~15%. This is the realistic "headline" cost.
3. **Always use int32 indices, not int64** — measured: int32 Gather memory line = 39,600 B; int64 = 43,200 B. Same param count (900 elements) reported either way.
4. **Constant folding works**: a graph that builds the index via `Range(0,900,1)` then uses it folds down to the same shape as one with a literal initializer. So the construction style doesn't matter — what matters is that the index is fully constant-derived.

**Non-power-of-2 handling.** Morton's recursive definition is naturally for power-of-2 sides. For 30×30 the standard fix (`https://github.com/plavin/morton`) is to embed the 30×30 grid in a 32×32 Morton lattice, then keep only the 900 indices that correspond to (r, c) with r,c < 30. **For NeuroGolf this is invisible**: we precompute the 900-element permutation off-line and store it as an initializer; the ONNX graph never runs the curve algorithm. The permutation is a standard `np.argsort(flat_to_morton)` call. Verified: my probe with this construction passes `onnx_tool.Model.shape_infer` and profiles cleanly.

**Where it pays off.** The win materializes only when downstream ops are *cheaper in 1-D form than 2-D form*:

- **Row-fill / column-fill / "extend the longest line"**: in 2-D this is a `Conv 1×1` with structured weights or a `Conv 3×3 + accumulate`; in Morton-1-D it can be a single `Slice` / `Concat` for an entire row block. **Best-case savings**: 90,100 (Conv 1×1) → 76,531 (Morton perm) ≈ 15% per applicable task.
- **Quadrant tasks** (e.g., "the answer is the quadrant with the most non-zero cells"): Morton's natural 2×2 self-similar block structure aligns with quadrant boundaries. A length-225 Slice on Morton-ordered data picks out a 15×15 quadrant; in 2-D it requires two `Slice` ops (along H and W).
- **Hilbert > Morton for general locality**: the Hilbert curve has strictly better locality preservation (no large jumps; max-distance neighbour rate = O(√n) vs O(n) for Morton). The Hilbert paper `https://arxiv.org/abs/2210.04984` ("Hilbert Flatten...") confirms Hilbert > Morton > Zigzag for 1-D-CNN reformulations of 2-D vision. **Same Gather cost in ONNX (length-900 int32 perm)**, so prefer Hilbert over Morton when the task wants generic local 2-D rules. Recommend trying both.

**Where it does NOT pay off:**

- **4-neighbour Conv 3×3 emulation**: vertically-adjacent cells in 2-D can be ~30 positions apart in Morton order. A `Conv1D` after Morton permute would need a kernel size ≈ 32 to reach all 4-neighbours, which costs *more* than the 2-D Conv 3×3. So Morton can only replace 1-D-natural rules.
- **Symmetry tasks**: D₄ symmetries are exactly captured by free `Transpose`/`Slice`; Morton has no equivalent free move.

**Estimated public-LB upside.** If I assume 10–15% of ARC public tasks have a row-fill / column-fill / quadrant-aligned structure, and Morton perm replaces an otherwise-needed Conv 1×1, the per-task delta is about 0.9 points (13.591 → 14.484). Over 400 tasks at 12% applicability, that's about 43 extra points — meaningful but not transformative. The leverage is bigger if the Morton/Hilbert layout enables a Conv 3×3 → Conv 1×1 downgrade (delta ≈ 2.2 points/task).

**Failure modes:**

1. The Morton perm is *not* an Identity for the `(>0)` threshold check. After threshold, equality check is per-cell. If the model emits `output = Reshape→Gather→Gather→Reshape` of input, the inverse Gather **must** be exact mathematical inverse — verify by running `argsort(argsort(perm)) == arange(900)` before submission.
2. For non-30×30 grids the decoder applies pad-color (10) to make `[1,10,30,30]`. A naive Morton permute scrambles the pad rows into random Morton positions, so the pad mask must be propagated alongside. Use a separate channel for the validity mask, or rely on the pad-color value being preserved by Gather.
3. The leaderboard score-jump anomaly (5,148 → 7,600 reported in `https://github.com/openai/codex/issues/19594`, traced to "ONNX Runtime profiler/accounting discrepancy") reportedly affected tasks 157, 177, 002, 337, 276, 309. The post-fix leaderboard (April 29, 2026 update) likely closed at least one Gather-related accounting bug — verify your Morton submissions against the *updated* validator before relying on these cost numbers.

## Sources

- **Morton's original paper** — G. M. Morton, "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing", IBM Tech Report (1966). Modern explainers: `https://en.wikipedia.org/wiki/Z-order_curve`, `https://www.bigocheatsheet.com/morton-encoding`.
- **Non-power-of-2 Morton implementation** — `https://github.com/plavin/morton` (C reference impl with arbitrary side lengths via embed-and-skip).
- **Hilbert vs Morton for CNN reformulations** — Wang & Zhang, "Hilbert Flattening: A Locality-Preserving Matrix Unfolding Method", arXiv:2210.04984. Reports Hilbert > Morton > Zigzag empirically.
- **ONNX Gather spec** — `since_version=13`, `T` includes float, `Tind ∈ {int32, int64}`. `https://onnx.ai/onnx/operators/onnx__Gather.html`.
- **onnx-tool 1.0.1 source** — `node.py` line 898 `GatherNode`, line 2580 `ReshapeNode`: both inherit default `profile() → [0, 0]`. Memory contributions come from value_info shape_infer fill-in.
- **My empirical probes** — `/home/claude/research/code/probe_j14.py` (included in delivery). All numbers above are measured outputs of `onnx_tool.Model('graph.onnx', {'constant_folding': True}).graph.profile()` followed by `print_node_map()`.
- **Leaderboard scoring discrepancy** — GitHub issue `https://github.com/openai/codex/issues/19594` documents a score jump from ~5,148 to ~7,600 attributed to "ONNX Runtime profiler/accounting discrepancy", with affected tasks 157, 177, 002, 337, 276, 309. Weekly Kaggle News #333 (`https://weeklykagglenews.substack.com/p/weekly-kaggle-news-333`) confirms the leaderboard was updated around April 29 to "fix evaluator issues" (Japanese: 評価ツールの不備を修正).

## Contradictions / corrections

- **COVERED claims pure shape-only ops cost 0**. As noted in J.13: this is true for `macs` and `params`; not true for `memory` whenever an intermediate value_info exists. Morton's apparent "free" status is therefore a 76k-cost minimum even in the best one-way case. Significant for budget planning.
- **COVERED's leaderboard reference (Top: Chris Deotte 7,006)** — the openai/codex issue suggests that some scores in the 7,600+ range exploited an evaluator accounting loophole that has since been patched. Treat ranking estimates as approximate; the "fixed" leaderboard distribution is the authoritative one for post-April-29 submissions.

## Confidence

**High** for cost numbers (measured directly). **Medium** for the upside estimate of ~43 extra points (depends on actual ARC task distribution that I haven't fully analyzed). **High** for "use int32 not int64", "use single Reshape→Gather→Reshape not double Gather round-trip", and "try Hilbert too — same cost".

## Code snippet — ready to use

```python
import numpy as np

def morton_perm_for_grid(H, W):
    """Returns int32 permutation of length H*W reordering row-major to Morton order."""
    bits = max(H, W).bit_length()
    flat_to_morton = np.zeros(H*W, dtype=np.int32)
    for r in range(H):
        for c in range(W):
            m = 0
            for i in range(bits):
                m |= ((r >> i) & 1) << (2*i)
                m |= ((c >> i) & 1) << (2*i + 1)
            flat_to_morton[r*W + c] = m
    # Permutation P such that out[k] = inp[P[k]] places the k-th Morton-rank cell
    return np.argsort(flat_to_morton).astype(np.int32)

def hilbert_perm_for_grid(N):
    """Length-N*N int32 Hilbert permutation. N must be power of 2 for canonical curve;
    for arbitrary N, embed in next power of 2 and keep valid indices."""
    # Standard reference Hilbert curve via rot-then-emit; pad to next pow2 if needed.
    M = 1 << (N-1).bit_length()  # next power of 2 ≥ N
    pos = np.zeros(M*M, dtype=np.int32)
    for d in range(M*M):
        x, y = 0, 0
        s = 1
        t = d
        while s < M:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        if x < N and y < N:
            pos[x*N + y] = d  # may overwrite for invalid; we skip those next
    # Build perm by sorting valid (x,y)<N according to their Hilbert index
    flat_to_hilbert = np.zeros(N*N, dtype=np.int32)
    for r in range(N):
        for c in range(N):
            flat_to_hilbert[r*N + c] = pos[r*N + c] if pos[r*N+c] != 0 or (r==0 and c==0) else 0
    return np.argsort(flat_to_hilbert).astype(np.int32)
```
