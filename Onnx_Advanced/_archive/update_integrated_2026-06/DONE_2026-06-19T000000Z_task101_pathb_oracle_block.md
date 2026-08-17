# task101 ScaleStampResize — Path-B (Resize/Conv stamp) is oracle-blocked, not cost-blocked

## What I discovered

For the "draw source sprite fully, stamp scaled blue template onto target red anchors"
family (task101, ARC-GEN `task_447fd412.py`), the prompt's radical lever was: replace the
team-best W14 TopK + rolling-ScatterElements renderer (cost 14039) with an
integer-scale stamp built as `Resize(nearest, scale=m)` / DepthToSpace of a tiny binary
source kernel, placed by separable anchor onehots, finished with `Where(fill,1,input)` —
deleting TopK, per-cell scatter, and the 30×30 working mask.

**The lever works on the cheap part and misses the expensive part.** The TopK/scatter
tail and the 17×20 working mask are only ~3.5 KB of the graph. The dominant charged
surfaces are the SOURCE-DETECTION + ANCHOR-RECOVERY planes: ~4046 B of `[1,1,17,14]`
working grids plus two `[1,1,17,14]` F32 input-read slices (952 B each — the two single
largest tensors, an unavoidable F32 input-read tax). Resize/Conv does nothing for those.

**The real blocker is rule-oracle fidelity.** I wrote three independent numpy decoders of
the exact generator rule. The cleanest formulation (slice source-window, source red =
red touching blue, derive blue template offsets relative to red anchor, then recover each
target's scale m and origin by pairing red anchor blocks) peaks at **48.5% vs ARC-GEN
truth** (W14's heuristic source-detect oracle: 30%). Team-best W14 hits 98.4%.

Root cause of the gap: **the two red anchor blocks of a single target sprite merge into
one connected component whenever they touch** — diagonally under 8-connectivity, or
edge-to-edge when the source's red-anchor delta vector is (0,±1)/(±1,0). At unknown scale
m∈{1,2,3} each anchor is an m×m block, so naive component-pairing cannot separate the two
anchors or recover (origin, m). W14 only reaches 98.4% because it spends ~890 lines on an
explicit edge-run-length classifier (row_len1..6 / col_len1..4 → square1-3 / hrect / vrect
+ partner row/col-delta masks). Any faithful Resize-stamp graph must re-incur exactly that
classifier, so Path-B saves the cheap tail while keeping the expensive detection — net not
a meaningful win, and high correctness risk.

## Where

- Task: task101. Builder of team-best: `The_400_Tasks/Task_by_Task/task101/prototypes/build_task101_structural_topk_scatter.py` (cost 14039, memory 13345 + params 694).
- My numpy oracles (left in place as future-worker reference): `ref101_exact.py`, `ref101_w14style.py`, `ref101_v3.py` in the same prototypes dir.

## Why it matters (trap avoided)

Stops the next worker from re-walking Path-B as a cost lever. The prerequisite is a
**≥99% numpy oracle of the rule**; without it the ONNX rewrite cannot beat W14 on the
PRIMARY ≥99%-truth gate. Generalizable lesson for "stamp/fill at unknown integer scale"
ARC families: connected-component-based anchor recovery fails because equal-color anchor
blocks merge; you need edge-run-length / explicit block-size classification, and that
logic — not the scatter/Resize tail — is the cost floor.

## What's missing from the library

`by_symptom.md` / `techniques.md` could use a short entry: "multiple equal-color marker
blocks of the same object merge under CC at unknown scale → recover per-marker block via
edge-run-length classification, not connected components." I did not find this stated;
task101.md now records it in its don't-do log.
