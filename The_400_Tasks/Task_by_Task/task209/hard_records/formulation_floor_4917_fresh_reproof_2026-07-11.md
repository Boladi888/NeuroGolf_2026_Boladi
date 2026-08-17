# task209 — FORMULATION FLOOR re-proven at 4917 (fresh, 2026-07-11, worker `bo_ng_cld_worker1`)

**Live bar:** cost **4917** (mem 4733 + params 184), sha `9662e34ad1df4e6d1de07a39c357b486776696c3fa561b91605b790bc26485e2`
(authoritative = `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`; `ng_current_bar.py --task 209`).
Current best = **v15** (`prototypes/task209_compress_v15.onnx`), byte-equal to shipped LIVE / `submission.zip`.

**Verdict:** WALL. No candidate strictly below 4917 found. Every dominant charged tensor re-derived
from scratch (ignoring the prompt's mismatched "TopK compaction ~1200" hint — task209 is an ARC
crop + scaled-sprite-overlay solver, NOT a TopK task) and confirmed pinned. Status = REJECT_LOCAL (no-gain).

**Discrepancy resolved:** `Team_Best_Onnx/task209.onnx` metadata showed cost 4564, but the file
(same sha `9662e34a…`) scores `(0,4733,184)=4917` exactly. 4564 was stale sidecar metadata; bar tool
already flagged it stale. Real bar = 4917.

## Per-tensor charged breakdown (dump of v15; mem 4733 + params 184 = 4917)

| tensor | dtype/shape | bytes | pin (why irreducible) |
|---|---|---:|---|
| `pad_497` canvas | u8[30,30] | 900 | Output is `Equal(canvas, chanidx)`→BOOL[1,10,30,30] (free tail). Canvas must be [30,30] to broadcast; u8 is min dtype. Bool-crop-first = 3200B. #81 alts (OneHot/Conv/QLC/Gather/ScatterND-as-output) all ≥900. |
| `conv_1` plane | f32[1,1,18,9] | 648 | One-hot F32 → integer color needs a float reduction. Strided Conv reads the FREE graph input and outputs only [18,9]. F16-out impossible on F32 input (needs 18kB input cast). 2-pack is the u8 ceiling. See 4-pack analysis below. |
| `gather_475` canv0 | u8[16,20] | 320 | Render crop. Crop is EXACTLY at max: fresh corpus 11/266 pairs hit tall=16, 4/266 hit wide=20. Cannot shrink [16,20]. |
| `scatternd_495` canv1 | u8[16,20] | 320 | 4 corners via ScatterND (4 sparse points). Where-mask alt = bool[16,20]=320 + mask-build ops (worse). |
| `cast_3` ciu | u8[1,1,18,9] | 162 | The packed plane. Content rows≤18, cols≤18 (fresh corpus). Minimal u8. |
| `einsum_5/_7` marginals | f32[30]×2 | 240 | Yellow box corners. Yellow spans rows 0–15, cols 0–19 (fresh corpus) — OUTSIDE the [18,18] plane, so must read raw F32 input. Einsum output inherently [30]. Plane-based yellow detect needs [16,10]=640B (worse). |
| `cast_471/_451` render idx | i32[20]+i32[16] | 144 | Gather indices. **Empirically (ORT 1.24.4): Gather rejects u8/i8/i16/u16 indices (INVALID_GRAPH); only i32/i64.** i32 is min legal. |
| `mul_493` cidx | i64[4,2] | 64 | ScatterND indices. **Empirically: ScatterND rejects i32 (INVALID_GRAPH); only i64.** |
| `gather_473` spcw | u8[5,20] | 100 | Separable double-Gather render row-expansion. GatherND alt = idx[16,20,2] i64 = 5120B. |
| moment matcher | f16 windows + 2 Conv | ~200 | Dual-moment = cheapest exact placement certificate (prior floor proof, unchallenged). |
| guide/anchor/mag + i8 constraint chains + ~50 scalar plumbing tensors | mixed i8/i32/i64 | ~1635 | Extensively i8-optimized v9–v15. onnxsim-clean (see scan). |

Hard-pinned subtotal (rows 1–9): 900+648+320+320+162+240+144+64+100 = **2898B**. Remainder 1835B = matcher + detection + plumbing.

## Fresh evidence gathered this session
1. **Corpus bounds (266 visible pairs, `GUI/client/public/tasks/task209.json`):** output tall max=16 (11 pairs), wide max=20 (4 pairs) → render canvas [16,20] is AT max. Yellow brow∈[0,7] bbot∈[0,15] bcol∈[0,13] bright∈[0,19]. Non-yellow content max row=col=18 → plane [18,9] minimal.
2. **ORT index-dtype kernels (1.24.4):** Gather = i32/i64 only; ScatterND = i64 only (u8/i8/i16/u16 all INVALID_GRAPH). Kills the "narrow the index lanes" idea (would have been −108B on Gather, −32B on ScatterND).
3. **Redundancy scan of v15:** zero no-op casts, zero Cast→Cast collapsible chains (the 2 found are i64→i32→i8 with BOTH outputs used), zero duplicate nodes (CSE), zero dead nodes. onnxsim-clean.
4. **Params 184:** onnxsim already pruned unused inits (six, iota3, iota16/20-i32, out4u). No same-dtype value-duplicates remain (v15 took the last one — reshape_141). init→computed conversion NEVER helps: params=1/elem vs memory≥1B/elem.
   - **CORRECTION (v16, same session):** the "184 fully pruned" claim was too strong. A *monotonic lookup table* can be folded to a positional encoding for free: `priU[3,5]=1+8r+c` (15 params) is strictly monotonic in the flat index, so `ArgMax(cons, select_last_index=1)` reproduces `argmax(priU over valid)` exactly — deleting priU (−15 params) → **v16 cost 4902**. Cost is params-only (the required u8 Cast replaces the old Where). NOT byte-equal: the all-zero-`cons` fallback shifts (gr0,gc0)→(gr2,gc4), but all such divergences are proven both-wrong (correctness-neutral) → PROBE_READY. Lesson: audit for *monotonic-weight tables → select_last/first_index* folds, not just duplicate inits. Memory floor 4733 is unchanged and stands.

## One remaining hypothesis (quantified — likely net-negative, DO NOT grind blindly)
**4-cell-packed F32 plane.** Pack 4 cells/element (c0+16c1+256c2+4096c3 ≤ 39321, exact in F32) →
conv output [1,1,18,5] F32 = **360B** (−288 vs 648), and drop the u8 cast (values overflow u8) → plane = 360B vs current 810B (−450B).
BUT downstream blows up: (a) all read-site reads become F32/i32 (4× the u8 nibble tensors) with i32 Div/Mod arithmetic-unpack replacing the 1-op BitShift/And — the [3,6] moment window alone goes ~72B→~234B (+160B), guide [3,5] +140B, anchor/mag +50B; (b) **column-extent resolution coarsens** — scol/gright/anc currently resolve to pair+nibble (2-cell granularity); 4-cell groups require re-deriving first/last-content-column at sub-group granularity (more i32 ops). Estimated downstream cost +400–500B. **Net ≈ −450 + ~450 = 0 to +50B (break-even to LOSS), high correctness risk (must stay byte-equal across a representation change), ~4h rewrite of half the graph.** 3-cell packing is strictly worse (overflows u8 → i32 plane 432B > u8 162B). Matches the independent v6-era verdict "4 nibbles need u16, no clean ORT path."

**Future-only unlock:** if a later ORT ships u8/i8 Gather-index kernels, the 144B index lanes → ~36B (−108B) becomes a clean win. Not available in 1.24.4.

## Prior floor proofs (consistent)
- 2026-06-22 worker-7 (beta_p21_claude): "FLOOR CONFIRMED, sub-2000 unreachable in this class." Per-tensor floor 900+648+240+matcher+render.
- 2026-06-24 v15: params-only −5 (last duplicate). "No absolute floor below 4917 claimed" — this record upgrades that to a re-proven WALL with fresh corpus + ORT-kernel + redundancy evidence.
