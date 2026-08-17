# DEAD ROUTE — task196 CarrierNarrowEqualTail (forced-floor ~720)

> **⚠️ 2026-07-12 UPDATE — the "WALL at 4482" conclusion below is OVERTURNED.**
> `bo_ng_cld_worker3` shipped **binpaint_k2 at cost 3167** (byte-equal to the Sho
> 4482 parent, ab --n2000 2266/2266, truth 2000/2000, numpy pipeline 200000/200000).
> The SECOND-PASS error was the premise *"detection must CLASSIFY ring size ⇒ 9
> channels ⇒ det[1,9,13,13]=1521 forced."* That is only true for a size-SPECIFIC
> stamp. A **binary** "complete-ring-here" detector needs just **K=2** conv channels
> (grouped by ring height: t∈{3,4} vs t=5 — provably separable, 9 canonical anchor
> windows vs 144,339 distinct negatives). The size-specific stamp is then replaced by
> a **size-AGNOSTIC** tail — paint a solid 5×5 block at each anchor, `Min(paint, ch1)`
> — which is exact because inter-box isolation is chebyshev≥2 and the paint is
> center-anchored (a 5×5 block covers the ring and never reaches a neighbor). det
> collapses 1521→338. The ~720 floor is still false, but **4482 was NOT the floor.**
> New best: `staged_candidates/task196_binpaint_k2_cost3167/`. Builder:
> `prototypes/build_task196_paint.py`; weights: `prototypes/robust_fit.py`.
> The forced-component argument below is WRONG on the `det 1521 forced` line only.

**Verdict: DROP (false floor). Do not re-explore the 720 route.**
Recorded 2026-07-10 by `bo_ng_cld_worker1` (Claude lane).

## Live bar (refreshed 2026-07-10 via `tools/ng_current_bar.py --task 196 --json`)
- **live_cost = 4482** — authoritative source = manifest, owner **Sho**, updated 2026-07-09T01:09:17Z, NOT stale.
- live sha256 = `b33b3a4412158c827d298bb8a5568b126cae34aaf566e905e1658ea71a068921` (`Team_Best_Onnx/task196.onnx`).
- manifest 4482, workbook 4482, team_best-onnx 4482 all agree. `local_task_state.csv` (4998) + local staged v4 (4989) are STALE/superseded.
- `Staged_Boladi/task196.json`: ABSENT (no Boladi baseline for this task).
- leaderboard reconcile 2026-07-11: current_public_score 7559.16.

## The prompt's premise is stale — the charged component is already GONE
The Direction/forced-floor targets a **"10-channel/F32 spatial carrier + repeated Conv masks"**
(the old 4998/4989 v4 = Slice→Cast→QLinearConv-degree + **five 3x3 masked-flood steps**).
Sho's live **4482** graph (`task196_disc_v3`, opset 18, 7 nodes) already replaced that entire
structure with a **2-QLinearConv template detector**, entirely in U8/INT8 — the carrier-narrow
rewrite the attack was meant to force has **already happened**:

```
[0] Slice   one_f  = input[:,1:2,:15,:15]          F32 [1,1,15,15]  900 B   (forced: read 1 F32 channel)
[1] Cast    one    = Cast(one_f, U8)               U8  [1,1,15,15]  225 B
[2] QLinearConv det = conv(one, wdet[9,1,5,5], bdet[9])  U8 [1,9,13,13] 1521 B  (detector core)
[3] QLinearConv stamp = conv(det, wstamp[1,9,5,5])       U8 [1,1,15,15]  225 B
[4] Cast    loop   = Cast(stamp, BOOL)             BOOL[1,1,15,15]  225 B
[5] Pad     cond   = Pad(loop, 15->30)             BOOL[1,1,30,30]  900 B   (forced: output mask)
[6] Where   output = Where(cond, onehot3, input)   F32 [1,10,30,30] (output, uncounted)
```
There is **no F32 carrier** (only the forced 900-B `one_f` channel read) and **no Conv-mask flood**.
Per the 2026-07-10 workflow rule ("if the charged component is already gone, return DROP"): **DROP**.

## Priced wall — 720 is below the irreducible I/O floor
`score_network_v2`: team_best = memory 3,996 + params 486 = **4,482** (confirmed).
Two sub-totals are FORCED by task I/O, independent of any detection:
- read one **F32** input channel at the 15x15 active grid → `one_f` F32[1,1,15,15] = **900 B**
  (input dtype is F32; Slice preserves dtype; casting the full input first is 9,000 B — worse).
- produce the 30x30 output-broadcast mask for the final `Where` → `cond` BOOL[1,1,30,30] = **900 B**
  (output is [1,10,30,30] copied from `input`; the mask must be 30x30; BOOL is the cheapest dtype).

**Empirical proof** — `prototypes/build_task196_iofloor_probe.py`
(`task196_iofloor_probe.onnx`, sha256 `a33ee236f3d0d199935be15b47ae0d5d7d3411688106c4cde27f9df92ac7cfc6`):
a shell that reads channel-1 and recolors EVERY color-1 cell (zero ring detection) prices at
**2,051** (memory 2,025 + params 26) = **+1,331 over the 720 claim**.
Any correct graph must do at least this I/O work, so **genuine floor ≥ 2,051**, and realistically
~4,000+ once real ring detection is added. **720 is a false floor by ~2.8x, before a single detection op.**

## Fast probe (kill rule = "prices above 720 OR falsifies") — DEAD on both
- Prices above 720: the I/O shell alone is 2,051. Priced wall hit → stopped (no truth/ab burned on a
  knowingly-wrong, over-floor shell, per gate hygiene).
- Falsifies: the "one-channel carrier + Equal-tail, no detection" formulation recolors all color-1
  cells — WRONG rule. The rule is **topological** (a color-1 component is recolored iff it forms a
  complete hollow rectangular ring, bbox>=3x3), not a per-cell carrier predicate. Already established
  falsified: local 2x2 signatures, trained 9x9/11x11/13x13/15x15 single-filter separators, Euler-feature
  trees, and cheap hole-neighborhood predicates all leave errors (see Don't-do log). Detection needs
  >=2 convs / a multi-channel stack — exactly what Sho's 4482 already uses.

## Two other from-scratch formulations evaluated (both DEAD vs 720)
1. **Single-conv direct ring stamp** (one QLinearConv detect+stamp → 15x15 → Pad → Where):
   floor = one_f 900 + conv_out 225 + cond 900 + wdet(~25) + onehot3 10 ≈ **2,060 > 720**, and single-filter
   detectors are falsified (Don't-do). Dead.
2. **Live 2-conv template detector (= Sho 4482 itself)**: already the carrier-narrow endpoint (U8 throughout).
   I/O sub-total 1,800 + detection sub-total 1,971 = 4,482. Cannot approach 720. Dead vs the floor claim.

## SECOND PASS (2026-07-10) — the sub-4482 hypotheses are now EMPIRICALLY DEAD. WALL at 4482.
The first pass left one open lever (reduce det channels / different assembly). The second pass
measured the corpus (`prototypes/probe_task196_floor_analysis.py`, 20k–50k real pairs) and killed
every sub-4482 route. **Sho's 4482 is the genuine floor — profile=static=4482 (no grader-gap).**

- **det channels are NOT reducible.** All **9** ring bbox sizes occur in the corpus — 3x3:4317,
  3x4:3240, 3x5:2511, 4x3:3275, 4x4:2351, 4x5:1647, 5x3:2551, 5x4:1836, 5x5:1161 (per 20k). Sho's
  9 INT8 templates = one per size (border≈128/perim, hole=-128, bias -127); dropping any size ⇒
  misses those rings ⇒ diverges from LIVE. The "3 square templates" idea is falsified.
- **Morphology / window detectors are DEAD.** Ring isolation is only **chebyshev distance 2**
  (min=2; ≤2 in 60% of pairs). Numpy-probed morphological closing (dilate r2 + erode + hole-adjacent)
  = **1092/20000 exact** (209k mismatched cells): a color-1 cell 2 away merges under dilation and
  fabricates false holes. Any non-connectivity-aware assembly leaks across the 2-cell gap. Sho's
  templates survive because their **zero-weight footprint ignores everything outside the ring bbox** —
  morphology cannot. (CC-rule, connectivity-aware, = 2000/2000 exact — detection REQUIRES component logic.)
- **Every dimension of Sho's graph is minimal.** Color-1 content spans the full **15x15**
  (rows/cols reach 14) ⇒ `one_f` F32[1,1,15,15]=900 forced. Ring top-left anchors reach **(12,12)**
  ⇒ detection must emit output at position 12 ⇒ `det`[1,9,13,13]=1521 forced (cannot shrink to 11x11).
  Output is 30x30 ⇒ `cond` BOOL[1,1,30,30]=900 forced. Casts/stamp (one/stamp/loop 225 each) and the
  weight banks (wdet 225 + wstamp 225, 5x5 forced by the 5x5 rings) are each at minimum.
- **Flood realization loses.** The connectivity-aware alternative to templates is a masked flood; the
  leanest measured flood (old v4) = **4989 > 4482**. Templates beat flood by 507. Dead.

### Forced-component floor (information-theoretic)
`I/O 1800 (one_f 900 + cond 900) + detection 2646 (det 1521 + one/stamp/loop 675 + wdet/wstamp 450) +
misc params 36 = 4482`. The detection term is forced by 9 corpus ring sizes × 13x13 anchor positions
(distance-2 isolation forbids the cheaper window/morphology realizations). **No correct sub-4482 graph
exists on this route.** WALL. Cheapest correct formulation = the live 4482 (profile 4482, no grader-gap).
