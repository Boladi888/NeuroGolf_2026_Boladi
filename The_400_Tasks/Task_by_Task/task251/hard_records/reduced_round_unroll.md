# DEAD ROUTE — reduced-round bit-packed dilation (unroll floor)

- **Date:** 2026-07-03 | worker `bo_ng_cld_worker1` (Claude lane), batch 42 slot 06
- **Live bar:** cost **3740** (Vamshi), sha256 `e99be2e84e3dde0bdc065418034917aa39cada740a2507ba38ac11b719db0844`
  (manifest.csv, Team_Best_Onnx/task251.onnx; profile=static=3740, memory 3602 + params 138)

## Route attempted
Reduce the number of unrolled BFS dilation rounds in the team-best bit-packed
flood-fill graph (`task251_packed_u16_flood10_finalbitand`). LIVE runs **R10**
(seed reach0 + 10 dilation rounds → reach10). Each round is exactly 9 intermediate
tensors of [1,12] UINT16 = **216 B**. Static costs of reduced variants:
| variant | rounds | static cost | Δ vs 3740 |
|---|---|---|---|
| R7 | 7 | 3092 | −648 |
| R8 | 8 | 3308 | −432 |
| R9 | 9 | 3524 | −216 |
| R10 (LIVE) | 10 | 3740 | 0 |

## Exact failed gate
Truth is all-or-nothing (need 200000/200000). Correctness of an R{k} graph ⇔
every corpus grid converges within k dilation rounds. Measured the **exact**
bit-packed convergence-round distribution over the **cached 200k truth corpus**
(`Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache\task251.npz`, seed_offset 2026 —
the same seeds `ng_gate truth --full` uses):

```
N=200000  max_convergence_round=10
tail(>=8): {8: 353, 9: 14, 10: 2}
count conv>=9 (break R8): 16
count conv>=10 (break R9): 2
```

The dilation is **8-connected** (vertical-then-horizontal bit spread), so it
converges ~1 round faster than a 4-connected model — but the 200k tail still
reaches **10**. Two 12×12 grids (cache idx **81470**, **109542**) require exactly
10 rounds. Real-ONNX check (`prototypes/wall_certificate.py`):

```
idx=81470  H=12 W=12  R9=FAIL  R10_LIVE=OK  R8=FAIL
idx=109542 H=12 W=12  R9=FAIL  R10_LIVE=OK  R8=FAIL
```

⇒ R9 truth = 199998/200000 = 99.999% → **REJECT**. R8 worse (fails 16). R7 fails 62/30k sample.
Model validated against real ORT: R9 = 30000/30000, R8 = 29999/30000 on a fresh 30k sample
(`prototypes/behavior_check.py`), matching the conv-round model exactly.

## Why it's dead (forced-component argument)
The truth corpus contains 12×12 enclosure mazes whose exterior-connected black
region has an 8-connected geodesic depth of **10** from the border seed. A
fixed-unroll graph must include **≥10** dilation rounds to reach those cells;
fewer rounds leave a rim of exterior cells unreached, which are then mislabeled
interior (color 1). **R10 is the minimal-safe unroll depth ⇒ the round-count
lever cannot produce a strictly-cheaper correct graph.** The 216 B/round is a
hard floor for this representation.

## One remaining hypothesis (NOT round reduction)
The only sub-3740 route left is a **structurally different dilation** that reaches
fixpoint in fewer materialized tensors — e.g. a log-depth / segmented horizontal
flood (fill a whole contiguous black run in one op instead of 1 bit/round) so the
vertical unroll shrinks, or a per-round tensor-count reduction (9→fewer helpers).
Both are new formulations, not covered by this dead route. Reviewer locked this
batch to round-reduction only; those are next-batch candidates.
