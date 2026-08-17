# Transferable insight — a render/stamp graph that carries one sprite in the BASE canvas has an always-no-op Nth stamp (byte-equal drop); a "render floor" WALL can still hide an over-provisioning shave

**Source**: task133 (`bo_ng_cld_worker3`, Claude lane, 2026-07-12). Audited LIVE **16,288**
(Boladi, sha `a622a54e…`). **NEW WIN**: byte-equal **15,208** (−1,080 B, +0.0686 pts), staged
`task133_drop_s3_topk3_cost15208` (sha `4389dbcb…`). Complements
`DONE_2026-07-10T220523Z_paramdeletion_not_totalcost_variablescale_stamp_render_floor.md` (same task:
the ~900 free-output floor IS a real Conv/render floor per lib #109/#110/#132) — that note was
right about the floor but a prior WALL still under-priced the graph by 1,080 B.

## Technique — drop the Nth stamp when the base canvas already carries one instance
Sprite-completion / multi-stamp render graphs (generator `task_57aa92db`: `num_sprites∈{2,4}`,
sprite 0 = full template, others = partials to complete) commonly build:
`TopK(scores, K=max_instances)` → K `Resize` stamps `s0..s{K-1}` → `Max(base_grid, s0..s{K-1})` →
`Equal(...)` output. The trap: the **base grid already contains sprite 0** (the full template, drawn
from the input). So only `num_sprites − 1` PARTIAL sprites are ever stamped — the **K-th stamp
(lowest TopK score) is ALWAYS a no-op**. LIVE over-provisioned `K=4`; max partials = 3.

**How to verify (do this, don't eyeball the scoring subgraph):** expose `s0..s{K-1}` as graph outputs,
run the deterministic generator across the full `num_sprites` range (I ran 5,998 arc-gen incl. 1,992
`num_sprites=4`, + all officials), and tally per-slot fire counts. Result: `s0=5992,s1=3974,s2=1992,
s3=0`. The last slot never fires ⇒ delete its branch and drop `TopK(K)→TopK(K−1)`. On task133:
remove `s3` Resize plane (U8[1,1,30,30]=900) + `roi3`(32) + `sh3s/sw3s/eh3s/ew3s/v3/p3`, then
`TopK(4)→(3)` trims the `[4]→[3]` per-sprite chain (`topi/topv/valg/sh4/sw4/eh4/ew4` + `mrg/mcg/hhg/
pp/qq`). Net **−1,080 B**, byte-equal.

## Meta-lesson — a correct "render floor" WALL can still over-count the forced tensor set
Two workers + the floor note pegged task133's forced floor at ≈16,150 ("4× Resize stamp 3,600 …")
and declared "no candidate below 16,290 by any non-epsilon route." Wrong by 1,080 B: it counted **4**
stamps as forced without checking the generator's actual max instance count vs what the base canvas
already carries. Before trusting "N render planes are forced," price N against the generator: is the
base/identity path already supplying one of them? The forced count is often `max_instances − (carried
by base)`, not `max_instances`.

## Gate note — a pre-existing LIVE crash can BLOCK `ab --n 2000`; prove byte-equality crash-tolerant
task133 LIVE has an unclamped flat-Gather col index (`flatf=r4*30+c4b`, `c4b` not clipped) → OOB
`idx=900` on ~0.1% of arc-gen (first at deterministic corpus-538). `ab --n 2000` / `truth --n 2000`
**abort** there because the harness runs BOTH graphs per example and has no per-example crash guard.
Since the candidate's flat-Gather is byte-identical, both crash on the SAME inputs — not a divergence.
Substitute proof: official `ab --n 538` (max clean) = 805/805 byte_equal, PLUS a crash-tolerant
comparator over `tools.ng_harness.corpus` (4000 examples: 3997/3997 byte-equal where both run, 3
IDENTICAL crashes, 0 behavior diffs). Don't read an aborted `--n 2000` as a candidate failure — check
whether LIVE crashes at the same index first.

## Rule of thumb
For any multi-stamp / multi-Resize / multi-scatter render graph: (1) expose the per-instance planes
and count fires across the generator's full instance range — the last slot is often a permanent no-op
when the base canvas carries one instance; drop it + shrink the selector `TopK/Gather` width, byte-equal.
(2) Treat "render floor / detection irreducible" WALLs as claims to re-price per-tensor, not gospel —
they fixate on the genuinely-forced planes and miss over-provisioned siblings.
