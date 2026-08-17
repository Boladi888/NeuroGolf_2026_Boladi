# Single-axis flood reach-safety + the gated-comb adversarial test

**Where:** task243 (4-conn seed flood). Builders `Task_by_Task/task243/prototypes/build_task243_rank1_alt.py`,
`adversarial_comb_check.py`. Staged `task243_rank1alt_safe64_cost6014/`. Contrasts the hidden-failed
staged max45 (`task243_regionfillrewrite_cost4718`).

## Insight 1 — single-axis alternating flood needs ≥2K rounds to be safe against a K-combined-round LIVE

For an iterated gated dilation (flood-fill) task, two Einsum representations exist:
- **combined-4-conn/round** (`Tri⊗I + I⊗Tri`, rank-2 bands `A,C[2,W,W]`): reaches path-distance = round-count,
  costs 3 einsum letters/round (≤15 rounds/einsum), params 2·W² per axis-pair.
- **single-axis alternating/round** (one band `M[W,W]`, VHVH schedule): costs 1 letter/round (≤~45 rounds/einsum),
  params only W² (one M). Cheaper params (task243: M=324 vs A+C=1296, **−972 B**).

**The reach trap:** a single-axis VHVH round only advances a path-cell when the round's axis matches the next
step's axis. Along a 1-wide gated corridor, a run of k same-axis cells needs k same-axis rounds, so the fixed
schedule **wastes up to half its rounds** → a path of 4-conn length L needs ≤2L alternating rounds. Therefore:

> **N alternating rounds reach path-distance ≥ N/2.** To provably ⊇ a K-combined-round LIVE (which reaches
> path-distance exactly K), you need **N/2 ≥ K, i.e. N ≥ 2K** (and ≥K per axis). Below that, the single-axis
> flood UNDER-REACHES gated winding paths and misses cells LIVE fills = **regression vs LIVE = hidden-fail**.

task243 numbers: LIVE = 28 combined. max45 (45 rounds ≈ 22/axis, 1 einsum, cheapest at 4718) has N/2=22 < 28 →
under-reaches → **hidden-failed** (submit record named the cause: "hidden >45-round snake"). rank1_alt (64 rounds
= 32/axis, 2 einsums, 6014) has N/2=32 ≥ 28 → provably ⊇ LIVE → 100% truth n=4000, 0 regressions. Round count
is FREE inside an einsum (§37), so the ONLY cost lever is einsum-count: ≥2K rounds forces >45 → 2 einsums → the
safe single-axis floor is the same memory as combined but −972 params. **Cheapest byte-safe = combined; cheapest
⊇-LIVE (more-correct, probe) = single-axis at ≥2K rounds.**

## Insight 2 — the gated-comb adversarial test (visible/arcgen/truth do NOT catch under-reach)

max45 passed visible 265/265, truth n=2000 100%, ab, AND a captain public probe (+0.392) — yet under-reached on
hidden. Reason: ARC-GEN/visible corpora rarely contain a 1-wide gated path whose **4-conn path-length ≫ Manhattan
distance**. Straight corridors cap at W (=18) so they never exceed reach; open regions test Manhattan reach, not
gated-path reach. The discriminating input is a **comb/maze**: vertical teeth (full-height corridors) joined by
single-cell connectors alternately at top/bottom, everything else a blocker. This forces a serpentine gated path
of length ~W·(#teeth) inside WxW.

Recipe (`adversarial_comb_check.py`): build combs of increasing teeth, run candidate + LIVE + true-flood, assert
`candidate_fill ⊇ LIVE_fill` per grid (regression = a cell LIVE fills that candidate misses). This reproduced
max45's failure (9/9 comb configs regress, e.g. LIVE=29 vs max45=25) and proved rank1_alt safe (0 regressions,
fills ≥ LIVE everywhere). **Any flood/dilation candidate that claims "more-correct-than-LIVE" should pass a gated
comb test before a probe** — it's the cheap local proxy for the hidden long-path risk.

## Insight 3 — dead-end record (so nobody re-tries it)

Folding a K-combined-round flood into ONE einsum to drop the inter-einsum state tensor is impossible for K>15:
the combined-round einsum uses 3 letters/round and the 52-letter budget caps it at 15 rounds (task243
`flood_direct_L15` uses exactly 52 letters). Empirically a 15-round single-einsum (task243 cost 5689, −1296) is
**visible 255/265 FAIL**. The 2nd einsum / 2nd state tensor is correctness-forced, not a design choice.

## What's missing from the library
I searched `by_symptom.md` / `techniques_catalog.md` for flood-reach / under-reach / single-vs-combined-axis
guidance and the adversarial-comb test — the reach-count arithmetic (N≥2K) and the comb test are not captured
as a reusable rule. The single-axis-vs-combined einsum cost/letter tradeoff is partially in #119 (free-output
Einsum micro-floors) but the **reach-safety count** and the **comb adversarial test** are new.

## Sorter note: traps.md — add new rule "single-axis alternating flood under-reach: N alternating rounds reach path-distance N/2; need N≥2K to ⊇ a K-combined-round reference; validate with a gated-comb test (adversarial_comb_check.py). Straight-corridor/open-region tests miss it."
## Sorter note: by_symptom.md — add route: symptom "flood/dilation candidate passes visible+truth+public-probe but hidden-fails" -> single-axis under-reach on gated long paths; gated-comb test.
## Sorter note: op_cost_patterns.md (or #119) — extend: single-axis einsum flood (one M[W,W] band, VHVH) costs 1 letter/round & W² params vs combined rank-2 (A,C[2,W,W], 3 letters/round, 2W² params); combined is byte-safe at K rounds, single-axis needs ≥2K rounds (2 einsums) to ⊇ LIVE but saves the identity-band params.
