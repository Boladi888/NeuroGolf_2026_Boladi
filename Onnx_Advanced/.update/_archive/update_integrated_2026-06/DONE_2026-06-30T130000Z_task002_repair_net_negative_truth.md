# Memorized visible-repair seeds can be NET-NEGATIVE on broad truth (task002)

**Where:** task002 (FloodFillColorMapRewrite), structural-LIVE / ambiguous-class
(LIVE truth ≈ 91.74% @100k). LIVE = `rayseed_bitpack_5` (cost 9545) =
ray-seed + 5 packed flood steps + **4 wall-row-signature "visible repair" seeds**
+ 3-lane unpack. Builders: `The_400_Tasks/Task_by_Task/task002/prototypes/
build_task002_rayseed_bitpack.py` (LIVE) and my from-scratch
`prototypes/work/build_param_repair.py` (parametric repair groups).

## The discovery

The 4 memorized repair seeds exist to clear 6 specific arc-gen officials that a
pure 5-step flood under-reaches. I measured the cost/accuracy frontier on the
ARC-GEN truth corpus:

| model | cost | officials (visible 268) | corpus truth @100k |
|---|---:|---|---|
| LIVE (i5 + 4 memo repairs) | 9545 | 268/268 ✓ | 91.736% |
| **pure i5 flood, repairs REMOVED** | **8852** | 262/268 ✗ | **91.841%** |
| i5 + 4 *auto* repair groups | 9542 | 268/268 ✓ | 91.665% |
| i4 + 5 auto repair groups | 8951 | 268/268 ✓ | 89.265% |

**Removing the repairs makes the model BOTH cheaper (−693) AND more accurate on
the broad corpus (+0.10pp, stable across 20k/100k).** The repairs are pure
overfit to the 6 visible arc-gen pairs; on random inputs their wall-row Equal
conditions occasionally fire spuriously and corrupt cells, costing a little
truth while costing 693 bytes.

## Why it matters / the trap

1. **`mode_ab` byte-equality streams the CORPUS, not just officials** (see
   `tools/ng_harness/modes.py` `mode_ab`: officials first, then `corpus.iter_tensors`).
   So any "byte-equal on the 268 officials" cost-golf candidate that diverges off-
   officials FAILS the ab gate. Don't assume ab = officials-only.
2. **The visible gate covers all 268 officials** (train+test+arc-gen), so the
   cheaper-and-more-accurate pure-flood model still fails `ng_gate visible`
   (262/268, §VISIBLE-FAIL AXIOM). A model can be net-more-correct on truth yet
   wrong on known official data — the two gates can point opposite directions on
   memorized-repair tasks.
3. **§DUAL-TRUTH "more-correct ⇒ shippable" requires a CLEAN superset**
   (cand_misses ⊆ LIVE_misses). Net-higher truth is NOT enough if the candidate
   also breaks pairs LIVE gets right (pure-i5 breaks 6). Mixed divergences don't
   qualify.

## Floor result

task002 is at a true floor of 9545 under the (visible-100% + ab) gate:
- Exact-semantics rebuild (own builder, LIVE's specs) = ab byte_equal=True,
  0/20268 div, cost **9545** → no leaner encoding of the same function exists.
- Forced components: 20-bit row mask must be **uint32** (uint16 has no ORT
  kernel) at 1 row/axis-2-elem ⇒ 80 B/flood-tensor; each 4-neighbour flood step
  needs ≥9 tensors (BitwiseOr is binary, Max≠bitwise-OR, can't fold the 5-input
  OR); 30×30 bool Where-mask = 900 B forced by output shape.
- Frontier is monotone (cost↓⇒flood↓⇒correctness↓); Path-B impossible (truth
  rises with flood depth: i5/i6/i7 = 91.85/93.25/94.05%, each +720 cost).

## Strategy note for the captain (above worker pay grade)

If ship policy ever weights hidden/corpus truth over official byte-match, the
**repair-free pure-i5 graph (8852, +0.10pp truth) strictly dominates LIVE** on
both axes that matter for Kaggle (cost-points + expected hidden accuracy), at
the price of 6 known arc-gen misses. Worth a one-shot Kaggle A/B if a slot is
cheap. Not staged by me because it fails the visible gate.

## Library gap

`traps.md` / `by_symptom.md` should note: **on structural-LIVE tasks whose LIVE
uses memorized visible-repair seeds, always measure pure-flood (repair-removed)
truth — the repair may be net-negative, and the ab gate includes the corpus so
officials-only byte-equality is not sufficient.** I did not find this stated.
