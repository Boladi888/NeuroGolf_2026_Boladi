# task219: ambiguous-class + hidden-sensitive — byte-equal-vs-LIVE is the ONLY safe compression gate

**Topic:** CLASS-CHECK / hidden-sensitive compression / stale "zero-weight" prompt framing.

## Facts established (Worker 6, 2026-06-24)

- task219 is **already unlocked**: authoritative `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv` row = cost **7175**, points **16.1216**, SHA `ad5e5f72…`. The v5 structural rewrite already earned +15.94 LB months ago. A prompt still labeled "EXPERT FRESH Mystery Solved / cherry13b zero-weight" for this task is STALE — do not treat task219 as zero-weight / asymmetric-upside.
- LIVE is **ambiguous-class**: `ng_gate truth --task 219` on the LIVE graph = **14222/20000 = 71.110%** vs ARC-GEN ground truth. Per traps §CLASS-CHECK, when LIVE itself is <100% on truth, **byte-equal-vs-LIVE replaces 100%-truth as the gold ship gate.** A candidate with higher truth % than LIVE is NOT necessarily safer — it diverges from the thing actually scoring points on hidden.

## The hard lesson (corroborates traps §14e / §GRADER-GAP)

This task has a documented Kaggle-NEGATIVE compression: the 2026-05-26 **7,107** build passed 265/265 visible AND **99.045%** (19809/20000) arbitrary-synth, yet shipped **-0.03 LB** on Kaggle (predicted +0.20). Cause: it was an *approximation* — it traded exact LIVE parity for cost, so it diverged from LIVE on ~0.23 LB of hidden cases.

**Rule of thumb reinforced:** for an ambiguous-class / hidden-sensitive task, ≥99% synth is provably insufficient. Only a candidate that is **byte-equal to LIVE on the full corpus** (AB `byte_equal=True`, 0 divergences over 200k) is a safe cost shave, because byte-equal ⇒ identical hidden behavior ⇒ identical points at lower cost.

## What worked

The `task219_structural_v5_probe_h1_edge_fillcrop` shave (zero the impossible ±10 row-shift slots + crop the internal fill-bit unpack from rows 0..14 to rows 2..14 before Pad) is **behavior-preserving, not approximate**: AB vs LIVE = **200265/200265 byte_equal=True**, candidate truth = LIVE truth = 71.110% (identical pass count), `_score_v2` 7175→**7094** (-81), `profile_real_cost --arcgen 266` static==profile==7094. Staged at `staged_submits/task219_v5_h1_edge_fillcrop_cost7094/` (SHA `b7216e41…`). Expected Δ ≈ +0.011 LB (near LB floor, may not register) with ZERO downside.

## Cross-refs
- traps §CLASS-CHECK (LIVE<100% truth ⇒ byte-equal is gold), §14e (≥99% synth ≠ hidden-safe), §GRADER-GAP.
- task219.md attempts + `_archive/task219_history.md`.
