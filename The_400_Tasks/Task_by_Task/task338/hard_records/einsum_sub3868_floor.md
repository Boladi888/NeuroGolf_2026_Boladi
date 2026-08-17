# task338 hard-record — sub-3868 wall for the free-output Einsum family

- **Live bar when explored**: 4776 (owner Sho, manifest, sha `2dd0b0bc428a…`, 2026-07-09). Non-stale (ng_current_bar).
- **This batch's result**: transpose-dedup refactor → **3868** (byte-equal to live champion, LOCAL_GOLD). See `staged_candidates/…_cost3868`.
- **Route explored for sub-3868**: shrink the two dominant operator planes of the live single-Einsum (`bianfu3`), whose cost is *pure params* (memory 0, free-output Einsum).

## Structure of the live/champion graph (decoded 2026-07-11)
Single `Einsum`, degree-3 in `input` (appears 3×/block), 6 near-identical blocks + linear tail.
Cost 4776 = param count. Dominant initializers:
- `v_w0 [3,30,30]=2700` = `[I, S(super-diag shift), Sᵀ(sub-diag)]`  (all {0,1}, full-rank)
- `v_w1 [2,30,30]=1800` = `[I, U(upper-tri ones / suffix-sum)]`     (full-rank)
- small channel/staircase tensors ≈ 276.

## What was tried and why sub-3868 is walled
1. **Separable / low-rank factorization of the 30×30 planes — DEAD.**
   All planes are permutation (I, S, Sᵀ) or triangular (U); exact rank 29–30.
   SVD/CP factor inner-dim ≥29 ⇒ ≥ the dense size. No param win (library #119 rank gate).
2. **Transpose-dedup of Sᵀ — WIN (this batch, −904).**
   The 2nd-`v_w0` operand of each block ("igr"-role) selects only `{I, Sᵀ}`; I is
   symmetric and Sᵀ = transpose of S, so drop slice 2 and swap the operand's spatial
   indices (`igr→irg`). `v_w0 [3,30,30]→[2,30,30]`, selectors `[2,3]→[2,2]`. Then v_w3==v_w4
   (both I₂) → share initializer (−4). ⇒ 3868.
3. **Identity-merge (share the single `I` plane across v_w0 & v_w1, → ~2982) — WALL (letter limit).**
   The `v_w1` role indexes its tensor directly by the size-2 block latent. Sourcing `{I,U}`
   from a merged `[I,S,U]` tensor needs a size-3 selector with a FRESH contracted index PER
   block (a shared letter couples the 6 blocks' products → wrong result). 6 fresh size-3
   indices required; the equation already uses **49 of 52** einsum letters (all 26 lowercase +
   A–W). Only X,Y,Z free ⇒ cannot add 6. Reducing blocks (the six [7,2] distance staircases)
   changes the computed function. Blocked.
4. **Identity-plane elimination via latent-split (→ ~2064) — WALL (memory blow-up).**
   Splitting the summed latent `a` into fixed a=0 (all-identity → index relabel, no plane)
   and a=1 (S,U) would drop both `I` planes. But a single Einsum is ONE product-sum; the two
   fixed-a terms are ADDED, which needs either (a) keeping both values in one tensor = the
   identity planes stay, or (b) two Einsums + Add = a charged `[1,10,30,30]` F32 intermediate
   (≥36,000 B memory) ≫ the 1,800 params saved. Dead.
5. **Non-Einsum / multi-node assembly — DEAD (charged intermediates).**
   Any multi-node graph materializes ≥1 `[1,10,30,30]` intermediate. The best known multi-node
   family for this exact rule (imos-2D, U8 planes) floors at **8468 = memory 7775 + params 693**
   — far above 3868. The Einsum's memory-0 is decisive.
6. **From-scratch crossing-parity rebuild — DEAD (documented).**
   Naive H∧V / 4-ray parity gives cross-frame false positives on multi-frame inputs
   (task338.md don't-do log, verified 10/267 visible fail historically). The champion's
   degree-3 structure exists precisely to avoid this; a fresh parity graph re-enters the trap.

## Forced-component floor argument
In the winning single free-output Einsum (memory 0), the credited cost = params. For the
"strict bbox-interior of each color-2 component" rule, robust multi-frame detection folded
into one Einsum forces: one shift plane `S` (900, neighbor/edge access) + one prefix plane `U`
(900, crossing-parity scan) + **two identity planes** (1800; the summed "self" terms cannot be
un-bundled without a 2nd charged Einsum, and cannot be merged to one under the 52-letter limit)
+ channel/staircase maps (~268). Σ ≈ **3868**.

**WALL at 3868 for known routes.** One remaining (unproven, next-batch) hypothesis: a
different, FEWER-block degree-3 structure that frees enough einsum letters to merge the two
identities into one plane (→ ~2982) *without* changing the computed function — but that needs a
full re-derivation of Sho's 6-block reflection ladder and carries high byte-exactness risk on
the 200k multi-frame corpus (the prompt's one-crack-stop rule for SWEEP tasks discourages
grinding it here).
