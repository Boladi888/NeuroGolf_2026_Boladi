# FLOOR — the 3-channel packed detector / nested-halo render (hdet 2700)

- **Live bar:** 14854 (Vamshi, sha `20657a86804d…`), 2026-07-03.
- **Target:** `hdet` U8 [1,3,30,30] = 2700 is the 2nd-largest tensor. Dropping to
  2 channels would save 900 B → ~13954.
- **Architecture (reverse-engineered from LIVE):**
  - `det_w` INT8 [3,1,1,12], 1×12 horizontal matched filters, `det_b`=[-199,-599,-999]:
    3 width-matched filters with fractional 31/32 tie-break weights that fire per
    maroon-run width band. `hdet` ∈ {0,1,64}.
  - `halo_w` U8 [1,3,11,20] = **3 nested halo stamps**: ch0 = 8×5 block (small r),
    ch1 = 16×9 block (medium r), ch2 = full 20×11 (large r). The graded detector
    selects which nested stamp(s) fire, rendering the exact 4r×4r halo per radius.
- **Why 3 → 2 is INFEASIBLE:** radii 1–5 all occur (don't-do log; ablation drops
  any width detector → ≤250/267 visible = certain hidden 0). The 5 distinct halo
  sizes are packed into **3 nested kernels + graded firing**. Two nested kernels
  encode at most ~3 distinct halo sizes (via on/off combinations); they cannot
  reproduce 5 exact 4r×4r halos. 3 channels is already the packed floor — this IS
  Vamshi's compression over our older 5-plane run-detect (15710).
- **Hidden-risk note:** even if a clever 2-channel matched filter *passed local
  gates*, a novel detector/halo geometry is exactly the distribution-dependent
  class that hidden-failed the MaxPool beam fuse (see
  `maxpool_beam_fuse_13414_HIDDEN_FAILED.md`). Do not stage a 2-channel detector
  on local-gate evidence alone.
- **Verdict:** hdet 2700 is a soft-but-strong floor; no safe reduction found.
