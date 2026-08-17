# DEAD ROUTE — perfect-25 / cost<=1 zero-param ceiling (task172)

- **Live bar:** cost **30** (manifest authoritative, sha256 `aa923630b6479cf1a1f5e7b7bfe2e4947e1c8ba8edba4f31d250b0b2f0c98dde`, owner Vamshi, 2026-07-02). Verified live via `ng_current_bar.py --task 172` on 2026-07-07.
- **Attempted route:** prompt `Task172_ZeroopsweepCeilingAudit` hypothesized a legal zero-param direct-output op (Identity / Transpose perm / attr-only) pricing <=1 (perfect-25, EV +3.40). A "rumor-hunt" slot.
- **Failed gate:** VISIBLE. Zero-param family scores cost 0 but **0/267 visible byte-equal**:
  - `Identity` → cost 0, 0/267
  - `Transpose perm=(0,1,2,3)` → cost 0, 0/267
  - `Transpose perm=(0,1,3,2)` → cost 0, 0/267
  - (only these 2 of 24 perms preserve the `[1,10,30,30]` output contract; both wrong.)
  - One visible miss = certain hidden 0 (traps §VISIBLE-FAIL). No truth/ab run.
- **Why it's dead (forced-component floor argument):** the transform is a vertical mirror-stack on rows, `out_rows = [r0,r1,r2, r2,r1,r0]` + 24 background rows. This requires row **duplication + reversal**. No attr-only op reorders rows: Identity=copy, Transpose=axis-permute, Reshape=no reorder, Pad has no numpy-`symmetric` mode (reflect/edge/wrap all mismatch — ORT-verified). Producing 30 reordered output rows forces a `Gather` with a 30-element index ⇒ **30 params minimum**; memory stays 0 only if that single Gather's output IS the graph output (any 2nd op materializes a counted spatial intermediate, smallest `[1,10,1,30]`=300 elems ≫ 30). INT8 idx → scorer returns `None` (traps §8c); params count elements not bytes so dtype can't shave it. ⇒ **cost 30 is the forced floor; cost<=1 is structurally impossible.**
- **Remaining hypothesis:** none. LIVE (1-node `Gather(input, idx INT64[30], axis=2)`) is correct (3000/3000 ARC-GEN oracle) AND optimal. Do NOT re-open the perfect-25 angle for this task; it is a false rumor for a mirror-stack transform. Twin task210 (uuid 8be77c9e) is byte-identical — same verdict.
- **Repro:** `prototypes/floor_probe.py` (full sub-30 sweep) + scratch zero-param sweep (Identity/Transpose). Full family: 113,116,164,311,172,210 all cost-30 mirror-stacks.
