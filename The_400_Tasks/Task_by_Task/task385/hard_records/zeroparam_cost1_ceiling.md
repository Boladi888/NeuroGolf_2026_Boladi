# DEAD ROUTE — task385 zero-param cost<=1 "perfect-25" family

**Live bar:** cost **30** (manifest, owner Vamshi, 2026-07-02).
sha256 `629a5673b6a8e745a4c5b1b5afcf3c5e7214f2fd7c54d7f5da27eb3e788ea5d1`.

**Attempted route (ZeroopsweepCeilingAudit rumor):** a legal zero-param /
attr-only op (Identity, all 24 Transpose perms, any no-initializer final-output
op) achieving perfect-25 at cost <=1, undercutting the Gather floor (30).

**Exact failed gate:**
- `zeroparam_sweep.py` (25 ops): every cost-0 op scores **0/2000** vs LIVE on
  full one-hots. Identity + Transpose(0,1,3,2) are the only shape-valid ones;
  the other 23 perms output a wrong shape (not [1,10,30,30]).
- `ng_gate visible --task 385`:
  - `Z_identity.onnx` → **[visible] 0/265 (0.00%) FAIL**
  - `Z_transpose_0132.onnx` (row/col swap) → **[visible] 0/265 (0.00%) FAIL**
- A visible 0 is a certain hidden 0 (traps §VISIBLE-FAIL) → no truth/ab/battery run.

**Why it's dead (forced-component argument):** the rule (ARC-GEN f25ffba3) is a
NON-MONOTONIC vertical row FOLD, `idx = max(r, 9-r)` for r<=9 then identity
(= `[9,8,7,6,5,5,6,7,8,9,10..29]`). Output rows 0-4 must become a MIRROR of rows
5-9 while rows 5-29 stay put. No zero-param op does this:
- Identity/attr-only-shape ops leave rows 0-4 unchanged → wrong.
- Transpose swaps axes (row↔col or across batch/channel) → wrong axis / wrong shape.
- The fold is numpy-`symmetric` (edge repeated), not ONNX `reflect`, and is
  non-monotonic, so it is NOT a single Slice(steps=±1) either (that path needs a
  monotonic index; only identity or full-reverse qualify, both wrong).
The minimal correct op is one dense `Gather(input, idx[30], axis=2)` → params 30,
mem 0 → **cost 30 = the proven hard floor** (full proof in `task385.md`, twice
re-confirmed independently, byte-equal to LIVE 2265/2265).

**One remaining hypothesis:** none. The floor is forced: any correct graph needs
either a 30-element index (Gather) or a >=300-B grid intermediate (Slice/Concat/
Scatter/Pad-reassembly, all measured 6k–36k). No cost<30 graph exists. **WALL.**

**Verdict:** no ship. LIVE already holds the optimal cost-30 Gather. Do not
re-run the zero-param/one-node sweep for task385.
