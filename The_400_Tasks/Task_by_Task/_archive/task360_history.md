# task360 — archived history

Moved here 2026-06-24 (worker w17, beta_p29) to keep `task360.md` manual content under the
size tripwire. The CURRENT floor proof lives in `task360.md` (the **340 free-output Einsum**
is the verified WALL). The block below is the OBSOLETE v7=1167 static-floor proof, written
against the older packbits family before the 340 free-output graph became the live best. It is
retained only for the concrete per-attack pricing, which is still instructive.

---

### 2026-06-12 worker-360: static-floor proof — v7 (1167) is the static optimum (WALL) [OBSOLETE — superseded by 340 free-output Einsum]

Re-priced every prompt-brief attack direction against the actual v7 surface (mem 720 + params 447).
No static candidate < 1167 exists. Concrete pricing:

- **Attack #1 "CP-factor the 447-param dense value-plane bank" — VOID PREMISE.** v7 has NO dense
  `[10,10,4]` bank to factor. The 447 is already a fully-separable #118 basis-matrix factorization:
  `ch_ids_f`[10] ⊗ `row_sel`[30,10]=300 ⊗ `pack_cols`[30,4]=120 inside the one 4-input Einsum
  `bchw,c,hx,wv->bxv`. Fusing the two basis matrices into one `[h,w→x,v]` factor = [30,30,10,4]=36000
  params; fusing row+chan = [10,30,10]=3000. Separable 420 is the proven minimum. Nothing to CP-factor.
- **Attack #2 "negpad-Conv color read (à la task321 v3/v6)" — KILLED on memory.** A 1×1 Conv reads ONE
  spatial cell, so it CANNOT contract/pack across the width axis nor mirror cols 5-8→8-5. It needs TWO
  separate F32 code planes (left + right), each [1,1,10,4]=160, plus a width-reversing Slice, before the
  Max+Equal tail: memory floor ≈ 880 (with immediate u8 casts) to 1040 (F32). The Einsum's `wv` factor
  does select+pack+mirror+contract-width in ONE charged 120-param operand that Conv structurally cannot
  replicate. task321's negpad-Conv wins there only because each panel is a contiguous no-mirror block.
- **Attack #3 "shave individual static tensors" — each at floor.** packed_f F32[1,10,4]=160 is the
  Einsum output (dtype follows F32 input; casting input to F16 births an 18000B [1,10,30,30] tensor —
  task321-floor dead-end). The four u8 unpack tensors (40 each) are the minimal Mod/Div/Max chain. core
  BOOL[1,10,10,4]=400 is the exact 10ch×10row×4col active one-hot region, already 1-byte BOOL.

- **v18 `slice_max` (built, scored 1310)**: drop `row_sel` by Einsum `bchw,c,wv->bhv` (keep h=30) then
  Cast-u8 + Slice rows 0:10 + Max tail. Saves 297 params but the [1,30,4] F32 `packed_full` (480) +
  [1,30,4] u8 (120) intermediates cost +600 memory. Net 1310 > 1167. Folding row-restriction into the
  free Einsum (the `row_sel` basis matrix) is strictly cheaper than any slice-after-materialize route;
  the 30-row F32 intermediate is unavoidable if `row_sel` is removed.
- **Sum/add overlay** still doubles equal-overlap colors (existing dead-end); **Max** requires both
  planes separately, and #70 digit-packing (radix 16, colors 0-9 exact in F32) is the cheapest way to
  carry both planes through one Einsum. Confirmed via #70/#74/#118 catalog reads.

WALL: v7 `packbits_max` = 1167 (mem 720 + params 447), static==profile==1167 over 266 arcgen, visible
266/266. Team-best is also 1167 (Boladi). No sub-1167 static route found; the dynamic-row family (v9-v17)
all reprice above 1167 on real arcgen (documented above). No new candidate staged.
