# FLOOR — the `ingrid` padding mask (900 B) cannot be folded away

- **Live bar:** 14854 (Vamshi, sha `20657a86804d…`), 2026-07-03.
- **Route attempted:** delete the `ingrid = And(rowu,colu)` [900] + `code = Max(…)`
  [900] planes by folding padding into a broadcast `Max(hmcode, beamcode, rowpad,
  colpad)` where `rowpad[1,1,30,1]`/`colpad[1,1,1,30]` carry the pad code only
  outside the active N×N grid (30-byte sentinels). Would have saved ~1740 B → ~13114.
- **Why it's IMPOSSIBLE (proven, not "tried hard"):**
  1. A broadcast Max/Min fold that sets a value *only outside* a 2-D region
     without a materialised 2-D mask requires the pad code to be **extremal**
     (strictly greater than every interior code, for Max; or strictly less, for Min).
  2. The interior U8 code range is **saturated with no free extremal slot**:
     `bg=0` (bottom), `blue=1`, `maroon=254`, `green=255` (top), `pad=253` sits in
     the *middle*. Measured intermediate `code` values: inside={0,1,254,255},
     padding={0,1,255} (`probe_intermediates.py`, 267 inputs).
  3. **Green (255) provably bleeds into the padding region** — 1681 padding cells
     across the corpus have code 255 (halo of a bottom-edge square extends into
     rows ≥ N). LIVE clips these to 253 via `Where(ingrid, code, 253)`. A Max fold
     with pad in the middle leaves them 255 → **divergence**.
  4. Making pad extremal requires re-encoding green (255) to a small value, which
     means changing the halo-render QLinearConv output. That conv is a **hard
     binary threshold** (y_scale=1/512 → multiplier 512 → sum≥64 saturates to 255,
     else 0; halou ∈ {0,255} exactly). You cannot make it emit a small *constant*
     (e.g. 3): a small target un-saturates the accumulator and reveals per-radius
     sum variation → non-constant → divergence. Any post-conv remap of {0,254,255}
     → {0,2,3} needs a masked/lookup op producing a full 900-B plane, exactly
     cancelling the 900 B saved from `ingrid`. Net zero at best.
  5. Two-Where padding (`Where(rowu,…,253)` then `Where(colu,…,253)`) is also
     net-zero: it trades `ingrid`(900) for an intermediate `cg1`(900).
- **Verdict:** `ingrid` (900 B) is a **hard floor** for this architecture because
  the U8 code alphabet is fully consumed [0..255] and pad is interior.
- **Remaining hypothesis:** only a from-scratch code alphabet with a reserved
  extremal pad slot AND a halo render that emits that alphabet directly could
  drop it — but the halo render can only cheaply emit a saturated binary {0,255}.
