# task097 — archived history

Superseded provenance moved out of `task097.md` on 2026-06-11 (worker-4) when the
floor was corrected from "17k byte-equal-to-LIVE" to "**910 forced**, DROP".
The 17k analysis below assumed the wrong objective (byte-equality to the
canonical same-colour LIVE model). The grader scores truth vs ARC-GEN ground
truth on the single-colour hidden set; team-best is a single Conv at cost 910.
See `task097.md` → Cost analysis (FLOOR CORRECTION) for the live proof.

---

## Historical 17k byte-equal-to-LIVE cost analysis (pre-2026-06-11)

**Canonical 71,292 = memory 71,200 + params 92.** Per-tensor (static == ORT-observed, no broadcast bloat):

| tensor | shape | dtype | bytes |
|---|---|---|---:|
| `_input_f16_cast` | [1,10,30,30] | F16 | 18,000 |
| `input_20`, `output_active` | [1,10,20,20] | F16 | 8,000 ea |
| `scnk_nz`,`neighbor_counts`,`keep_mask`,`kept_colors` | [1,9,20,20] | F16 | 7,200 ea (×4 = 28,800) |
| `has_same_neighbor_b` | [1,9,20,20] | BOOL | 3,600 |
| 6× `[1,1,20,20]` (orig/kept/removed/ch0…) | | F16 | 800 ea (4,800) |

So the prompt's "kill the 18,000 cast" is only ~25% of the cost — the **real
dominator is the five `[1,9,20,20]` per-colour Conv tensors (~32k)**.

**Two byte-equal rewrites built (both 0-divergence vs canonical):**
- **v1 single-channel = 17,160** (memory 17,084). canvas Conv `[1,1,30,30]` F32
  (3,600) + cast/crop (1,300) ≈ 4,900 entry; same-colour core = halo pad (484) +
  8 shift `Slice` (3,200) + 8 `Equal` (3,200) + 7 `Or` (2,800) = 9,684; output
  tail (nonbg/keep/active/where/pad) ≈ 2,500.
- **fallback in-vocab = 44,307**: U8 entry (9,000) + F16 Conv pair (14,400) +
  9-ch U8 tensors. Floor for the Conv approach (ORT Conv needs ≥F16).

**The prompt's "floor ~2,000–3,000 / target ≤2,500" is NOT achievable byte-equal
— it implicitly assumed the cheap any-neighbour single-channel Conv** (count all
non-bg neighbours in one `[1,1,*]` Conv). Same-colour safety (mandatory, §14d)
forces EITHER per-colour Conv (9-channel, ~7,200 B/tensor) OR ~8 explicit
single-channel `Equal`-on-shift compares (~9,700 B). Both floor the working set
near **~12–17k**, not 2.5k. v1's 17,160 is the measured floor. The "edge-based"
4-compare variant (`task097_v2_edge`, share 1 `Equal` per orientation, expand to
2 endpoints) was BUILT and verified byte-equal — but scored **17,979 > 17,160**:
the 8 edge-input slices + 8 endpoint `Pad`s cost more than v1's 8 shifts + 8
`Equal`s, so it's a confirmed dead end. The canvas-Conv entry (3,600 F32, forced
— Conv on the full F32 input is the cheapest 10→1 collapse) + the 8-neighbour
detection (~9,700) + output tail (~2,500) are each at their own floor.
Measured via `tools/_score_v2.score_network_v2`.

NOTE: this whole analysis is the WRONG objective — retained only for provenance.
The byte-equal-to-LIVE constraint it assumes is not a grader requirement.
