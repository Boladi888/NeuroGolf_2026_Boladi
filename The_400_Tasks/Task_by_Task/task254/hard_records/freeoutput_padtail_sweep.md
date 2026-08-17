# DEAD ROUTE — task254 FreeOutputPadTailSweep (prompt 5609, 2026-07-09)

**Live bar:** 329 (manifest, authoritative; `ng_current_bar.py --task 254`).
SHA `9e932cc6721e80f92d7fe06b6ac1f58f2ba3ab2c21644fa8b1f4d680621417dc`, owner Aditya.
Baseline re-verified this session: `extract_task 254` → static=329 profile=329
(mem 121 + params 208), visible **265/265 PASS**.

**Prompt hypothesis (FALSIFIED):** "329 is a formulation artifact; floor_est ~131.
Delete the charged 10-channel prepad tensor; emit one label/mask channel and make
Equal/Pad the graph output." EV +0.92, class SWEEP, confidence HIGH.

**Why it's dead — the output contract kills the premise.** `ng_gate`/`mode_visible`
compares `np.array_equal( (out>0) , encode_grid(target) )` over the FULL `[1,10,30,30]`
one-hot (`tools/ng_harness/modes.py:44`, `session.run_raw`, `encode.py`). So:
1. **"One label/mask channel" is impossible.** Every cell needs exactly one hot channel;
   color-0 (background) requires an EXPLICIT hot channel-0 plane over the 9×9 grid
   (all-zero cell → decodes to −1 → mismatch). A single label plane cannot satisfy this.
2. **The graph output is already FREE at any width.** Aditya's 329 renders the full
   `[1,10,30,30]` directly from one Einsum (no canvas, no pad). Shrinking to a 9×9 inner
   + Pad does not delete a charged tensor — it ADDS the pad-input `[1,10,9,9]` intermediate.

## Attempted formulations (3 independent, from scratch) — all price ≥ 329

| route | builder | static=profile | verdict | why |
|---|---|---:|---|---|
| free-output Einsum, shared P[5,30] (Aditya) | slot5 (prior) | **329** | tie-only | the floor; adds 0.000 |
| **direct-30 masks** (no slot basis) | `prototypes/build_task254_direct30.py` | **844** | KILL (cost) | dropping P[5,30]→122 params but placement moves to data-dependent width-30 F32: `mask_f[3,30]`=360 + `colheight[1,30]`=120 + masks → mem 722 |
| **pad-tail** 9×9 inner + Pad (prompt's named dir) | `prototypes/build_task254_padtail_probe.py` | **3,256** | KILL (cost + `[1,10,H,W]`) | pad-input `inner[1,10,9,9]` = 3,240 alone; free output makes pad pure regression |

Analytical (not built — dominated):
- **reshape-collapse** (colheight[1,30]→reshape[1,5,2]→sum→slots): still materializes
  `colheight[1,30]` F32 = 120 (≈ the whole 150-param P) AND needs slot→width re-expansion
  for the render → never < 150 placement.
- **Scatter/Gather** (prior worker 2026-06-21): ScatterND floors 306–384; falsified.

## Forced-component floor (why 329 is structural, 131 unreachable)
`329 = mem 121 + params 208`. The dominant, irreducible term is **P[5,30] = 150 params**:
- **rank 5 forced** — up to 5 distinct bar columns (`2·idx+offset`); a linear placement
  of 5 distinct positions is rank-5 (SVD of P: `[4.69,1.41,1.41,1.41,1.41]`, prior proof).
  K<5 cannot resolve 5 heights for argmax≠argmin.
- **width 30 forced** — the Einsum contracts P's w-axis against the F32-locked width-30
  input. Slicing input to width 9 = charged `[1,10,30,9]`=10,800. Params count ELEMENTS
  (dtype-free), so a sparse/F16 P is still 150.
- **free-output makes 150 the CHEAPEST placement** — every alternative materializes a
  data-dependent width-30 operand (≥360, direct-30 proved 722) or a 9×9 pad-input
  (≥810 bool / 3,240 F32, pad-tail proved 3,256). 150 < all of them.
Routing 52 (S/A/B/E, 2-view minimal; direct 3-term=60) + consts + mem 121 (w_basis F32[3,5]=60,
col_sum_f F32[1,5]=20, all F32 forced by the input-dtype lock) are each already minimal.

**Remaining hypothesis: NONE viable.** The only sub-329 path would need a placement basis
< 150 that avoids both width-30 data-dependent intermediates and rank loss — proven
impossible under the F32-input lock + full-one-hot output contract. A candidate at 329
ties (adds 0.000). **VERDICT: WALL. Do not re-explore free-output / pad-tail / label-channel
/ direct-width / scatter on this task.** The prompt's ~131 is an over-optimistic EV heuristic.
