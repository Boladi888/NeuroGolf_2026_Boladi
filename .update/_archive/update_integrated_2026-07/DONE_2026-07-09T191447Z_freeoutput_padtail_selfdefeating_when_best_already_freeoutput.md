# Transferable insight — the "FreeOutputPadTail / color-index collapse" sweep is SELF-DEFEATING when the current best already renders a free output (Einsum case)

**Source**: task254 batch56 (`bo_ng_cld_worker3`, 2026-07-09). Audited LIVE **329** (Aditya);
this is a **floor re-confirmation** (329 exhaustively proven, 3 independent from-scratch families),
not a new win. Companion to `DONE_2026-07-09T184440Z_gather_index_beats_free_output_for_permute_tile.md`
(task249) — same trap, different already-free graph (Gather there, **free-output Einsum** here).

## The trap it kills
The EV/prompt-writer instinct: *"329 is a formulation artifact; delete the charged 10-channel prepad
tensor; emit one label/mask channel and make Equal/Pad the graph output (free-output). floor_est ~131."*
Correct ONLY when the live graph still materializes a charged `[1,C,H,W]` canvas/prepad. When the best
is **already a free-output construction**, the premise is false in two independent ways:

1. **The output is FREE at any width already.** A free-output Einsum (`bchw,…→bohw`) renders the full
   `[1,10,30,30]` for zero charge. Shrinking to a 9×9 inner + `Pad` does NOT delete a charged tensor —
   it **adds** the pad-input `inner[1,10,9,9]` (= 3,240 B f32; measured 3,256 total). Pad-tail is pure
   regression whenever the render is already free. (task254 pad-tail probe: **3,256** vs 329.)
2. **"One label/mask channel" cannot satisfy a full-one-hot output contract.** The grader compares
   `np.array_equal((out>0), encode_grid(target))` over the WHOLE `[1,10,30,30]` (ng_harness/modes.py
   `mode_visible`; session.run_raw). Every cell needs exactly one hot channel; **background (color 0)
   needs an EXPLICIT hot channel-0 plane** (all-zero cell → decodes to −1 → mismatch). A single label
   plane is structurally impossible here.

## The mechanism (why the free-output basis is already the floor)
For "recolor 2 selected columns + keep a one-hot background base" (tall→c1, short→c2, rest→c0):
- The placement is a shared **`P[5,30]` param basis = 150** (rank-5: 5 distinct bar columns, SVD
  `[4.69,1.41,1.41,1.41,1.41]`; width-30: contracts the F32-locked width-30 input). Params count
  ELEMENTS, so sparse/F16 P is still 150.
- **P=150 is the CHEAPEST placement precisely BECAUSE the output is free.** Every alternative that
  drops P materializes a **data-dependent width-30** operand (≥360) or a **9×9 pad-input** (≥810 bool /
  3,240 f32):
  - *direct-30 masks* (no P; colheight[1,30]=120 + mask stack[3,30] f32=360): **844** (measured).
  - *reshape-collapse* (colheight[1,30]→[1,5,2]→sum): still pays colheight[1,30] f32=120 ≈ whole P,
    then needs slot→width re-expansion. Dominated.
  - *pad-tail 9×9 inner*: **3,256** (measured).
  - *ScatterND / gather-selector* (prior worker): floors 306–384.

## Rule of thumb
Before taking a "free-output / delete-the-canvas / color-index collapse" sweep prompt at face value,
**inspect the live graph first** (`inspect_graph.py`). If its output already comes straight from an
Einsum/Gather (no charged `[1,C,H,W]` prepad), the canvas is already dissolved and the residual is a
**forced placement basis** (`P[k,W]` param, or a forced int32 gather index). A pad-tail or
label-channel rewrite will REGRESS. Free-output-Equal wins only for a genuine per-(r,c) 1-D/separable
render (library #109), never for a full one-hot recolor whose background is an explicit channel. The
EV model's `floor_est ~= best/2.5` "SWEEP" heuristic does NOT survive the full-one-hot output contract
+ F32-input lock — treat such HIGH-confidence estimates as unproven when the best is already free-output.
