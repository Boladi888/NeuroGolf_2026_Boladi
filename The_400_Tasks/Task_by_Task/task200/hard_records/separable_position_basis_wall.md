# task200 — WALL: position-basis / separable formulation is already the live 350 graph

**Date:** 2026-07-03 · worker `bo_ng_cld_worker1` (Claude lane, batch48 slot02)

## Live bar (authoritative)
- `python tools/ng_current_bar.py --task 200 --json` → **live_cost 350**, sha256 `389ab76bdf5983b1226a402980560c037b2ba190db6f81f129b884f14ddb09dc`, owner Vamshi, source=manifest (workbook agrees 350; local task_state 613 is STALE — ignored).
- No lower rung staged (`Staged_Boladi/` empty for task200; manifest = 350).
- Live profile: `static=350 profile=350 (memory 204 + params 146)`.

## Prompt premise is STALE
The prompt ("kill the A/B LUTs [10,10,10], target ~120") was authored against the **old 613 A/B-LUT graph** (`task200.md` diagnostic still shows `A_table`/`B_table` [10,10,10]). The **live team-best (350, Vamshi)** already dropped the LUTs — it is `task200_selected_scalar_moments_state2`: a moment-decode + Conv + separable Einsum, i.e. exactly the position-basis / separable formulation the prompt requests. **The structural win the prompt targets was already realized.** `floor_est ~120` was measured relative to 613, not to the current 350 separable graph.

## Rule (confirmed on all 84 pairs, 0 conflicts)
Single marker (color c, always row 9, col k). Output = fixed template T[r, j], j=col−k (right-only, clipped), recolored: label1→c (vertical lines at even j, ALL rows = rank-1), label2→5 (fives: row0 at j∈{1,5,9}, row9 at j∈{3,7} = rank-2). Decode = (k, c) two scalars. Grader decodes by per-cell channel argmax (background = all-zero → ch0).

## Cost model (measured from neurogolf_utils)
cost = params (initializer **element count**, dtype-free) + memory (intermediate **bytes** = elems×itemsize; **input & output tensors are FREE**). ⇒ never materialize a full 30×30 (=900-elem) intermediate; emit output in the final op.

## Independent formulations tried (all price ABOVE 350)
| # | formulation | correct | params | memory | cost | verdict |
|---|---|---|---|---|---|---|
| 1 | rank-3 CP `sum_t Cf[t,ch]·Rf[t,r]·Wf[t,w]`, Conv-shifted comb, F32 | 84/84 | 153 | 880 | **1033** | rank-3 + F32 intermediates lose |
| 2 | live-shave: drop k-moment round-trip, feed direct column one-hot into same Conv | 84/84 | **127** | 408 | **535** | −18 params but +204 memory |
| 3 | scalar-k via combined `Einsum('bchw,c,w->b', input,nonbg,colcoord)` (analysis only) | — | ~=live | ~=live | ~=350 | no net gain vs live's shared-color_coord Div trick |

Prototypes: `prototypes/rank3cp/build_rank3cp.py` (#1), `prototypes/rank3cp/build_liveshave.py` (#2).

## Forced-component argument (why 350 is at/near floor)
`ng_graph_cost_diff` on #2 vs live (named, reconciled):
```
              memory  params  cost
live             204     145   349
candidate(#2)    408     127   535   (delta +204 mem / -18 par / +186)
```
- Dropping `column_coord[30]`+`position_*` saves 18 params, but forces materializing the width-30 column one-hot: `kvec`(F32 120B)+`kplane`(F16 60B)+`kvec16`(F16 60B) = **+240 B**, removing only `seed_position`/`position` (~36 B). Net regression +186.
- **Memory floor is the single width-30 Conv-output `states` (F16, 120 B).** Any formulation that shifts a fixed template by a *decoded* offset k must materialize ≥1 width-grid intermediate. Live has exactly one; carrying the shift instead as a scalar-k basis inside the free final Einsum costs `[2,10,30]=600` params (dead). So the Conv trade (pay ~120 B memory to save ~580 params) is forced, and live already takes it.
- Live's **scalar-first k extraction** (moment → scalar → OneHot(10) → Conv) keeps every pre-Conv intermediate tiny; the direct column-one-hot route (#2) is strictly worse on memory. Live also reuses `color_coord` as the moment weight (Div by c) so k+c share one 30-coordinate = 40 params; a separate `nonbg` indicator adds 10 with no offset.

## Verdict
**WALL at 350.** Live is the forced floor of the separable position-basis family for this task; the prompt's ~120 target reflects the retired 613 LUT graph, not the current graph. Nothing staged — never stage a tie/regression. Do not re-explore "kill the A/B LUTs" (already done) or direct-column-one-hot (memory regression, priced above).

## One remaining (low-confidence) hypothesis for a future batch
Only conceivable sub-350 lever: a formulation that emits the output with **zero width-grid intermediate** by folding the k-shift into the FREE final Einsum without a 600-param basis — e.g. exploit periodicity (even-column comb = a col-parity indicator) so the shift is a scalar phase rather than a full basis. Unproven; parity-from-k likely needs its own width-grid intermediate too, so expect it to also wall.
