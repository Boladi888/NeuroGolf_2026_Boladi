# task045 hard record: final-Einsum input-width carry

- Live bar: 90 bytes, sha256 `151314a926f7a39e4ab3bef82481b135f2e5382cef065996d7d59c1a6115db84`, source `python tools/ng_current_bar.py --task 45 --json` / manifest owner Vamshi, updated 2026-07-12T15:15:54Z.
- Reviewer-locked route: start from the live one-node Vamshi `Einsum` and carry output axis `w` from the free input operand, not from any learned `[*,30]` output-width basis. Do not retry label-tail, diff2, nonmatch Where, sparse, packed bitset, or rank-2 basis.
- Algebra simulation: `python The_400_Tasks/Task_by_Task/task045/prototypes/analyze_task045_input_width_carry.py`.
- Exact result:
  - `domain_pairs=82`
  - `cell_color_width_path_active_misses=72`
  - `left_color_active_mask_path_active_misses=648`
  - `required_column_roles=left,interior,right rank=3`
  - `minimum_full_width_role_basis_cost=90`
  - `equal_tail_fallback_cost>=900`
  - `verdict=NO_SUB90_INPUT_WIDTH_CARRY`
- Why dead: the row rule is an exact mux. Unequal endpoints require the current-cell path (`output(w)=input(w)`); equal endpoints require the left-color active-mask path (`output(w)=left_color*active(w)`). A terminal `Einsum` that carries `w` only from input can supply one path, but an exact one-node mux between them still needs spatial target roles for left endpoint, interior, and right endpoint. Those three role vectors have rank 3 over the fixed 30-wide output ABI, so the cheapest full-width learned role/basis family is `3*30=90` params, tying the live graph rather than beating it. A full delta selector or Equal-tail fallback is larger (`>=900` for Equal-tail by #81).
- Remaining hypothesis: only an external scorer/accounting change that makes a three-role selector cheaper than 90, or a non-Einsum final op outside this locked request, could move below the current bar.
