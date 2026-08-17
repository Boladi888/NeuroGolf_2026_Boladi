task043 hard record - shared-selector 54 blocked by STRICT one-hot decode (marker lines)

Live bar:
- cost 84, sha256 1e0d69485acbed45aafb3f94ab63b182f635896031144d8fb7fa9865c6c8ad01
- source: `python tools/ng_current_bar.py --task 43 --json` (manifest authoritative, owner Vamshi)

Live graph = one free-output Einsum (score_v2 = (0,0,84)):
  equation `bqrw,lq,barx,ka,x,bdyw,kd,y,km,mc->bcrw`
  inits: C[2,10]=20, M[2,2]=4, right_col[30]=30 (=e9), top_row[30]=30 (=e0).  memory=0 (free output).
Decoded rule: top-row(row0) gray-5 -> marked COLS; right-col(col9) gray-5 -> marked ROWS;
  output = keep input markers, paint red-2 at every (marked row x marked col) interior intersection.

Only sub-84 lever: share the two length-30 selectors into ONE initializer referenced by
Einsum operands 4 and 7 -> cost 54 (C20+M4+S30). Selectors cannot be shortened (contracted
against input's 30 spatial axis; crop/sparse routes already dead) and C=20, M=4 are forced
(see floor proof in task043.md).

DECODE CONTRACT (the trap): the harness decodes NOT by argmax. `encode.decode_to_grid`
assigns a cell to color c iff EXACTLY ONE channel is > 0 (npos==1). So the 10-channel output
must be a true one-hot: target channel > 0, all others <= 0. Only channels {0,2,5} are ever
non-zero (C has support only there), so per cell: out[target] > 0 AND the other two of {0,2,5} <= 0.

Attempted route + prototype:
- `prototypes/task043_shared_selector_joint_cost54.onnx`
  sha256 d9bdc944f12ec493fd31f119c6d28aa31aad8236c847908c29886fa4f1d35040, score_v2 (0,0,54).
  Built from an ARGMAX-feasible (C,M,S) joint solve (0/266 bad under argmax).
- Exact failed gate:
  `python tools/ng_gate.py visible --task 43 --cand .../task043_shared_selector_joint_cost54.onnx`
  -> `[visible] 0/266 (0.00%)  FAIL`
- Mechanism (verified via ORT): the argmax solution yields npos==2 (two positive channels) on
  95/100 active cells -> strict >0 decode rejects it. Argmax-optimal != one-hot-valid.

Why the route is DEAD (strict one-vs-rest shared selector is infeasible in practice):
- With shared S, interior reads reduce to (any S support): u_U=v_U=p ; u_M=p+S9*d ; v_M=p+S0*d
  where p=total*C[:,0], d=C[:,5]-C[:,0]. The GENERIC interior 4-combo strict system (UU/UM/MU->bg,
  MM->red) IS strictly feasible (fast solver `t43_generic.py` finds it in <5 restarts).
- BUT seeding the full 266-example strict solve from generic-feasible points drives interior
  cells to EXACTLY 0 failures while failures pin to row0 (1179) and col9 (1219); interior=0.
  Root cause: on the marker lines the shared selector's "leak" IS the signal. rowdet(k,0) and
  coldet(k,9) become data-dependent weighted sums over the marker pattern (Sum_x S[x]*read),
  varying with which/where markers sit; a one-hot e0/e9 isolates a single cell, a shared S cannot.
- Independent formulations all fail on the marker lines: argmax-joint (valid argmax, invalid
  decode), strict one-vs-rest multistart (>=228/266), homotopy separate->shared (breaks as
  Scol->Srow tie), scale-invariant seeded-from-generic (interior 0, row0/col9 ~1200 each).
  Prior batch's `shared_selector_live_coeff.md` reached the same 0/266 wall.

One remaining (thin) hypothesis:
- A radically different output-axis basis (not the live bilinear rowmask (x) colmask read) could
  in principle re-price the marker-line reads, but every known variant still needs one length-30
  contraction per axis (60) unless shared, and sharing is what fails here. Treat 84 as the floor
  for this einsum family absent a genuinely new contraction structure.
