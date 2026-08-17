# task052 hard records — perfect-25 (cost≤1) and rank-2 (cost 20) both DEAD

- **Live bar**: 40, sha `06e5f6b27e01fc5a4c012e2d748a51684b55f91b9d0757f203f09ff708891591` (manifest, Aditya, 2026-06-12).
- **Rule**: per-row monochromatic → color 5, else color 0. Grids 3×3; input palette {1,2,3,4,6,7,8,9} (never 0/5, verified over 200k); output {0,5}.

## DEAD: perfect-25, cost ≤ 1 (the prompt's SPEC hypothesis)
- **Attempted route**: one-node / attr-only direct-output op (Identity, all Transpose perms, no-payload op preserving the ARC contract) priced ≤ 1.
- **Why dead (forced-component argument)**:
  1. Output colors {0,5} never appear in the input → output channels must be *generated*, forcing ≥1 constant to select channels 0/5.
  2. The transform is a within-row equality predicate (mono?), a genuine nonlinear/multiplicative computation — not a shape-only rearrange. Local pre-scan found no whole-grid zero-param identity/transpose/flip/rotate match.
  3. No no-payload op computes "row equal → paint 5 else 0". Cost ≤ 1 is unreachable.
- **Failed gate**: n/a (killed at design — a cost-≤1 op cannot even be correct). Do not build one-node sweeps for this task.
- **One remaining hypothesis**: none for cost≤1. The task floor is bounded below by the channel-routing factor.

## DEAD: rank-2 single-Einsum, cost 20
- **Attempted route**: same live structure `bahx,bdhy,bjhw,at,dt,kt->bkhw` with `U[10,2]` (params 20).
- **Why dead**: `out[k]=Σ_t U[k,t]·P[t]²` with P∈R² is a diagonal quadratic form in (P0²,P1²)≥0. Getting channels 0 and 5 to take *opposite* signs across the three row-classes (mono, (2,1), (1,1,1)) for every color combo is infeasible at rank 2. Best fit: 80.3% strict decode on the 266 real rows; 95.4% over the complete palette set. min_target went negative.
- **Failed gate**: never reached visible (fit could not achieve a positive strict-decode margin). Do not re-run truth/ab on rank-2.
- **One remaining hypothesis**: a *non-symmetric* factorization (distinct input-projection V and output-routing W matrices) might reach a smaller total, but W must still generate all 10 output channels (shape [10,·]) so params ≥ 10·rank(W); combined with V it is not obviously < 30. Untried; low EV vs the confirmed rank-3 (cost 30) win.

## LIVE win found this batch
Rank-3 re-fit → **cost 30** (−10 vs live 40), 100% truth (200000/200000 closed-form + official gate), visible 267/267. Staged at `staged_candidates/task052_rank3_einsum_cost30/`.
