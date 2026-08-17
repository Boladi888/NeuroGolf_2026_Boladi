# R2 selector-bank rank execution

Date: 2026-06-07

Scope: live shipped graphs from `submissions/submission.zip`, with task notes and `live_costs.json` used for context. The root `task304.onnx` and `task312.onnx` files differ from the live zip, so the rank inventory below uses the packaged/live ONNXs.

## Executive result

Task312 is shrinkable. The shipped rank-19 CP bank has flattening ranks `[10, 10, 10]`, so the cheap kill-test does not prove minimality. A signed bounded-integer exact decomposition over `{0,+1,-1}` with rank 18 was found, rebuilt, fully gated, and staged:

`staged_submits/task312_rankcut_cost570/task312.onnx`

Cost moves from 600 to 570 (`memory=0, params=570`), for `ln(600/570) = +0.051293` score points. The full theoretical gap to rank 10 is not closed; rank <=17 remains an open exact-rank question because no MIQCP/CP-SAT/SMT solver is installed locally (`gurobipy`, `ortools`, `z3` all absent).

## Inventory and verdicts

| task | bank | bank_shape | current_rank/elements | flattening_ranks | min_exact_rank_found | alphabet | new_elements | point_gain | status |
|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| 312 | effective `BANK[k,c,d] = sum_t u[k,t] vw[c,t] w[d,t]` | `[10,10,10]` | `T=19`, 570 factor elems (+ `e0[30]` = 600 total) | `[10,10,10]` | 18 | `{0,+1,-1}` | 540 factor elems (+ `e0[30]` = 570 total) | +0.051293 | SHRUNK-staged |
| 304 | `H2` channel map | `[10,10]` | dense 100 elems; matrix rank 10 | `[10,10]` | none cheaper | n/a | n/a | 0 | DEAD-proven-minimal for factorization cost |
| 304 | `Ddiv` selector | `[30,3]` | dense 90 elems; matrix rank 3 | `[3,3]` | none cheaper | n/a | n/a | 0 | DEAD-proven-minimal for factorization cost |
| 304 | `Dmod` selector | `[30,3]` | dense 90 elems; matrix rank 3 | `[3,3]` | none cheaper | n/a | n/a | 0 | DEAD-proven-minimal for factorization cost |
| 357 | live `packbank` lookup | `[9,5,1]` | packed lookup 45 elems; not an einsum CP bank | `[5,5,1]` | n/a | n/a | n/a | 0 | DEAD-not-a-live-CP-bank |

R1 worklist check: no additional live selector-bank CP-rank target was added. The R1 hits mentioning `einsum`, `selector`, `rank`, or `bank` are formulation rewrites or future task attacks, not already-shipped selector CP banks needing R2 rank execution.

## Task312 derivation

The shipped bank is:

```text
BANK[k,c,d] = (c != 5 ? d == c : d == k)
```

The old construction used 19 nonnegative terms:

```text
9 terms:  ones(k) * e_c(c) * e_c(d), c != 5
10 terms: e_k(k) * e_5(c) * e_k(d)
```

Flattening ranks are `[10,10,10]`, so the unconditional CP lower bound is only 10. The rank-19 construction is therefore not killed by Step 0.

Rank-18 signed construction, anchor `a=0`:

```text
For each c != 5:
  ones(k) * (e_c(c) + [c == a] e_5(c)) * e_c(d)

For each j != a:
  e_j(k) * e_5(c) * (e_j(d) - e_a(d))
```

At slice `c=5`, this gives:

```text
ones * e_a^T + sum_{j != a} e_j * (e_j - e_a)^T
= I
```

All factor entries are `-1`, `0`, or `+1`; all products and accumulated sums stay tiny integers, well inside the binary32 exactness bound. The builder asserts exact tensor equality with the original bank.

Implementation: `The_400_Tasks/Task_by_Task/task312/prototypes/build_task312_v4.py::build_v7`.

Staged artifact:

```text
staged_submits/task312_rankcut_cost570/task312.onnx
sha256 D6A0ACB72238A23D3E0847F6713B6F4C7C6EA8BEAF661CB9DBB8A712B0F95EFE
```

## Gate results for staged task312

All gates were run against the staged path.

```text
visible:       265/265 PASS
truth --full: 200000/200000 PASS
ab --full:    200265/200265 byte_equal, divergences=0
score_v2:     (macs=0, memory=0, params=570), cost=570
profile real: static=570 profile=570 (memory 0 + params 570; 3 real inputs)
```

AB reference was the live packaged graph extracted from `submissions/submission.zip` to `tmp/task312_live_zip.onnx`.

## Task304 notes

Live task304 has:

```text
Einsum('begv,be,gr,vk,xr,yk,xm,hm,yn,wn,bdhw,cd->bcxy',
       input, sgn, Dmod, Dmod, Ddiv, Ddiv, Dmod, Dmod, Dmod, Dmod, input, H2)
```

`H2[10,10]` has exact matrix rank 10. A two-factor rank decomposition would need at least `10*(10+10) = 200` elements, worse than the dense 100-element initializer. The `Ddiv` and `Dmod` matrices each have rank 3; factoring either costs at least `3*(30+3) = 99`, worse than the dense 90-element initializer. No smaller exact decomposition was found or worth staging.

## Task357 notes

Live task357 is no longer an einsum-final selector bank. It uses a packed position lookup:

```text
packbank[9,5,1] uint8 = 45 elems
```

The graph unpacks nibbles with `Mod`/`Div`, compares against `col[10]`, builds `X[1,1,10,10]`, and ends with `ConvInteger` as the output placer. The older dense/permutation-bank contraction route is already killed in `task357.md`: dense `P[9,10,10]` is 900 params, flattening max rank is 10, rank-10 rectangle support is disproven, and even the optimistic rank-11 route is about 319 params before extraction overhead. That does not beat the current packed lookup path.

## Solver availability

Installed locally: `onnx`, `numpy`, `onnxruntime`, `sympy`, `scipy`.

Not installed locally: `gurobipy`, `ortools`, `z3`, `cvxpy`.

Because the exact solver stack requested by L1/M2 is unavailable here, I did not claim an infeasibility certificate for task312 ranks `17..10`. The rank-18 construction is exact and gated; lower ranks remain open until a bounded-integer MIQCP/CP-SAT/SMT run is available.

## Task312 verdict

The +0.51 theoretical residual was real, not proven-minimal: the live `T=19` bank shrinks exactly to `T=18`, saving 30 params and staging a cost-570 candidate. The full rank-10 lower-bound target is still unproved and not reached in this run.
