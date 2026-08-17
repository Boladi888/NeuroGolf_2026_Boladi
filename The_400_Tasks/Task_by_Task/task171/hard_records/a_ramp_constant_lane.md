# task171 hard record: A[30]-only final-Einsum rewrite

- Live/team-best bar: cost 78, sha256 `f81e8726f351bb2df475f70b40f788e2941b5fef7d9f8e00a0bb88d02e7db3de`, source `python tools/ng_current_bar.py --task 171 --json` / manifest owner Vamshi updated 2026-07-05T04:54:56Z.
- Reviewer-locked route: start from `Team_Best_Onnx/task171.onnx`; replace live `B[2,30] = [coord, ones]` with a single coordinate ramp `A[30]`; keep one final `Einsum -> output`, zero charged intermediates, no sparse initializer, no Equal/Where/full-canvas condition, no second `[2,30]` basis.
- Live contraction: `bcij,bdrs,beuv,ti,tj,tpk,pi,kr,tql,qj,lv,tmn,o->boij`.
- Load-bearing subfactor: `sum_{p,k} R[0,p,k] * B[p,i] * B[k,r] = i - 2r`; the column side is the same shape, `j - 2v`.

## Capped coefficient solves

Script: `prototypes/prove_a_ramp_constant_lane.py`.

1. Candidate solve 1, A-only product family: `c*i*r`
   - fitted `c = -0.0508474576271`
   - `max_abs_residual = 58`
   - exact: `False`
   - verdict: cannot preserve the live row factor; visible would be a certain fail.

2. Candidate solve 2, expanded monomial family: `a*ir + b*i + c*r + d`
   - fitted coefficients: `a ~= 0`, `b = 1`, `c = -2`, `d ~= 0`
   - `max_abs_residual = 3.2e-14`
   - exact: `True`
   - ONNX final-Einsum blocker: terms `b*i`, `c*r`, and `d` require multiplying the other axis by an all-ones lane inside the same contraction (`i*1_r`, `1_i*r`, `1_i*1_r`). That all-ones lane is exactly `B[1,:]`.

## Why the route is dead

Scalar coefficients can scale monomials, but they cannot carry an axis label. In one final `Einsum`, adding `i` and `r` terms without separate large output nodes requires a term/basis operand whose rows are `[coord, ones]` over the 30 axis. Replacing that with:

- another `[2,30]`/term-selector basis violates the locked hypothesis and recreates `B`;
- a dense `[30,30]` row-factor table costs 900 params;
- separate expanded output-sized terms would materialize charged `[1,10,30,30]` or `[1,1,30,30]` tensors;
- sparse initializers were already rejected by scorer/sanitization.

No valid ONNX candidate was built under the cap because both allowed solves fail before construction: solve 1 is algebraically wrong; solve 2 is exact only by reintroducing the forbidden constant lane.

Remaining hypothesis: a sub-78 dense win would need a different final-output algebra that avoids the `i - 2r` / `j - 2v` linear factors entirely, not a fold of `B[1,:]` into scalars.
