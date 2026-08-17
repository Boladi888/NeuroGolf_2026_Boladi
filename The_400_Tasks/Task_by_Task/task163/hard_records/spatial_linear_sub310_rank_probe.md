# task163 spatial_linear_sub310_rank_probe

- Live bar: **310** params/memory 0, sha `d784d1c2166c70d4255aa51164f685d47b12b3f17a075424b740d7bfcd5bf64a`, source `python tools/ng_current_bar.py --task 163 --json` / team manifest owner Vamshi.
- Attempted route: replace the current `RL[30,4] + RB[30,3] + TL[4,3,2] + TB[3,3,2]` spatial carrier with a lower-dimensional exact linear coordinate basis below 310.
- Exact failed gate: algebraic rank pre-gate before building. The current row relation
  `R[h,i,r,t] = RL[h,m] TL[m,x,t] RB[h,o] TB[o,y,t] RB[r,x] RL[r,d] RB[i,y] RL[i,d]`
  has active-region flattening ranks for the copy term `t=0`: `h|ir=9`, `i|hr=11`, `r|hi=11`, `hi|r=11`. The combined `hi|rt` rank is `15`.
- Why dead: the copy relation is full-rank over the 11 active row states for the exact linear carrier class. A smaller exact linear basis would have to recreate an 11-rank row/source/output relation, so it cannot simply alias or linearly compress the current block/local factors. A second node is not viable because it would charge a full output-sized intermediate.
- One remaining hypothesis: a genuinely nonlinear/sign-rank spatial construction could still beat 310, for example a polynomial or simplex code that uses threshold margins instead of exact 0/1 row-relation equality. No concrete legal one-node graph was found in this pass.

