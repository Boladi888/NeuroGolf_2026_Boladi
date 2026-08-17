# task123 hard record: sub-475 factorization / QLinear routes

- live bar used: 475, sha256 `20ca07e240d729b997878ee4259fc241a7d59dafd757ad8ba58dae9ccf58d2c0`, from `python tools/ng_current_bar.py --task 123 --json` / team manifest.
- attempted route: beat `direct_diag_factoredH` by collapsing the remaining `P[10,30]`, `Srow[2,10,5]`, or branch carrier into QLinear/Conv or a smaller free-output Einsum factorization.
- exact failed gate: pre-build price/rank gate, followed by tie check against current best (`ng_graph_cost_diff` live vs prototype delta `0`; `profile_real_cost` `static=475 profile=475`). No full truth/ab was run because no candidate priced below 475.
- why dead:
  - QLinearConv carrier route has no U8 feed in the current graph; direct graph input is F32, and `Cast(input->U8)` costs the full one-hot frame (~9000 B), far above the 475 bar.
  - One-node final Conv/QLinearConv is only a linear threshold per output cell/channel. The needed K mux contains `f=(d4_is_0 AND d1c) OR (d4_nonzero AND d0c)` for q=5, which is not linearly separable: inequalities for positives `(x=1,y=0,z=1)`, `(x=0,y=1,z=0)` and negatives `(x=0,y=1,z=1)`, `(x=1,y=0,z=0)` force `c>0` and `c<0`.
  - `Srow[2,10,5]` flatten ranks are 5 over `(x,q)|i`, 8 over `(x,i)|q`, and 2 over `(q,i)|x`; exact CP factorization therefore needs at least 8 terms and prices above the dense 100-element bank. Branch-delta pricing is also worse: K5 rank 5 plus K4-K5 delta rank 4 plus branch selector exceeds 100.
  - The spatial ring masks require up to matrix rank 6; a general bilinear latent basis with L=6 prices at `U[6,30] + C[2,5,6,6] = 540` params, worse than the current `P + Srow = 400` renderer.
- one remaining hypothesis: a non-Einsum op with a true nonlinear mux that can write directly to `output` from F32 input, without an intermediate full grid and without U8 casting, would be a different route; no such ORT/scorer-supported op was identified in this pass.
