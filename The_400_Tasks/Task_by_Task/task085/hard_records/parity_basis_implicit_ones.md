# parity_basis_implicit_ones route

- Live/team-best bar used: cost/profile 136, sha256 c2d0c80df4932d3b3c782185af15b9d2749d2f6caee9c1fbfe9a5e74895873b7, source workbook + team manifest refreshed 2026-07-03. No task-local or Boladi staged-pending candidate was present.
- Reviewer-locked hypothesis: replace the live `parity_basis = [ones; s_i]` (`s_i=(-1)^i`, 60 params) with one 30-param sign vector and keep a single terminal free-output `Einsum`, using #89/#119 in-kernel factors only.

Live equation:

```text
bkhw,ph,uw,up,pr,uc,uad,ak,do,bkrc->borc
```

With `P[0,i]=1`, `P[1,i]=s_i` and `row_weight = [[1,0],[1/6,-1/2]]`, the nonzero `(u,p)` terms expand to exactly three spatial monomials:

```text
u=0,p=0: 1                         * C0
u=1,p=0: s_w * s_c                 * (1/6) C1
u=1,p=1: s_h * s_w * s_r * s_c     * (-1/2) C1
```

Single symbolic reconstruction attempted:

```text
bkhw, H_t[h], W_t[w], R_t[r], C_t[c], G[t,a,d], ak, do, bkrc -> borc
```

where `t` enumerates the three monomials above and `G` carries the `C0`, `(1/6)C1`, and `(-1/2)C1` coefficients. This preserves the #89/#119 free-output boundary only if all term selection lives inside the final `Einsum`.

Param-count pass:

```text
current: color_basis 40 + parity_basis 60 + row_weight 4 + channel_core 32 = 136
sign-only before term selectors: color_basis 40 + sign_vector 30 + factored term/core 38 = 108, but this is not expressive: it can emit only one fixed sign monomial.
required selectors: H/R rows [1,1,s], W/C rows [1,s,s]
best selector form: color_basis 40 + factored term/core 38 + two axis-shared T x 30 selectors 180 = 258
full selector form: color_basis 40 + G[t,a,d] 48 + four T x 30 selectors 360 = 448
```

Verdict: dead. The all-ones row can be implicit for one fixed monomial by omitting the sign operand or by using `s_i*s_i=1`, but the live formula needs a sum of three different monomials inside the same terminal `Einsum`. Selecting which axes use ones vs signs requires term-conditioned `T x 30` selector rows, which is the reviewer's explicit stop condition and also prices above the 136 bar. Splitting the three monomials into separate `Einsum` nodes would materialize/add full `[1,10,30,30]` outputs and violates the single-terminal-free-output constraint.

No ONNX was built and no visible/truth/A-B gates were run for this route because the symbolic pass failed before the build gate.
