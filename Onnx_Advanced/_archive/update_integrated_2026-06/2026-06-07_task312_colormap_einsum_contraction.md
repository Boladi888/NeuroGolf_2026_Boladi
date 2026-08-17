# Per-row color-map as ONE Einsum-as-output — in-contraction selector + CP-factored bank (task312, 2026-06-07)

**Instance of techniques #68 ADDENDUM (contraction finals + params-only class; task304
precedent). task312: 12,211 LIVE → 600 (memory ZERO), truth 200k/200k, ab byte-equal.**

## Pattern

Any rule of the form "recolor each cell by a color map that is selected per-row (or
per-column / per-grid) by a key readable from the input" is ONE einsum:

```
output = Einsum('v,bkhv,kcd,bchw->bdhw', e0[30], input, BANK[10,10,10], input)
       = Σ_{c,k,v} input[c,h,w] · input[k,h,v]·e0[v] · BANK[k,c,d]
```

- `input` appears TWICE (both free); `output` is the node result (free) ⇒ **memory = 0**.
- `e0` = one-hot column selector (30 params) — the key (here col0 color) is read INSIDE
  the contraction; no Gather, no materialized per-row P, no canvas, no collapse.
- `BANK[k,c,d]` = the color map for key k (here: identity except 5↦k) = 1,000 params dense.
- **CP-factorization of the bank**: BANK = Σ_t U[k,t]·V[c,t]·W[d,t] with T = (#identity
  colors) + (rank of the key-dependent block) = 9+10 = 19 → operands `kt,ct,dt` of
  [10,19] = 570 params total (cost 600 with e0). Keep the reconstruction EXACTLY 0/1
  (assert in the builder); exact tensor rank of I_n is n, so the δ(d,k) block needs 10.

## Correctness discipline

- One-hot inputs ⇒ exactly ONE nonzero monomial per output cell ⇒ F32-exact, no
  accumulation/cancellation (the task191 F16 multi-operand drift warning doesn't bite).
- Padding cells/rows: all-zero input ⇒ zero products ⇒ all-False decode. Check the
  key-readable-at-padding case (here col0oh=0 ⇒ P=0 ⇒ 0 ✓).
- **Operand ORDER = smallest first**: ORT reduces pairwise; `e0×input → [1,10,30]` then
  `×U/V/W → [1,30,·,·]` then `×input → output` keeps internals ≤ a few KB. A bad order
  (input×input first) makes a 10.8 MB internal.
- ORT ran 6 operands over 200k+ pairs cleanly (task304: 9 operands).

## When to reach for it (#65 question FIRST)

Ask: **is the output FULLY-GENERATED-expressible** — every cell a (possibly identity) map
of input content? Then contraction finals compete and usually win; copy-semantics floors
(Where-as-output passthrough arguments) DO NOT BIND. This exact blind spot held task312 at
2,775 for three hardening rounds. Color Mapping family is full of candidates: any
"recolor X by rule keyed on a row/col/grid feature" task.
