# task346: occupancy-packed window sums + max-bound existence test (2026-06-07)

Two composable tricks from the task346 floor campaign (9,178 → 523, all gates gold,
ladder in `staged_submits/task346_v*`). Both generalize; candidates for #-entries
extending #69/#70/#71.

## 1. 128-spacing occupancy pack — ONE conv plane carries (sum, occupancy)

Where a detector needs BOTH a window's value-sum AND its occupancy count (the #71
moment pair is the usual answer), pack them into one kernel when value range allows:

    w_pack(c) = c + K*[c >= 1],   K > max possible window value-sum
    ringval   = A + K*occ         (A = idx-sum, occ = #nonbg, one Conv plane)

Any (occ, A) pair decodes uniquely because A < K (task346: A <= 72, K = 128, F16/F32
dyadic-exact after /8 prescale). A "solid window of color v with occupancy n" test
becomes ONE Equal against `n*v + K*n` — partial windows can't fake it (gap K beats
the whole A range), and for the 2-colors-present case full-mixed windows can't either
(exchange argument: ka + (n-k)b == n*a  =>  k == n). Halves the #71 plane bill
(no second-moment conv, no Mul-square, no variance Equal).

## 2. Max-bound existence test — exists(solid-of-MAX-color) with NO fire mask

When the question is "does a solid window of color `a` exist" and `a` is the MAXIMUM
present color, every cell value <= a bounds the packed statistic:

    ringval <= n*a + K*n,  equality IFF some window is solid-a
    exists  ==  Equal(ReduceMax(ringval), n*va + K*n)     # 4 B + 1 B, scalar-land

No comparison plane, no Where plane, no BOOL->U8->Reduce hop — the aggregation IS the
detector. Multi-fire immune by construction (exists-semantics). Pair with a structural
guarantee ("a solid block always exists") to resolve the complement branch:
`answer = Where(exists, other, va)`. task346 assembles `other = ReduceSum(present_ids)
- va` because exactly 2 colors are present (presence via Sign(ReduceSum counts) —
count-weighted max is WRONG, see task346.md don't-do).

## Supporting datapoints

- 4-operand dummy-axis einsum `'bchw,c,d,e->bcde'` profiled EXACT (537 == 537) before
  being replaced by plain ReduceSum (per #74 "don't einsum unweighted reductions").
- Three S41-negpad Convs (begin+end) in one shipped-class graph profile exact at 0% gap
  (more begin-pad evidence for the #41 addendum).
- BOOL graph output + Pad-as-free-output (#67) reconfirmed.
