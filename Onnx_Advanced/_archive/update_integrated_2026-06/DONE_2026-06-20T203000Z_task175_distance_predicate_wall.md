# task175 — #108 distance-predicate collapse is a WALL when the per-cell value is non-separable

**What I discovered (a refinement/counter-case for technique #108 and zero_param_generators.md):**

Technique #108 says "collapse N stored mask/shell/ring planes into ONE per-cell distance
predicate." That works when the planes are BOOLEAN masks that a `Greater`/`Less` + tiny Gather
emits straight into a free-output Equal-tail. It does NOT work when the stored plane is a
**multi-valued INT offset table whose value is a genuine 2-D (non-rank-1) function of position**
and the downstream consumer is not a free-output contraction.

task175's dominant param is `offset_table` (INT8 [1,1,21,21] = 441 B). I decoded its EXACT
closed form (byte-verified over all 441 cells):

```
m = min(r,c); d = |r-c|
O = d // (m+2) - 1      for in-wedge  d >= m+2
O = 0                    on diagonal  d == 0
O = -1                   off-wedge (sentinel)
```

The table is symmetric and fully a function of (s=r+c, d=|r-c|). So it IS an arithmetic
predicate — #108 "should" apply. But pricing it kills it:

- Computing O on-graph needs Min, Sub, Abs, Add, Div, Sub, mask(Greater/Equal), Where —
  **≥8 surviving [1,1,21,21] planes**. Under the byte-memory scorer each is 441 B (INT8/BOOL),
  and `Div` may force INT32 (1764 B each). Measured floor = **3,528 B** of intermediates to
  replace ONE 441-B stored init (+ the 441-B `Mod` plane the real graph already has).
- Net ≈ **+2,600 B** → ~4,600 total, vs the stored-table candidate at 1,979 (team-best 1,277).

**Why #109 free-output doesn't rescue it:** the actual output color is `(base+O) mod period`,
a data-dependent, non-low-rank per-cell value (rank-tested: not a row⊗col outer product). There
is no K/R/C Einsum decomposition, so the arithmetic planes cannot be absorbed into a free-output
contraction — they survive and get charged.

**Durable rule (suggest adding as a #108 discriminator / a zero_param_generators.md caveat):**
A per-cell distance/offset PREDICATE collapses cheaply ONLY when (a) it is BOOLEAN (or feeds a
free-output Equal/Einsum tail), or (b) the generated tensor replaces a LARGER init without adding
≥1 full-size intermediate per op. For a SINGLE multi-valued plane consumed by a charged Mod/Pad/
Equal canvas, the cheapest carrier under this scorer is the stored packed init itself; regenerating
it arithmetically pays one extra full-plane intermediate per op and always loses. "It has a closed
form" is necessary but NOT sufficient — price the surviving-intermediate memory first.

**Bonus (family-label correction):** task175 is an INPAINT rule — `output == input` wherever
`input != 0`; only the `0`-holes are filled by the offset field. The "Euclidean wedge from a
center" label describes the visual, not the operative transform. Neighbor-copy fill of the holes
fails (best single-shift covers 91/194 holes; interior hole blocks have hole neighbors), so the
offset field is the genuine mechanism.

**Where:** `The_400_Tasks/Task_by_Task/task175.md` (Rule + Attempts + Don't-do updated this
session). Local best accessible candidate stays v21 @ 1,979
(`staged_submits/task175_shifted_offset_masked_floor_cost1979_sha76A6392A/`). Team-best 1,277
(Aditya) graph is NOT in this repo, so I could not independently attack ITS structure — only
confirm the assigned #108 lever is a wall and that v21 (1,979) > 1,277 leaves nothing stageable.
