# Einsum all-ones contraction elimination: 3-op → 2-op for implicit axis summation

**What I discovered**: In any N-operand Einsum where one operand is a vector of all-1s used solely to sum over a dimension, dropping that operand and changing the equation to implicitly sum that axis is mathematically equivalent AND saves the params cost of the all-ones vector. The 2-operand form is ALSO more trusted than ≥3-operand form (§74/§42 caveat: ≥3-op Einsum may materialize partials).

**Concrete case (task379 v3)**:
```python
# v2: 3-operand with ones30=[1,1,...,1] (30 F32 elements, 30 params)
Einsum(input, cyan_sel, ones30, "nchw,c,h->nw")  # sums over c and h
Einsum(input, cyan_sel, ones30, "nchw,c,w->nh")  # sums over c and w

# v3: 2-operand, h/w implicitly summed (0 additional params)
Einsum(input, cyan_sel, "nchw,c->nw")  # h is free index in nchw, not in output nw → summed
Einsum(input, cyan_sel, "nchw,c->nh")  # w is free index in nchw, not in output nh → summed
```
Math is identical: with ones30[h]=1 for all h, the 3-op form reduces to the 2-op implicit sum. Verified byte-equal (200266/200266 AB gate) after change.

**Where encountered**: task379, file `The_400_Tasks/Task_by_Task/task379/prototypes/build_task379_v3.py`, staged at `staged_submits/task379_v3_2op_einsum_cost8427/`. Gain: -30 params, cost 8457→8427.

**Why it matters**: Any graph that uses a 3-operand Einsum with an all-ones vector for contraction can be simplified to 2-operand. This saves params AND makes the Einsum unconditionally trusted (not subject to §42/§74 ≥3-op caveat). Transferable to: any task where a "row sum" or "column sum" is computed as `Einsum(input, channel_selector, ones_N, "nchw,c,h->nw")` or similar.

**Trap to avoid**: Do NOT replace with Range+Cast at runtime. A [30]-element Float Range costs 3 scalar params + 120B Float32 intermediate = net +93 cost vs the -30 param save. The implicit equation form costs nothing extra.

**What's missing from the library**: Technique §74 mentions 3-operand Einsum trust but doesn't explicitly document the "all-ones contraction → implicit sum" simplification. The relevant pattern is: if an Einsum operand is all-ones and its contracted index appears in the equation's implicit-sum (not in the output), you can drop that operand. This could be added as a sub-bullet to §74 or as a note in techniques_catalog.md.

## Sorter note: techniques.md §74 — extend with "all-ones operand → implicit axis sum" simplification (2-operand form, saves params AND removes ≥3-op trust caveat)
