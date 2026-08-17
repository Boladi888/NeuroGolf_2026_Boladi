# Transferable insight: counting distinct NONZERO colours via sum(presence) - 1

## What I discovered

When a task needs `k = number of distinct nonzero colours`, the common pattern is:

```
presence    = ReduceMax(input, axes=[2,3])     # [1,10,1,1] per-channel presence
presence_nz = presence * ch_nonzero            # zero out channel 0  (ch_nonzero = 10-param mask)
k           = ReduceSum(presence_nz)
```

This pays a **10-param `ch_nonzero` initializer** and a **40-byte `presence_nz` intermediate**.

For ARC tasks where the active grid is padded into the 30x30 one-hot canvas (i.e.
the grid is smaller than 30x30, so the surrounding region is background), the
**background channel 0 is ALWAYS present** in every input. That makes the
nonzero-mask redundant:

```
presence = ReduceMax(input, axes=[2,3])        # [1,10,1,1]
s        = ReduceSum(presence)                 # [1]   == k + 1   (ch0 always counted)
sm1      = Sub(s, one)                          # [1]   == k       (one = 1-param scalar)
```

Saves the 10-param `ch_nonzero` mask + the 40-byte `presence_nz` tensor, costing
only a 1-param `one` scalar + a 4-byte `sm1`. **Net -45 bytes** on task289
(679 -> 634), purely in the k detector, no change to the rest of the graph.

## Precondition (must verify per task)

ch0 (background) present in EVERY scored input. For any task whose grid is
strictly smaller than 30x30 this is automatic. **Verify before relying on it** —
e.g. a full-canvas task could in principle have zero background. On task289 I
checked all 268 official inputs: 0 had ch0 absent, and `sum(presence)-1 == k`
held on all 268.

## Where

- Build: `The_400_Tasks/Task_by_Task/task289/prototypes/build_task289_einsum_sum_minus_one/build_task289_einsum_sum_minus_one.py`
- Staged: `staged_submits/task289_einsummicro_cost634/` (cost 634, profile 634, byte-equal vs LIVE on 200268 inputs)

## Why it matters

Generalises to any distinct-nonzero-colour count where background is guaranteed
present. The "subtract the always-present background" trick converts a
per-channel mask (>=10 params + a [1,10,1,1] product intermediate) into a single
scalar subtract (1 param + 4 bytes). Pairs naturally with the free-output Einsum
expansion family (#89/#119).

## What's missing from the library

I did not find this "k = sum(presence) - 1 when background always present" lever
in `techniques.md`, `cost_floor_reference.md`, or `zero_param_generators.md`. The
closest is the general distinct-colour-count discussion, but it always carries
the nonzero mask. Worth a one-line note in the k-detector / colour-count section.
