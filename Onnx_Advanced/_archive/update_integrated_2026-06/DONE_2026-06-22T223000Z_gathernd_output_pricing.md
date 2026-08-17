# GatherND "free output" probe — confirmed CHARGED unless IS the graph output node

## What was discovered

Task110 beta_p21 session (worker-5): probed the "modular GatherND direct output" angle — replace the U8-carrier + Equal tail by GatherND of the FREE F32 input at modular indices (r%p_v, c%p_h), feeding output directly.

**Result: GatherND output IS charged as intermediate unless it IS the graph output node (dtype must match).**

Specifically:
- `GatherND(F32_input[1,10,30,30], idx[1,10,30,30,2])` → F32[1,10,30,30] — if there is a `Cast(F32→BOOL)` afterwards before the BOOL graph output, the F32[1,10,30,30] intermediate = **36,000B charged** (NOT free). profile==static==54,000.
- Only way to make GatherND output free = make it the graph output node directly. But the output must be BOOL, and GatherND on F32 gives F32 — type mismatch. No direct path.

## Where encountered

- Task 110, worker-5, `The_400_Tasks/Task_by_Task/task110/prototypes/w5/probe_gathernd.onnx`
- profile_real_cost --arcgen 266 → static=54,000 profile=54,000 (memory 36,000 + params 18,000)

## Why it matters

Validates the #109 free-output rule: the output is only FREE if the node producing it IS the graph output (last op). Any intermediate on the path to the graph output — even one step before a Cast — is fully charged. The "free output carve-out" is very narrow: it applies ONLY to the tensor that is literally the graph output node's output.

## Secondary finding: compact-tile tail beats direct-indexing

For a p×29 compact tile approach:
- Compact [1,1,p,29] (p≤9, e.g., 261B for p=9) → Pad to [1,1,p+1,30] (≤300B) → Gather rows → [1,1,30,30] (900B) → Equal → FREE output

vs. direct indexing from a 29×29 tensor:
- GatherND/Gather twice on [1,1,29,29]: 841+841+900 = 2,582B intermediate for just the tail.

The compact-tile approach saves because p≤9 rows (261B) is 3× cheaper than 29 rows (841B) per Gather step.

## What's missing from the library

Existing #109 "free-output" says: "the gathered one-hot is FREE if it feeds the output." This needs a clarification: FREE only if there is NO op between GatherND and the graph output node (including Cast). A Cast to change dtype breaks the free-output — the pre-Cast intermediate is charged in full.
