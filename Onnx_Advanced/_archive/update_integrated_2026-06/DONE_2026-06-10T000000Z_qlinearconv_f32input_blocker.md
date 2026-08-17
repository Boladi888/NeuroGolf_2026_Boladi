# #137 QLinearConv prerequisite: U8 input required — application limit + named-input bridge trap

## What was discovered

Task321 (task_cf98881b, 3-panel priority select) has a v6 architecture where three 1x1
negative-pad Convs run directly on the free F32 graph input (`[1,10,30,30]` F32) and emit
`[1,1,4,4]` F32 priority-code tensors (64 B each). Technique #137 would save 3×48=144 B by
making those tensors U8 (16 B each) via QLinearConv.

However, QLinearConv requires INT8/UINT8 input. The graph input is F32 (fixed by competition).
Three routes to U8 input were tried:

1. **Direct QLinearConv(F32 graph input)**: ONNX checker rejects at `check_model(full_check=True)`:
   `TypeInferenceError: input and zero_point pair is expected to have be same type`. ORT also
   rejects at load: `INVALID_GRAPH: Type 'tensor(float)' of input parameter (input) of operator
   (QLinearConv) in node () is invalid.` No workaround.

2. **Cast(input→U8) before QLinearConv**: Produces `[1,10,30,30]` UINT8 = 9,000 B charged
   intermediate. 9,000 B >> 144 B savings. Cost explodes.

3. **Named-input bridge trick** (attempted first in task192 v14, now confirmed on task321):
   Rename the graph formal input from "input" to "raw" (F32), add `Cast("raw", "input", to=U8)`.
   The scorer's `calculate_memory` has: `if tensor_name in ['input', 'output']: continue` — the
   Cast output named "input" IS skipped. But `sanitize_model` renames the graph's F32 formal
   input from "input" to "safe_name_0", and "safe_name_0" is NOT in the exemption list. So the
   original F32 graph input (`[1,10,30,30]` F32 = 36,000 B) becomes charged. Probe on task321
   scored 36,248 (memory=36,224 = 36,000 + 4×16 + 160). The same trap was seen in task192 v14
   (scored 39,395; the gap is from task192's extra intermediate tensors).

   Root cause: `sanitize_model` renames `graph.input[0].name` away from "input", making it a
   billable intermediate. The "input"/"output" string exemption only saves the Cast output, not
   the now-renamed graph input. Net result is ~80× worse than v6.

**Key rule confirmed**: #137 QLinearConv only applies when there is an EXISTING U8 intermediate
(from a prior Cast or QLinearConv step) to feed the downstream Conv. It cannot be applied to the
FIRST Conv in a graph that directly reads the F32 one-hot input.

## Where encountered

- Task: task321 (Worker 5 session 2026-06-10)
- Compared to: task192 v13/v14 (same failure, documented in task192.md don't-do)
- Build script probe: not saved (probe was run inline)
- Floor confirmed: v6 cost 434 = true floor; #137 does not change it

## Why it matters

The task321 prompt specifically directed #137 as the "untried lever" (the floor proof only killed
F16, panel-fusion, and extraction-merge). This investigation closes that gap definitively. The
result generalizes: any single-Conv graph where the Conv reads the F32 graph input directly cannot
be cheapened via QLinearConv.

The named-input bridge trap is also worth documenting — workers may try to exploit the
`['input','output']` accounting exemption. It does not work because `sanitize_model` renames
the formal graph input to `safe_name_N`, which is then treated as a billable intermediate.

## What's missing from the library

**§137 (techniques.md)** should add a prerequisite note: "#137 applies ONLY when there is a
pre-existing U8 intermediate. It cannot be applied to a Conv that directly reads the F32 graph
input — QLinearConv rejects F32 input at the type-check level, and all three paths to U8 input
(direct cast, Cast-to-U8 intermediate, named-input bridge) either cost more or fail."

**traps.md** (or traps_appendix.md): add a trap row: "Named-input bridge does not make a
Cast(F32→U8) free — sanitize_model renames the F32 graph formal input from 'input' to
'safe_name_N', which is then billable (task192 v14: 39,395; task321 v7 probe: 36,248). The
['input','output'] exemption only saves the Cast output node, not the renamed original input."

## Sorter note: techniques.md §137 — extend existing with application prerequisite + named-input bridge failure
## Sorter note: traps.md — add trap row for named-input bridge accounting trap (with task192 v14 + task321 v7 probe as evidence)
