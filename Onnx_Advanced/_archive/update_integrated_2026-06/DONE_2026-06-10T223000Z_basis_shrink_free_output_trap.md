# Basis-shrink + Pad is NOT cheaper than full-width basis (free-output trap)

## What I discovered

Task059 direction (prompt): "shrink [5,30] bases to [5,11] — the output pre-Pad is 11×11, not 30×30,
so the basis needs only 11 columns; Pad-to-30 is free."

**This is WRONG.** The "Pad-to-30 is free" claim misunderstands how free-output Einsum scoring works.

In the free-output Einsum approach:
- The ONNX graph output tensor named `'output'` is exempt from memory scoring.
- ANY OTHER tensor — including the Einsum output that feeds the Pad node — is a CHARGED intermediate.

When you add a Pad node after the Einsum:
- The Einsum output (pre-Pad, [1,10,11,11] F16) becomes a CHARGED intermediate = 2420 bytes.
- The Pad node's input (Slice output, [1,10,11,11] F32) is ALSO charged = 4840 bytes.
- Net extra memory = 7260 bytes.
- Params saved = ~169 (from [5,11] vs [5,30] basis, 11 columns vs 30).
- Net cost: 861 + 7260 - 169 = 7952 — much WORSE.

The CORRECT insight (#132, techniques.md): "make the basis map straight to the full 30-axis so the
einsum output IS the 30×30 canvas." The [5,30] free-output approach is OPTIMAL because:
- The Einsum output is `'output'` = FREE
- The [5,30] basis encodes zero-padding natively (columns 11-29 are all zero)
- No Pad node is needed, so no charged pre-Pad intermediate exists

Empirically verified (2026-06-10): built task059_basis11_shrunk_v2.onnx, scored cost=7952.

## The misunderstanding in one line

"Pad-to-30 is free" means "Pad PARAMS are cheap" (the pad_vals initializer = 8 elements).
It does NOT mean "Pad creates no memory cost." The Einsum output feeding Pad is a charged
intermediate if the final graph output is named 'output' and comes AFTER the Pad.

## Where encountered

- Task: task059
- Build: `tmp/task059_basis11_shrunk_v2.onnx` (test-only, not staged)
- Prompted by: `The_400_Tasks/Task Prompts/4/Prompt_2026-06-10T133000Z_Task059_BasisShrink.md`
- Current best: `staged_submits/task059_einsum_basis5_cost861/`

## Why it matters

This trap can misdirect a worker for hours. The phrasing "Pad-to-30 is free" is ambiguous and
appears in task prompts. A future agent might see a similar prompt and attempt basis-shrink.

The trap is a variant of the FREE-OUTPUT SEPARABILITY TRAP documented in WORKER_PROTOCOL.md §87:
"splitting a free-output conv materializes a charged intermediate." Same principle: splitting the
free-output Einsum (by inserting a Pad after) materializes a charged intermediate.

## What's missing from the library

Techniques.md #132 hints at this but doesn't explicitly forbid basis-shrink + Pad:
"make the basis map straight to the full 30-axis" is the positive formulation.

The negative formulation should be added: "Shrinking the basis to M < 30 and adding a Pad-to-30
node makes the Einsum output (pre-Pad) a CHARGED intermediate. The full-width [K,30] basis where
the Einsum output IS the graph output is always at least as cheap."

## Sorter note: techniques.md §132 — extend existing with "Basis-shrink + Pad is always worse" addendum

Add to the end of §132:

**ANTI-PATTERN: Basis-shrink + Pad**: shrinking the basis from [K,30] to [K,M] (M < 30) and adding
a Pad-to-30 node LOSES cost because (a) a Slice input [1,C,M,M] intermediate is created = C·M·M·4
bytes (F32), and (b) the Einsum output [1,C,M,M] F16 intermediate is created = C·M·M·2 bytes — both
CHARGED. The full-width basis encodes the zero-padding natively (columns M+1..29 are zero) so no
Pad or Slice is needed. task059 empirical proof: [5,30]→[5,11] + Pad = cost 7952 vs cost 861 for
[5,30] free-output (9.2× worse). The "Pad-to-30 is free" phrasing in task prompts refers only to
the Pad initializer params (~8 elements), NOT to memory costs.
