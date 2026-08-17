# hard_record: task220 QLinearTinyCarrier blocker at 437

- **Date:** 2026-07-14
- **Worker:** bo_ng_worker1
- **Live bar used:** cost **437**, sha256 `b6dfd2919336138dfcae4c4b65ffbbe2d9b37cc9ed31722ba4f335a6c28772d9`, owner Vamshi, updated `2026-07-12T15:15:54Z`, source = `python tools/ng_current_bar.py --task 220 --json` / team manifest.

## Attempted Route
Probe the prompt's QLinear/U8 tiny carrier idea against the current parent.

## Failed Gate
**Pre-build ORT dtype/price gate:** current parent inspection shows no Conv/F32 carrier plane to narrow. It is a scalar polynomial/Einsum graph (`Einsum` x10, `Mul` x3, `Add` x3, `Concat` x3) with real profile **132 memory + 305 params = 437**. `QLinearConv` cannot consume the NeuroGolf F32 graph input directly; the required full-frame `Cast(input -> U8)` would materialize a `[1,10,30,30]` U8 tensor (**9000 B**) before any QLinearConv work, already >20x the live cost. This is the traps §8g / #137 prerequisite blocker, not an ORT tuning issue.

## Why Dead
The current best has only tiny F32 scalar/vector intermediates (`row*`, `col*`, `feat*`), not a full-resolution threshold-only carrier. Adding a U8 bridge would erase the dtype win before visible/truth gates matter.

## Remaining Hypothesis
The remaining floor is polynomial-specific: remove or factor scalar features/selector params inside the single terminal Einsum. The staged `task220_shifted_absence_scalar_cost407` candidate follows that route by dropping the explicit presence component.
