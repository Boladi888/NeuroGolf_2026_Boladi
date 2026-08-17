# task327 — `add_chain_to_sum` (origin)

| Field | Value |
|---|---|
| Task | task327 |
| Pattern name | `add_chain_to_sum` (techniques.md §6) |
| Cost reduction | -5,760 (origin); +0.350 pts |
| Detector | `tools/spread_patterns/pass_add_chain_to_sum.py` |

## Before

```
sum0 = Add(a, b)
sum1 = Add(sum0, c)
sum2 = Add(sum1, d)
sum3 = Add(sum2, e)
result = Add(sum3, f)
```

Five Add nodes producing four intermediate tensors. Each intermediate is the same shape as the inputs.

## After

```
result = Sum(a, b, c, d, e, f)
```

One Sum node, no intermediates. ONNX `Sum` accepts ≥1 inputs with the same broadcast semantics as Add.

## Why it works

Add is associative + commutative. `((((a+b)+c)+d)+e)+f == sum(a,b,c,d,e,f)` exactly. ONNX Sum's output is the broadcasted sum of all inputs.

## Generalizes to

`techniques.md` §6. Mechanically detected. Detector requires:
- A chain of Add nodes where each Add's first input is the prior Add's output (strict left-fold).
- Each prior Add's output is consumed ONLY by the next Add (no other consumer).
- Chain length ≥ 2 (single Add → Sum(2) is a no-op refactor).

## Implementation note: unique node names

The auto-detector had a bug where multiple emitted Sum nodes shared the same name (when source Add chains had unnamed head nodes). Fixed 2026-05-20 by adding a counter (`_uniq("prefix")`) for emitted names. **If you write any pattern that emits multiple nodes per task, always use unique names** — ORT rejects models with duplicate node names at load.

## Detection trap (NeuroGolf-specific)

Tree-shaped Add reductions (`Add(Add(a, b), Add(c, d))` — branching, not left-fold) are NOT handled by the basic detector. Sum could still apply, but the rewrite would change first-parent identity. Out of scope for the current detector.

## Reproducibility

Run `tools/spread_scan.py --tasks taskNNN`. Detector is fully automated. Manual application: trace any sequence of Adds where the SAME tensor flows through as the first input — that's the left-fold chain.

## Cross-ref

- task013 — longest single chain documented (-17,280 bytes); no exemplar written, see `The_400_Tasks/Task_by_Task/task013.md` + techniques #55.
- `task163_add_chain.md` — -4,840 bytes.
- `techniques.md` §6 — the full technique description.
