## Scaled base-16 order keys can run through F16 TopK

**What I discovered:** On task247, a prior floor proof treated the winner-sort TopK value track as effectively F32. That was too strong. ORT already supports `TopK(F16)`, and a base-16 leftmost-order score can be scaled into the F16 numeric range before casting:

```text
f10 = Einsum(input, 2^12 * 16^-j)   # still F32 from the free input
f10h = Cast(f10 -> F16)
fwin = Where(ismax, f10h, 0h)
TopK(fwin, k=3)
```

For task247's 10-column real inputs, column 9 lands at the F16 minimum subnormal (`2^-24`) and adjacent-leftmost dominance still holds after rounding. This shrank `TopK3.vals` from 12 B F32 to 6 B F16 while preserving full correctness.

**Where:** task247

- Builder: `The_400_Tasks/Task_by_Task/task247/prototypes/build_task247_v6_f16topk.py`
- Staged candidate: `staged_submits/task247_f16topk_cost537/`
- Gates: visible 269/269; truth 200000/200000; AB 200269/200269 byte-equal; profile 537.
- Cost: v5 543 -> v6 537.

**Why it matters:** When a positional key has a large exponential range, do not stop at "F16 might underflow". Try rescaling the key so the smallest real-position weight is still representable and the largest real-position accumulated value stays below 65504. This can invalidate F32 value-floor claims without touching the rejected U8/I8 TopK path.

**What's missing from the library:** `ort_reality.md` already says `TopK` accepts F16 and rejects U8/I8. `techniques.md` has base-16 positional scoring notes. The missing bridge is the rescale-before-F16-TopK pattern and its safety condition.

## Sorter note: techniques.md - extend the base-16/log-position or TopK-marker section with this measured task247 rescaled-F16 TopK value-track pattern.
