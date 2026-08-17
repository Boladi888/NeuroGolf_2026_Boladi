# DEAD ROUTE — class-sum / full30 same-class Einsum (task197)

- **Live bar when tested:** 277 (manifest-authoritative, owner Sho, SHA256
  `4f8eb859bab9ded1783047fd8fa0dc8ae202b4fe0aa3afe9a9e53f9124f45132`, updated
  2026-07-09T01:09:17Z). Live profile == static == 277.
- **Worker:** `bo_ng_worker1`, task197, 2026-07-14/15 session.

## Attempted route
Avoid the first-different-column scalar and final `[30]` INT32 Gather index by
using all source columns in the same pattern class:

`output[b,c,h,j] = sum_s input[b,c,h,s] * Equal(pattern_id[s], pattern_id[j])`

Fresh ARC-GEN numpy check passed (`class_sum_numpy fresh_arcgen 1000/1000 ok`),
because every seed row exposes at least one source cell from each pattern class
before later cells go blank.

Prototype:
`The_400_Tasks/Task_by_Task/task197/prototypes/build_task197_classsum_full30_v1.py`
→ `task197_classsum_full30_v1.onnx`.

## Exact failed gate

`python tools/ng_gate.py visible --task 197 --cand The_400_Tasks/Task_by_Task/task197/prototypes/task197_classsum_full30_v1.onnx`

Result: `[visible] 0/267 (0.00%)  FAIL`

No truth/AB/battery gates were run after the visible miss.

## Price evidence

`python -c "from tools._score_v2 import score_network_v2; print(score_network_v2('The_400_Tasks/Task_by_Task/task197/prototypes/task197_classsum_full30_v1.onnx'))"`

Result: `(0, 4740, 10)` ⇒ static cost **4750**, far above the 277 bar.

Dominant charged tensors:

| tensor | shape | dtype | bytes |
|---|---:|---:|---:|
| `same30_f` | `[1,1,30,30]` | F32 | 3600 |
| `same30` | `[1,1,30,30]` | BOOL | 900 |
| `pat30` | `[1,1,1,30]` | F32 | 120 |
| `pat30_col` | `[1,1,30,1]` | F32 | 120 |

## Why it is dead

The route replaces the 120-byte legal Gather index with a full dynamic
same-class carrier. Keeping the final Einsum as `output` avoids a charged output
plane, but the F32 `[30,30]` selector alone costs 3600 bytes. A normalized/binary
tail would require `Greater`/`Clip` after the Einsum, which would charge the full
`[1,10,30,30]` F32 Einsum result. Cropping to width 10 forces either a charged
`[1,10,30,10]` source crop or 30x10 source/target basis initializers, both above
the current 277-byte Gather-index family.

## One remaining hypothesis

Only revisit this route if a legal ONNX op can threshold or normalize an Einsum
inside the final output kernel, or if a sub-120-byte data-dependent axis-3
selection primitive appears. Otherwise the current `[30]` INT32 Gather index
remains strictly cheaper.
