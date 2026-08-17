# DEAD ROUTE — task380 pure no-payload transpose

- **Live bar:** cost **66**, profile **66** (static=66, memory 0, params 66), owner Vamshi, manifest
  2026-07-02. LIVE sha256
  `f68be2221a8a3688bd779f2ff146190449943c226439072b19829fa3369b7cf4`.
- **Worker:** `bo_ng_cdx2_worker1`, task380.
- **Probe artifact:** `../prototypes/task380_pure_transpose_probe.onnx`, sha256
  `02089aa35f8cf34315cce8693a741ca591bb601e8b985a91bd231a07f73b9c28`, static cost 0.

## Attempted Route

Tested the remaining low-confidence "implicit coordinate / no initializer payload" escape as a single
`Transpose(input -> output)` graph:

```text
output[n,c,i,j] = input[n,c,j,i]
```

This checks whether task380 can be treated as a no-payload spatial transpose instead of a transpose plus
top-left 3-column reversal.

## Failed Gate

```text
python tools/ng_gate.py visible --task 380 --cand The_400_Tasks/Task_by_Task/task380/prototypes/task380_pure_transpose_probe.onnx
[visible] 37/267 (13.86%)  FAIL
```

## Why Dead

Task380 requires `output[n,c,i,j] = input[n,c,j,2-i]` on the top-left 3x3 patch, with zero tail. A pure
shape op can swap row/column axes but cannot express the `2-i` reversal or the 3x3-only zero tail without
adding a charged payload/operator family. Per the visible-fail axiom, do not run truth/AB/battery on this
family.

## One Remaining Hypothesis

Only a future ORT-supported single output op with implicit subrange reversal/rotation and zero-tail semantics
could undercut the current 66-byte fused `Einsum`; no such legal op is present in the current scorer surface.
