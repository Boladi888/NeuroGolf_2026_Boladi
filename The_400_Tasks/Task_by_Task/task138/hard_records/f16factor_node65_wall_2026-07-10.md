# task138 — WALL (gated): reviewer's F16-factor of node 65 REGRESSES

**Recorded:** 2026-07-10 by `bo_ng_cld_worker1` (Claude lane), task138 only. Executed the reviewer's ONE
locked crack (F16 two-stage factor of node 65). Result: mechanically valid, but a **+5644 B regression**.
This is the reviewer's WALL condition #1 ("combined residue coefficient materializes a LARGE intermediate"),
now **gated + named + priced**.

## Live bar
`ng_current_bar --task 138` → manifest **4311**, Sho, sha `72ffa3463b99f3a41b11a2d1ee431771769274fa99ec94a416ddfc022b3520a9`.
Static score_v2 = memory 2814 + params 1478 = **4292**; real-input profile = 2830 + 1481 = **4311** (both bases agree the candidate regresses hugely).

## The lever, and why it cannot pay (tensor-network topology)
Node 65 is a 56-operand fused free-output Einsum. The reviewer's plan: pre-combine the residue-space
coefficient operands {Ac, Bc, VT, selR1/R2/C1/C2, aLr, aHr} (676 B of F32 intermediates) in a SEPARATE
F16 Einsum → small combined coefficient → cast F32 → feed the final expansion. Save ~338 B.

**Symbolic index accounting (`scratchpad/node65_wall.py`) falsifies the "small coefficient" premise:**
the coefficient operands are **independent spokes into different sub-blocks**, not a contractible cluster.
Each carries only *external* labels (shared with its own expansion partner, not with the other
coefficients):
- `selR1..selC2` → each into its own `CHC` block (labels a/d/g/h); `internal=∅`.
- `aLr` → `SHA` (label W); `aHr` → `SHB` (label S); `VT` → channel bases `rxHI10/rxLO10` (label k).
- `Ac[TyZ]` → `rxLO30` (T,y); `Bc[zYZ]` → `rxHI30` (z,Y). Only shared internal index in the whole group: **`Z`** (Ac↔Bc).

Contracting the full 9-operand group over its one internal index `Z` leaves the outer product of all 12
external axes = **8,640,000 elements = 17,280,000 B (F16)** — ~51,000× the intended 338 B saving.

Even the **only** pair with a genuine shared index, `Ac ⊗ Bc` over `Z(2)`, **EXPANDS**:
`Ac[6,6,2] (72) + Bc[5,5,2] (50) = 122 elems  →  Pcoef[6,6,5,5] = 900 elems` (÷ the size-2 contracted axis,
× the four free axes). The "combined coefficient" is 7× **larger** than the operands it replaces.

## The ONE build (best case) — gated regression certificate
`prototypes/build_task138_f16factor_acbc.py` → `task138_f16factor_acbc.onnx`: factors the single
best-case pair (Ac,Bc → F16 → `Pcoef_h = Einsum('TyZ,zYZ->TyzY')` → Cast F32 `Pcoef`), reduces node 65
to 55 operands with `Pcoef` replacing Ac/Bc. `Z` is provably internal to {Ac,Bc} (appears nowhere else),
so the split is exact.
- **visible 266/266 (100%) PASS** — the two-stage factor is algebraically correct; ORT accepts it at
  opset-17; output decode-identical. (So the *mechanism* works — it just doesn't pay.)
- **profile_real_cost --arcgen 266: static=9955 profile=9955 = memory 8474 + params 1481** — **+5644 B
  (+131%) vs 4311.** Not a WIN; a large regression. No `truth/ab/battery` run (regressing candidate — a
  `--full` on a losing graph is pure burn per gate hygiene). No `ng_safety` (not a SHIP candidate).

**Named charged-structure delta** (`ng_graph_cost_diff`, live→candidate):
```
              memory  params    cost
live            2814    1478     4292
candidate       8458    1478     9936
delta          +5644      +0    +5644
  +3,600 B  Pcoef   [f32 6,6,5,5]  newly materialised   <- the "combined coefficient" (900 elems, F32 to feed input-fused Einsum)
  +1,800 B  Pcoef_h [f16 6,6,5,5]  newly materialised
  +  144 B  Ac_h    [f16 6,6,2]    newly materialised
  +  100 B  Bc_h    [f16 5,5,2]    newly materialised
```

## Root cause (Onnxadv #89 cuts BOTH ways)
"Multi-operand Einsum temps are in-kernel/free; operand order = contraction path." The fused node 65 keeps
every coefficient/expansion temp *inside the kernel* along ORT's optimal path — that is exactly why the
live memory is only 2830 B with a 56-operand contraction. **Splitting a sub-contraction out to get an F16
Einsum FORCES those free in-kernel temps to become CHARGED materialized tensors.** The input-fusion "lock"
is therefore not breakable by factoring: any split that isolates an F16 sub-Einsum materialises a tensor
(900 elems best case → 8.64 M elems full group) that the fused kernel never formed. The F16 dtype halves
that materialised tensor but cannot make it smaller than the free temp it replaced.

## Verdict
**WALL at 4311 — the F16-factor lever is falsified by a real gate.** visible 266/266 (correct) yet
profile 9955 (+5644 B). Every grouping regresses: full 9-op = 17.28 MB intermediate; best pair Ac_Bc =
+5644 B gated; spokes (sel/aLr/aHr/VT) share no internal index (no contraction at all). Per the reviewer's
STOP CONDITION this priced result is a real floor. **Do NOT grind a third route** — the next reviewer pass
should DROP. Cheapest correct formulation remains the live **Sho 4311** (unchanged).
