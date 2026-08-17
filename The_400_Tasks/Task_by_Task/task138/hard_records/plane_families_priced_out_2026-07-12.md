# task138 — every genuinely-different (plane-based) formulation is priced strictly worse than the free-output win (3983)

**Recorded:** 2026-07-12 by `bo_ng_cld_worker2` (Claude lane), forced 2nd pass. Companion to
`letter_budget_wall_2026-07-12.md`. Purpose: kill the "just render an index/fill plane" instinct with
measured prices so the next batch does not re-explore it.

## The staged win to beat
`staged_candidates/task138_foldAcBc_cost3983/` — **profile==static 3983** (mem 2502 + params 1481),
byte-equal to live Sho 4311 (`ab` 566/566, ng_safety PASS). A **free-output** separable Einsum: the whole
`[1,10,30,30]` one-hot output is the graph output (FREE); it materialises **no 2-D plane**.

## Why any plane formulation loses: it ADDS a ≥900 B charged plane on top of the SAME cost
The output is multi-color (4 frame borders in their own colors + interior ray-fill in color M + markers) —
NOT a single mask. Producing it needs (a) an F32 detection island (~1000 B, forced — it contracts the F32
input), (b) the ray-shape rendering (SHA/SHB/CLC/CHC ≈ 926 params + F32 operand intermediates), and (c) the
one-hot expansion. The free-output Einsum pays (a)+(b) and gets (c) FREE. A plane formulation pays (a)+(b)
too **and** materialises the plane for (c) — strictly worse. Measured prices of the plane tail alone:

| genuinely-different formulation | forced tensor(s) | price | verdict |
|---|---|---:|---|
| **index-plane → `OneHot`** | `idxf` F32[1,30,30] 3600 **+ `idx64` INT64[1,30,30] 7200** | **10813** (probe) | DEAD — ORT `OneHot` accepts **int64 indices ONLY** (uint8/int8/int32 = NOT_IMPLEMENTED, measured this session); the idx plane feeding OneHot is forced to INT64 7200 |
| index-plane → `Equal(idx,arange)`→`Cast` | BOOL `[1,10,30,30]` 9000 | 9000+ | DEAD — the pre-cast one-hot is a full 9000 B intermediate |
| CumSum directional scan (prior probe) | 4 scan planes F16 | 7200 | DEAD (prior `below_4311_freeoutput_wall` pass) |
| bit-pack + prefix/suffix LUT-Gather (#110) | "paint30 BOOL 900 B canvas" (structurally forced, per #110) + border/crop machinery | >4023 | LOSES — even the ray mask alone (900 B) + the multi-color border render exceeds the plane-free 3983 |
| U8 color-index plane (best case, no OneHot) | `idx` U8[1,30,30] 900 + border/ray render planes | >4635 est | LOSES — the 900 B plane already exceeds the ~328 B of channel machinery it would replace; and the ray render still needs the same shape tensors |

Probe: `prototypes/build_task138_indexplane_onehot_floorprobe.py` → `task138_indexplane_onehot_floorprobe.onnx`
(static memory 10800 + params 13 = **10813**, JUST the collapse+OneHot tail, before any detection/borders/rays).

## Forced-component floor argument for < 3983
`cost = Σ(intermediate bytes) + params`; the `[1,10,30,30]` output is free ONLY as the graph output.
1. **Detection F32 island ≈ 1000 B** (`rcnt,ccnt,cntin,scr,scc,inRf,inCf,...`) — forced F32 (contracts the
   F32 graph input; an F16 input copy = 18,000 B). Formulation-independent.
2. **Ray-shape params ≈ 926** (`SHA/SHB/CLC/CHC`) — the ray fill is a per-column/row INTERVAL (threshold
   matrix, rank up to 30, non-separable); rendering it free-output needs the CRT shape tensors. Rank-5/3
   factorable but **letter-budget-blocked** (see companion record).
3. **F32-locked node-65 operands** (`inRT,inCT,mR,mC,selectors,QK`) — forced F32 because node 65 shares the
   F32 `input` (input-fusion); can't narrow to F16 without a 18,000 B input copy.
4. **Output expansion is free ONLY via the final free-output Einsum** (or a final OneHot, which forces INT64).

The free-output Einsum hits this floor with **no plane**; the 3983 win = the forced floor minus the two
foldable slices (`Ac,Bc`) that the 52-letter budget was hiding. **The only sub-3983 route** is a
letter-footprint-reducing re-derivation of node 65 (merge the 4 per-direction ray blocks → free 2-3 letters
→ fold the SHB/CLC/CHC rank factors, est. ~3700) — an algorithmic rewrite that loses byte-equality and needs
200k-truth (infeasible in the ~1.2 s/input foreground window). **WALL at 3983 for verifiable formulations.**
