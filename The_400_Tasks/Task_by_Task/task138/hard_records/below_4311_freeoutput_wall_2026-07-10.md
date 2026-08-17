# task138 — WALL: strict-below-4311 (Sho free-output family), forced second pass

**Recorded:** 2026-07-10 by `bo_ng_cld_worker1` (Claude lane), task138 only. Forced second pass after
the pass-1 DROP; this pass adds a *priced + rank-proven + cost-model* wall, not just a structural DROP.

## Live bar
`ng_current_bar --task 138` → manifest **4311**, owner **Sho**, sha256
`72ffa3463b99f3a41b11a2d1ee431771769274fa99ec94a416ddfc022b3520a9`, updated 2026-07-09.
Fresh profile: **static==profile==4311 = memory 2830 + params 1481** (266 real inputs, clean).

## What 4311 is (fully decoded this pass)
A **free-output separable Einsum**. The whole `[1,10,30,30]` output is emitted by ONE 57-operand
Einsum (node 65) that contracts F32 `input`'s spatial dims with a **CRT-factored positional code**:
30 = 6 × 5, so positions are encoded by `rxLO30[30,6]` (6 groups of 5) and `rxHI30[30,5]` (5 groups of
6), combined by 3-way shape couplings `SHA[6,6,6,2]`, `SHB[5,5,5,2]` (marker×frame×output position in
each residue system) and `CLC/CHC`. Detection: `rcnt/ccnt→ArgMax` (frame), `cntin→ArgMax→OneHot sOH`
(single marker color), border `Gather+Greater` (ray side). **Zero 2D-spatial intermediates** (largest
intermediate `Ac[6,6,2]`=288 B; all spatial work on 1D profiles).

## The two cost levers, and why both are dead

### Lever 1 — reduce params 1481 (the render basis). DEAD: proven minimal-rank + dtype-immune.
- **`params` in score_v2 is an ELEMENT COUNT, not bytes** (Sho params=1481; F32×4 would read ~5924).
  Proof: removing `zero_f`(F32[1]) was "−4 B / **−1 param**" and cost fell by **1**, not 4
  (`staged_candidates/…cast_marker_bool` manifest). ⇒ **Casting the basis to F16 saves 0 params.** The
  only way down is fewer ELEMENTS = lower rank.
- **The positional basis is provably full-rank** (`scratchpad/rank_probe.py`, SVD of the fixed inits):
  `rxLO30[30,6]` rank 6/6, uniform σ=√5; `rxHI30[30,5]` rank 5/5, uniform σ=√6; `rxLO10[10,5]` rank 5/5;
  `rxHI10[10,2]` rank 2/2; `SHA` mode-0 rank 6/6; `SHB` mode-0 rank 5/5. The uniform spectra = **exact
  orthogonal partitions** (each column an indicator for a disjoint position group). Dropping any column
  deletes 5–6 positions from the addressable set ⇒ wrong output. **30 = 6×5 is the balanced minimal
  factorization** (max factor 6, shape tensor ∝ 6³·2); 6×6 or 2×3×5 cost more. `rank4_shared_basis_probe.md`
  independently showed the *data* row/col bases need rank ≥5 on 4702/5000 inputs. **No rank slack.**

### Lever 2 — reduce memory 2830 by F16-narrowing intermediates. DEAD: F32-forced by input-fusion.
- Every F32 intermediate is **produced by an Einsum that contracts the F32 graph `input`**
  (`rcnt,ccnt,cntin,scr,scc` — nodes 0,1,18,24,25) **or consumed by node 65 alongside `input`**
  (`Ac,Bc,inRT,inCT,mR,mC,sOHT,VT,aLr,aHr,selectors`). ONNX Einsum requires one dtype for all operands;
  `input` is fixed F32 and a F16 copy of `input[1,10,30,30]` = **18,000 B** ≫ any saving.
- Note: the basis values are all **[0,1] (absmax=1) — F16-SAFE**; the block is *structural* (input
  fusion), NOT the old exp-ramp overflow (`below_5283_freeoutput_narrowing_wall.md`, now superseded).

## Genuinely-different families tried this pass — both priced above 4311
1. **Index-plane** (pass 1, `task138_indexplane_floorprobe.onnx`): forced `[1,30,30]` plane = **3600 B**
   (F32) / 3600 (INT32 to one-hot) — > Sho's whole 2830 memory, before any logic.
2. **CumSum-scan canvas** (this pass, `task138_cumsum_canvas_floorprobe.onnx`): the data-dependent ray
   side (1 of 4) forces **4 static directional scan planes** `sx_f/sx_r/sy_f/sy_r` = 4 × 1800 F16 =
   **7200 B**, + marker plane 1800 + F32 collapse 3600. Forced non-output intermediates **≈18,000 B**,
   ~6× Sho's memory — before frame-detect / interior-mask / direction-select / frame-copy / one-hot.

**Root cause both lose:** the free-output separable Einsum's whole advantage is it **never materializes a
2D spatial plane** — it renders straight to the (free) graph output. Any "spatial logic on a plane"
approach materializes ≥1 (usually ≥4) `[30,30]` planes at ≥1800 B (F16) each, exceeding the entire live
memory budget on the first plane.

## Family floor estimate (why 900 is a false floor)
Absolute floor of the free-output family ≈ **params 1481 (minimal-rank, proven) + forced F32 detection
memory (~1000 B: the input-contracting `rcnt/ccnt/scr/scc/cntin` + argmax/mask chain) ≈ 2500**, even with
a *perfect* F16 render of every node-65 operand. The prompt's **~900 = the class-SWEEP heuristic
(4311 ÷ 4.79)**, not a construction; it is **~3× below the proven family floor and unreachable by any
route examined.** FALSE FLOOR.

## Verdict
**WALL at 4311.** No correct sub-4311 candidate found. Params are minimal-rank (SVD-proven) and
dtype-immune; memory is F32-forced by input-fusion; the two plane-based alternatives price at 3600 B and
7200 B of *forced* intermediates respectively. Cheapest correct formulation = the live **Sho 4311**.

## One remaining hypothesis (real upside, but out of scope + high risk — NOT attempted)
**Factor `input` out of node 65 and render in F16.** `input`'s two appearances each locally contract to a
tiny tensor (e.g. `sum_{n,R,C} input·sOHT·inRT·inCT → [b,t]`); if both are pre-computed as small F16-safe
tensors, the remaining basis-only Einsum could run in F16, halving the ~1476 B of node-65 F32 operand
intermediates → **est. ~3700 (a ~600 B / ~14% win, params unchanged at 1481).** Blocked here because:
(a) it is error-prone hand-surgery on a 57-operand hand-tuned Einsum that must re-pass 200k truth + a raw
nan/inf scan; (b) even perfect, it lands ~3700 ≫ the 900 target and the params floor (1481) dominates;
(c) the 2026-07-10 workflow update de-scopes open-ended re-derivation for this SWEEP/false-floor prompt
("one crack, stop on a priced wall"). A future *dedicated* (non-recapture) pass could attempt it.
