# task382 — floor / wall record (2026-07-12, worker bo_ng_cld_worker1, Claude lane)

## Live bar at time of work
- **live_cost = 2870** (manifest.csv, owner Sho, updated 2026-07-09), SHA `57fba1f3614bf9cd86791f6a74e1f9c8648a8a8ecdf6660248c8b596af2a07f9`.
- Local `Team_Best_Onnx/task382.onnx` == that SHA (the live model). Graph name `task382_v2b`, opset 17, 43 nodes.
- **NOTE the local task doc was STALE**: context_pack/task382.md described a "6275/6362 safe floor, CLOSED-at-safe-floor". That is superseded — Sho blew past it to 2870 (v2b) on 2026-07-09 via a from-scratch single-Einsum free-output formulation. The "sub-5k is a non-byte-equal gamble / DO NOT PUSH" note no longer describes reality.

## What I shipped
- **v2c = 2800 (memory 2456 + params 344), byte-equal to LIVE (ab 2266/2266).** −70 B. Staged at `staged_candidates/task382_v2c_onehot_byteequal_cost2800/`. LOCAL_GOLD (PATH_A_BYTE_EQUAL).
- Change: the two `Equal→Cast` one-hot constructions (index one-hot and Conv-kernel one-hot) → `OneHot` with int32 index. Eliminates materialized BOOL intermediates `indb[1,5,30]=150B` and `Wgb[6,1,9]=54B`; adds int32 index tensors `inc_i32[1,30]=120B` + `posW_i32[6,1]=24B` (OneHot only accepts int32/int64 idx in ORT — F16 idx NOT implemented, tested). Net −60 memory, −10 params.

## Why floor_est ~900 is FANTASY (priced)
The card's ~900 came from an old "[30,30] BOOL output plane = 900 B" heuristic. It does not apply: v2b keeps the [30,30]×10 output as the **uncharged graph output** (final Einsum emits it directly). The charged state is the small edge-ingredient set + machinery. Priced hard floor for THIS (already-optimal) formulation:

| Charged component | bytes | why forced |
|---|---:|---|
| F32 reduction rows (RH2f 2 + RW2f 2 + cynH 1 + cynW 1 = 6 rows × 30 × 4B) | 720 | input is F32 → any op consuming it emits F32; casting input to F16 = 18 000 B [1,10,30,30] intermediate (dead). The 6 rows are semantically DISTINCT: {ch2-marker-h, active-edge-h, ch2-marker-w, active-edge-w, ch8-anchor-h, ch8-anchor-w}. Cannot merge (element count conserved) or drop. |
| Ind16 [1,5,30] F16 shift one-hot | 300 | depth 5 = max displacement 4 (generator: ≤4 red rows, `inc∈{0..4}`) + 1. Proven minimal. |
| Shf6 [1,6,30] F16 Conv taps | 360 | 6 taps = 5 shift magnitudes + 1 constant/bias plane (sentinel tap value 20 → out-of-range shift → 0, +bias6[5]=1). ALL 6 used: B2[:,:,j] norms = [1,1,1,1,1,9] (j=5 used broadly to rebuild non-displaced channels). |
| RH3+RW3 F16 casts (ingredients) | 240 | needed F16 for the F16 final Einsum. |
| cumsum + cyn casts + conv-kernel + flags machinery | ~600 | forward+reverse CumSum, seed, edge slices, kernel one-hot, orientation flags. |
| **priced floor** | **≈ 2400–2800** | |

**Merging never saves** (cost = elements × dtype_size; element count conserved). **Narrowing is blocked**: F32 reductions forced by F32 input; all Einsum operands already F16; BOOL/U8 impossible for arithmetic operands. So there is NO path to ~900 in this formulation, and any other correct formulation must still pay the ~720 B F32 edge-read + ~660 B ingredient floor.

## Routes tried & priced-dead (do not re-explore)
1. **OneHot with F16 index** — ORT `NOT_IMPLEMENTED` (idx must be int32/int64). int32 idx is 4B/elem, so it only nets −30/site vs the BOOL it removes. Captured.
2. **Drop a Shf6 or Ind16 channel byte-equal** — DEAD: all B1 k-slices (norm 10) and all B2 j-slices (incl sentinel j=5, norm 9) active. No dead channel.
3. **Narrower F32 reduction (6→4 rows)** — DEAD: the 6 rows are distinct quantities (2 marker + 2 active + 2 anchor). Removing any loses the clip mask or an orientation.
4. **Input-passthrough additive compose** (`out = input + disp*(e8−e0)`) — DEAD: materializes a [1,10,30,30] correction/sum intermediate (18 000–36 000 B charged). v2b's "reconstruct everything in the final Einsum" is provably better because the only [10,30,30] tensor is the uncharged output.
5. **F32→F16 cast elimination** — DEAD: the F16 casts feed the F16 Einsum; the F32 originals feed nothing the Einsum can consume (mixed dtype not allowed).

## One remaining (unproven, non-byte-equal, HIGH RISK) hypothesis
The 7-mode `s`-basis (B1[2,7,5], B2[2,7,6], BrH[7,2], Ca[7,3] = ~299 params) MIGHT compress if two of the 7 modes are linearly dependent for all reachable inputs → could shave ~27 params/mode. This needs exact re-derivation of the case algebra and is a **non-byte-equal formulation change on a HIDDEN-FRAGILE task (−12.29 LB scar)** — a Kaggle-probe gamble, not a local win. Not pursued; flagged for a future batch only if a Kaggle probe budget exists. Even if it worked it is a ~50–100 B win, not a path to 900.

## PASS-2 (2026-07-12): genuinely-different formulations tried, all priced dead by ORT reality + doctrine

Forced 2nd pass pushed for a different dominant-intermediate assembly (not a micro-tweak). Tested the two levers that could break the F16 ingredient floor:

1. **INT8/UINT8-assembly** (narrow the F16 Einsum ingredients `Ind16` 300→150, `Shf6` 360→180, reduction casts to 1 byte) — **DEAD.** ORT `Einsum` has NO INT8/UINT8/INT16 kernel (empirically: `NOT_IMPLEMENTED`; library `ort_reality.md:55,645` task354, `techniques.md:2892,3811`). **F16 is the hard 2-byte floor for any Einsum operand.** `Conv` INT8/U8 = `INVALID_GRAPH` (Shf6 F16-locked). `CumSum` INT8/16 = invalid (F16/INT32 only). So the whole assembly + shift path is ≥ F16.
2. **Gather-BOOL end-to-end output** (technique #95, task253: sidestep the Einsum-U8 wall with `Gather(oh[C,K], cellmap, axis=2)` → BOOL output) — **DEAD for task382.** That trick needs a `cellmap[h,w]` **Gather index**, which must be INT32/INT64 → a `[30,30]` INT32 plane = **3,600 B**. It only wins when the rearrangement is a FIXED geometric map (task253 corners); task382's displacement is **DATA-DEPENDENT** (per-row `inc[r]`), so the cellmap is a full charged `[30,30]` INT32 plane, far worse than v2b's factored 660 B F16 ingredients. Materializing the `[30,30]` displacement in ANY dtype (900 BOOL / 1800 F16 / 3600 I32) loses to the factored `Ind16`+`Shf6`.

**Doctrinal wall (BoBrain trap §23a, verbatim):** *"task382: −12.29 scar — gets the byte-equal compression of a Kaggle-validated version (v22 ≡ v9 anchor), NOT a sub-floor gamble ... Fragile tasks don't get the [non-byte-equal] bet."* So even if a cheaper non-byte-equal formulation existed, shipping it is FORBIDDEN here — byte-equal only. The only remaining non-byte-equal idea (compress the 7-mode s-basis, ~50 B params) is exactly the forbidden "sub-floor gamble" and is NOT a path to 900 anyway.

**Byte-equal chain PROVEN (4000/4000 each, bool-normalized):** `v2c ≡ v2b(LIVE) ≡ v22 ≡ v9`. So v2c is the byte-equal compression of the **+2.17-LB proven Kaggle anchor (v9)** → strongest §23a safety, LOCAL_GOLD grounded.

## Verdict
**WALL at 2800 (byte-equal) — with a strict −70 B win banked (v2c).** Two passes, ~10 priced-dead routes (INT8-Einsum, Conv-INT8, Gather-BOOL for data-dependent shift, crop20, input-passthrough, narrower-reduction, Gather-shift, dead-channel, s-basis-gamble). The incumbent v2b is minimal in every reducible dimension; the ONLY byte-equal slack was the 2 `Equal→Cast` BOOL one-hots, captured. `floor_est ~900` is unreachable by any formulation: F32 edge-read floor 720 + F16 ingredient floor 660 (Einsum-U8 wall forces F16) ≈ 1,380 B before machinery; 2,800 is essentially the formulation floor for the only class (byte-equal) this fragile task permits.
