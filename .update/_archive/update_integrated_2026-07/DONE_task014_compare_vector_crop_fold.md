# Transferable insight — fold the color-map into the final `Equal` compare-vector to drop a redundant crop copy; and the static-cost-diff blind-spot for dynamic crop tensors

**Source**: task014 batch 54 (`bo_ng_cld_worker3`, 2026-07-08). LIVE 2693 (Boladi v18, sha `dcadc581…d9664`) → **staged v19 profile 2558 (−135 B)**, byte-identical to v18 on all 200266 corpus cases.

**Task**: rarest non-bg color → crop input to its bbox, keep interior (with bg holes). One-hot `[1,10,30,30]` BOOL output. Single-channel crop is rule-exact for this generator (monochrome bbox).

## Lever 1 — compare-vector color-fold (the −135 B win, reusable)
A single-color-index crop→canvas→`Equal(channel_ids, canvas)` tail materializes **two** `[1,1,h,w]` crop copies: `rare_crop=Greater(cropF32,0)` (BOOL) + `color_crop=Where(rare_crop, tgt, 0)` (U8). Collapse to **one** copy by moving the color into a data-dependent compare vector:
- `crop_u8 = Cast(tgt_crop_f, u8)`  → raw `{0,1}` mask (one copy)
- `mask_grid = Pad(crop_u8, pads, const=2)`  → canvas `{0=bg-in-crop, 1=rare, 2=outside}`
- `compare[1,10,1,1] = Where(onehot(tgt)_bool, 1_u8, base)` where `base=[0,3,3,…]` (u8 init) → `{0@ch0, 1@tgt, 3 else}`
- `output = Equal(compare, mask_grid)`: ch0 lights on mask==0, tgt on mask==1, pad(2)/others match nothing.

Removes `rare_crop`(168 runtime) + `color_crop`(168) → `crop_u8`(168) + ~30 B compare chain. Byte-exact reproduction of the one-hot. **Requires `tgt != bg`** (guaranteed by the bg_penalty ArgMin) so `compare[0]=0` is always the bg channel.
- **Opset trap**: u8 `Mul`/`Sub` need opset ≥14 (ort_reality §41). At opset 12 build `compare` with **`Where`** (type-agnostic selection), not arithmetic.

**Applies to**: any "recolor / mask → one-hot canvas → `Equal`" tail where the pre-canvas tensor is duplicated once as a mask and once as a color-index. Fold the color into a `[1,C,1,1]` compare-vector and pad the raw mask.

## Lever 2 — `ng_graph_cost_diff` (static) is BLIND to dynamic crop tensors; trust `profile_real_cost`
For dynamic-shape graphs the crop/slice intermediates are declared `[1,1,1,1]` in value_info but run at `[1,1,h,w]`. `ng_graph_cost_diff` (and any static scorer) sees ~1 B; the gate-5 profiler charges the **runtime** max (here 168 B for a mask, 672 B for the F32 crop). So the diff tool reported v19 as a static **+31 "regression"** while the real credited cost fell **−135 B**. The team credits the **profile** (v18 is credited at profile 2693, not its static 7085). **Always compare candidates by `profile_real_cost --arcgen K`, never by the static diff, whenever a Slice/Pad/dynamic tensor is present.**

## Lever 3 — coordinate-rank test as a fast "is-this-foldable" probe (kills PositionFormulaFold classes)
Before attempting to fold a 2D crop/region output into marginal scalars + a coordinate formula, measure the **matrix rank of the interior mask** over the corpus. task014's rare-color interior: mean rank **4.8**, 91.7% rank≥3, 262/266 have bg holes → high-rank 2D data, NOT a separable `rowmask⊗colmask` outer product. A pure-formula solid-rectangle output prices ~1062 static (near the ~700 floor estimate) but **visible-fails 262/266** (passes only the 4 rank-1 solid cases). **rank>1 common ⇒ a spatial carrier (crop tensor + 30×30 canvas) is forced ⇒ any "avoid the 2D plane" floor estimate is wrong.** Cheap probe, decisive kill (see `Task_by_Task/task014/hard_records/formula_fold_rank_wall.md`). Same family as #109/#110 rank gates and #132.
