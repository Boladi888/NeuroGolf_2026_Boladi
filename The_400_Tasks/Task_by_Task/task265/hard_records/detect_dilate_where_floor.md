# detect_dilate_where_floor

- live bar: 3637, sha `0c5b087e1708ee1e348809c46c3bc1e5dcb94234fc4c11444875f9696cc4dcf3` (`ng_current_bar.py --task 265 --json`, manifest source). Current staged floor after this session: **v9 = 3349** (`task265_v9_graded_u8_cast_cost3349`, byte-equal, LOCAL_GOLD).
- attempted route: drive task265 well below 3349 by deleting one of the two forced-size planes of the gapfiller formulation (the 1156-B F32 detection score or the 900-B BOOL Where-condition).
- exact failed gate: pre-build cost feasibility (priced by op semantics), plus the two existing hard records (`signed_final_conv.md`, `bool_maxpool_path.md`). No sub-3349 formulation reaches a buildable graph.

## Why the two big planes are FORCED

**score_f32 [1,1,17,17] = 1156 B (detection).** The official input is F32 `[1,10,30,30]`. "All-black 2×2 window" is a 2×2 windowed reduction → a 17×17 map. Every ONNX op that does a windowed reduction (`Conv`, `MaxPool`, `AveragePool`, `LpPool`) returns the **input dtype = F32** → 17×17 F32 = 1156. The only integer-output windowed reducer is `ConvInteger`, which needs a U8/I8 input; casting the F32 input to U8 first costs a single-channel `Slice` (F32 `[1,1,30,30]` = 3600) + `Cast` (900) ≫ 1156. Element-wise compares (`Greater`/`Less`/`Equal`) produce BOOL directly but cannot window (no pooling). ⇒ one 17×17 F32 plane is unavoidable. Confirmed: v5's QLinearConv route (U8 output) needed a charged `gray18` slice+cast and priced **higher** (3791 > 3641).

**red30_bool [1,1,30,30] = 900 B (output overlay).** Output is `[1,10,30,30]`; painted cells are data-dependent and scattered across the top-left 18×18 active grid. `Where(cond, red_onehot, input)` is the cheap assembly (references `input` free), but needs a 30×30 BOOL `cond` = 900. Folding the mask into a final output `Conv` is **LP-infeasible** (`hard_records/signed_final_conv.md`: red/background is not linearly separable even at 5×5/7×7 context). Reconstructing the output per-channel needs expensive F32 `Slice`s of `input`. Data-dependent `ScatterND` into the output needs a 30×30-resolution index and is dynamic (cost fiction). ⇒ a 30×30 1-byte structure is unavoidable; BOOL (1 B/elem) is already minimal.

## Scaffolding is also minimal (2× fix planes, 2× dilate planes)

- Fix (position (8,11), data-dependent legality kill — **mandatory**, unconditional fix fails 6/266 visible): `ScatterND` needs a base + a result plane → 2×289. Doing it in F32 (before the U8 cast) makes the scattered plane 1156 (worse). U8-domain fix (v9) is optimal.
- Dilate+threshold: `MaxPool` rejects BOOL (`hard_records/bool_maxpool_path.md`) so dilation is U8 → `red_u8` 324; `Where` needs BOOL so a `Cast`/`Equal` → `red_bool` 324. Doing either at 30×30 costs 900 (worse). 18×18 is optimal.

## Priced lower bound

1156 (score_f32) + 900 (red30_bool) + 289 + 289 (fix) + 324 + 324 (dilate+threshold) + ~64 params ≈ **3346**. v9 = 3349 (the +3 is three 1-byte scalar tensors for the fix predicate: `anchor_left`, `keep_bool`, `keep_u8`). v9 is essentially on the floor.

## why dead / prompt-model mismatch

The prompt frames task265 as "DepthToSpace block stamp, floor_est ~1200". That estimate assumes a **disjoint 2×2 block-tiling** expansion of a tiny channel-packed motif (DepthToSpace: `[N,C·4,H,W]→[N,C,2H,2W]`). The **actual decoded rule** (generator `a8d7556c`, "gapfiller") is a **stride-1 OVERLAPPING-window morphological dilation** of all-black 2×2 windows, plus a data-dependent (8,11) legality kill, plus an input-overlay output. DepthToSpace expresses none of these (overlap ≠ disjoint tiling; traps also ban SpaceToDepth on U8 and DepthToSpace-via-Concat/Tile). So ~1200 is a wrong-task-model artifact; the real forced floor for this rule is ~3346.

## one remaining hypothesis

The only unpriced lever is the `signed_final_conv` "remaining hypothesis": a two-stage bitset association that computes the paint state in one compact plane AND assembles the 30×30 output without re-introducing an equivalent full 30×30 mask. Every concrete attempt so far reintroduces either a 30×30 BOOL (the Where cond) or a 30×30 index (dynamic → cost fiction). Treat as DEAD unless someone exhibits a buildable, all-static assembly that reads a <900-B structure at output resolution.
