# task091 hard record — selector-as-index formulation is at its floor (WALL @ 2486)

- **Live bar when recorded:** 2486, manifest sha `872f4ce27db095cb9a76abfd1aab326e76263d36998daf4f080dca468aa979f1`, owner Boladi (`tools/ng_current_bar.py --task 91`). Same ONNX = `Team_Best_Onnx/task091.onnx` (local score_v2 reports 2485 vs manifest 2486; 1-B reconciliation noise, not a separate rung).
- **Route attempted (2026-07-08 worker1):** prompt "EqualTailSelectorFold" — fold channel selectors into a color-index Equal tail / delete redundant one-hot selector planes, target floor_est ~800.
- **Verdict:** WALL. No strict-below-2486 formulation exists. Prompt's ~800 est is infeasible by ~2.8×.

## Exact gate that kills it (scoring mechanics, not a runtime failure)

`neurogolf_utils.calculate_memory`: memory = Σ over every named node-output tensor (except literal `input`/`output`) of `num_elements × np.dtype(dtype).itemsize`; the graph `output` is free. **Min itemsize = 1 byte.**

Measured (`prototypes/probe_task091_floor.py`):
- dtype itemsizes: `UINT4=1, INT4=1, BOOL=1, UINT8=1, F16=2, F32=4, INT32=4, INT64=8` → **no sub-byte packing possible**.
- **skeleton** (bare selector-as-index core: `Conv→Cast→STATIC Slice→Pad→Equal`, zero bbox/geometry) = **2258** (idx_f 900 + idx_u8 225 + crop_u8 196 + canvas30 900 + params 37). This is the irreducible core BEFORE any rule logic.
- **tench_pad** (delete `canvas30` by Equal-at-14×14 then free Pad→output) = **3317** (the 10-ch BOOL crop is 10·14·14 = 1960 > 900).

## Why both 900-B planes are forced

- `canvas30` (30×30 scalar pre-Equal plane) = 900: the final free `Equal(plane[1,1,30,30], ids[1,10,1,1])→output` broadcasts channel but cannot upsample H/W, so the plane must be full 30×30; 30·30·1 = 900 for ANY dtype (min itemsize 1). The only "delete" route (10-ch Pad) is 1960 > 900.
- `idx_f` (15×15 F32 color-index source) = 900: the 10→1 channel combine (Conv/MatMul/Einsum) outputs F32 → 15·15·4 = 900. SRC=15 is required (a valid 14-wide crop reads source index 14). Alternatives all ≥900: ArgMax→INT64=1800; two per-channel F32 slices=900 each; QLinearConv/ConvInteger need a U8 input = ~9000 Cast.

v16 (2486) is only **228 B** above the 2258 skeleton; that 228 B is the forced dynamic rule (4× ArgMax INT64=32, INT32 arithmetic ~48, dynamic Slice starts/ends INT32=32, dynamic Pad INT64[8]=64, U8 bbox reduces=30) — each at its op-mandated floor.

## One remaining hypothesis (low confidence)

The only region not at a proven hard floor is the ~228 B of dynamic geometry between the 2258 skeleton and 2486. A micro-shave of scalar arithmetic (merge a Sub/Add, share an endpoint) could in principle trim ~8–16 B for a marginal strict win, but: (1) it is NOT the STRUCTURAL 3× rewrite the task wants; (2) it keeps the v10-class dynamic-Slice ORT dummy-shape warning; (3) it requires a full re-gate (throttled truth+ab --full) and risks breaking byte-equal — net-negative EV. Not pursued. If a future batch wants it, start from `build_task091_v16_u8_bbox_reduce.py` and target the 4 ArgMax→INT32 cast chain.

## Do NOT re-explore
- Sub-byte packing (UINT4/INT4) — grader itemsize floor is 1 B.
- Separable Einsum crop (v15, 11797) · GridSample crop (v17/v18, 5979/6146) · 10-ch Pad tail (3317) — all measured > 2486.
- QLinearConv/ConvInteger U8 source — needs ~9000-B input Cast.
