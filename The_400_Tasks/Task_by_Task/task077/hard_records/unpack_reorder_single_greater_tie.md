# task077 hard record: U8-lane unpack reorder tie

- live bar: 6,524, SHA `BF353D101BCFD46F5DFE1A13ED27C3137220097110714E52BE0E76B438F07E1D` from `python tools/ng_current_bar.py --task 77 --json` (`manifest`, 2026-06-24).
- attempted route: `prototypes/task077_probe_unpack_single_greater.onnx`, built by `prototypes/build_task077_probe_unpack_single_greater.py`; reorder U8-lane exit from `3x Greater -> Concat BOOL` to `Concat U8 -> 1x Greater`.
- exact failed gate: cost/profile gate after visible pass. `ng_gate visible` = `[visible] 266/266 (100.00%)  PASS`; `profile_real_cost.py --arcgen 266` = `static=6524 profile=6524 (memory 6424 + params 100; 266 real inputs)`.
- why dead: not strictly below the 6,524 bar. The graph-cost diff is a zero-sum charged swap: adds `cat_u8 [1,20,21] UINT8` 420 B and removes lane BOOL tensors `lo_b/mid_b/hi_b` totaling 420 B; net cost delta 0. Do not stage or run full truth/AB for this family.
- one remaining hypothesis: only a structurally different detector that deletes part of the U32 reach core or avoids the locked 900 B full-canvas Where condition can beat 6,524; further U8-lane exit orderings are exhausted unless they remove a materialized `[1,20,21]` plane entirely.
