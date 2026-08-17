# task014 — count-mask bbox shortcut REJECT_LOCAL

## Live bar
- Authoritative bar: **1306** from `python tools/ng_current_bar.py --task 14 --json`, source `manifest`, owner Sho, updated `2026-07-10T09:55:00Z`.
- Parent: `Team_Best_Onnx/task014.onnx`
- Parent sha256: `2b7ec8991a744169495b1e10dfe6f5de29a5309917d082728695146a3cccdedb`

## Attempted route
`prototypes/task014_count_mask.onnx` replaces the final Einsum's `mr=(iota30<=t1)` and `mc=(iota30<=l1)` masks with the existing positive row/column count vectors `rows` and `cols`. This deletes `t1`, `l1`, `mrb`, `mcb`, `mr`, `mc`, and `iota30`.

## Exact failed gate
- `ng_gate visible --task 14 --cand prototypes/task014_count_mask.onnx`: **266/266 PASS**
- `profile_real_cost --task 14 --cand prototypes/task014_count_mask.onnx --arcgen 266`: **profile=960** (`memory 604 + params 356`)
- `ng_gate ab --task 14 --cand prototypes/task014_count_mask.onnx --ref Team_Best_Onnx/task014.onnx --n 2000`: **2266/2266**, divergences 0, `byte_equal=True`
- `ng_gate ab --task 14 --cand prototypes/task014_count_mask.onnx --ref Team_Best_Onnx/task014.onnx --full --no-watch-queue`: **200257/200266**, divergences **9**, `byte_equal=False`, `first_div=16719`

## Why dead
The shortcut assumes the selected rare color touches every row and every column inside its bbox. That is true on the 266 visible pairs, but false in the full cached ARC-GEN corpus:

- `row_miss`: 8 / 200000
- `col_miss`: 7 / 200000
- `any_miss`: 9 / 200000
- first corpus miss: index 16453, matching A/B `first_div=16719` after the 266 official examples.

One-axis shortcuts are also dead: row-only would still miss 8 corpus cases; col-only would still miss 7 corpus cases. Do not stage or full-truth this family unless the interval mask is restored, which gives back the current parent topology.

## Remaining hypothesis
A different exact interval-fill mask that is cheaper than `ArgMax(select_last) + LessOrEqual(iota30, end) + Cast(F32)` could still shave the parent. Prefix/suffix/CumSum-style fills price worse on this graph unless they avoid materializing an extra `[30]` vector per axis.
