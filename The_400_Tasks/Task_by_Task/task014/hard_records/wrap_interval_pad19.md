# task014 — opset19 Pad(mode="wrap") interval-mask hypothesis REJECT_LOCAL

## Live bar
- Authoritative bar: **1306** from `python tools/ng_current_bar.py --task 14 --json`.
- Source: team manifest, owner Sho, updated `2026-07-10T09:55:00Z`.
- Parent: `Team_Best_Onnx/task014.onnx`
- Parent sha256: `2b7ec8991a744169495b1e10dfe6f5de29a5309917d082728695146a3cccdedb`

## Locked reviewer hypothesis
Replace only the parent row/column interval-fill masks with an exact opset-19 `Pad(mode="wrap")` endpoint construction. Keep the rest of the live parent graph unchanged. No count-mask shortcut, no Tile/Concat-chain wrap emulation, no Range/modulo tables.

## Prototype 1: dynamic Pad-wrap interval
- Builder: `prototypes/build_task014_wrap_interval_probe.py`
- Candidate: `prototypes/task014_wrap_interval_probe.onnx`
- Method: for each endpoint `e`, `Pad(mode="wrap")(ones30, [0, e-29])` crops to length `e+1`, then zero `Pad([0, 29-e])` restores the prefix mask to length 30.
- Gate: `ng_gate visible --task 14 --cand ... --no-watch-queue` -> **266/266 PASS**
- Failure: `profile_real_cost --arcgen 266` -> **FAIL: score_network returned None on real-input trace**
- Shape proof: ONNX shape inference leaves `r_prefix_dyn`, `mr_wrap`, `c_prefix_dyn`, `mc_wrap` as symbolic `unk__*`, not fixed `[30]`.

## Prototype 2: dynamic Pad-wrap interval with fixed value_info
- Builder: `prototypes/build_task014_wrap_interval_static_vi_probe.py`
- Candidate: `prototypes/task014_wrap_interval_static_vi_probe.onnx`
- Method: same computation as prototype 1, but declares `r_prefix_dyn`, `mr_wrap`, `c_prefix_dyn`, `mc_wrap` as F32 `[30]`.
- Gate: `ng_gate visible --task 14 --cand ... --no-watch-queue` -> **266/266 PASS**
- Score/profile: `static=1647 profile=1647 (memory 1228 + params 419; 266 real inputs)` -> **LOSES** to live 1306 by **+341**.
- Runtime warning during profile: repeated ORT `Shape mismatch attempting to re-use buffer. {30} != {5..25}` warnings, caused by the first Pad's true dynamic crop length despite the fixed value_info.

## Named charged-structure delta
`ng_graph_cost_diff` on prototype 2 reports static regression **+308** vs the local static parent view:
- Added charged `[30]` F32 tensors: `r_prefix_dyn`, `mr_wrap`, `c_prefix_dyn`, `mc_wrap` = +480 B.
- Removed parent `mr`, `mc`, `mrb`, `mcb` = -300 B.
- Added pad scaffolding tensors and params (`*_pads`, `*_after_v`, `ones30_f`, `axes0_i64`, `zero_i64_v`, `c29_i64`) = net positive.

## Verdict
The exact Pad-wrap endpoint route is not a sub-1306 candidate. Without fixed value_info it is unscoreable; with fixed value_info it profiles at 1647. Per reviewer instruction, no `ab --n 17000` or full gates were run after the profile loss.

## Remaining hypothesis
Only a Pad-wrap construction with fixed-shape intermediates and no separate dynamic cropped prefix tensor could reopen this. Plain `Pad(mode="wrap")` does not provide that for variable-length interval fill; it transports/crops shape, and the scorer charges or rejects that shape.
