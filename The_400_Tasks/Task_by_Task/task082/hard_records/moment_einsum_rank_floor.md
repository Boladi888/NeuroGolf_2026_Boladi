# hard_record: task082 - moment/free-output Einsum route is DEAD against the 190 bar

- **Date:** 2026-07-15 (`bo_ng_cdx2_worker2`, task082 only)
- **Live bar used:** 190 from `python tools/ng_current_bar.py --task 82 --json`, authoritative source `manifest`, sha256 `e050837db177e7d6b2e5ad5c4cd76bcf19e85a0aedf3fc265056d75870d80446`, owner Sho, updated `2026-07-09T01:09:17Z`.
- **LIVE graph checked:** `Team_Best_Onnx/task082.onnx` is one grouped `Conv` writing graph `output`, `conv_w[10,1,6,3]` (180 params) + `conv_b[10]` (10 params), group=10, pads `[5,1,0,1]`, memory 0, profile cost 190.

## Attempted route

Prompted route: collapse row-0 seeds into row/column/color moments or a compact marginal vector, then render the 6-row diamond lattice with a final free-output Einsum instead of the grouped Conv.

## Exact failed gate

Failed the prebuild rank/cost screen before any candidate was worth emitting. Probe script:

`python The_400_Tasks/Task_by_Task/task082/prototypes/moment_einsum_rank_probe.py`

Report path:

`The_400_Tasks/Task_by_Task/task082/prototypes/moment_einsum_rank_probe.json`

Key result lines:

- Observed official pairs: 265; all inputs are `6xW`, `W in 5..15`; nonzero input rows are row 0 only; seed gaps are `{3:217, 4:207}`.
- Horizontal placement rank at max active width 15: center identity rank 15; odd left+right rank 14.
- Materialized row-0 profile/marginal skeleton: `[1,10,30]` F32 = 1200 bytes, or even impossible active-only `[1,10,15]` F32 = 600 bytes.
- Direct free-output rank factorization lower bound with output axes pinned to 30: active-width rank lower bound `15*(30+30)=900` params; full-width lower bound `30*(30+30)=1800` params.
- Dense #119 shift-bank formulation: `[3,30,30]` center/left/right bases = 2700 params.
- Current grouped Conv remains 190 params with zero charged intermediate memory.

## Why it is dead

The rule is separable in color and row parity, but not in the horizontal source-column to output-column coupling. Even rows require the identity placement matrix and odd rows require the adjacent-column matrix. The identity alone has rank equal to active width, so a low-term free-output Einsum cannot represent the placement. If row-0 state is materialized as a marginal/profile first, that tensor already costs at least 600 bytes, far above the 190 bar. If it is kept inside a terminal Einsum, the required horizontal rank/basis params are at least 900 before color, row, bias/background, or padding details.

This is different from the drop-bias wall: the moment/Einsum route loses even before addressing the forced background bias contradiction.

## Remaining hypothesis

Only a one-node, zero-intermediate formulation with fewer than 190 parameters can improve this task. Known remaining families still look losing: Conv/ConvTranspose need the forced `10*6*3` stencil plus bias, while any final-Einsum route must carry a rank-W horizontal placement basis.
