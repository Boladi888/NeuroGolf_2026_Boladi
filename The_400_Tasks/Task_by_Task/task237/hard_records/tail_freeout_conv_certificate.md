# task237 tail free-output Conv certificate

Worker: bo_ng_cdx_worker3  
Date: 2026-07-07  
Route: structural tail rewrite from `task237_rankone_cost696`, deleting `region_select` and `code_vec2` by moving the 9x9 L-trail render into the final output op.

## Bars

- Authoritative live bar: `705`, SHA-256 `a59a2fa610fa58f67f8026812f7a160bf82f9fb1b65955c50fb91dde71bf538e`, source `python tools/ng_current_bar.py --task 237 --json` / manifest.
- Lower task-local bar to beat: `696`, SHA-256 `64549e67db188f7dceb7ccfd133fd2d058158845f3c1fd138ec668060fad33e6`, source `staged_candidates/task237_rankone_cost696/manifest.json`.

## Attempted structural rewrite

Prototype:

`The_400_Tasks/Task_by_Task/task237/prototypes/task237_tail_freeout_conv_certificate.onnx`

Builder:

`The_400_Tasks/Task_by_Task/task237/prototypes/build_task237_tail_freeout_conv_certificate.py`

The final op is a free-output F32 `Conv` with pads `[0,8,21,29]`, so the op itself renders the 9-row carrier to `[1,10,30,30]`. It removes the current charged 2-D tail:

- `code_vec2` UINT8 `[1,2,9,9]` = 162 B
- `region_select` BOOL `[1,1,9,9]` = 81 B
- `selected_vec` UINT8 `[1,2,9,1]` = 18 B
- `col_active`, `col_active_u8`, `active_col_code` = 27 B

Correct classification inside one final linear op is not separable from just `(selected_t, selected_t^2, row_threshold, width, 1)`. The smallest LP-feasible row-feature set found was:

`t, t^2, q, W, t*q, t*W, q*W, 1`

Quantized `QLinearConv` is still infeasible for that set under its coefficient range:

- `M=127`: infeasible at output channel 2
- `M=255`: infeasible at output channel 4
- `M=512`: infeasible at output channel 8
- first feasible bounded solve needed `M=1024`, outside uint8 weight-zero-point coefficient span

So the built certificate used F32 `Conv`, which makes the extra cost explicit.

## Exact cost/profile

Score:

`(0, 659, 843) 1502`

Profile:

`task237 task237_tail_freeout_conv_certificate.onnx: static=1502 profile=1502 (memory 659 + params 843; 266 real inputs)`

Largest added charged structures:

- `tail_features_f32` F32 `[1,8,9,1]` = 288 B
- `tail_features_u8` UINT8 `[1,8,9,1]` = 72 B
- `width_rows`, `tq`, `tW`, `qW` = 36 B total
- `tail_conv_w` F32 `[10,8,1,9]` = 720 params
- `tail_conv_b` F32 `[10]` = 10 params
- `one_rows` UINT8 `[1,1,9,1]` = 9 params

Net vs 696:

- Memory: `659 - 551 = +108`
- Params: `843 - 145 = +698`
- Total: `1502 - 696 = +806`

This is not close to the reviewer target `<=615`; it is also above the existing 696 rung.

## Gate result

Visible:

`[visible] 37/266 (13.91%)  FAIL`

Per gate hygiene, this representation family is dead after the visible miss; no truth/ab/full gates were run.

## Why the final-output formulation is dead

The final output must bind three facts at each cell: selected color `t`, trail threshold `q <= col`, and active width `col <= W`. A one-node linear final renderer cannot express the color-region product without product features. The first algebraically feasible product-feature formulation requires at least 8 row features and a dense `10 x 8 x 9` output kernel, which alone costs 720 params. That param bill more than cancels deletion of `code_vec2` and `region_select`.

A direct #109/#119 separable Einsum formulation has the same wall in another form: it must materialize or factor both color-by-row and threshold-by-row state, then place the 9x9 result into 30x30. The minimum explicit factor sketch pays at least color row one-hot 90 B, threshold row one-hot 81 B, active-column 9 B, triangular threshold factor 81 params, plus row/column placement factors `[9,30] + [9,30] = 540 params`, before background-complement terms. That is already `>=801` added structure on top of the upstream carrier and is above the 696 rung.

## Remaining hypothesis

Only a different supported op that can compute the color-region product inside the final free output without dense per-column kernels could reopen this route. I did not find such an op under #67/#81/#89/#109/#119; do not retry scalar microshaves, zero-point bias absorption, tensor-max Clip, or active-col false-branch replacement.
