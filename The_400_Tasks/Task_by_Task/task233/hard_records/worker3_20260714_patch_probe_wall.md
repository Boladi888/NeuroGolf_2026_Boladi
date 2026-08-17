# hard_record: task233 worker3 2026-07-14 patch/probe micro-audit - DEAD

## Live bar used
- Authoritative bar from `python tools/ng_current_bar.py --task 233 --json`: cost **24624**, manifest owner Sho, updated 2026-07-09T01:09:17Z, manifest sha `2b822338b46698102baef1dfbac706879e93b0df529c28c06473cc795356e121`.
- Working static task-local live copy used for graph inspection: `prototypes/_live_shipped_24624.onnx`, sha `fbe0832f409e6b032f13ae820a3f2c1e894b55e24861075529364bce544a20ba`, profile cost **24624** (`memory 24307 + params 317`), visible **266/266 PASS**.

## Attempted route
Fresh micro-audit of the current 24624 graph after prior carrier/sigmap/morphology walls:
1. Run `largest_intermediate.py` on live: dominant tensors remain `ch2_f32` 3600B, `source_score_3_f32` 3136B, `ch2_u8` 900B, `labels30` 900B, `ch2_crop_f16` 800B, five `match_score_1_N_*` F16 planes at 648B each.
2. Run `polish_candidate.py` on the task-local live copy as `prototypes/task233_worker3_20260714_polished_probe.onnx`.
3. Inspect whether the fixed visible-patch probes can be redirected from raw F32 `input` `GatherND` to the already-materialized `ch2_u8` carrier.
4. Reprice the existing sparse/carrier floor probes in `prototypes/worker3_20260711_floorprobe/`.

## Exact failed gate / result
- Polish probe: `original_cost=24624`, `final_cost=24624`; onnxsim `no-savings`, safe_opt `no-op`, const_to_init `no-op`, clip_mul_to_min `no-op`, scalar_broadcast_ones `no-op`, slice_axis_collapse `no-op`.
- Static audit: no unused initializers, no duplicate initializer groups, and only one unused node output: `constraint_sort_values` from `TopK`, already known non-removable because `TopK` requires the values output.
- Patch-probe audit: `patch0_input_idx_i64=[[0,3,9,14]]`, `patch1_input_idx_i64=[[0,3,13,7],[0,4,11,1]]`, `patch3_input_idx_i64=[[0,3,11,9]]`; the probes check decoration channels 3 and 4, not channel 2, so they cannot reuse `ch2_u8`. Keeping equivalent sparse reads from `input` is the floor for this patch family.
- Sparse/carrier probe prices:
  - `A_argmax_collapse.onnx`: `(0, 8100, 0)`
  - `B_1x1conv_collapse.onnx`: `(0, 4500, 10)`
  - `C_3x3conv_current.onnx`: `(0, 3136, 90)`
  - `D_redread_forced.onnx`: `(0, 3600, 1)`
  The sparse/color-index carrier path is still a net loss versus the incumbent 3136B source Conv.

## Why dead
The current graph has no byte-equal polish slack in initializers, value_info, or safe algebraic passes. The only apparent patch-detector reuse would need a pre-existing channel-3/channel-4 carrier, but creating any such carrier is larger than the existing 1/2-cell raw F32 `GatherND` probes. The sparse mask/carrier rebuild also still costs more than the original mask family.

## Remaining hypothesis
Only a genuinely new exact matcher/source detector could matter: remove or replace either the forced 3136B fractional source Conv or the five 648B F16 match-score TopK planes without falling into the known visible-fail sigmap, local-morphology, or 13% truth lookup families.
