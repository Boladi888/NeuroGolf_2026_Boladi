# task334 — DEAD ROUTE: final-route fusion microshaves

- **Date:** 2026-07-14 (bo_ng_cdx2_worker1)
- **Live bar at audit:** cost **63** (manifest, owner Boladi, sha `31fd85a9bfe4b3052edca36ffd3248e48485d05881fb0deb8cba5aed7c4502d7`, updated 2026-07-11T00:40:30Z). `profile_real_cost.py --arcgen 266` on `Team_Best_Onnx/task334.onnx` reports `static=63 profile=63`.
- **Route audited:** delete/fuse the charged final `Where(UINT8 2/0) -> ConvInteger(zp=1)` route.

## Verdict: rejected locally

Two cheaper static formulations were built, but both fail visible 0/271. Per §VISIBLE-FAIL, neither is a candidate and no truth/ab/profile/safety gates were run.

## Exact failed gates

- `prototypes/cand_direct_small_where_output.onnx` (sha `cea14db3520c3c2569745c95c19e21749e30cebadf4b351f68e70d890e3094e8`) — `_score_v2`: cost 52 = memory 28 + params 24. `ng_gate visible --task 334 --cand ...` -> **0/271 (0.00%) FAIL**.
- `prototypes/cand_cast_u8_convint.onnx` (sha `c0271fae8428cc64034be308e94d64045c75e04f98405c5fb9f494971fb9c0ed`) — `_score_v2`: cost 60 = memory 37 + params 23. `ng_gate visible --task 334 --cand ...` -> **0/271 (0.00%) FAIL**.

## Why the route is dead

- Direct small output makes a legal-looking `[1,10,3,3]` glyph and deletes the final `ConvInteger`, but the official target encoding is always `[1,10,30,30]`; shape mismatch makes every visible pair fail.
- `Cast(gray BOOL -> UINT8) -> ConvInteger` deletes `gray_u8`, `black_u8`, and `x_zp`, but it only paints color 5 glyph cells. The target also requires channel-0 black cells inside the 3x3 patch. The live signed route (`0/2` patch with zero-point 1) encodes both black and gray while ConvInteger padding still emits no-color outside the 3x3 active output. A gray-only carrier cannot satisfy all three states: black, gray, and no-color.

## Named charged-structure deltas

- Direct small output: removed `signed_patch` (9 B), `gray_u8` (1 param), `black_u8` (1 param), and `x_zp` (1 param), but lost the required padded 30x30 output contract.
- Cast route: replaced `signed_patch` with `patch_u8` at the same 9 B and removed `gray_u8`/`black_u8`/`x_zp`, but lost the black-channel semantics.

## One remaining hypothesis

The only plausible residual shave is a different way to encode the three-state final patch (`black`, `gray`, `outside`) with fewer than the live `Where`'s two U8 scalar branch constants plus one shared zero-point. `Cast` alone is insufficient; any complement-channel or QLinearConv-bias variant prices above the live route unless it avoids materializing a second 3x3 patch.
