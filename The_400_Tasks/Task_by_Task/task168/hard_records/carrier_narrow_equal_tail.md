# CarrierNarrowEqualTail stale-route wall — 2026-07-10

## Authoritative bar

- Live/team-best cost: **1508** (`memory=1268`, `params=240`).
- Source: `python tools/ng_current_bar.py --task 168 --json` → authoritative `manifest` row, Aditya, updated `2026-06-22T06:32:53Z`.
- Parent: `Team_Best_Onnx/task168.onnx`.
- Parent SHA-256: `d5af2d4f19d87dbb694e4d08482cfd2d44f55e518f6206990b1b43a58f2ec5cd`.
- Full real-input profile: `static=1508 profile=1508 (memory 1268 + params 240; 265 real inputs)`.
- The task-local 4673 state/library note is stale. `Staged_Boladi/task168.json` was absent from the repo, team repo, BoBrain root, and provider root checked during this pass.

## Attempted route

Prompted fast route: collapse a 10-channel/F32 spatial carrier and repeated Conv masks to a one-channel U8 carrier, then use an Equal output tail, targeting approximately 720.

## Exact failed gate

**PREBUILD_ROUTE_PRECONDITION: FAIL.** The authoritative live graph no longer contains the charged family named by the prompt. Its first op is `Einsum`; it has no repeated `Conv`, no 10-channel spatial F32 carrier, no `Where`, no `Equal`, and no charged full canvas/template. It already contracts the input to `[1,1,30,1]` F32 (120 B), narrows that to U8 (30 B), performs packed bitwise L detection, renders packed rows with one F16 `ConvTranspose`, and makes the final `BitwiseAnd` the free graph output.

**FAST PRICE GATE: FAIL.** The library §81 Equal-tail requires a charged `[1,1,30,30]` U8 canvas, **900 B minimum**, already greater than the entire 720 target before detection, color selection, active/padding sentinel handling, or parameters. The 10×10 small-one-hot inversion is 1000 B. Per the prompt's stale-route/price kill rule, no ONNX was built and no visible/truth/AB gate was burned.

Dominant live charges are already packed-tail costs: `chan_rows30` U16 600 B, `chan_rows10` U16 200 B, `fg_rows30` F32 120 B. The old carrier/canvas/template charge is gone.

## Why this route is dead

Carrier narrowing and QLinearConv cannot remove a carrier that is no longer present. QLinearConv also cannot consume the F32 graph input directly; a full-frame U8 cast would cost 9000 B. Equal-tail cannot reach the requested floor because its canvas alone exceeds it. This assignment is a false-floor attack, not a 1508-minus-epsilon shave, so the correct verdict is **DROP / REJECT_LOCAL** with no staged candidate.

## One remaining hypothesis

A different assignment could test a direct-output fusion that eliminates the 600 B `chan_rows30` carrier and the runtime-color Scatter tail simultaneously. A read-only in-memory audit found a smaller, non-byte-equal micro-shave concept—implicit channel-0 ties plus direct `ScatterND(zeros_rows16, color_idx, paint_rows16)`—projected at 1431 (−77) and decoded-equal on the 265 bundled pairs. It was deliberately not materialized or staged here because it is Path-B, unproved on fresh truth, and outside this prompt's required false-floor route.

