# task338 — "sentinel VALUE is free": reuse an existing constant as the Pad fill to drop a dedicated sentinel init (−1 param)

Session: bo_ng_cld_worker1 (Claude lane), task338, 2026-06-30. Verdict: a genuine
strict win (8469 → 8468, −1 B), byte-identical to the shipped graph. Small, but
the underlying fold is transferable to any sentinel-Pad / out-of-region masking graph.

## The fold (transferable)

Many "render into a fixed crop, then Pad to the [*,30,30] output and `Equal`
against per-channel `ch_ids`" graphs use a **sentinel value** to force out-of-
region cells (border padding AND undersized-grid cells inside the crop) to match
NO `ch_id` (i.e. decode to no-cell). The classic shape:

    n10   = Div(s_u8, D)        # out-of-grid s∈{160..164} -> SENTINEL ; in-grid 0..k -> 0
    canvas= Add(n10, parity)    # in-grid {0,1} ; out-of-grid {SENTINEL, SENTINEL+1}
    padded= Pad(canvas, fill=SENTINEL)
    output= Equal(padded, ch_ids)   # ch_ids never contains SENTINEL or SENTINEL+1

**Key realization: the sentinel VALUE is a free parameter.** Any value works as
long as it (and sentinel+parity) is absent from `ch_ids`. So:

1. Set `SENTINEL` equal to a u8 constant the graph ALREADY has for another op
   (here `two_u8`, used by `Mod(cum2, 2)`), and feed that same init as the `Pad`
   fill value — initializers can be consumed by multiple nodes for free.
2. Retarget the `Div` divisor `D` so the out-of-grid band maps to that sentinel
   (here `160..164 // 80 == 2`; in-grid `0..4 // 80 == 0`, unchanged).
3. The dedicated sentinel constant (`ten_u8`) is now dead → delete it.

task338: constants `{2,3,10,16}` → `{2,3,80}`, **−1 param**, output byte-identical
(ab 200267/200267, truth 200000/200000, visible 267/267, profile 8468==static).

## When it applies / when it doesn't

- Applies whenever a Pad-fill sentinel and a Div/threshold-derived sentinel share
  the SAME conceptual "out-of-region" role and the sentinel value is unconstrained.
- The divisor just needs to keep in-grid values → 0 and map the out-of-grid band
  → the chosen sentinel: pick `D ∈ (hi/(SENTINEL+1), lo/SENTINEL]` (here
  `(164/3, 160/2] = (54.6, 80]`). Any integer in that window is one init; you do
  not save the divisor, you save the *separate sentinel* init by reusing one you
  already pay for (the Mod / Equal operand).
- Does NOT apply if every existing constant is too small to also serve as a legal
  sentinel (must differ from all `ch_ids` AND from the in-grid value set {0,1}).
- Pure micro-fold (−1 param). It does not touch memory or the algorithm; it's a
  last-mile param squeeze once a graph is otherwise at its plane floor.

## Context: task338 is otherwise WALLED at 8,468

Memory 7,775 is forced (F32 conv-entry plane 2,500 B; seven 625-B U8 planes; 900-B
output canvas — all individually argued in `task338.md`). After this fold the
remaining 693 params are all forced (Wq 625 full 25×25 i8 2D-prefix kernel —
separable split is net +50; K 40 = Conv weight in_ch must == 10 input channels;
ch_ids 10; pad_parms 8; 6 QLinearConv quant scalars; B 1 = +160 bias offset;
{2,3,80} 3). Sub-8468 needs to delete the 2,500-B F32 entry, which requires a
float-in/int-out conv ai.onnx does not have. Corpus check: 150164/200000 inputs
have a dim < 25, so the out-of-grid sentinel machinery is load-bearing.
