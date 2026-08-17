# Two transferable findings from task094 (worker-094, 2026-06-18)

Task094: per-cell rule, F32 one-hot input `[1,10,30,30]`, U8 canvas + Pad-sentinel + free
Equal output. Static baseline v8 = 3022. Pushed to v9 = **2812** (-210 B), gated green
(visible 265/265, truth 200k/200k, AB 200265/200265 byte-equal, profile==static==2812).

## 1. The 2D `Or` line-plane is often REDUNDANT — paint via nested `Where` off the 1D axis vectors

A very common pattern for "color a full-width + full-height cross at detected rows/cols" is:

```
line_b = Or(row_b[1,1,H,1], col_b[1,1,1,W])      # broadcasts to a 2D BOOL [1,1,H,W] plane (H*W bytes)
canvas = Where(line_b, fg, bg)
```

The `Or` MATERIALIZES a full 2D BOOL plane (225 B at 15x15, 900 B at 30x30) purely to feed one
`Where`. You can DELETE that plane by nesting two `Where`s directly on the 1D vectors:

```
wc          = Where(col_b[1,1,1,W], fg, bg)      # broadcasts col -> [1,1,H,W]
line_canvas = Where(row_b[1,1,H,1], fg, wc)      # overlays row
```

Same plane count on the paint side (2 planes either way: `line_b`+`canvas` vs `wc`+`line_canvas`),
but the nested form drops the separate `Or` node AND its dedicated 2D plane — net -1 plane (-H*W bytes).
Measured -210 B on a 15x15 grid (the saving exceeds the bare `line_b` size because the scorer also
stops billing the intermediate the Or produced). Reach for this whenever you see `Or`/`And` of two
broadcast axis vectors feeding a single `Where`/select. (Do NOT instead fold to arithmetic
`bg - delta*line_u8`: that needs `Cast`+`Mul`+`Sub` = MORE planes than it removes, and scored worse/None.)

## 2. #137 carrier-narrow-to-U8 does NOT apply when the source is a plain F32 `Slice`

Prompt routed task094 to #137 (narrow a threshold-only F32 carrier to U8). It does NOT work here, and
the reason generalizes: #137 works in task192 because the carrier was a **Conv output** that could be
re-emitted as a **QLinearConv** producing U8 directly — the F32 plane is never materialized. But when
the F32 plane comes from a plain `Slice` of an F32 input (here `ch1 = Slice(input)[:,1:2,:15,:15]`),
`Slice` preserves dtype, so:

- `Cast(ch1, U8)` charges BOTH the 900 B F32 slice AND the 225 B U8 plane → measured **4147 > 3022** (double-charge).
- `Cast(input, U8)` first = 9000 B for the full one-hot (the #137 documented blocker).
- Rebuilding the plane from 1D counts needs a `Conv` → re-materializes a 900 B F32 plane.

So a plain F32 `Slice`-read of a single channel (`channels*h*w*4` at the active grid) is an **irreducible
input-read floor**; #137 only helps when a U8/INT8 feed already exists OR the F32 producer is itself a
Conv convertible to QLinearConv. Before routing a task to #137, check the carrier's PRODUCER: Conv →
maybe; Slice/Gather of F32 input → no (the F32 source is already charged, the U8 cast just adds on top).
This refines #137's existing "use only when a U8/INT8 feed already exists or is cheaper than the carrier
it deletes" note with the concrete Slice-source double-charge measurement.
