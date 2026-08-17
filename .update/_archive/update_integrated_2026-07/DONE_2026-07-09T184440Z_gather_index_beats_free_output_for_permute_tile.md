# Transferable insight — for a row-independent column permutation/tile, a direct int32 gather-index BEATS every free-output (Equal/label-map/MatMul) reformulation

**Source**: task249 batch56 (`bo_ng_cld_worker2`, 2026-07-09). Audited LIVE **219** (Sho, 2026-07-09);
this is a **floor confirmation** (219 exhaustively proven optimal across 3 independent from-scratch
families), not a new win. Kills the tempting "free-output Pad-tail / color-index collapse" direction
when the current graph is ALREADY free-output.

## The trap it kills
On a "copy/tile/permute columns of a one-hot input into a fixed `[1,10,30,30]` canvas" task, the
prompt-writer's instinct is: *"delete the charged 10-channel canvas; emit one label/mask channel and
make Equal/Pad the graph output (free-output)."* That instinct is correct ONLY when the live graph
still materializes a charged `[1,C,H,W]` canvas. Once someone has already collapsed it to a
**gather-index** graph (final `Gather(input, idx, axis=3)` = the free output), the free-output-Equal
route is a **regression**, and it is easy to burn a whole session "dissolving" a canvas that is
already gone.

## The mechanism (why gather-index wins and is a hard floor)
For output that is a **row-independent** column permutation of a one-hot input
(`output[:,:,r,c] = input[:,:,r, idx[c]]`, same `idx` for every row):

1. **The permutation's minimal charged control tensor is a length-W_out int index.** For a 30-col
   output that is `idx[30]`. `Gather`/`GatherElements`/`GatherND` indices must be **int32/int64** —
   tested int8/uint8/int16/uint16 all → `INVALID_GRAPH` (ORT has no narrow-index Gather kernel). So
   `idx[30]` = **120 B** is a hard floor (int64 = 240, worse).
2. **Every free-output alternative is strictly larger:**
   - *Color-index label map* `Equal(label[1,1,H,W], colors[10,1,1]) → output`: the output color at
     `(r,c)` depends on BOTH `r` and `idx[c]`, so `label` is a full `[1,1,30,30]` = **≥900 B** (uint8),
     7.5× the gather index. NOT row-separable → cannot shrink.
   - *Selection matrix* `MatMul(input, P[30,30])`: `P` ≥ 900 B (bool) / 3600 B (f32). Worse.
   - *Dynamic Slice+Tile+Pad*: needs a `[1,10,30,W]` active slice ≤ 6000 B. Worse.
3. **You cannot un-charge `idx` by exposing it as a 2nd graph output** — the grader's
   `calculate_memory` returns `None` for any multi-input/multi-output graph
   (`neurogolf_utils.py` line 197: `if len(graph.input) > 1 or len(graph.output) > 1: return None`).
   Only the single tensor literally named `output` (and `input`) is free.

**Corollary**: keep the one-hot input AS one-hot and column-gather it directly. Do NOT argmax it to a
label map and re-`Equal` it — that throws away the compactness (the `[30]` index becomes a `[30,30]`
map) AND adds an argmax + charged intermediate.

## Measured (task249, W∈{3,4,5} horizontal 2× tile)
- Live gather-index (Sho): **219** = memory 176 (idx[30] i32 120 + idx10[10] i32 40 + probe 16) + params 43.
- `arith_u8` (arithmetic mod-index, no LUT): **227** — an extra u8[30] tensor exists before the forced int32 cast.
- `direct_bank30` (single `[3,30]` LUT gather, no Pad): **236** — the `[3,30]` bank = 90 params.
- `free_equal` (free-output color-index Equal — the "suggested" direction): **56046** — charged perm/label surface.

## Rule of thumb
Before taking a "free-output / delete-the-canvas" prompt at face value, **inspect the live graph
first**. If its final op is already `Gather(input, idx)` (or any op whose output IS the graph output),
the canvas is already dissolved and the residual cost is a **forced int32 index** whose bytes =
`4 × output_width`. A free-output-Equal/label-map/MatMul rewrite of a row-independent permutation
will regress. Only pursue free-output-Equal when the permutation/render is genuinely a per-(r,c)
formula (a 1-D curve or separable field), NOT a full column copy — cf. library #109 2-D-region
boundary. And never expect a narrower-than-int32 gather index.
