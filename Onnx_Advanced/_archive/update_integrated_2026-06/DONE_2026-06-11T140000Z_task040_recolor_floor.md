# task040 colorcanvas — confirmed forced floor 1785 (half-space two-color probe recolor)

## What I discovered (a clean floor law for the "channel-collapse + per-cell 2-way recolor + Equal-tail" class)

task040 (`task_2204b7a8`, fixed 10×10, two opposite border-strip colors recolor interior green probes by column-half, random transpose) is at a **forced 1785-byte floor**. The graph is the canonical:
`Conv crop+collapse (F32 [1,1,10,10]=400) → Cast U8 (100) → recolor 3 planes (300) → Pad to U8 canvas [1,1,30,30] (900) → Equal(canvas, ids)-as-output (free)`.

The durable, transferable piece is the **recolor-plane floor count**: a per-cell DATA-DEPENDENT 2-way select on a K×K crop costs exactly **4 planes** (source crop + BOOL condition + replacement + result), each K·K bytes U8/BOOL. There is no 3-plane or arithmetic-blend shortcut when:
- the two outcomes collide in value space with the background (here green↔bg both collapse-collide, killing Min/Add fusion), AND
- ONNX `Where` forces a materialized BOOL condition tensor (no inline predicate), AND
- the replacement is itself a data-dependent broadcast of two differently-shaped 1-D vectors (`Where(col_axis, col_repl[1,1,1,W], row_repl[1,1,H,1])`) — col_axis being data-dependent (random transpose) forces the [H,1]×[1,W] broadcast to materialize the full K×K plane; it cannot stay 1-D.

## Routes priced and REJECTED (measurements, not vibes)
- **Where-as-output over input** (`Where(probe30, repl_onehot, input)`, the #81/292 sparse-substitution route): loses here because the replacement is data-dependent (color0/color1 read from corners, not a fixed e_k). repl_onehot needs `Equal(repl_canvas[1,1,30,30], ids)` = a 900B canvas PLUS a separate 900B probe mask ⇒ ≥1800B > the 900B canvas+Equal. **Lever note: Where-as-output only wins when the substituted value is a FIXED channel e_k; a data-dependent replacement color reintroduces the canvas.**
- **Non-F32 channel-collapse to dodge the 400B F32 crop:** ArgMax(axis=channel) emits INT64 (800B > 400B F32). QLinearConv-U8 is blocked by #137's prerequisite (needs a U8 feed; `Cast(input→U8)` on the full one-hot = 9000B). So the F32 crop-collapse 400B is forced (same class as task400/138/383).
- **Nibble-pack (#83) the recolor planes:** fights this rule specifically — the column-half boundary c<5 straddles pack p=2 (cols 4,5), the random transpose flips the packed axis, and the output Equal needs an unpacked 30×30 canvas anyway. Net-neutral-or-worse on a fixed 10×10.

## Why it matters
Converts task040 from "tried hard" to a forced-component proof at the 338/054/080 standard (written into `The_400_Tasks/Task_by_Task/task040.md` ## FLOOR PROOF). Team-best/LIVE both = 1785 (sha `95395df1…`), byte-equal 200266/200266, visible 266/266, static==profile==1785.

## What's possibly missing from the library
`cost_floor_reference.md` has the 900B canvas law (:164-168) and the F32-collapse floor entries (task400/138/383), but I did not find an explicit **"per-cell data-dependent 2-way recolor = 4 planes (source+cond+repl+result), no <4 shortcut"** law. Suggest adding it near the canvas law, with the green↔bg value-collision and the data-dependent-broadcast (`Where(col_axis, vec_w, vec_h)` can't stay 1-D) as the two reasons the count is forced. Also worth a one-line note on #81 that Where-as-output requires a FIXED e_k — a data-dependent replacement color reintroduces the full canvas.
