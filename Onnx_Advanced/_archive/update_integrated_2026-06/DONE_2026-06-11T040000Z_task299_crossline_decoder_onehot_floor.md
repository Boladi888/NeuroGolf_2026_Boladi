# task299 cross-line fill: 2-input row/col decoder + forced-ch0 one-hot floor (1022 -> 923)

**Transferable insight (cross-line "extend two perpendicular segments" micrographs):**

Rule family: a horizontal color-A segment in one row + a vertical color-B segment in one
column; output extends both lines full-grid, the intersection becomes a third color C,
background = 0. The cheap, byte-equal expression is a **2-input boolean decoder** of two
1-D presence vectors, NOT an And/Or chain:

```
row_a = ReduceMax(slice_chA, over W)   # which row has A   [1,1,h,1]
col_b = ReduceMax(slice_chB, over H)   # which col has B   [1,1,1,w]
nr=Not(row_a); nc=Not(col_b)
chC = row_a & col_b ; chA = row_a & nc ; chB = nr & col_b ; ch0 = nr & nc
Concat -> [1,10,h,w] one-hot (zero filler channels) -> Pad-as-output
```

This 2 Not + 4 And replaces the prototype's 3 And + 2 Not + 2 Or (7 intermediates -> 6, and
the two Not are on small 1-D vectors `[1,1,h,1]`/`[1,1,1,w]`, not h×w). The four masks are the
minimal distinct h×w channels the Concat needs.

**Forced-ch0 floor (reinforces the task386 law, two-axis variant):**
LIVE sets ch0=1 on every background cell of the active grid (probed: 25/25 bg cells of a 6×6,
padding cells all-zero). Byte-equal vs `(out>0)` therefore FORCES the full 10-channel one-hot
incl the `ch0 = nr & nc` background mask. The `[1,10,h,w]` one-hot (BOOL = 10·h·w bytes) cannot
be shrunk to the 4 active channels.

**The F32 channel-read floor (the other half of the floor, often missed):**
To get row_a / col_b you must Slice the A- and B-channels. `Slice preserves the input dtype
(F32)`, and the spatial crop to the active h×w is MANDATORY — a channel-only slice (axes=[1])
of the full 30×30 canvas is `[1,1,30,30] F32 = 3600 B` each. Cropped it is `h·w·4` (144 at 6×6).
Two channels = 288 B that no fp16/bool trick removes (casting the full input is 9000 B; bool
ReduceMax is unsupported in ORT; reduce-then-slice keeps all 10 channels = 240/axis, worse).

**Net floor for a cross-line byte-equal one-hot output at grid h×w:**
`10·h·w (one-hot) + 2·(h·w·4) (two F32 channel reads) + 4·(h·w) (masks)` in memory.
At 6×6 that is 360 + 288 + 144 + small = ~864 -> total ~923. A "~500" target that assumes a
4-channel output is INFEASIBLE here because ch0 is forced; 360+288 alone exceeds 500.

**Param micro-trims:** drop the Slice `steps` initializer (defaults to 1); drop any unused
zero-filler dtype duplicate. Concat does NOT broadcast, so the zero filler channel must be a
full `[1,1,h,w]` init (h·w params), reused by reference across all zero slots.

**Gates:** visible 21/21, ab 200021/200021 byte-equal, profile==static==923. Static graph =>
local cost == Kaggle exact (no probe needed; ran gate-5 anyway, profile==static confirmed).
