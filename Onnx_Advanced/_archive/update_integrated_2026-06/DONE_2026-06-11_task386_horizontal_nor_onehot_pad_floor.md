# task386 horizontal-NOR micrograph: Where-as-output one-hot+Pad floor (1150 -> 296)

**Transferable insight (small-grid one-hot+Pad is the floor when byte-equal needs ch0):**

A near-floor micrograph (`Mul:2|Slice:2|Add:1|Pad`, 1150) dissolved to 296 by replacing
the TWO 10-channel F32 intermediates (`colored_delta`[1,10,4,3] + `combined`[1,10,4,3],
480 B each) with a SINGLE `Where(mask_bool, e_k[1,10,1,1]_u8, e_bg[1,10,1,1]_u8)` ->
[1,10,4,3] u8 = 120 B, then Pad-as-output. The Add-based "bg_vec + delta*mask" two-tensor
assembly is strictly dominated by one Where (one charged tensor instead of two).

**The ch0-completeness forcing argument (why you can't go below the one-hot):**
Byte-equal vs LIVE compares `(out>0)` decoded grids. LIVE sets ch0=1 on background cells.
A construction that leaves bg cells all-zero decodes them to sentinel-10 ("no color"),
NOT color-0 -> diverges. So the background channel is FORCED, which:
- kills the free-output Conv-placer (#78): a constant-free linear conv (#85 bias-leak law)
  can emit ch_fg=mask for free but cannot emit ch0=(1-mask) complement;
- the Equal-tail (#67) *can* produce ch0 via sentinel canvas, but its 900 B padded
  [1,1,30,30] canvas loses to the 120 B one-hot below the #81 9.5x9.5 small-grid crossover.

So for a tiny two-color output grid where byte-equal demands the background channel:
**decoded one-hot ([1,10,h,w] u8) + free Pad is the floor**; both free-output escapes are blocked.

**Mask micro-law:** `Greater(slice_L)+Greater(slice_R)+And` (3 bool tensors, 12 B each = 36)
beats `Mul(F32,48)+Cast(bool,12) = 60` for an AND of two F32 0/1 strips. Slice dtype = input
dtype (F32), and non-contiguous panels (separator column between) force two separate slices.

**Gates:** visible 268/268, ab 200268/200268 byte-equal, profile==static 296. Where(bool,u8,u8)
ORT-supported; u8 output decodes via >0 byte-equal to LIVE's F32 (decode is dtype-agnostic).
