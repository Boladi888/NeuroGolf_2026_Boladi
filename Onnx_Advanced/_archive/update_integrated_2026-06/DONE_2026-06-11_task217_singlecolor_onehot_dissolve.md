# DONE 2026-06-11 — single-color fractal/recolor dissolve to a 1-channel pipeline

**Source:** task217 ("fractal2" gen `8f2ea7aa`), Worker-2. LIVE 13301 -> staged 2668,
Path-A byte-equal (200266/200266) + 3000/3000 ARC-GEN true-oracle + 100% adversarial.

## Transferable technique: collapse a single-color recolor task to ONE channel + a free color vector

When the rule has exactly ONE non-bg color and the output is `mask x color` (recolor,
fractal, fill, stamp...), you do NOT need any [1,10,H,W] carrier except the final
output one-hot. The collapse:

1. **Color index is FREE.** `per_ch = ReduceMax(input, axes=[2,3])` over the FULL
   30x30 canvas (padding is 0, so max == max over the active grid). No crop. Drop ch0
   with a Mul mask. This [1,10,1,1] vector IS the color one-hot.
2. **Foreground via 1-ch0 on a SINGLE-channel crop.** If the grid size is bounded and
   the crop contains NO padding cells (i.e. grid fills the crop exactly), then
   `fg = 1 - ch0` is EXACT — the 1-ch0 trap only mis-fires on padding cells, which
   can't exist inside a full grid. So crop ONLY ch0 ([1,1,H,W]) instead of 9/10
   channels. (task217: [1,1,9,9]=324 F32 vs the old [1,10,9,9]=3240 F32 crop.)
3. **Build the output one-hot in ONE `Where(mask_bool, color_vec[1,C,1,1], ch0_vec)`**
   broadcasting a [1,1,H,W] cond against [1,C,1,1] color/bg vectors. Replaces the
   classic `color_out + bg_out` pair (two [1,C,H,W] tensors) with a single output
   tensor. Valid whenever bg = complement of the color mask (no padding inside crop).
4. **Carry the spatial mask chain (shifts, kron, tiling) as BOOL** — Gather is
   dtype-agnostic; `And(bool,bool)` replaces `Mul` for AND-style combines (ORT has no
   uint8 Mul kernel but And/Gather work). BOOL [1,1,H,W] = H*W bytes vs F16 = 2*H*W.
   Keep ONE F16 copy only where a numeric ReduceMax/ArgMax is needed (topleft detect).
5. **Output one-hot as UINT8 (opset>=13 for Pad(uint8)).** `Where(bool,u8,u8)` is
   ORT-supported; the decoder thresholds >0 so a {0,1} uint8 one-hot decodes to the
   right color (channel identity carries color). [1,C,H,W] uint8 = C*H*W, half of F16.

Net on task217: every [1,10,9,9] carrier (cropped 3240, cast 1620, color_out 1620,
bg_out 1620, ch_nonbg 1458) dissolved; only the irreducible output one-hot (810 u8)
and ~1-channel work remain.

## Calibration note: trust context_pack/show_task current_cost, NOT the prompt narrative

The prompt said "team-best Aditya ~4059, from ~146k" but `context_pack`/`show_task`
reported `current_cost = 13301` (the crop9+F16 graph was already LIVE in
submission.zip). The prompt narrative was a stale snapshot. Per WORKER_PROTOCOL the
live tool number is the EV baseline. Always re-derive team-best from the tool, not the
prose. (Saved a phantom "+0.81 at 1800" estimate from being mis-scoped.)

## Boolean-reformulation safety (re: core.md task080 caution)

The BOOL chain here is NOT a rule reformulation — it carries the SAME exact {0,1}
mask through Gather/And. It's hidden-safe because (a) byte-equal vs LIVE on 200k,
(b) exact vs the INDEPENDENT ARC-GEN generator on 3000 fresh seeds (the generator
that makes the hidden set), (c) the generator is fully decoded and the grid bound is
FORCED (size=3 default, no override). When all three hold, a bool/uint8 mask carry is
safe; the task080 danger is bool reformulations whose RULE could silently differ.
