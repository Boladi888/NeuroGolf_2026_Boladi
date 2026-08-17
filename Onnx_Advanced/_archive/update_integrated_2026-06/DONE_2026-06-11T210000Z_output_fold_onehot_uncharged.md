# Output-fold: make the final one-hot Equal the *graph output* (uncharged), not a [1,C,h,w] intermediate

**What I discovered (transferable, generalizes well beyond task377).**
When a graph ends by expanding a small color-value map into a `[1,10,H,W]` one-hot via
`Equal(value, channel_ramp[1,10,1,1])` and then `Pad`-ing it to the canvas, the usual build
materializes the one-hot as a *charged intermediate* (BOOL [1,10,h,w] + a U8 Cast = 2×810 on a
9×9 tile). That whole expansion can be FOLDED into the graph output:

```
# OLD (charged):  v_oh = Equal(val[1,1,h,w], ramp[1,10,1,1]) -> [1,10,h,w] BOOL (810)
#                 v_u8 = Cast(v_oh -> U8)                    -> [1,10,h,w] U8   (810)
#                 output = Pad(v_u8, ...)                    -> [1,10,30,30]
# NEW (folded):   vp   = Pad(Cast(val->U8)[1,1,h,w], fill=SENTINEL) -> [1,1,30,30] U8 (900)
#                 output = Equal(vp, ramp_u8[1,10,1,1])              -> [1,10,30,30]  == graph output
```

`calculate_memory` SKIPS the tensors named exactly `input` / `output` (core.md), so when the
final `Equal` (or `Cast`/`Where`/`Greater`) IS the output node, its `[1,10,30,30]` result is
**not charged** at all. You only pay for the small padded value-canvas `vp` that feeds it.
Pad the value-canvas with a SENTINEL outside the palette (e.g. 10) so out-of-shape/padding
cells `Equal`-compare false on every channel → decode to "no color" (blank), preserving the
background. Output dtype becomes BOOL (decoder thresholds `>0`, identity on 0/1) — fine.

**Net on task377:** v5 charged 1,620 B for the `[1,10,9,9]` BOOL+U8 one-hot pair; v6's `vp`
U8 `[1,1,30,30]` is 900. **−720 B (9,887 → 9,166)**, byte-equal on 200,266/200,266 corpus
inputs (pure encoding swap, no algorithm change). It also drops the trailing `Cast` node.

**Tradeoff to note:** the folded value-canvas is `[1,1,30,30]` (full output spatial, needed to
broadcast), so the win only materializes when the unfolded one-hot pair `2 × C × h × w` exceeds
`(min-dtype) × H × W`. For task377: 1,620 > 900. For a tiny output tile with many channels the
fold always wins; for a large tile it can lose — price both.

**Where:** task377; builder `The_400_Tasks/Task_by_Task/task377/prototypes/build_task377_rewrite_v6.py`;
staged `staged_submits/task377_renderfold_out_cost9166/`.

**Library gap I hit:** the prior task377 floor proof explicitly called the `[1,10,9,9]` output
one-hot "irreducible" — it only considered replacing the renderer with a cheb-map Gather, never
the output-fold. I did not find a general "fold the terminal one-hot into the uncharged output
tensor" entry while grepping `by_symptom.md` / `techniques_catalog.md` (closest is §81 tail
pricing / §FREE-OUTPUT, which is about free-output Einsums, not this Equal-as-output fold).
Suggest a short technique entry or a `by_symptom.md` row: "graph ends in Equal/Cast→one-hot→Pad
→ make the Equal the output node; pay only for the [1,1,H,W] padded value, not the [1,C,h,w]
one-hot." Cross-ref core.md "Tensors named exactly 'input'/'output' are SKIPPED".

**Also re-confirmed (dead-ends, for the task377 don't-do log):** F32 graph `input` forces the
front-end `Conv`/`Gather` outputs to F32 (no F16/INT narrow without the 18,000 B canvas cast —
measured 19,808 for an F16 front-end); `CumSum` still rejects F16 in opset 11 (accepts I32, but
I32==F32 bytes so no gain). These bound task377 at the 9,166 floor.
