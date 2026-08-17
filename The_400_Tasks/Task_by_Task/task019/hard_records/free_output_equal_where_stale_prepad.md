# task019 — free-output Equal/Where stale-prepad route

- Live bar: **1473** (`memory=1134`, `params=339`), SHA-256 `efdb9c123b02111d6f0805ba1d45b48efeb5d7d95f97bb31add7add29393582f`; authoritative source: `python tools/ng_current_bar.py --task 19 --json` → team manifest, confirmed by workbook.
- Attempted route: the prompt's one-channel predicate/color-index followed by a free-output `Equal` or `Where` tail, intended to delete the old `[1,10,12,12] F16` pre-Pad canvas.
- Exact failed gate: **workflow premise / priced-wall gate**. `inspect_graph.py Team_Best_Onnx/task019.onnx` shows the live graph has no `Pad`, `Equal`, or `Where`; its final `QLinearConv` writes `[1,10,30,30] UINT8 output` directly. `largest_intermediate.py` shows the largest carrier is now `qpt_pair_u8 [1,2,12,12] U8 = 288 B`; the cited 2880-B F16 canvas is absent.
- Why dead: a legal index tail must retain `idx12 [1,1,12,12] U8 = 144 B`, then `Pad(..., sentinel=10)` to a charged `[1,1,30,30] U8 = 900 B`, before terminal `Equal`; tail floor **>=1044 B** before detection, tiling, color selection, or params, already above the prompt's 420 kill bar. Reversing the order charges `[1,10,12,12] BOOL = 1440 B`. A terminal `Where` needs a charged full `[1,1,30,30] BOOL` condition (900 B), and its free `input` passthrough branch is invalid because task019 outputs a 2x2 tile rather than identity.
- Stop rule: prompt workflow update says to return `DROP` when the named charged component is already gone; no candidate, visible, truth, AB, full, safety, or staging gate is warranted.
- One remaining hypothesis: only a fundamentally different packed **single-plane state** that preserves distinct padding/background/marker codes and lets one terminal output-writer derive marker color plus diagonal cyan without a full spatial Equal/Where operand could beat this mechanism. Prove local separability first; the existing one-plane 3x3/5x5/7x7 searches and rank>1 halo evidence make this low confidence.

Measured live lines:

```text
Team_Best_Onnx/task019.onnx -> mem=1134 params=339 COST=1473
task019 task019.onnx: static=1473 profile=1473 (memory 1134 + params 339; 267 real inputs)
```
