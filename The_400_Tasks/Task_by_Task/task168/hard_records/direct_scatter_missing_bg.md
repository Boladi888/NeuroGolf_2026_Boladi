# DirectScatterMissingBg wall - 2026-07-12

## Authoritative bar

- Live/team-best cost: **1508** (`memory=1268`, `params=240`).
- Source: `python tools/ng_current_bar.py --task 168 --json` -> authoritative `manifest` row, Aditya, updated `2026-06-22T06:32:53Z`.
- Parent: `Team_Best_Onnx/task168.onnx`.
- Parent SHA-256: `d5af2d4f19d87dbb694e4d08482cfd2d44f55e518f6206990b1b43a58f2ec5cd`.
- Parent profile: `task168 task168.onnx: static=1508 profile=1508 (memory 1268 + params 240; 265 real inputs)`.

## Attempted route

Materialized the remaining hypothesis from `carrier_narrow_equal_tail.md`: remove the channel-0 background update tail and directly scatter only `paint_rows16` into the runtime foreground channel:

`ScatterND(zeros_rows16, color_idx, paint_rows16) -> chan_rows10`

Prototype: `The_400_Tasks/Task_by_Task/task168/prototypes/task168_direct_scatter_probe.onnx`.
Candidate SHA-256: `dbfef50122a6029263791ee76f0420d501e7210563b5bab3bed712244c705cce`.

Static score was cheaper: `_score_v2 = (macs=0, memory=1192, params=239)`, cost **1431**. `ng_graph_cost_diff` named the removed charged structures as `scatter_updates_nd` (-40 B), `bg_rows16` (-20 B), `scatter_nd_idx` (-16 B), and `zero_idx1` (-1 param), net -77.

## Exact failed gate

`python tools/ng_gate.py visible --task 168 --cand The_400_Tasks/Task_by_Task/task168/prototypes/task168_direct_scatter_probe.onnx`

Result: **`[visible] 0/265 (0.00%)  FAIL`**.

## Why this route is dead

The static saving is real but removes required raw one-hot background. A local diagnostic on the first bundled example showed:

- `foreground_equal_excluding_ch0 = True`
- `cand_bg_ones = 0`
- `expected_bg_ones = 88`
- `bad_cells_ch0 = 88`

So the probe is only cropped-grid/decode tempting; it fails the official raw tensor comparison because background cells must have channel 0 set. Per the visible-fail axiom, this representation family is hidden-zero and no truth/AB/profile gate should be run.

## One remaining hypothesis

Only a final-op fusion that preserves channel-0 background while avoiding the charged `chan_rows30` carrier could beat 1508. No static ORT op route found in this pass can both unpack the U16 row masks to full `[1,10,30,30]` output for free and apply the runtime color/channel-0 selection without first materializing a charged `[10,30,1]` or larger carrier.
