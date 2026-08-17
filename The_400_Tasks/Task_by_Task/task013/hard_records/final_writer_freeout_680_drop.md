# task013 DROP - final writer free-output enumeration at 680

## Bar
- Authoritative live/team-best bar: 680 = memory 565 + params 115.
- Source: `python tools/ng_current_bar.py --task 13 --json`, authoritative source `manifest`.
- Owner/update: Boladi, 2026-07-10T22:11:58Z.
- Parent/live SHA-256: `64ccf42b47a7d774932f89238c048b6bfc1f5ad2f93e81da13e9b936de7406d5`.
- Local `Team_Best_Onnx/task013.onnx` has the same SHA; stale helper displays may show cost 679, but manifest/workbook bar is 680.

## Locked Hypothesis
Only node 42 / the final output writer was tested. The detector, scalar moments, period chain, and U8 line-code chain through `line_code` were left intact.

The current #89 final `Einsum` consumes raw F32 `input` as the active-grid mask and consumes `line_feat` F32 `[2,30]` for channel comparison. `line_feat` costs 240 B, with `line_feat_u8` costing 60 B before it.

## Prototype 1 - comparison tail underprice
- Path: `The_400_Tasks/Task_by_Task/task013/prototypes/task013_cdx2w1_proto1_rank_compare_min.onnx`
- SHA-256: `741d06ab2a00063fca59a52370480111d2aacf95d5b173194ff6bf32243d1ddb`
- Score: `(0, 565, 152) total=717`
- Delta vs live: `+37` bytes by `score_v2`; `ng_graph_cost_diff` reports the same structure delta with its known one-byte scalar skew.

This is an intentionally underpriced #67 Comparison/BOOL-tail:
- Removes `line_feat_u8` 60 B and `line_feat` 240 B.
- Adds the minimum charged channel comparison `line_eq` BOOL `[1,10,1,30]` = 300 B.
- Adds only a fake constant row-active vector, not runtime orientation or real inactive-grid detection.

Even this one-orientation, not-correct version is already 717. A correct version needs the missing runtime orientation and inactive-grid sentinel, so it cannot be below 680.

## Prototype 2 - Equal-tail canvas underprice
- Path: `The_400_Tasks/Task_by_Task/task013/prototypes/task013_cdx2w1_proto2_equal_canvas_min.onnx`
- SHA-256: `172e9ddc4fd38236ddc87af24711ab85b7c3a16acfd0210e755c9dfd79170768`
- Score: `(0, 1165, 152) total=1317`
- Delta vs live: `+637` bytes by `score_v2`; `ng_graph_cost_diff` reports the same structure delta with its known one-byte scalar skew.

This is an intentionally underpriced #67 Equal-tail:
- Removes `line_feat_u8` 60 B and `line_feat` 240 B.
- Adds the required `color_canvas_u8` `[1,1,30,30]` = 900 B before final `Equal`.
- Does not yet add runtime orientation or inactive-grid sentinel repair.

The required canvas alone makes this family far above 680.

## #67/#68 Enumeration Answer
- Equal-tail / OneHot-style tails require a full value canvas or index canvas so padding can be sentinelled. Minimum U8 canvas is 900 B; INT32/INT64 OneHot indices are worse.
- Comparison / BOOL tails require a charged `[1,10,axis]` channel-comparison condition before the final graph-output op. The underpriced one-axis case is 717 before real orientation and active-grid repair.
- Broadcast-Add / Mul / Min / Where finals need either the same charged channel-comparison plane or a full active/canvas condition. `Where(BOOL,BOOL,BOOL)` is not an ORT kernel; U8 Where routes back through a U8 value canvas plus Equal-tail.
- BitwiseAnd set-intersection is not a clean fit for 10 channel classes in one U8 lane, and still needs orientation plus active/sentinel gating; the minimum charged masks are no cheaper than the comparison tail.
- QLinearConv/#84/#85 is ruled out for this locked hypothesis by traps §8g / ort_reality: the graph input is F32, and a full input Cast to U8 loses before it can replace the final writer.

## Gate Policy
No visible, profile, truth, AB, or safety gates were run on these prototypes because both score above the 680 bar. Per the reviewer request, this stops the ladder before burning full gates.

## Verdict
DROP for the locked final-writer hypothesis. The current #89 F32 `Einsum` final is cheaper because it keeps the active-grid mask and channel comparison inside an uncharged output contraction. The #67/#68 alternatives either materialize the forbidden full canvas/active mask or materialize a charged `[1,10,*]` comparison, and the cheapest underpriced versions are already `717` and `1317`.
