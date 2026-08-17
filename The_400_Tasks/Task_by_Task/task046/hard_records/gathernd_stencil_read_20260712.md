live bar: 1574, sha d785ffe94a9e8b0147f9a7d3ce79f77d4056c4f8a54c8066d2fc268f1ed32276 (`python tools/ng_current_bar.py --task 46 --json`, manifest authority, owner Sho).

Attempted route: prompt-suggested GatherND/stencil-read formulation: sample endpoint/corner/column-summary features instead of materializing the 3x19 color surface, then reconstruct the snake.

Failed gate: feature-vector uniqueness probe, before ONNX build. Command: `python The_400_Tasks/Task_by_Task/task046/prototypes/probe_task046_gathernd_stencil_floor.py`. Result: `column_summary_equal=true`, `outputs_equal=false`, first diff `{row:0,col:1,a:2,b:0}` for two valid generator instances with the same width, occupied/gap columns, gray-joint rows, and per-column non-gray max color.

Why dead: task046's output depends on each segment's interior turn column. Endpoint/corner/column features do not identify that turn. The only fixed GatherND route that is information-complete must color-decode all 3*19 candidate cells by reading all 10 one-hot channels: 570 F32 reads, 1710 INT64 index params + 2280 B output = 3990 B before the decision tail. The Sho live graph's Conv color decode for the same 3x19 surface is 228 B F32 + 57 B U8 + 10 params = 295 B, and the whole live graph is 1574.

One remaining hypothesis: a non-GatherND rewrite that keeps the Sho full-surface Conv decode but fuses row-roll and final one-hot emission could still shave the tail; a sparse fixed stencil route needs new information, not just endpoints or column summaries.
