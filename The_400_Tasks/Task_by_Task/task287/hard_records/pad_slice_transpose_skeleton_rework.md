# pad_slice_transpose_skeleton_rework dead route

Live bar used: cost/profile cost 190, sha256 `d8a937d60fabb9a7e847cf1ec19fa2425e7157607022d94bb3cf51fb0e8e50a2` from `python tools/ng_current_bar.py --task 287 --json` / manifest.

Reviewer-allowed route: build a Pad-wrap/Slice/Transpose skeleton only if it deletes `low/high` or replaces them with a strictly cheaper charged structure.

Attempted route: delete the live `low[30,4] + high[30,2]` basis entirely and emit a one-node terminal 180-degree full-canvas `Slice`. Charged structure is only four INT64 Slice vectors: `starts[2]`, `ends[2]`, `axes[2]`, `steps[2]` = 8 params, zero memory if correct.

Prototype: `The_400_Tasks/Task_by_Task/task287/prototypes/task287_slice180_skeleton_v1.onnx`; builder: `The_400_Tasks/Task_by_Task/task287/prototypes/build_task287_slice180_skeleton.py`.

Exact gates:
- builder/checker: `saved ... task287_slice180_skeleton_v1.onnx bytes=262`
- visible: `[visible] 0/267 (0.00%)  FAIL`

Why dead: a pure terminal spatial skeleton deletes `low/high`, but it cannot perform the task287 completion rule. The full-canvas reverse moves the top-left 16x16 active grid into the padded region, and no Pad-wrap/Slice/Transpose-only skeleton suppresses channel 4 or unions the source with its 180-degree partner. A crop-first or Pad-wrap-on-active variant would need a pre-output crop/Pad/Where/Add tensor; the smallest useful active one-hot crop is `[1,10,16,16]` F32 = 10,240 bytes before output, far above the 190 live bar. Profile/truth/ab were not run after the visible fail.

One remaining hypothesis: only a terminal one-node contraction that both reads the F32 input and encodes the eight mirror classes can compete. Pad-wrap/Slice/Transpose can only replace fixed spatial movement, not the axis-code membership relation plus channel-4 suppression.
