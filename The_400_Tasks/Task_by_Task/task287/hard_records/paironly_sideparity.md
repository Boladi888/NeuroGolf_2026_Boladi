# paironly_sideparity dead route

Live bar used: cost 190, sha256 `d8a937d60fabb9a7e847cf1ec19fa2425e7157607022d94bb3cf51fb0e8e50a2` from `python tools/ng_current_bar.py --task 287 --json` / manifest.

Attempted route: encode the observed lower-level rule `threshold(non4_mask * (input + rot180(input)))` as a single terminal `Einsum`, adding a side-parity feature so the shipped full-orbit relation only keeps identity plus 180-degree partners.

Prototype: `The_400_Tasks/Task_by_Task/task287/prototypes/task287_paironly_sideparity_v1.onnx`; builder: `The_400_Tasks/Task_by_Task/task287/prototypes/build_task287_paironly_sideparity.py`.

Exact gates:
- builder/checker: `saved ... task287_paironly_sideparity_v1.onnx bytes=1342`
- visible: `[visible] 267/267 (100.00%)  PASS`
- score: `score_network_v2 -> (0, 0, 250)`, cost 250

Why dead: the terminal graph is correct on visible officials, but the side-parity selector adds `side[30,2]` = 60 params to the shipped `low[30,4] + high[30,2] + mask[10]` structure. It lands at 250, worse than the current 190 cost/profile-cost bar, so it is not stageable and should not receive truth/ab heavy gates.

One remaining hypothesis: beat 190 only by replacing the forced 8-class row/col equality code itself; any route that adds a side/parity selector on top of the shipped full-orbit code is dead by construction.
