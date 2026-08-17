live bar: cost 100, sha256 0bfa37796a89ad38a7655d8e98dbb256aba81674632133fbae9b8b9a137c49fe (ng_current_bar.py, manifest authority, Vamshi, 2026-07-12T15:15:54Z)

attempted route: `prototypes/task318_v5_group5_underroute.onnx`, a sub-100 direct free-output Conv using `group=5`, weight shape `[10,2,2,1]`, dilation `[5,1]`, theoretical dense params 40.

failed gate: `python tools/ng_gate.py visible --task 318 --cand The_400_Tasks/Task_by_Task/task318/prototypes/task318_v5_group5_underroute.onnx` -> `[visible] 4/267 (1.50%)  FAIL`.

why dead: Conv group partitioning is binding. With `group=5`, output channel 3 can see only global input channels 2 and 3, so it cannot route top-half color 1 into output color 3. A visible miss is a certain hidden zero.

one remaining hypothesis: dense Conv needs `group=2` so output channels 0 and 3 can both read input channels 0, 1, and 2; with two vertical taps that forces `10*5*2*1 = 100` params unless a different legal output op replaces Conv.
