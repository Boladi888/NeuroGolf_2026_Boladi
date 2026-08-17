# task263 sparse initializer shave

Live/team bar: 472 cost, SHA-256 `8ddcdb4887dc6ad862198b673ec921675d830a239844bb40edf1b96a08ac7ea7`, source `Team_Submission/Backend/manifest.csv` and workbook row `Tasks!265`.

Attempted route: replace dense `zero_chan_u8` `[1,10,1,1]` with a sparse initializer containing the single nonzero background-channel value. If accepted, this would save 9 params.

Exact failed gate: `python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker2_sparse_zero_chan.onnx` -> `ERROR: ... Invalid model. Node input 'safe_name_59' is not a graph input, initializer, or output of a previous node.`

Why dead: the scorer sanitizer does not preserve sparse initializer names as valid node inputs, so even a sparse Where branch is unshippable. Do not retry sparse `sig_W`; Conv sparse weights are already globally documented as rejected and would hit the same sanitizer/checker class.

One remaining hypothesis: a win must reduce dense runtime tensors or dense initializer element counts without sparse initializers.
