live bar: manifest cost 90, sha256 3ca372ef43947b9f7b0abf5d12ea7a6dadd0956b176bab87a3e408f35639309e (`python tools/ng_current_bar.py --task 26 --json`)

attempted route: replace Sho's final 10-param `Conv(signed, w_out)->output` with `ConvTranspose(signed, w_pair)->output`, reusing the first `Conv`'s `[1,10,1,2]` initializer so the static cost drops to 80 (`memory 60 + params 20`).

exact failed gate: `python tools/ng_gate.py visible --task 26 --cand The_400_Tasks/Task_by_Task/task026/prototypes/task026_shared_convtranspose.onnx` -> `[visible] 0/268 (0.00%)  FAIL`.

why dead: the shared tensor couples first-stage input-channel weights to final output-channel weights. Any nonzero weight needed to read foreground color 9 also becomes a positive/negative channel-9 output weight in the free tail; because the signed carrier must take both signs to separate color 0 vs color 8, channel 9 inevitably lights for some visible cases. Boundary columns also force the same contradiction. Cost diff confirmed only the `w_out` deletion (`-10 params`); correctness is the blocker.

one remaining hypothesis: a different final op that can both pad to `[1,10,30,30]` and unpack a packed row/bit code in the output node could beat 90, but `Conv`/`ConvTranspose` are linear and `BitwiseAnd`/`Greater` cannot also perform the free spatial pad without a large constant or charged pre-pad canvas.
