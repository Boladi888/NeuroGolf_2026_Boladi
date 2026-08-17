live bar: cost 100, sha256 91294db3a76db1cf55efd0eeb6e6c475503684a7d282ec1b5611be5a48729242

attempted route: remove the bias input from the one-node grouped Conv team-best, dropping params from 100 to an estimated 90 while keeping the same kernel.

failed gate: `python tools/ng_gate.py visible --task 314 --cand The_400_Tasks/Task_by_Task/task314/prototypes/task314_conv_no_bias.onnx` -> `[visible] 0/266 (0.00%)  FAIL`

why dead: without the per-channel negative threshold, any single same-color endpoint in the dilation-3 Conv neighborhood can fire under the NeuroGolf `>0` decoder; the visible set shows the bias is load-bearing.

one remaining hypothesis: a different single-output op with an explicit threshold source might beat 100 only if it supplies the bias for fewer than 10 charged elements without using unsafe subdeclared Conv bias.
