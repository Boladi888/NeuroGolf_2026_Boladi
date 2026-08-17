# task031 channel passthrough / v_w1 delete

- Live bar used: 400, sha256 `963fb663b0272cda192e00965a1681bfe006b13d5da05c178d6c3dd6dc9d8904`, source `python tools/ng_current_bar.py --task 31 --json` / OnnxlyFans manifest.
- Attempted route: delete the dense `v_w1[10,10]` output-channel/source-channel coupling from Sho's one-node `Einsum`, change the source input term from `nkrc` to direct output-channel passthrough `norc`, and score a 300-param graph.
- Candidate: `The_400_Tasks/Task_by_Task/task031/prototypes/worker3_einsum_passthrough/task031_worker3_channel_passthrough.onnx`, sha256 `841279bd2019051e68b1035611e7218b1073de84040f3fade721b80ae4a39cab`.
- Exact failed gate: `python tools/ng_gate.py visible --task 31 --cand The_400_Tasks/Task_by_Task/task031/prototypes/worker3_einsum_passthrough/task031_worker3_channel_passthrough.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the `v_w1` signed channel-coupling matrix is load-bearing in this contracted formulation; simply carrying the source channel through changes every visible output. Per the visible-fail axiom, no truth/AB/profile follow-up is valid for this family.
- One remaining hypothesis: an exact lower-param decomposition of `v_w1` would need a different output-channel coupling, but the 10x10 matrix is full-rank; prefer small shared-factor rewrites like `v_w8` rather than deleting `v_w1`.
