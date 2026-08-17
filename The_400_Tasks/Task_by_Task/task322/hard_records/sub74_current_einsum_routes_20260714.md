# task322 sub-74 routes vs current Einsum live — rejected

Live bar used: **74** bytes, owner Vamshi, sha256
`5f9a8513d653a8fd3ea5c74440eb587bce7e2e25e3ae5b96907e99fab8d0fddc`, from
`python tools/ng_current_bar.py --task 322 --json` (`authoritative_source=manifest`,
updated `2026-07-12T15:15:54Z`). Current live graph is one terminal `Einsum`
with zero intermediates and params `R[2,30]=60 + V[10]=10 + T[2,2]=4`; real-input
profile confirmed `static=74 profile=74`.

## Routes attempted

1. **Drop `V` channel vector: same-channel rank-2 Einsum, static cost 64.**
   - Candidate: `prototypes/task322_probe_samechannel64.onnx`
   - SHA-256: `95d9e50b657664e01e509857d4cfd7ebbcae84446b1a0f9dff99630a60218955`
   - Exact failed gate: `python tools/ng_gate.py visible --task 322 --cand The_400_Tasks/Task_by_Task/task322/prototypes/task322_probe_samechannel64.onnx`
   - Result: `[visible] 0/266 (0.00%)  FAIL`
   - Why dead: removing `V=[1,-3,...,-3]` loses the foreground/background signed row summary needed to make channel 0 a cumulative-AND while color channels are cumulative-OR. A bounded no-`V` rank-2 feasibility search did not find a satisfying sign system; the representative graph is visible-zero, so do not build variants in this no-channel-vector family without a new proof.

2. **Sparse `R` initializer: theoretical cost about 20 if Einsum accepted sparse input.**
   - Candidate: `prototypes/task322_probe_sparseR20.onnx`
   - SHA-256: `7b98df4bde00e65239700239061f2f5a3253b46c4f2388a00576a92ba94863ba`
   - Exact failed gate: `python tools/ng_gate.py visible --task 322 --cand The_400_Tasks/Task_by_Task/task322/prototypes/task322_probe_sparseR20.onnx`
   - Result: `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_2' is not a graph input, initializer, or output of a previous node.`
   - Why dead: scorer sanitization renames dense graph inputs/initializers but does not preserve the sparse initializer as the node input. This matches the global sparse-initializer trap; dense `R` is the effective floor for this Einsum.

3. **Single terminal depthwise `ConvTranspose` with `K<=6`, cost <=70.**
   - Artifact: inline LP route; no ONNX emitted because no feasible geometry exists.
   - Exact failed proof: row-0 color must reach output rows 0, 1, and 2, forcing dilation 1 and pad begin `0..K-3`; exhaustive LP over `K=3..6`, stride `1..9`, legal output padding/pad-end, with all 30 output rows constrained found `hits 0`.
   - Why dead: the shared transpose kernel cannot fill lower rows for an object at row 0 without leaking the same tap from row 1/2 into off-grid rows, and cancellation/bias constraints do not make any `K<=6` geometry feasible.

## Remaining hypothesis

Only a different single terminal op or a mathematically proven lower-rank `Einsum` factorization can beat 74. Multi-op Range/ConstantOfShape/Gather rewrites are structurally priced out here: any generated basis, crop, gather result, or mask becomes a counted intermediate and exceeds the zero-intermediate live bar before it can recover the 10-byte saving.

Probe manifest: `prototypes/task322_worker1_20260714_probe_manifest.json`.
