# task047 hard record — sparse-C initializer probe (2026-07-14, bo_ng_cdx2_worker1)

**Live bar:** cost **60**, sha
`6b007e6d6665d31a941916cb0bdb5b7eb482a88e99164ab1e7db49cdc15b3a3c`
(Vamshi, manifest 2026-07-02; `ng_current_bar.py --task 47 --json`).
Team-best graph is one output `Einsum` with dense `F[3,10]` + dense `C[10,3]`;
`score_network_v2 -> (0, 0, 60)` and `profile_real_cost --arcgen 266 -> static=60 profile=60`.

## DEAD ROUTE — sparse output coefficient matrix for final Einsum

Attempted route: keep dense `F[3,10]` and replace dense `C[10,3]` with a
`SparseTensorProto` initializer containing only its 12 nonzero values. Nominal accounting
would be `30 + 12 = 42`, below live, if the sparse operand were legal.

Prototype:
`The_400_Tasks/Task_by_Task/task047/prototypes/build_task047_sparse_C_probe.py`
builds
`The_400_Tasks/Task_by_Task/task047/prototypes/task047_sparse_C_probe.onnx`.

Exact failed gate:

```text
python tools/ng_gate.py visible --task 47 --cand The_400_Tasks/Task_by_Task/task047/prototypes/task047_sparse_C_probe.onnx
[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_1' is not a graph input, initializer, or output of a previous node.
```

Why dead: the scorer sanitizer renames normal graph inputs/initializers but does not
make the sparse initializer available under the renamed node input. This matches the
library's sparse-initializer warning: the NNZ param discount is unreachable for a real
op consumer under the official scorer path.

No truth/AB/profile run: first gate failed, so this is not a candidate and must not be
staged.

One remaining hypothesis: only a non-sparse algebraic compression of the dense
rank-3 coefficient factorization could beat 60. The prior `perfect25_and_rank2.md`
record already proves rank 2 infeasible; any revisit must name a concrete rank-3
factor sharing that reduces the dense `F[3,10] + C[10,3]` element count without adding a
second graph node or a charged full-canvas intermediate.
