# task376 — input → BOOL whole-graph propagation

| Field | Value |
|---|---|
| Task | task376 |
| Family | Object_Based / MID_GATHER |
| Cost before | 42,836 |
| Cost after | 24,626 |
| Bytes saved | -18,210 (-42.5%) |
| Estimated LB | +0.55 |
| Verify | 39/39 (100%) |
| Adversarial | 582/582 (100%) |
| Pattern name | `cast_input_to_bool_gather_and_mask` |
| Staged | `.hermes/board/staged_submits/task376_bool_t_8d70c38f_on_5835_30/` |

## Before

Original graph chain (simplified):

```
input [1, 10, 30, 30] F32                          (36,000 bytes IF counted, but i/o is skipped)
  → Gather(input, ch_indices, axis=1)              gathered [1, K, 30, 30] F32 = K × 3,600 bytes
  → various ops on gathered
  → valid_row_mask_construction → mask F32         [1, 1, 30, 30] = 3,600 bytes
  → Mul(gathered, valid_row_mask)                  intermediate [1, K, 30, 30] F32
  → final ops → output F32
```

Largest intermediate: F32 tensors at `[1, K, 30, 30]` resolution. With K=10 channels gathered, that's 36,000 bytes per intermediate.

## After

```
input [1, 10, 30, 30] F32
  → Cast(input, BOOL)                              input_b [1, 10, 30, 30] BOOL = 9,000 bytes (4× smaller)
  → Gather(input_b, ch_indices, axis=1)            gathered_b [1, K, 30, 30] BOOL = K × 900 bytes
  → various ops on gathered_b (Gather, ReduceMax, etc. all accept BOOL)
  → valid_row_mask_b construction → mask_b [1, 1, 30, 30] BOOL = 900 bytes
  → And(gathered_b, valid_row_mask_b)              instead of Mul; BOOL intermediate, no F32 promotion
  → final ops → output BOOL (declare graph output dtype as BOOL)
```

Every intermediate that was F32 (4 bytes/elem) is now BOOL (1 byte/elem). `run_network` does `(out > 0.0).astype(float)` — BOOL {True, False} produces the same {0, 1} mask, so the output is byte-exact identical to the original under the official decoder.

## Why it works

- The input is one-hot encoded: every cell has exactly one channel set to 1.0 (or zero channels for padding cells). Values are exactly {0.0, 1.0}.
- Cast(input, BOOL) maps {0.0, 1.0} → {False, True} losslessly.
- Every downstream op in the chain accepts BOOL input as ORT-supported:
  - **Gather** — channel selection works on any dtype.
  - **ReduceMax / ReduceSum** — **REJECT BOOL.** This task's actual ONNX **casts to UINT8 before any ReduceMax/ReduceMin** (UINT8 works at opset 13+) and **casts to F16 or INT32 before any ReduceSum** (Sum/Mean reject BOOL/UINT8/INT8). The earlier claim "ReduceMax stays BOOL" was wrong — the BOOL chain must break at Reduce boundaries. See `ort_reality.md §7` for the per-reducer dtype table.
  - **Where(c, x, y)** — accepts BOOL `c` always; here `x` and `y` were originally `F32 mask, 0_F32`; with the rewrite, they become `BOOL mask, False_BOOL` via the `And` replacement (NOT via Where directly — Where(BOOL,BOOL,BOOL) is rejected; rewrite uses Or(And(c,x), And(Not(c), y))).
  - **Slice, Concat, Reshape, Transpose** — dtype-agnostic.
  - **And/Or/Not** — native BOOL ops.
- `Mul(gathered, mask)` was the only F32-requiring node. Replacing with `And(gathered_b, mask_b)` keeps BOOL throughout.

## Generalizes to

`techniques.md` rule 8 (input → BOOL whole-graph). **Two tracks now confirmed (2026-05-22 update)**:

### Track A — input→BOOL whole-graph (the original task376 pattern, more restrictive)

Propagates BOOL throughout the entire graph. Requires bool-tolerance on every internal op.

**Eligibility checklist (Track A)**:
- [ ] Model input is one-hot encoded (NeuroGolf default: F32 [1,10,30,30] with 0/1 values per channel). True by definition.
- [ ] Downstream chain contains NO Conv, MatMul, Gemm, or F32 arithmetic ops (Add/Sub/Mul/Div) on the input lineage.
- [ ] Every Mul that gates a value can be rewritten as And or Where.
- [ ] The output's `(out > 0.0)` thresholding is identity on the new dtype.

Eligibility: ~10% of NeuroGolf tasks (MID_GATHER family, simple Gather+Mask chains). Saves bytes throughout (4× compression from F32→BOOL).

### Track B — output→BOOL declaration only (the task069 generalization, much broader)

Declares only the GRAPH OUTPUT as BOOL. Internals stay in any dtype (F16/F32/INT8/etc.).

**Eligibility checklist (Track B)**:
- [ ] Output values are deterministically {0, 1} after the final op (one-hot decoded).
- [ ] Final Pad to [1, 10, 30, 30] (or task's expected output shape) is preserved.
- [ ] Downstream `run_network` decoder applies `(out > 0.0).astype(float)` thresholding.

Eligibility: ~70% of NeuroGolf tasks. Internals can include Conv, MatMul, F32 arithmetic, ConvTranspose — none of those disqualify Track B.

Saves: only the final Cast(intermediate→F32) op + the F32-output dtype declaration. Typical savings ~18-25k bytes (same magnitude as the F16-output trick in `techniques.md` §3, but deeper compression because BOOL is 1 byte/elem vs F16's 2).

**Cross-task confirmation chain (2026-05-22)**:
- task376 (MID_GATHER, no Conv on input lineage): Track A applied, +0.55 LB clean
- task069 (Crop+Conv+ConvTranspose internals): Track B applied, **+2.54 LB clean** (Kaggle 2026-05-22T20:44Z)

If Track A's checklist fails because of Conv/MatMul on input lineage, **try Track B first** — it's much wider applicable.

## Artifacts

- Manifest: `.hermes/board/staged_submits/task376_bool_t_8d70c38f_on_5835_30/manifest.json` (the `method` field has the worker's notes in full).
- Candidate ONNX: `.hermes/board/staged_submits/task376_bool_t_8d70c38f_on_5835_30/task376.onnx`.
- Build prototypes: `The_400_Tasks/Task_by_Task/task376/prototypes/` (also has `build_task376_crop17.py`, `build_task376_crop21h.py`, etc. — earlier crop variants the worker rejected in favor of the bool whole-graph).

## Reproducibility

To apply this to another task:

1. Load the canonical ONNX. Inventory ops in order via `onnx.load(...).graph.node`.
2. Verify NO Conv, MatMul, Gemm, or F32 arithmetic ops on the input lineage. If any exist, this technique is disqualified — try crop-to-active-grid or other techniques first.
3. Insert `Cast(input, BOOL)` as the first node. Rename its output (e.g., `input_b`).
4. Walk the graph and rewire every subsequent node that previously consumed `input` to now consume `input_b`.
5. Replace any `Mul(gathered, mask)` with `And(gathered_b, mask_b)`.
6. Declare `graph.output[0].type.tensor_type.elem_type = TensorProto.BOOL`.
7. `graph.ClearField("value_info")` — ORT will fail to load otherwise (stale F32 cached shapes).
8. `verify_against_live` must return 100% M/M.
9. `_score_v2 --adversarial` — run and report, but **advisory only** per `core.md` 2026-05-22 update. Empirical: task064 candidates shipped at 7-9% adversarial for +1.20 LB. Gate applies only for documented hidden-correctness-sensitive tasks (task008 style).

If verify fails on one or two pairs, check for non-bool-tolerant ops you missed (especially F32 thresholds or comparisons). Surgery may need an explicit `Cast(BOOL, F32)` insertion at the boundary if a downstream consumer truly needs F32.
