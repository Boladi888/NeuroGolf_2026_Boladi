# task302 worker1 direct feature-fold probes

- **Live bar**: 1574, Vamshi, sha256 `366237800121a80b62e22e725646ec0d3c7a6a56a5ec3dd71a57d350e6498528`, source `ng_current_bar.py --task 302 --json` / team manifest.
- **Effective local reviewer bar**: 1573, `staged_candidates/task302_regionfillrewrite_cost1573`, sha256 `f39b26ef41e9d40cb7cfd1ecb511d289e57af66377cbb0e188ef87669b21937e`.
- **Attempted route**: delete or replace the charged feature carrier below 1573 by keeping the ch5 crop/cast and replacing the `frame5_bin -> fill8 -> Concat` stack with direct final classifiers or a no-bias final classifier.
- **Failed gate**: builder feasibility, before ONNX gating.
  - `build_task302_no_bias_final.py`: deleting `final_b` with final `x_zero=1` failed channel 0 MILP as infeasible on 1377 feature patches. A wider x-zero scan was capped and produced no artifact; it is not a staged candidate.
  - `build_task302_direct9x9_qconv.py`: direct binary 9x9 over `ch5_u8` failed channel 0 MILP as infeasible even on the 266 official pairs. A larger sampled training set reached the MILP time cap before producing an artifact.
  - `build_task302_tristate8x8_qconv.py`: tri-state 8x8 (`2*ch5_u8`, final `x_zero=1`) failed channel 0 MILP as infeasible on the 266 official pairs for both even-kernel alignments (`pad_head=3` and `pad_head=4`).
- **Why dead**: channel 0/background is not linearly separable from border/fill/outside cases in any tested one-plane direct window that still prices below 1573. The existing two-feature stack carries real rank, not only redundant memory.
- **One remaining hypothesis**: a non-linear low-memory replacement for `fill8` might still work if it creates a rank-2 signal without materializing a full second 12x12 feature plane; any version that adds a full U8 plane must shrink final params enough to offset the 144-byte memory cost.
