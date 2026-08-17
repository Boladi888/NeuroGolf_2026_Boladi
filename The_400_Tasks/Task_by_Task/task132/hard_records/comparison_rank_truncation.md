# task132 comparison-rank truncation dead route

- Live bar: 516, SHA256 `0c6b6d8bbe081c188c1cb4b6f37cfca44e825a3e05fcad1b9be451feeca1b774` from `python tools/ng_current_bar.py --task 132 --json`.
- Attempted route: after the valid three-bucket high zero-tail, truncate low/high comparison carrier ranks: `f` rank 3->2 (`task132_v22_h3_zero_tail_f2.onnx`, static cost 402) and `l` rank 2->1 (`task132_v22_h3_zero_tail_l1.onnx`, static cost 409).
- Failed gates: `f2` visible -> `[visible] 0/267 (0.00%)  FAIL`; `l1` visible -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: both carrier rank axes are essential to the existing interval-comparison tensor factorization.
- One remaining hypothesis: a new factorization of the same comparison predicate may exist, but simple prefix truncation of either bridge rank is dead.
