# task132 low-residue truncation dead route

- Live bar: 516, SHA256 `0c6b6d8bbe081c188c1cb4b6f37cfca44e825a3e05fcad1b9be451feeca1b774` from `python tools/ng_current_bar.py --task 132 --json`.
- Attempted route: after the valid three-bucket high zero-tail, truncate the low coordinate bases: modulo-3 3->2 (`task132_v22_h3_zero_tail_p3_2.onnx`, static cost 364) and parity 2->1 (`task132_v22_h3_zero_tail_p2_1.onnx`, static cost 361).
- Failed gates: `p3_2` visible -> `[visible] 0/267 (0.00%)  FAIL`; `p2_1` visible -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: visible pairs exercise all residue information required by the CRT-style coordinate comparison.
- One remaining hypothesis: a different coordinate basis with fewer total params could still beat the CRT `p2+p3` basis, but dropping residue states from this basis is dead.
