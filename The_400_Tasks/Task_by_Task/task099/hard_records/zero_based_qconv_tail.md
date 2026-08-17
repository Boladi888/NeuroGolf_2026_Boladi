# task099 hard record: zero-based QLinearConv tail

- Live bar used: 855 bytes, SHA-256 `5cb6de54a603f7678251fea24985cb2ecd45392e5004be8c201c0efa6f25aaa3`, from `python tools/ng_current_bar.py --task 99 --json` authoritative manifest source.
- Attempted route: `The_400_Tasks/Task_by_Task/task099/prototypes/build_zero_based_qconv_tail.py` removed `rl_p1 = render_labels + 1` and retuned the final `QLinearConv` to read zero-based labels.
- Candidate: `The_400_Tasks/Task_by_Task/task099/prototypes/task099_zero_based_qconv_tail.onnx`, SHA-256 `93d7ebe37d31d6ce295b3eb0b34003b62cbc9e467a2f9890f6d52c39e9c2b64e`.
- Exact failed gate: `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: the final `QLinearConv` implicit padded area also has label 0, so active background label 0 and outside-pad label 0 collapse together. The comparator requires the outside 20-row/20-col output pad to remain all-zero across one-hot channels, but this route lights the background channel outside the active 10x10 region. A nonzero input-zero-point sentinel variant was checked conceptually/brute-force over plausible sentinel shifts; the two-feature parabola encoding did not separate all ten labels plus the pad sentinel.
- One remaining hypothesis: a different final renderer that avoids materializing `u_grid`/`u_sq`/`feat2` entirely may lower the floor; zero-based QConv-tail variants should not be re-explored without a new pad-sentinel separation mechanism.
