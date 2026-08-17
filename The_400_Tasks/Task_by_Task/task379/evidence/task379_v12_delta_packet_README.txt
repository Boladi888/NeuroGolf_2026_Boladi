task379_v12_fp32_bool_shift_masks delta packet for review

Candidate ONNX: task379.onnx
Candidate SHA256: 37bffe749c8c1200ad8a2ef7d0ed9dd9c9b6f7fc631b3e109f012a8b1d8e155f
Worker-base package SHA256: 5ecc4a4be545e9bc6665dfba6f716afa73096362efc46dad70703af953e539fa
Current canonical package SHA256: 99f2ea256c20e0382a78c4566392c89e801ee39aa8b8221d8f4d50017544be14
Current-canonical candidate package SHA256: a60d9a0ddddcb281e8973709a6b3fa54646c84a130a84932c936017fae823a43

Cost: 108414 -> 94646 (delta -13768; predicted points 13.4063 -> 13.5421, +0.1358).
Parity: standard 266/266 exact, extended 3200/3200 exact, verify_against_live prototype-vs-live 266/266.

Rewrite: v10-safe FP32/boolean lane plus exact bool Slice/Concat row-shift replacement for the two 2x1 Conv masks and bool Or unions for v70/v76. Avoids FP16, zero-channel replacement, final mask-union rewrite, and boundary-cast sinking.
